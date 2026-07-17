"""Modern PyTorch training, DDP execution, checkpoint continuation, and recovery."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import time
from collections.abc import Iterator, Sized
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler

from .checkpoints import load_checkpoint, random_state, save_checkpoint
from .distributed import (
    DistributedContext,
    synchronize_module_buffers,
    unwrap_model,
    wrap_ddp,
)
from .exceptions import ConfigurationError
from .metrics import ClassificationAccumulator
from .types import CostRates, DatasetBundle, ResourceBudget, TrialResult


class ExactDistributedSampler(Sampler[int]):
    """Shard evaluation data without padding or duplicating examples."""

    def __init__(self, dataset: Sized, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = max(0, len(self.dataset) - self.rank)
        return math.ceil(remaining / self.world_size)


def resolve_trial_config(
    adapter: Any, model_config: dict[str, Any], dataset: DatasetBundle
) -> dict[str, Any]:
    """Resolve derived adapter values before validation, training, and serialization."""

    resolver = getattr(adapter, "resolve_config", None)
    if callable(resolver):
        config = dict(resolver(model_config))
    else:
        merger = getattr(adapter, "merged", None)
        config = (
            dict(merger(model_config)) if callable(merger) else {**adapter.baseline, **model_config}
        )
    config["num_classes"] = dataset.metadata.num_classes
    config["image_size"] = dataset.metadata.image_size[0]
    return config


def stable_configuration_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def seed_everything(seed: int, reproducibility: str = "balanced", rank: int = 0) -> None:
    effective_seed = seed + rank
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)
    if reproducibility == "strict":
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    elif reproducibility == "fast":
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
    else:
        torch.use_deterministic_algorithms(False)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True


def git_state(cwd: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None


def _per_device_batch_size(
    config: dict[str, Any], execution: dict[str, Any], distributed: DistributedContext
) -> tuple[int, int]:
    requested = int(config.get("batch_size", 128))
    if requested <= 0:
        raise ConfigurationError("batch_size must be positive")
    scope = str(execution.get("batch_size_scope", "global")).lower()
    accumulation = max(1, int(config.get("gradient_accumulation", 1)))
    if scope not in {"global", "per_gpu", "per_device"}:
        raise ConfigurationError("execution.batch_size_scope must be global or per_gpu")
    if distributed.world_size > 1 and scope == "global":
        if requested % distributed.world_size != 0:
            raise ConfigurationError(
                f"global batch_size={requested} must be divisible by world_size={distributed.world_size}"
            )
        per_device = requested // distributed.world_size
        global_batch = requested * accumulation
    else:
        per_device = requested
        global_batch = requested * distributed.world_size * accumulation
    if per_device <= 0:
        raise ConfigurationError("per-device batch size resolved to zero")
    return per_device, global_batch


def _worker_seed(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _loader(
    dataset: Dataset[Any],
    config: dict[str, Any],
    execution: dict[str, Any],
    device: torch.device,
    train: bool,
    distributed: DistributedContext | None = None,
) -> tuple[DataLoader[Any], Sampler[int] | None]:
    context = distributed or DistributedContext()
    workers = int(config.get("num_workers", execution.get("num_workers", 0)))
    pin = bool(execution.get("pin_memory", device.type == "cuda")) and device.type == "cuda"
    per_device_batch, _ = _per_device_batch_size(config, execution, context)
    sampler: Sampler[int] | None = None
    shuffle = train
    if context.enabled:
        if train:
            sampler = DistributedSampler(
                dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=True,
                seed=int(execution.get("split_seed", 42)),
                drop_last=bool(config.get("drop_last", False)),
            )
        else:
            sampler = ExactDistributedSampler(dataset, context.rank, context.world_size)
        shuffle = False
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": per_device_batch,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": workers,
        "pin_memory": pin,
        "drop_last": train and bool(config.get("drop_last", False)),
        "persistent_workers": workers > 0 and bool(execution.get("persistent_workers", True)),
        "worker_init_fn": _worker_seed,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = int(execution.get("prefetch_factor", 2))
        process_context = execution.get("multiprocessing_context")
        if process_context:
            kwargs["multiprocessing_context"] = str(process_context)
    return DataLoader(**kwargs), sampler


def _precision(device: torch.device, requested: str) -> tuple[bool, torch.dtype | None]:
    requested = requested.lower()
    if requested in {"auto", ""}:
        if device.type == "cuda":
            requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            requested = "fp32"
    if requested == "fp32":
        return False, None
    if requested == "bf16" and device.type in {"cuda", "cpu"}:
        return True, torch.bfloat16
    if requested == "fp16" and device.type == "cuda":
        return True, torch.float16
    raise ConfigurationError(f"precision {requested!r} is unsupported on {device.type}")


def _autocast(device: torch.device, enabled: bool, dtype: torch.dtype | None):
    if not enabled:
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=True)


def _budget_reached(
    budget: ResourceBudget,
    start_time: float,
    epoch: int,
    steps: int,
) -> bool:
    if budget.kind == "epochs":
        return epoch >= int(math.ceil(budget.value))
    if budget.kind == "steps":
        return steps >= int(math.ceil(budget.value))
    return time.perf_counter() - start_time >= budget.value


def _distributed_max(value: float, device: torch.device, context: DistributedContext) -> float:
    tensor = torch.tensor(value, dtype=torch.float64, device=device)
    context.all_reduce(tensor, dist.ReduceOp.MAX)
    return float(tensor.item())


def _distributed_sum(
    value: float | int, device: torch.device, context: DistributedContext
) -> float:
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    context.all_reduce(tensor, dist.ReduceOp.SUM)
    return float(tensor.item())


def _rank_random_states(context: DistributedContext) -> list[dict[str, Any]] | None:
    if not context.enabled:
        return None
    gathered: list[dict[str, Any] | None] = [None] * context.world_size
    dist.all_gather_object(gathered, random_state())
    return [state for state in gathered if state is not None]


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    max_batches: int | None = None,
    distributed: DistributedContext | None = None,
) -> tuple[float, float, float]:
    context = distributed or DistributedContext()
    synchronize_module_buffers(model, context)
    evaluation_model = unwrap_model(model) if context.enabled else model
    evaluation_model.eval()
    accumulator = ClassificationAccumulator(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            with _autocast(device, amp_enabled, amp_dtype):
                logits = evaluation_model(images)
                loss = criterion(logits, labels)
            accumulator.update(logits, labels, loss)
    accumulator.synchronize(context)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if context.enabled:
        elapsed = _distributed_max(elapsed, device, context)
    return accumulator.mean_loss, accumulator.accuracy, elapsed


def run_trial(
    adapter: Any,
    model_config: dict[str, Any],
    dataset: DatasetBundle,
    budget: ResourceBudget,
    *,
    study_name: str,
    profile: str,
    stage: str,
    trial_id: str,
    seed: int,
    device: torch.device,
    execution: dict[str, Any],
    cost_rates: CostRates,
    checkpoint_path: Path,
    rung: int | None = None,
    resume_checkpoint: Path | None = None,
    max_train_steps: int | None = None,
    max_validation_batches: int | None = None,
    evaluate_test: bool = False,
    distributed: DistributedContext | None = None,
    assigned_gpu_indices: tuple[int, ...] = (),
) -> TrialResult:
    context = distributed or DistributedContext()
    seed_everything(seed, str(execution.get("reproducibility", "balanced")), context.rank)
    config = resolve_trial_config(adapter, model_config, dataset)
    scaling = str(execution.get("learning_rate_scaling", "none")).lower()
    if scaling not in {"none", "linear", "sqrt"}:
        raise ConfigurationError("execution.learning_rate_scaling must be none, linear, or sqrt")
    if context.world_size > 1 and scaling != "none" and "learning_rate" in config:
        base_learning_rate = float(config["learning_rate"])
        factor = float(context.world_size) if scaling == "linear" else math.sqrt(context.world_size)
        config["base_learning_rate"] = base_learning_rate
        config["learning_rate"] = base_learning_rate * factor
        config["learning_rate_scaling"] = scaling
    adapter.validate_config(config)
    _, global_batch_size = _per_device_batch_size(config, execution, context)
    base_model = adapter.build_model(config)
    description = adapter.describe_model(base_model, config)
    model: nn.Module = base_model.to(device)
    if device.type == "cuda" and bool(execution.get("channels_last", adapter.name == "cnn")):
        model = model.to(memory_format=torch.channels_last)
    model = wrap_ddp(model, context, device)
    compile_enabled = bool(execution.get("compile", False)) and hasattr(torch, "compile")
    if (
        context.enabled
        and compile_enabled
        and not bool(execution.get("compile_distributed", False))
    ):
        compile_enabled = False
    if compile_enabled:
        model = torch.compile(model, mode=str(execution.get("compile_mode", "default")))
    train_loader, train_sampler = _loader(dataset.train, config, execution, device, True, context)
    validation_loader, _ = _loader(dataset.validation, config, execution, device, False, context)
    test_loader = None
    if dataset.test is not None:
        test_loader, _ = _loader(dataset.test, config, execution, device, False, context)
    optimizer = adapter.build_optimizer(model, config)
    steps_per_epoch = max(
        1, math.ceil(len(train_loader) / int(config.get("gradient_accumulation", 1)))
    )
    scheduler = adapter.build_scheduler(optimizer, config, budget, steps_per_epoch)
    amp_enabled, amp_dtype = _precision(device, str(execution.get("precision", "auto")))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16 and device.type == "cuda"
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    start_epoch = 0
    optimization_steps = 0
    examples_processed = 0
    best_accuracy = float("-inf")
    best_loss = float("inf")
    best_epoch = -1
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None and resume_checkpoint.exists():
        context.barrier()
        metadata = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
            random_state_index=context.rank if context.enabled else None,
        )
        start_epoch = int(metadata.get("epoch", 0))
        optimization_steps = int(metadata.get("optimization_steps", 0))
        examples_processed = int(metadata.get("examples_processed", 0))
        best_accuracy = float(metadata.get("best_accuracy", best_accuracy))
        best_loss = float(metadata.get("best_loss", best_loss))
        best_epoch = int(metadata.get("best_epoch", best_epoch))
        history = list(metadata.get("history", []))
        context.barrier()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    process = psutil.Process(os.getpid())
    peak_cpu = process.memory_info().rss / 2**20
    started = time.perf_counter()
    data_seconds = compute_seconds = evaluation_seconds = checkpoint_seconds = 0.0
    early_patience = int(config.get("early_stopping_patience", 0))
    epochs_without_improvement = 0
    training_loss = training_accuracy = None
    test_loss = test_accuracy = None
    status = "completed"
    failure_reason: str | None = None
    git_commit, git_dirty = git_state(Path.cwd())
    _, global_batch_size = _per_device_batch_size(config, execution, context)

    try:
        epoch = start_epoch
        stop = False
        while not stop:
            reached = _budget_reached(budget, started, epoch, optimization_steps)
            if context.enabled:
                reached = context.broadcast_bool(reached if context.is_main else False, device)
            if reached:
                break
            epoch += 1
            if isinstance(train_sampler, DistributedSampler):
                train_sampler.set_epoch(epoch)
            model.train()
            accumulator = ClassificationAccumulator(device)
            optimizer.zero_grad(set_to_none=True)
            accumulation = max(1, int(config.get("gradient_accumulation", 1)))
            fetch_started = time.perf_counter()
            for batch_index, (images, labels) in enumerate(train_loader):
                data_seconds += time.perf_counter() - fetch_started
                if max_train_steps is not None and optimization_steps >= max_train_steps:
                    stop = True
                    break
                if budget.kind == "steps" and optimization_steps >= int(math.ceil(budget.value)):
                    stop = True
                    break
                if budget.kind == "seconds":
                    time_stop = time.perf_counter() - started >= budget.value
                    if context.enabled:
                        time_stop = context.broadcast_bool(
                            time_stop if context.is_main else False, device
                        )
                    if time_stop:
                        stop = True
                        break
                images = images.to(device, non_blocking=device.type == "cuda")
                labels = labels.to(device, non_blocking=device.type == "cuda")
                if device.type == "cuda" and bool(
                    execution.get("channels_last", adapter.name == "cnn")
                ):
                    images = images.contiguous(memory_format=torch.channels_last)
                should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(
                    train_loader
                )
                sync_context = contextlib.nullcontext()
                if context.enabled and not should_step and hasattr(model, "no_sync"):
                    sync_context = model.no_sync()  # type: ignore[union-attr]
                compute_started = time.perf_counter()
                with sync_context, _autocast(device, amp_enabled, amp_dtype):
                    logits = model(images)
                    loss = criterion(logits, labels) / accumulation
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at epoch {epoch}, batch {batch_index}"
                        )
                    scaler.scale(loss).backward()
                if should_step:
                    gradient_clip = float(config.get("gradient_clip", 0.0))
                    if gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()
                    optimization_steps += 1
                compute_seconds += time.perf_counter() - compute_started
                accumulator.update(logits, labels, loss * accumulation)
                peak_cpu = max(peak_cpu, process.memory_info().rss / 2**20)
                fetch_started = time.perf_counter()
            accumulator.synchronize(context)
            training_loss, training_accuracy = accumulator.mean_loss, accumulator.accuracy
            examples_processed += accumulator.example_count
            validation_loss, validation_accuracy, eval_time = evaluate(
                model,
                validation_loader,
                criterion,
                device,
                amp_enabled,
                amp_dtype,
                max_validation_batches,
                context,
            )
            evaluation_seconds += eval_time
            improved = (validation_accuracy > best_accuracy) or (
                validation_accuracy == best_accuracy and validation_loss < best_loss
            )
            if improved:
                best_accuracy, best_loss, best_epoch = validation_accuracy, validation_loss, epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "training_accuracy": training_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "optimization_steps": optimization_steps,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            rank_states = _rank_random_states(context)
            checkpoint_started = time.perf_counter()
            if context.is_main:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    {
                        "epoch": epoch,
                        "optimization_steps": optimization_steps,
                        "examples_processed": examples_processed,
                        "best_accuracy": best_accuracy,
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "history": history,
                        "config": config,
                        "world_size": context.world_size,
                        "global_batch_size": global_batch_size,
                    },
                    rank_random_states=rank_states,
                )
            context.barrier()
            checkpoint_seconds += time.perf_counter() - checkpoint_started
            early_stop = early_patience > 0 and epochs_without_improvement >= early_patience
            if context.enabled:
                early_stop = context.broadcast_bool(
                    early_stop if context.is_main else False, device
                )
            if early_stop:
                break
        if evaluate_test:
            if test_loader is None:
                raise ConfigurationError(
                    "test evaluation requested but the dataset has no test split"
                )
            test_loss, test_accuracy, eval_time = evaluate(
                model, test_loader, criterion, device, amp_enabled, amp_dtype, distributed=context
            )
            evaluation_seconds += eval_time
    except KeyboardInterrupt:
        status = "interrupted"
        failure_reason = "keyboard interrupt"
    except torch.cuda.OutOfMemoryError as exc:
        status = "failed"
        failure_reason = f"CUDA out of memory: {exc}"
    except (RuntimeError, FloatingPointError, ConfigurationError) as exc:
        status = "failed"
        failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak_device = (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        )
        if context.enabled:
            elapsed = _distributed_max(elapsed, device, context)
            peak_device = _distributed_max(peak_device, device, context)
            peak_cpu = _distributed_max(peak_cpu, device, context)
            data_seconds = _distributed_max(data_seconds, device, context)
            compute_seconds = _distributed_max(compute_seconds, device, context)
            evaluation_seconds = _distributed_max(evaluation_seconds, device, context)
            checkpoint_seconds = _distributed_max(checkpoint_seconds, device, context)

    gpu_count = context.world_size if device.type == "cuda" else 0
    gpu_hours = elapsed / 3600 * gpu_count
    cpu_threads = max(1, int(execution.get("cpu_threads", execution.get("intraop_threads", 1))))
    cpu_hours = elapsed / 3600 * cpu_threads * context.world_size
    checkpoint_size_mb = checkpoint_path.stat().st_size / 2**20 if checkpoint_path.exists() else 0.0
    storage_cost = checkpoint_size_mb / 1024 * cost_rates.storage_gb_month_usd
    cost = gpu_hours * cost_rates.gpu_hour_usd + cpu_hours * cost_rates.cpu_hour_usd + storage_cost
    throughput = examples_processed / max(compute_seconds + data_seconds, 1e-9)
    estimated_flops = (
        description.flops_per_forward * examples_processed * 3
        if description.flops_per_forward is not None
        else None
    )
    result = TrialResult(
        study_name=study_name,
        adapter=adapter.name,
        profile=profile,
        stage=stage,
        rung=rung,
        trial_id=trial_id,
        status=status,
        architecture_id=description.architecture_id,
        configuration_id=stable_configuration_id(config),
        config=config,
        budget={"kind": budget.kind, "value": budget.value},
        seed=seed,
        device=str(device),
        validation_accuracy=None if best_accuracy == float("-inf") else best_accuracy,
        validation_loss=None if best_loss == float("inf") else best_loss,
        training_accuracy=training_accuracy,
        training_loss=training_loss,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        best_epoch=None if best_epoch < 0 else best_epoch,
        epochs_completed=len(history),
        optimization_steps=optimization_steps,
        examples_processed=examples_processed,
        elapsed_seconds=elapsed,
        data_loading_seconds=data_seconds,
        compute_seconds=compute_seconds,
        evaluation_seconds=evaluation_seconds,
        checkpoint_seconds=checkpoint_seconds,
        throughput_examples_per_second=throughput,
        parameter_count=description.parameter_count,
        trainable_parameter_count=description.trainable_parameter_count,
        flops_per_forward=description.flops_per_forward,
        estimated_training_flops=estimated_flops,
        peak_memory_mb=peak_device,
        peak_cpu_memory_mb=peak_cpu,
        cpu_hours=cpu_hours,
        gpu_hours=gpu_hours,
        estimated_cost_usd=cost,
        checkpoint_path=str(checkpoint_path),
        checkpoint_size_mb=checkpoint_size_mb,
        failure_reason=failure_reason,
        git_commit=git_commit,
        git_dirty=git_dirty,
        history=history,
        worker_pid=os.getpid(),
        gpu_indices=list(assigned_gpu_indices),
        world_size=context.world_size,
        global_batch_size=global_batch_size,
        parallel_mode="ddp" if context.enabled else str(execution.get("parallel_mode", "serial")),
    )
    del (
        model,
        base_model,
        optimizer,
        scheduler,
        scaler,
        train_loader,
        validation_loader,
        test_loader,
    )
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_checkpoint(
    adapter: Any,
    config: dict[str, Any],
    dataset: DatasetBundle,
    checkpoint: Path,
    device: torch.device,
    execution: dict[str, Any],
) -> tuple[float, float]:
    model = adapter.build_model(config).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    if dataset.test is None:
        raise ConfigurationError("test evaluation requested but no test split exists")
    loader, _ = _loader(dataset.test, config, execution, device, False)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(config.get("label_smoothing", 0.0)))
    amp_enabled, amp_dtype = _precision(device, str(execution.get("precision", "auto")))
    loss, accuracy, _ = evaluate(model, loader, criterion, device, amp_enabled, amp_dtype)
    return loss, accuracy


def safe_run_trial(
    adapter: Any,
    model_config: dict[str, Any],
    dataset: DatasetBundle,
    budget: ResourceBudget,
    **kwargs: Any,
) -> TrialResult:
    """Run a trial and convert setup/runtime errors into auditable records."""

    try:
        return run_trial(adapter, model_config, dataset, budget, **kwargs)
    except KeyboardInterrupt:
        status, reason = "interrupted", "keyboard interrupt during trial setup"
    except ConfigurationError as exc:
        status, reason = "rejected", f"ConfigurationError: {exc}"
    except torch.cuda.OutOfMemoryError as exc:
        status, reason = "failed", f"CUDA out of memory during trial setup: {exc}"
    except (RuntimeError, ValueError, OSError) as exc:
        status, reason = "failed", f"{type(exc).__name__} during trial setup: {exc}"
    gc.collect()
    device = kwargs["device"]
    if device.type == "cuda":
        torch.cuda.empty_cache()
    try:
        config = resolve_trial_config(adapter, model_config, dataset)
    except (ConfigurationError, RuntimeError, ValueError):
        config = {**getattr(adapter, "baseline", {}), **model_config}
    context = kwargs.get("distributed") or DistributedContext()
    assigned = list(kwargs.get("assigned_gpu_indices", ()))
    return TrialResult(
        study_name=str(kwargs["study_name"]),
        adapter=str(adapter.name),
        profile=str(kwargs["profile"]),
        stage=str(kwargs["stage"]),
        rung=kwargs.get("rung"),
        trial_id=str(kwargs["trial_id"]),
        status=status,
        architecture_id="invalid",
        configuration_id=stable_configuration_id(config),
        config=config,
        budget={"kind": budget.kind, "value": budget.value},
        seed=int(kwargs["seed"]),
        device=str(device),
        failure_reason=reason,
        worker_pid=os.getpid(),
        gpu_indices=assigned,
        world_size=context.world_size,
        parallel_mode="ddp"
        if context.enabled
        else str(kwargs.get("execution", {}).get("parallel_mode", "serial")),
    )
