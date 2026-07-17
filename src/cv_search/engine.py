"""Shared three-stage search engine with persistent Optuna studies and Pareto reporting."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .config import StudyConfig
from .data import fraction_subset
from .estimation import RuntimeEstimate, project_from_pilots, write_estimate
from .exceptions import ConfigurationError, InsufficientResources
from .hardware import choose_device, inspect_hardware
from .interactive import choose_candidates, confirm
from .logging_utils import JsonlLogger
from .objectives import (
    comparable_results,
    constraint_violations,
    objective_values,
    pareto_front,
    select_trial,
)
from .parallel import ParallelTrialExecutor, TrialTask, WorkerResult
from .plotting import generate_plots
from .reporting import write_reports
from .samplers import AskedCandidate, StudyCoordinator
from .search_space import SearchSpace, parameter_from_dict
from .storage import StudyStorage
from .training import evaluate_checkpoint, safe_run_trial
from .types import DatasetBundle, ResourceBudget, TrialResult


class SearchEngine:
    """Run hardware preview, proxy search, halving, confirmation, and optional final test."""

    def __init__(self, adapter: Any, dataset: DatasetBundle, config: StudyConfig) -> None:
        self.adapter = adapter
        self.dataset = dataset
        self.config = config
        self.storage = StudyStorage(config.study_dir)
        self.logger = JsonlLogger(self.storage.paths.logs / "events.jsonl")
        self.hardware = inspect_hardware(self.storage.paths.root / "hardware.json")
        requested_device = str(config.execution.get("device", "auto"))
        gpu_index = config.execution.get("gpu_index")
        self.device = choose_device(
            requested_device, int(gpu_index) if gpu_index is not None else None
        )
        recommended = dict(self.hardware["recommended"])
        for key, value in recommended.items():
            self.config.execution.setdefault(key, value)
        self.config.execution.setdefault("cpu_threads", recommended["intraop_threads"])
        self.config.execution.setdefault("reproducibility", "balanced")
        self.config.execution.setdefault("precision", recommended["precision"])
        self.config.execution.setdefault(
            "trial_concurrency",
            self.config.search.get("concurrency", "auto"),
        )
        self.config.execution.setdefault("parallel_mode", "auto")
        self.config.execution.setdefault("gpus_per_trial", 1)
        self.config.execution.setdefault("batch_size_scope", "global")
        self.config.execution.setdefault("distributed_backend", "auto")
        self.config.execution.setdefault("worker_imports", [])
        self.config.execution.setdefault(
            "split_seed", int(self.config.dataset.get("split_seed", 42))
        )
        torch.set_num_threads(max(1, int(self.config.execution.get("intraop_threads", 1))))
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(
                max(1, int(self.config.execution.get("interop_threads", 1)))
            )
        self.session_started = time.perf_counter()
        self.space = self._resolve_space()
        self.runtime_estimate: RuntimeEstimate | None = None
        self._prepare_files()
        self.logger.write(
            "study_initialized",
            study=self.config.name,
            adapter=self.adapter.name,
            profile=self.config.profile,
            device=str(self.device),
        )

    def _stage_execution(self, stage_name: str) -> dict[str, Any]:
        """Resolve stage-specific parallel settings without mutating the study config."""

        execution = dict(self.config.execution)
        stage_execution = self.config.stages.get(stage_name, {}).get("execution", {})
        if isinstance(stage_execution, dict):
            execution.update(stage_execution)
        mode = str(execution.get("parallel_mode", "auto")).lower()
        concurrency_raw = execution.get("trial_concurrency", "auto")
        concurrency = (
            max(1, int(self.hardware["recommended"]["trial_concurrency"]))
            if str(concurrency_raw).lower() == "auto"
            else max(1, int(concurrency_raw))
        )
        gpu_indices_raw = execution.get("gpu_indices")
        available_gpus = list(self.hardware.get("gpus", []))
        available_indices = {int(gpu["index"]) for gpu in available_gpus}
        if gpu_indices_raw is None:
            ordered = sorted(
                available_gpus,
                key=lambda gpu: (
                    float(gpu.get("memory_free_mb", gpu.get("memory_total_mb", 0))),
                    -float(gpu.get("utilization_percent", 0)),
                ),
                reverse=True,
            )
            gpu_indices = [int(gpu["index"]) for gpu in ordered]
        else:
            gpu_indices = [int(index) for index in gpu_indices_raw]
            missing = sorted(set(gpu_indices) - available_indices)
            if missing:
                raise InsufficientResources(
                    f"selected CUDA indices are not visible: {missing}; visible={sorted(available_indices)}"
                )
        requested_device = str(execution.get("device", "auto")).lower()
        cuda_requested = requested_device.startswith("cuda") or (
            requested_device == "auto" and bool(self.hardware.get("cuda_available"))
        )
        if mode == "auto":
            mode = (
                "parallel_trials"
                if cuda_requested and len(gpu_indices) > 1 and concurrency > 1
                else "serial"
            )
        if mode not in {"serial", "parallel_trials", "ddp", "hybrid"}:
            raise ConfigurationError(
                "execution.parallel_mode must be auto, serial, parallel_trials, ddp, or hybrid"
            )
        if mode == "parallel_trials" and cuda_requested:
            concurrency = min(concurrency, max(1, len(gpu_indices)))
        gpus_per_trial_raw = execution.get("gpus_per_trial", 1)
        gpus_per_trial = (
            max(1, len(gpu_indices))
            if str(gpus_per_trial_raw).lower() == "all"
            else max(1, int(gpus_per_trial_raw))
        )
        if mode in {"ddp", "hybrid"} and cuda_requested:
            if len(gpu_indices) < gpus_per_trial:
                raise InsufficientResources(
                    f"{mode} requires {gpus_per_trial} GPUs per trial but only {len(gpu_indices)} were selected"
                )
            concurrency = min(concurrency, max(1, len(gpu_indices) // gpus_per_trial))
        physical = max(1, int(self.hardware.get("physical_cpu_cores") or 1))
        reserve = max(0, int(execution.get("cpu_core_reserve", 1)))
        usable = max(1, physical - reserve)
        execution["intraop_threads"] = max(
            1, min(int(execution.get("intraop_threads", usable)), usable // concurrency)
        )
        execution["cpu_threads"] = execution["intraop_threads"]
        total_workers = int(
            execution.get("num_workers_total", execution.get("num_workers", 0) * concurrency)
        )
        execution["num_workers"] = max(0, total_workers // concurrency)
        execution["multiprocessing_context"] = (
            "spawn" if mode != "serial" else execution.get("multiprocessing_context", "spawn")
        )
        execution["parallel_mode"] = mode
        execution["trial_concurrency"] = concurrency
        execution["gpu_indices"] = gpu_indices
        execution["gpus_per_trial"] = gpus_per_trial
        return execution

    def _executor(self, stage_name: str) -> ParallelTrialExecutor:
        execution = self._stage_execution(stage_name)
        return ParallelTrialExecutor(
            mode=str(execution["parallel_mode"]),
            gpu_indices=list(execution["gpu_indices"]),
            concurrency=int(execution["trial_concurrency"]),
            gpus_per_trial=int(execution.get("gpus_per_trial", 1)),
            ddp_processes=int(execution.get("ddp_processes", 1)),
            persistent_workers=bool(execution.get("persistent_trial_workers", True)),
        )

    def _make_task(
        self,
        *,
        stage_name: str,
        config: dict[str, Any],
        budget: ResourceBudget,
        trial_id: str,
        seed: int,
        checkpoint_path: Path,
        rung: int | None = None,
        resume_checkpoint: Path | None = None,
        max_train_steps: int | None = None,
        max_validation_batches: int | None = None,
        evaluate_test: bool = False,
        train_fraction: float = 1.0,
    ) -> TrialTask:
        execution = self._stage_execution(stage_name)
        serial_gpu_indices: tuple[int, ...] = ()
        if execution["parallel_mode"] == "serial" and self.device.type == "cuda":
            serial_gpu_indices = (int(self.device.index or 0),)
        return TrialTask(
            adapter_name=self.config.adapter,
            dataset_name=str(self.config.dataset.get("name", "cifar10")),
            dataset_config=dict(self.config.dataset),
            model_config=dict(config),
            budget_kind=budget.kind,
            budget_value=budget.value,
            study_name=self.config.name,
            profile=self.config.profile,
            stage=stage_name,
            rung=rung,
            trial_id=trial_id,
            seed=seed,
            execution=execution,
            cost_rates=asdict(self.config.cost_rates),
            checkpoint_path=str(checkpoint_path),
            resume_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
            max_train_steps=max_train_steps,
            max_validation_batches=max_validation_batches,
            evaluate_test=evaluate_test,
            train_fraction=train_fraction,
            split_seed=int(self.config.dataset.get("split_seed", 42)),
            gpu_indices=serial_gpu_indices,
            worker_imports=tuple(str(item) for item in execution.get("worker_imports", [])),
        )

    def _execute_task_inline(self, task: TrialTask) -> WorkerResult:
        dataset = self.dataset
        if task.train_fraction < 1.0:
            dataset = replace(
                dataset,
                train=fraction_subset(dataset.train, task.train_fraction, task.split_seed),
            )
        result = safe_run_trial(
            self.adapter,
            task.model_config,
            dataset,
            ResourceBudget(task.budget_kind, task.budget_value),
            study_name=task.study_name,
            profile=task.profile,
            stage=task.stage,
            rung=task.rung,
            trial_id=task.trial_id,
            seed=task.seed,
            device=self.device,
            execution=task.execution,
            cost_rates=self.config.cost_rates,
            checkpoint_path=Path(task.checkpoint_path),
            resume_checkpoint=Path(task.resume_checkpoint) if task.resume_checkpoint else None,
            max_train_steps=task.max_train_steps,
            max_validation_batches=task.max_validation_batches,
            evaluate_test=task.evaluate_test,
            assigned_gpu_indices=task.gpu_indices,
        )
        return WorkerResult(result, result.worker_pid or 0, task.gpu_indices)

    def _run_tasks(self, stage_name: str, tasks: Any) -> Any:
        execution = self._stage_execution(stage_name)
        started = time.perf_counter()
        executions = 0
        device_seconds = 0.0
        try:
            if execution["parallel_mode"] == "serial":
                iterator = (self._execute_task_inline(task) for task in tasks)
                for worker_result in iterator:
                    executions += 1
                    device_seconds += worker_result.result.elapsed_seconds * max(
                        1, worker_result.result.world_size
                    )
                    yield worker_result
                return
            with self._executor(stage_name) as executor:
                for worker_result in executor.run(tasks):
                    executions += 1
                    device_seconds += worker_result.result.elapsed_seconds * max(
                        1, worker_result.result.world_size
                    )
                    yield worker_result
        finally:
            wall = time.perf_counter() - started
            slots = max(1, int(execution["trial_concurrency"]))
            efficiency = device_seconds / max(wall * slots, 1e-9)
            self.logger.write(
                "parallel_stage_completed",
                stage=stage_name,
                parallel_mode=execution["parallel_mode"],
                trial_concurrency=slots,
                executions=executions,
                wall_seconds=wall,
                allocated_device_seconds=device_seconds,
                scheduler_efficiency=min(1.0, efficiency),
            )

    def _prepare_files(self) -> None:
        shutil.copy2(self.config.source_path, self.storage.paths.root / "resolved_config.toml")
        environment = {
            "hardware": self.hardware,
            "command_line": sys.argv,
            "adapter": self.adapter.name,
            "profile": self.config.profile,
            "dataset": self.dataset.metadata.__dict__,
            "search_space_size": self.space.finite_size(),
            "search_space": {name: spec.__dict__ for name, spec in self.space.parameters.items()},
            "parallel_execution": {
                stage: self._stage_execution(stage) for stage in ("proxy", "halving", "full")
            },
        }
        (self.storage.paths.root / "environment.json").write_text(
            json.dumps(environment, indent=2, default=str), encoding="utf-8"
        )
        free_gb = float(self.hardware.get("disk_free_gb", 0))
        required_gb = float(self.config.constraints.get("minimum_free_disk_gb", 0))
        if required_gb and free_gb < required_gb:
            raise InsufficientResources(
                f"only {free_gb:.2f} GB disk is free; study requires {required_gb:.2f} GB"
            )

    def _resolve_space(self) -> SearchSpace:
        if self.config.profile == "simple":
            space = self.adapter.simple_search_space()
        elif self.config.profile == "thorough":
            space = self.adapter.thorough_search_space()
        else:
            space = SearchSpace({})
        parameters = dict(space.parameters)
        for name, raw in self.config.parameters.items():
            parameters[name] = parameter_from_dict(name, raw)
        return SearchSpace(parameters, self.config.explicit_configs)

    def preview(
        self, *, calibration_steps: int = 25, skip_calibration: bool = False
    ) -> dict[str, Any]:
        finite_size = self.space.finite_size()
        rejected_grid_candidates = 0
        if str(self.config.search.get("sampler", "tpe")) == "grid":
            valid_grid, rejected_grid_candidates = self._valid_grid_candidates()
            finite_size = len(valid_grid)
        preview: dict[str, Any] = {
            "study": self.config.name,
            "adapter": self.adapter.name,
            "device": str(self.device),
            "search_space_size": finite_size,
            "invalid_grid_combinations": rejected_grid_candidates,
            "sampler": self.config.search.get("sampler", "tpe"),
            "hardware": self.hardware,
        }
        if skip_calibration:
            preview["runtime_estimate"] = {
                "status": "not calibrated",
                "reason": "--skip-calibration was requested",
            }
        else:
            pilots = self._calibrate(calibration_steps)
            proxy = self.config.stages.get("proxy", {})
            halving = self.config.stages.get("halving", {})
            full = self.config.stages.get("full", {})
            proxy_trials = int(proxy.get("trials", self.config.search.get("trials", 10)))
            proxy_steps = int(proxy.get("max_train_steps", calibration_steps))
            reduction = max(2, int(halving.get("reduction_factor", 2)))
            candidates = int(proxy.get("top_k", max(1, proxy_trials // reduction)))
            steps_per_epoch = max(
                1,
                math.ceil(
                    self.dataset.metadata.train_examples
                    / max(1, int(self.adapter.baseline.get("batch_size", 128)))
                ),
            )
            halving_steps: list[int] = []
            for budget in halving.get("budgets", []):
                halving_steps.append(candidates * int(float(budget) * steps_per_epoch))
                candidates = max(int(halving.get("minimum_candidates", 1)), candidates // reduction)
            full_runs = int(full.get("top_k", 1)) * len(full.get("seeds", [42]))
            full_steps = int(full.get("epochs", 1)) * steps_per_epoch
            concurrency = int(self._stage_execution("proxy")["trial_concurrency"])
            self.runtime_estimate = project_from_pilots(
                pilots,
                proxy_trials=proxy_trials,
                proxy_steps=proxy_steps,
                halving_candidate_steps=halving_steps,
                full_runs=full_runs,
                full_steps=full_steps,
                concurrency=concurrency,
                cost_rates=self.config.cost_rates,
                parallel_efficiency=float(
                    self.config.execution.get("parallel_efficiency_estimate", 0.80)
                ),
            )
            write_estimate(self.storage.paths.root / "runtime_estimate.json", self.runtime_estimate)
            preview["runtime_estimate"] = self.runtime_estimate.to_dict()
        self.logger.write(
            "runtime_preview",
            calibration_skipped=skip_calibration,
            search_space_size=finite_size,
            device=str(self.device),
        )
        (self.storage.paths.root / "preview.json").write_text(
            json.dumps(preview, indent=2, default=str), encoding="utf-8"
        )
        return preview

    def _calibrate(self, steps: int) -> list[TrialResult]:
        from .data import fake_classification

        calibration_dataset = fake_classification(
            {
                "train_examples": max(64, steps * 8),
                "validation_examples": 32,
                "test_examples": 16,
                "image_size": self.dataset.metadata.image_size[0],
                "num_classes": self.dataset.metadata.num_classes,
                "split_seed": 731,
            }
        )
        pilots: list[TrialResult] = []
        for index, candidate in enumerate(self.adapter.calibration_configs()):
            config = {**candidate, "batch_size": 8, "num_workers": 0, "early_stopping_patience": 0}
            result = safe_run_trial(
                self.adapter,
                config,
                calibration_dataset,
                ResourceBudget("steps", max(1, steps)),
                study_name=self.config.name,
                profile=self.config.profile,
                stage="calibration",
                trial_id=f"calibration-{index}",
                seed=731 + index,
                device=self.device,
                execution={**self.config.execution, "compile": False, "num_workers": 0},
                cost_rates=self.config.cost_rates,
                checkpoint_path=self.storage.paths.checkpoints / "calibration" / f"{index}.pt",
                max_train_steps=max(1, steps),
                max_validation_batches=2,
            )
            pilots.append(result)
        (self.storage.paths.root / "calibration_results.json").write_text(
            json.dumps([result.to_dict() for result in pilots], indent=2, default=str),
            encoding="utf-8",
        )
        return pilots

    def execute(self) -> list[TrialResult]:
        results = self.storage.load()
        if not any(result.stage == "proxy" for result in results):
            proxy_results = self._proxy()
            results.extend(proxy_results)
        else:
            proxy_results = [result for result in results if result.stage == "proxy"]
        proxy_survivors = self._recommended(
            proxy_results,
            int(self.config.stages.get("proxy", {}).get("top_k", 4)),
        )
        if self.config.mode == "interactive":
            proxy_survivors = choose_candidates(
                "after proxy search", proxy_results, proxy_survivors
            )
        if not proxy_survivors:
            self._finalize(results)
            return results

        halving_enabled = bool(self.config.stages.get("halving", {}).get("enabled", True))
        existing_halving = [result for result in results if result.stage == "halving"]
        if halving_enabled and not existing_halving:
            halving_results, survivors = self._halving(proxy_survivors)
            results.extend(halving_results)
        elif existing_halving:
            highest_rung = max((result.rung or 0) for result in existing_halving)
            final_rung = [
                result for result in existing_halving if (result.rung or 0) == highest_rung
            ]
            survivors = self._recommended(
                final_rung, int(self.config.stages.get("halving", {}).get("minimum_candidates", 1))
            )
        else:
            survivors = proxy_survivors
        if self.config.mode == "interactive" and (existing_halving or halving_enabled):
            all_halving = [result for result in results if result.stage == "halving"]
            survivors = choose_candidates("after successive halving", all_halving, survivors)
        if not survivors:
            self._finalize(results)
            return results

        full_enabled = bool(self.config.stages.get("full", {}).get("enabled", True))
        existing_full = [result for result in results if result.stage == "full"]
        if full_enabled and not existing_full:
            full_results = self._confirmation(survivors)
            results.extend(full_results)
        else:
            full_results = existing_full
        if full_results:
            self._optional_test(full_results)
        self._finalize(results)
        return results

    def _proxy(self) -> list[TrialResult]:
        stage = self.config.stages.get("proxy", {})
        trials = int(stage.get("trials", self.config.search.get("trials", 10)))
        sampler_name = str(self.config.search.get("sampler", stage.get("sampler", "tpe")))
        finite_size = self.space.finite_size()
        coordinator_sampler = sampler_name
        coordinator_explicit = self.config.explicit_configs
        if sampler_name == "grid":
            valid_grid, rejected_grid = self._valid_grid_candidates()
            finite_size = len(valid_grid)
            self.logger.write(
                "grid_validated",
                valid_combinations=finite_size,
                invalid_combinations=rejected_grid,
            )
            if finite_size > trials:
                raise ConfigurationError(
                    f"grid has {finite_size} valid combinations but proxy.trials={trials}; "
                    "increase the explicit limit or choose random/TPE. The grid is not silently truncated."
                )
            coordinator_sampler = "explicit"
            coordinator_explicit = valid_grid
        coordinator = StudyCoordinator(
            database=self.storage.paths.database,
            study_name=f"{self.config.name}-proxy",
            sampler_name=coordinator_sampler,
            objectives=self.config.objectives,
            search_space=self.space,
            seed=int(self.config.search.get("seed", 42)),
            explicit_configs=coordinator_explicit,
            constant_liar=self._stage_execution("proxy")["parallel_mode"] != "serial",
        )
        target_trials = trials
        if sampler_name == "grid" and finite_size is not None:
            target_trials = finite_size
        elif sampler_name == "explicit":
            target_trials = len(self.config.explicit_configs)
        remaining = max(0, target_trials - coordinator.completed_count)
        results: list[TrialResult] = []
        candidates: dict[str, AskedCandidate] = {}

        def tasks() -> Any:
            for _ in range(remaining):
                if self._global_budget_exhausted(self.storage.load() + results):
                    return
                try:
                    candidate = coordinator.ask()
                except StopIteration:
                    return
                trial_id = f"proxy-{candidate.trial_number:05d}"
                candidates[trial_id] = candidate
                budget = ResourceBudget("epochs", float(stage.get("epochs", 3)))
                if stage.get("resource") == "steps":
                    budget = ResourceBudget("steps", float(stage.get("max_train_steps", 600)))
                elif stage.get("resource") == "seconds":
                    budget = ResourceBudget("seconds", float(stage.get("max_wall_seconds", 300)))
                yield self._make_task(
                    stage_name="proxy",
                    config=candidate.config,
                    budget=budget,
                    trial_id=trial_id,
                    seed=int(self.config.search.get("seed", 42)) + candidate.trial_number,
                    checkpoint_path=self.storage.paths.checkpoints / "proxy" / f"{trial_id}.pt",
                    max_train_steps=int(stage["max_train_steps"])
                    if stage.get("max_train_steps")
                    else None,
                    max_validation_batches=int(stage["max_validation_batches"])
                    if stage.get("max_validation_batches")
                    else None,
                    train_fraction=float(stage.get("data_fraction", 1.0)),
                )

        for worker_result in self._run_tasks("proxy", tasks()):
            result = worker_result.result
            self._decorate(result)
            candidate = candidates.get(result.trial_id)
            if candidate is not None:
                values = [
                    result.objective_values[objective.name]
                    for objective in self.config.objectives
                    if objective.enabled
                ]
                coordinator.tell(
                    candidate,
                    values if result.status == "completed" else None,
                    failed=result.status != "completed",
                    user_attrs={"result": result.to_dict()},
                )
            self.storage.append(result)
            self.logger.write(
                "trial_finished",
                trial_id=result.trial_id,
                stage=result.stage,
                status=result.status,
                validation_accuracy=result.validation_accuracy,
                elapsed_seconds=result.elapsed_seconds,
                worker_pid=result.worker_pid,
                gpu_indices=result.gpu_indices,
                parallel_mode=result.parallel_mode,
            )
            results.append(result)
        return results

    def _halving(
        self, candidates: list[TrialResult]
    ) -> tuple[list[TrialResult], list[TrialResult]]:
        stage = self.config.stages.get("halving", {})
        budgets = [float(value) for value in stage.get("budgets", [5, 15, 30])]
        resource = str(stage.get("resource", "epochs"))
        reduction = max(2, int(stage.get("reduction_factor", 2)))
        minimum = max(1, int(stage.get("minimum_candidates", 1)))
        continue_checkpoints = bool(stage.get("continue_checkpoints", True))
        all_results: list[TrialResult] = []
        active = candidates
        previous_paths: dict[str, Path] = {
            candidate.trial_id: Path(candidate.checkpoint_path)
            for candidate in active
            if candidate.checkpoint_path
        }
        for rung, value in enumerate(budgets):
            rung_results: list[TrialResult] = []
            sources: dict[str, TrialResult] = {}

            active_rung = list(active)
            current_rung = rung
            current_value = value

            def tasks(
                active_candidates: list[TrialResult] = active_rung,
                current_results: list[TrialResult] = rung_results,
                current_sources: dict[str, TrialResult] = sources,
                rung_index: int = current_rung,
                rung_budget: float = current_value,
            ) -> Any:
                for index, candidate in enumerate(active_candidates):
                    if self._global_budget_exhausted(
                        self.storage.load() + all_results + current_results
                    ):
                        return
                    trial_id = f"halving-r{rung_index}-{candidate.trial_id}"
                    current_sources[trial_id] = candidate
                    checkpoint = (
                        self.storage.paths.checkpoints
                        / "halving"
                        / candidate.configuration_id
                        / "checkpoint.pt"
                    )
                    yield self._make_task(
                        stage_name="halving",
                        config=candidate.config,
                        budget=ResourceBudget(resource, rung_budget),
                        trial_id=trial_id,
                        seed=int(self.config.search.get("seed", 42)) + index,
                        checkpoint_path=checkpoint,
                        rung=rung_index,
                        resume_checkpoint=previous_paths.get(candidate.trial_id)
                        if continue_checkpoints
                        else None,
                    )

            for worker_result in self._run_tasks("halving", tasks()):
                result = worker_result.result
                self._decorate(result)
                rung_results.append(result)
                source = sources.get(result.trial_id)
                if source is not None and result.checkpoint_path:
                    previous_paths[source.trial_id] = Path(result.checkpoint_path)
            keep = max(minimum, math.ceil(len(rung_results) / reduction))
            promoted = self._recommended(rung_results, keep)
            promoted_ids = {result.trial_id for result in promoted}
            for result in rung_results:
                if result.status == "completed":
                    result.status = "promoted" if result.trial_id in promoted_ids else "pruned"
                self.storage.append(result)
                self.logger.write(
                    "trial_finished",
                    trial_id=result.trial_id,
                    stage=result.stage,
                    rung=result.rung,
                    status=result.status,
                    validation_accuracy=result.validation_accuracy,
                    elapsed_seconds=result.elapsed_seconds,
                    worker_pid=result.worker_pid,
                    gpu_indices=result.gpu_indices,
                    parallel_mode=result.parallel_mode,
                )
            all_results.extend(rung_results)
            active = [sources[result.trial_id] for result in promoted if result.trial_id in sources]
            if not active or self._global_budget_exhausted(self.storage.load() + all_results):
                break
        final_candidates: list[TrialResult] = []
        if all_results:
            final_rung = max(result.rung or 0 for result in all_results)
            final_candidates = self._recommended(
                [result for result in all_results if (result.rung or 0) == final_rung],
                minimum,
            )
        return all_results, final_candidates

    def _confirmation(self, candidates: list[TrialResult]) -> list[TrialResult]:
        stage = self.config.stages.get("full", {})
        top_k = int(stage.get("top_k", 2))
        seeds = [int(seed) for seed in stage.get("seeds", [7, 21, 42])]
        selected = self._recommended(candidates, top_k)
        results: list[TrialResult] = []

        def tasks() -> Any:
            for finalist_index, candidate in enumerate(selected):
                for seed in seeds:
                    if self._global_budget_exhausted(self.storage.load() + results):
                        return
                    trial_id = f"full-{finalist_index}-seed-{seed}"
                    yield self._make_task(
                        stage_name="full",
                        config=candidate.config,
                        budget=ResourceBudget("epochs", float(stage.get("epochs", 100))),
                        trial_id=trial_id,
                        seed=seed,
                        checkpoint_path=self.storage.paths.checkpoints / "full" / f"{trial_id}.pt",
                        evaluate_test=False,
                    )

        for worker_result in self._run_tasks("full", tasks()):
            result = worker_result.result
            self._decorate(result)
            self.storage.append(result)
            self.logger.write(
                "trial_finished",
                trial_id=result.trial_id,
                stage=result.stage,
                status=result.status,
                validation_accuracy=result.validation_accuracy,
                elapsed_seconds=result.elapsed_seconds,
                worker_pid=result.worker_pid,
                gpu_indices=result.gpu_indices,
                parallel_mode=result.parallel_mode,
            )
            results.append(result)
        return results

    def _optional_test(self, full_results: list[TrialResult]) -> None:
        stage = self.config.stages.get("full", {})
        if not bool(stage.get("evaluate_test", False)):
            return
        if self.config.mode == "interactive" and not confirm(
            "Evaluate the test set for the selected final configuration?"
        ):
            return
        policy = str(self.config.selection.get("policy", "weighted_pareto"))
        winner = select_trial(full_results, self.config.objectives, policy)
        if winner.checkpoint_path is None:
            return
        test_loss, test_accuracy = evaluate_checkpoint(
            self.adapter,
            winner.config,
            self.dataset,
            Path(winner.checkpoint_path),
            self.device,
            self.config.execution,
        )
        winner.test_loss = test_loss
        winner.test_accuracy = test_accuracy
        (self.storage.paths.root / "selected_test_result.json").write_text(
            json.dumps(winner.to_dict(), indent=2, default=str), encoding="utf-8"
        )

    def _decorate(self, result: TrialResult) -> None:
        violations = constraint_violations(result, self.config.constraints)
        result.constraint_violations = violations
        result.constraints_satisfied = not violations
        result.objective_values = objective_values(result, self.config.objectives)
        if result.status == "completed" and violations:
            result.status = "rejected"
            result.failure_reason = "; ".join(violations)

    def _recommended(self, results: list[TrialResult], count: int) -> list[TrialResult]:
        feasible = [
            result
            for result in results
            if result.validation_accuracy is not None
            and result.status not in {"failed", "rejected", "interrupted"}
            and result.constraints_satisfied
        ]
        return sorted(
            feasible,
            key=lambda result: (
                -float(result.validation_accuracy or -1),
                float(result.validation_loss or float("inf")),
                result.elapsed_seconds,
                result.peak_memory_mb,
            ),
        )[:count]

    def _global_budget_exhausted(self, results: list[TrialResult]) -> bool:
        constraints = self.config.constraints
        total_runtime = time.perf_counter() - self.session_started
        total_gpu = sum(result.gpu_hours for result in results)
        total_cpu = sum(result.cpu_hours for result in results)
        total_cost = sum(result.estimated_cost_usd for result in results)
        checks = [
            ("max_search_runtime_seconds", total_runtime),
            ("max_search_gpu_hours", total_gpu),
            ("max_search_cpu_hours", total_cpu),
            ("max_search_cost_usd", total_cost),
        ]
        return any(
            constraints.get(name, 0) > 0 and observed >= constraints[name]
            for name, observed in checks
        )

    def _finalize(self, results: list[TrialResult]) -> None:
        parallel_events: list[dict[str, Any]] = []
        if self.logger.path.exists():
            for line in self.logger.path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("event") == "parallel_stage_completed":
                    parallel_events.append(record)
        if parallel_events:
            summary = {
                "stages": parallel_events,
                "total_wall_seconds": sum(
                    float(event.get("wall_seconds", 0.0)) for event in parallel_events
                ),
                "total_allocated_device_seconds": sum(
                    float(event.get("allocated_device_seconds", 0.0)) for event in parallel_events
                ),
                "mean_scheduler_efficiency": sum(
                    float(event.get("scheduler_efficiency", 0.0)) for event in parallel_events
                )
                / len(parallel_events),
            }
            (self.storage.paths.root / "parallel_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        self.storage.write_csv(self.storage.paths.leaderboard_csv, results)
        front = pareto_front(comparable_results(results), self.config.objectives)
        self.storage.write_csv(self.storage.paths.pareto_csv, front)
        generate_plots(
            results,
            png_dir=self.storage.paths.plots_png,
            svg_dir=self.storage.paths.plots_svg,
            html_dir=self.storage.paths.plots_html,
        )
        write_reports(
            self.storage.paths.root,
            results,
            self.config.objectives,
            str(self.config.selection.get("policy", "weighted_pareto")),
            self.hardware,
            self.runtime_estimate.to_dict() if self.runtime_estimate else None,
        )
        self.logger.write(
            "study_finalized",
            executions=len(results),
            pareto_trials=len(front),
            result_directory=str(self.storage.paths.root),
        )
