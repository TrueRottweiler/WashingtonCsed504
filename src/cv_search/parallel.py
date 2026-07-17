"""Process-isolated trial scheduling and local multi-process DDP launch support."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import CostRates, ResourceBudget, TrialResult


@dataclass(frozen=True)
class TrialTask:
    """Serializable description of one isolated training invocation."""

    adapter_name: str
    dataset_name: str
    dataset_config: dict[str, Any]
    model_config: dict[str, Any]
    budget_kind: str
    budget_value: float
    study_name: str
    profile: str
    stage: str
    trial_id: str
    seed: int
    execution: dict[str, Any]
    cost_rates: dict[str, float]
    checkpoint_path: str
    rung: int | None = None
    resume_checkpoint: str | None = None
    max_train_steps: int | None = None
    max_validation_batches: int | None = None
    evaluate_test: bool = False
    train_fraction: float = 1.0
    split_seed: int = 42
    gpu_indices: tuple[int, ...] = ()
    worker_imports: tuple[str, ...] = ()
    result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable task payload."""

        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrialTask:
        """Construct a task from a persisted JSON payload."""

        values = dict(raw)
        values["gpu_indices"] = tuple(values.get("gpu_indices", ()))
        values["worker_imports"] = tuple(values.get("worker_imports", ()))
        return cls(**values)


@dataclass(frozen=True)
class WorkerResult:
    """One trial result plus the process and devices that produced it."""

    result: TrialResult
    worker_pid: int
    gpu_indices: tuple[int, ...] = ()


_DATASET_CACHE: dict[str, Any] = {}


def _cache_key(name: str, config: dict[str, Any]) -> str:
    return json.dumps([name, config], sort_keys=True, default=str)


def _load_components(task: TrialTask) -> tuple[Any, Any]:
    """Load registered adapter/dataset components inside a worker process."""

    from dataclasses import replace

    from . import adapters as _adapters  # noqa: F401
    from . import data as _data  # noqa: F401
    from .data import fraction_subset
    from .registry import create_dataset, create_model_adapter

    for module_name in task.worker_imports:
        importlib.import_module(module_name)
    adapter = create_model_adapter(task.adapter_name)
    key = _cache_key(task.dataset_name, task.dataset_config)
    dataset = _DATASET_CACHE.get(key)
    if dataset is None:
        dataset = create_dataset(task.dataset_name, dict(task.dataset_config))
        _DATASET_CACHE[key] = dataset
    if task.train_fraction < 1.0:
        dataset = replace(
            dataset,
            train=fraction_subset(dataset.train, task.train_fraction, task.split_seed),
        )
    return adapter, dataset


def execute_trial_task(task: TrialTask) -> WorkerResult:
    """Execute one task in the current process on its assigned device."""

    import torch

    from .training import safe_run_trial

    execution = dict(task.execution)
    threads = max(1, int(execution.get("intraop_threads", execution.get("cpu_threads", 1))))
    torch.set_num_threads(threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(max(1, int(execution.get("interop_threads", 1))))
    if task.gpu_indices:
        physical_index = int(task.gpu_indices[0])
        local_index = 0 if execution.get("_cuda_visible_remapped", False) else physical_index
        torch.cuda.set_device(local_index)
        device = torch.device("cuda", local_index)
    else:
        requested = str(execution.get("device", "cpu"))
        device = torch.device("cpu" if requested == "auto" else requested)
    adapter, dataset = _load_components(task)
    result = safe_run_trial(
        adapter,
        dict(task.model_config),
        dataset,
        ResourceBudget(task.budget_kind, task.budget_value),
        study_name=task.study_name,
        profile=task.profile,
        stage=task.stage,
        rung=task.rung,
        trial_id=task.trial_id,
        seed=task.seed,
        device=device,
        execution=execution,
        cost_rates=CostRates(**task.cost_rates),
        checkpoint_path=Path(task.checkpoint_path),
        resume_checkpoint=Path(task.resume_checkpoint) if task.resume_checkpoint else None,
        max_train_steps=task.max_train_steps,
        max_validation_batches=task.max_validation_batches,
        evaluate_test=task.evaluate_test,
        assigned_gpu_indices=task.gpu_indices,
    )
    return WorkerResult(result, os.getpid(), task.gpu_indices)


def _failed_result(task: TrialTask, reason: str) -> TrialResult:
    from .training import stable_configuration_id

    return TrialResult(
        study_name=task.study_name,
        adapter=task.adapter_name,
        profile=task.profile,
        stage=task.stage,
        rung=task.rung,
        trial_id=task.trial_id,
        status="failed",
        architecture_id="worker-failed",
        configuration_id=stable_configuration_id(task.model_config),
        config=dict(task.model_config),
        budget={"kind": task.budget_kind, "value": task.budget_value},
        seed=task.seed,
        device="ddp"
        if len(task.gpu_indices) > 1
        else (f"cuda:{task.gpu_indices[0]}" if task.gpu_indices else "cpu"),
        failure_reason=reason,
        gpu_indices=list(task.gpu_indices),
        world_size=max(1, len(task.gpu_indices)),
        parallel_mode="ddp" if len(task.gpu_indices) > 1 else "parallel_trials",
    )


def _subprocess_environment(task: TrialTask) -> dict[str, str]:
    """Build an isolated environment with source imports and GPU visibility."""

    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    current_path = environment.get("PYTHONPATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    if source_root not in path_entries:
        environment["PYTHONPATH"] = os.pathsep.join([source_root, *path_entries])
    if task.gpu_indices:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in task.gpu_indices)
    threads = max(
        1,
        int(task.execution.get("intraop_threads", task.execution.get("cpu_threads", 1))),
    )
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["MKL_NUM_THREADS"] = str(threads)
    environment.setdefault("PYTHONUNBUFFERED", "1")
    environment.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    environment.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
    return environment


def _task_artifact_paths(task: TrialTask, suffix: str) -> tuple[Path, Path, Path]:
    checkpoint = Path(task.checkpoint_path)
    task_dir = checkpoint.parent / ".tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", task.trial_id)
    result_path = task_dir / f"{token}.{suffix}.result.json"
    task_path = task_dir / f"{token}.{suffix}.task.json"
    log_path = task_dir / f"{token}.{suffix}.log"
    return result_path, task_path, log_path


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass(frozen=True)
class _PreparedTask:
    command: list[str]
    environment: dict[str, str]
    result_path: Path
    task_path: Path
    log_path: Path
    process_count: int = 1


@dataclass
class _WorkerDaemon:
    process: subprocess.Popen[str]
    log_handle: Any
    indices: tuple[int, ...]


@dataclass
class _ActiveTask:
    task: TrialTask
    prepared: _PreparedTask
    processes: list[subprocess.Popen[str]]
    log_handles: list[Any]
    started_at: float
    persistent: bool = False


def _prepare_task(task: TrialTask, *, ddp: bool) -> _PreparedTask:
    suffix = "ddp" if ddp else "worker"
    result_path, task_path, log_path = _task_artifact_paths(task, suffix)
    execution = dict(task.execution)
    if not ddp:
        execution["_cuda_visible_remapped"] = bool(task.gpu_indices)
    payload = task.to_dict() | {"result_path": str(result_path), "execution": execution}
    task_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    module = "cv_search.cli.trial_worker" if ddp else "cv_search.cli.single_trial_worker"
    command = [sys.executable, "-m", module, "--task-file", str(task_path)]
    process_count = (
        (len(task.gpu_indices) or int(task.execution.get("ddp_processes", 1))) if ddp else 1
    )
    return _PreparedTask(
        command,
        _subprocess_environment(task),
        result_path,
        task_path,
        log_path,
        max(1, process_count),
    )


def _popen_kwargs(log_handle: Any, environment: dict[str, str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
        "text": True,
        "env": environment,
        "cwd": str(Path.cwd()),
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


def _start_task(task: TrialTask, *, ddp: bool) -> _ActiveTask:
    prepared = _prepare_task(task, ddp=ddp)
    processes: list[subprocess.Popen[str]] = []
    log_handles: list[Any] = []
    if ddp:
        port = _free_local_port()
        for rank in range(prepared.process_count):
            rank_log = prepared.log_path.with_name(
                f"{prepared.log_path.stem}.rank-{rank}{prepared.log_path.suffix}"
            )
            log_handle = rank_log.open("w", encoding="utf-8")
            environment = prepared.environment.copy()
            environment.update(
                {
                    "MASTER_ADDR": "127.0.0.1",
                    "MASTER_PORT": str(port),
                    "WORLD_SIZE": str(prepared.process_count),
                    "LOCAL_WORLD_SIZE": str(prepared.process_count),
                    "RANK": str(rank),
                    "LOCAL_RANK": str(rank),
                    "GROUP_RANK": "0",
                    "ROLE_RANK": str(rank),
                    "ROLE_WORLD_SIZE": str(prepared.process_count),
                }
            )
            processes.append(
                subprocess.Popen(prepared.command, **_popen_kwargs(log_handle, environment))
            )
            log_handles.append(log_handle)
    else:
        log_handle = prepared.log_path.open("w", encoding="utf-8")
        processes.append(
            subprocess.Popen(
                prepared.command,
                **_popen_kwargs(log_handle, prepared.environment),
            )
        )
        log_handles.append(log_handle)
    return _ActiveTask(task, prepared, processes, log_handles, time.monotonic())


def _start_daemon(task: TrialTask) -> _WorkerDaemon:
    prepared = _prepare_task(task, ddp=False)
    indices = task.gpu_indices
    token = "cpu" if not indices else "gpu-" + "-".join(str(index) for index in indices)
    log_path = prepared.log_path.parent / f"slot-{token}.daemon.log"
    log_handle = log_path.open("a", encoding="utf-8")
    command = [sys.executable, "-m", "cv_search.cli.worker_daemon"]
    kwargs = _popen_kwargs(log_handle, prepared.environment)
    kwargs["stdin"] = subprocess.PIPE
    process = subprocess.Popen(command, **kwargs)
    return _WorkerDaemon(process, log_handle, indices)


def _submit_daemon_task(daemon: _WorkerDaemon, task: TrialTask) -> _ActiveTask:
    prepared = _prepare_task(task, ddp=False)
    if daemon.process.poll() is not None or daemon.process.stdin is None:
        raise RuntimeError("persistent worker daemon is not available")
    daemon.process.stdin.write(str(prepared.task_path) + "\n")
    daemon.process.stdin.flush()
    return _ActiveTask(
        task,
        prepared,
        [daemon.process],
        [],
        time.monotonic(),
        persistent=True,
    )


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            if os.name == "nt":
                process.kill()
            else:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)


def _stop_daemon(daemon: _WorkerDaemon) -> None:
    if daemon.process.poll() is None and daemon.process.stdin is not None:
        with contextlib.suppress(OSError):
            daemon.process.stdin.write("__quit__\n")
            daemon.process.stdin.flush()
            daemon.process.stdin.close()
        try:
            daemon.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _terminate_process(daemon.process)
    daemon.log_handle.close()


def _stop_task(active: _ActiveTask) -> None:
    for process in active.processes:
        _terminate_process(process)
    for handle in active.log_handles:
        with contextlib.suppress(OSError):
            handle.close()


def _task_complete(active: _ActiveTask) -> bool:
    if active.persistent:
        if active.prepared.result_path.exists():
            return True
        return active.processes[0].poll() is not None
    exit_codes = [process.poll() for process in active.processes]
    if any(code not in {None, 0, 2} for code in exit_codes):
        for process in active.processes:
            _terminate_process(process)
        return True
    return all(code is not None for code in exit_codes)


def _task_timed_out(active: _ActiveTask) -> bool:
    timeout = float(active.task.execution.get("worker_timeout_seconds", 0.0))
    return timeout > 0 and time.monotonic() - active.started_at > timeout


def _collect_task(active: _ActiveTask) -> WorkerResult:
    if active.persistent:
        exit_codes = [active.processes[0].poll()]
    else:
        exit_codes = [process.wait() for process in active.processes]
        for handle in active.log_handles:
            handle.close()
    if not active.prepared.result_path.exists():
        reason = (
            f"trial workers exited with codes {exit_codes}; logs share prefix "
            f"{active.prepared.log_path}"
        )
        return WorkerResult(
            _failed_result(active.task, reason),
            active.processes[0].pid,
            active.task.gpu_indices,
        )
    try:
        raw = json.loads(active.prepared.result_path.read_text(encoding="utf-8"))
        result = TrialResult(**raw)
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        reason = f"invalid worker result after exit codes {exit_codes}: {type(exc).__name__}: {exc}"
        result = _failed_result(active.task, reason)
    return WorkerResult(
        result,
        result.worker_pid or active.processes[0].pid,
        active.task.gpu_indices,
    )


def _run_task(task: TrialTask, *, ddp: bool) -> WorkerResult:
    active = _start_task(task, ddp=ddp)
    try:
        while not _task_complete(active):
            if _task_timed_out(active):
                _stop_task(active)
                return WorkerResult(
                    _failed_result(task, "worker timeout exceeded"),
                    active.processes[0].pid,
                    task.gpu_indices,
                )
            time.sleep(0.05)
        return _collect_task(active)
    except BaseException:
        _stop_task(active)
        raise


def launch_single_task(task: TrialTask) -> WorkerResult:
    """Launch one isolated Python subprocess for a single-device trial."""

    return _run_task(task, ddp=False)


def launch_ddp_task(task: TrialTask) -> WorkerResult:
    """Launch one local DDP trial with one rank process per assigned device."""

    return _run_task(task, ddp=True)


@dataclass
class _Slot:
    indices: tuple[int, ...]
    active: _ActiveTask | None = None
    daemon: _WorkerDaemon | None = None


class ParallelTrialExecutor:
    """Dynamically replenish dedicated device slots as trial processes complete."""

    def __init__(
        self,
        *,
        mode: str,
        gpu_indices: list[int],
        concurrency: int,
        gpus_per_trial: int = 1,
        ddp_processes: int = 1,
        persistent_workers: bool = True,
    ) -> None:
        self.mode = mode
        self.gpu_indices = gpu_indices
        self.concurrency = max(1, concurrency)
        self.gpus_per_trial = max(1, gpus_per_trial)
        self.ddp_processes = max(1, ddp_processes)
        self.persistent_workers = persistent_workers
        self.slots: list[_Slot] = []

    def __enter__(self) -> ParallelTrialExecutor:
        if self.mode == "serial":
            return self
        if self.mode in {"ddp", "hybrid"} and (self.gpus_per_trial > 1 or self.ddp_processes > 1):
            groups = [
                tuple(self.gpu_indices[index : index + self.gpus_per_trial])
                for index in range(0, len(self.gpu_indices), self.gpus_per_trial)
                if len(self.gpu_indices[index : index + self.gpus_per_trial]) == self.gpus_per_trial
            ]
            if not groups and self.ddp_processes > 1:
                groups = [tuple()]
            self.slots = [_Slot(group) for group in groups[: self.concurrency]]
        else:
            devices = [(index,) for index in self.gpu_indices]
            if not devices:
                devices = [tuple() for _ in range(self.concurrency)]
            self.slots = [_Slot(device) for device in devices[: self.concurrency]]
        if not self.slots:
            raise RuntimeError("parallel execution requested but no worker slots are available")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for slot in self.slots:
            if slot.active is not None and not slot.active.persistent:
                _stop_task(slot.active)
            slot.active = None
            if slot.daemon is not None:
                _stop_daemon(slot.daemon)
                slot.daemon = None
        self.slots.clear()

    def _with_slot(self, task: TrialTask, slot: _Slot) -> TrialTask:
        execution = dict(task.execution)
        if not slot.indices and self.mode in {"ddp", "hybrid"}:
            execution["ddp_processes"] = self.ddp_processes
        return TrialTask.from_dict(
            task.to_dict() | {"gpu_indices": slot.indices, "execution": execution}
        )

    def _uses_ddp(self, slot: _Slot) -> bool:
        return self.mode in {"ddp", "hybrid"} and (len(slot.indices) > 1 or self.ddp_processes > 1)

    def _fill_slot(self, slot: _Slot, task: TrialTask) -> None:
        if self.persistent_workers and not self._uses_ddp(slot):
            if slot.daemon is None or slot.daemon.process.poll() is not None:
                if slot.daemon is not None:
                    slot.daemon.log_handle.close()
                slot.daemon = _start_daemon(task)
            slot.active = _submit_daemon_task(slot.daemon, task)
        else:
            slot.active = _start_task(task, ddp=self._uses_ddp(slot))

    def run(self, tasks: Iterable[TrialTask]) -> Iterator[WorkerResult]:
        """Yield results as soon as each dedicated device slot becomes free."""

        iterator = iter(tasks)
        if self.mode == "serial":
            for task in iterator:
                yield execute_trial_task(task)
            return

        def fill(slot: _Slot) -> bool:
            try:
                task = self._with_slot(next(iterator), slot)
            except StopIteration:
                return False
            self._fill_slot(slot, task)
            return True

        for slot in self.slots:
            fill(slot)
        try:
            while any(slot.active is not None for slot in self.slots):
                completed_any = False
                for slot in self.slots:
                    active = slot.active
                    if active is None:
                        continue
                    if _task_timed_out(active):
                        if active.persistent and slot.daemon is not None:
                            _stop_daemon(slot.daemon)
                            slot.daemon = None
                        else:
                            _stop_task(active)
                        slot.active = None
                        completed_any = True
                        yield WorkerResult(
                            _failed_result(active.task, "worker timeout exceeded"),
                            active.processes[0].pid,
                            active.task.gpu_indices,
                        )
                        fill(slot)
                        continue
                    if not _task_complete(active):
                        continue
                    completed_any = True
                    slot.active = None
                    result = _collect_task(active)
                    if active.persistent and active.processes[0].poll() is not None:
                        if slot.daemon is not None:
                            slot.daemon.log_handle.close()
                        slot.daemon = None
                    yield result
                    fill(slot)
                if not completed_any:
                    time.sleep(0.05)
        except BaseException:
            for slot in self.slots:
                if slot.active is not None and not slot.active.persistent:
                    _stop_task(slot.active)
                slot.active = None
                if slot.daemon is not None:
                    _stop_daemon(slot.daemon)
                    slot.daemon = None
            raise
