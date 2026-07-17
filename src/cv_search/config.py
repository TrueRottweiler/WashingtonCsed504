"""Validated TOML study configuration."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError
from .types import CostRates, ObjectiveSpec


@dataclass
class StudyConfig:
    source_path: Path
    name: str
    output_dir: Path = Path("results")
    mode: str = "continuous"
    adapter: str = "cnn"
    profile: str = "simple"
    dataset: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    explicit_configs: list[dict[str, Any]] = field(default_factory=list)
    objectives: list[ObjectiveSpec] = field(default_factory=list)
    constraints: dict[str, float] = field(default_factory=dict)
    cost_rates: CostRates = field(default_factory=CostRates)
    selection: dict[str, Any] = field(default_factory=dict)

    @property
    def study_dir(self) -> Path:
        return self.output_dir / self.name


def _parse_objectives(raw: dict[str, Any]) -> list[ObjectiveSpec]:
    if not raw:
        return [ObjectiveSpec("validation_accuracy", "maximize", True, 1.0)]
    objectives: list[ObjectiveSpec] = []
    for name, values in raw.items():
        if not isinstance(values, dict):
            raise ConfigurationError(f"objectives.{name} must be a table")
        direction = values.get("direction", "maximize" if "accuracy" in name else "minimize")
        if direction not in {"maximize", "minimize"}:
            raise ConfigurationError(f"objectives.{name}.direction must be maximize or minimize")
        weight = float(values.get("weight", 1.0))
        if weight < 0:
            raise ConfigurationError(f"objectives.{name}.weight cannot be negative")
        objectives.append(ObjectiveSpec(name, direction, bool(values.get("enabled", True)), weight))
    enabled = [objective for objective in objectives if objective.enabled]
    if not enabled:
        raise ConfigurationError("at least one objective must be enabled")
    return objectives


def _validate_execution(values: dict[str, Any], prefix: str = "execution") -> None:
    parallel_mode = str(values.get("parallel_mode", "auto")).lower()
    if parallel_mode not in {"auto", "serial", "parallel_trials", "ddp", "hybrid"}:
        raise ConfigurationError(
            f"{prefix}.parallel_mode must be auto, serial, parallel_trials, ddp, or hybrid"
        )
    batch_scope = str(values.get("batch_size_scope", "global")).lower()
    if batch_scope not in {"global", "per_gpu", "per_device"}:
        raise ConfigurationError(f"{prefix}.batch_size_scope must be global or per_gpu")
    backend = str(values.get("distributed_backend", "auto")).lower()
    if backend not in {"auto", "nccl", "gloo"}:
        raise ConfigurationError(f"{prefix}.distributed_backend must be auto, nccl, or gloo")
    concurrency_value = values.get("trial_concurrency", "auto")
    if str(concurrency_value).lower() != "auto" and int(concurrency_value) < 1:
        raise ConfigurationError(f"{prefix}.trial_concurrency must be positive or 'auto'")
    gpus_per_trial_value = values.get("gpus_per_trial", 1)
    if str(gpus_per_trial_value).lower() != "all" and int(gpus_per_trial_value) < 1:
        raise ConfigurationError(f"{prefix}.gpus_per_trial must be positive or 'all'")
    gpu_indices = values.get("gpu_indices")
    if gpu_indices is not None and (
        not isinstance(gpu_indices, list) or any(int(index) < 0 for index in gpu_indices)
    ):
        raise ConfigurationError(f"{prefix}.gpu_indices must be nonnegative integers")
    for key in ("worker_timeout_seconds", "distributed_timeout_seconds"):
        if key in values and float(values[key]) < 0:
            raise ConfigurationError(f"{prefix}.{key} cannot be negative")
    if "parallel_efficiency_estimate" in values:
        efficiency = float(values["parallel_efficiency_estimate"])
        if not 0 < efficiency <= 1:
            raise ConfigurationError(f"{prefix}.parallel_efficiency_estimate must be in (0, 1]")


def load_config(path: str | Path) -> StudyConfig:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise ConfigurationError(f"configuration file not found: {source}")
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    study = raw.get("study", {})
    model = raw.get("model", raw.get("profile", {}))
    adapter = str(model.get("adapter", study.get("adapter", "cnn"))).lower()
    profile = str(model.get("profile", model.get("name", "simple"))).lower()
    mode = str(study.get("mode", "continuous")).lower()
    if profile not in {"simple", "thorough", "custom"}:
        raise ConfigurationError("model.profile must be simple, thorough, or custom")
    if mode not in {"continuous", "interactive"}:
        raise ConfigurationError("study.mode must be continuous or interactive")
    search = dict(raw.get("search", {}))
    sampler = str(
        search.get("sampler", raw.get("stages", {}).get("proxy", {}).get("sampler", "tpe"))
    )
    if sampler not in {"tpe", "random", "grid", "explicit"}:
        raise ConfigurationError("search.sampler must be tpe, random, grid, or explicit")
    search["sampler"] = sampler
    execution = dict(raw.get("execution", {}))
    execution.setdefault("trial_concurrency", search.get("concurrency", "auto"))
    _validate_execution(execution)
    stages = {key: dict(value) for key, value in raw.get("stages", {}).items()}
    for stage_name, stage_values in stages.items():
        stage_execution = stage_values.get("execution")
        if stage_execution is not None:
            if not isinstance(stage_execution, dict):
                raise ConfigurationError(f"stages.{stage_name}.execution must be a table")
            _validate_execution({**execution, **stage_execution}, f"stages.{stage_name}.execution")
    if profile == "thorough":
        bounded = any(
            float(search.get(key, 0) or 0) > 0
            for key in ("trials", "timeout_seconds", "max_gpu_hours", "max_cpu_hours")
        ) or any(
            float(raw.get("constraints", {}).get(key, 0) or 0) > 0
            for key in ("max_runtime_seconds", "max_gpu_hours", "max_cpu_hours", "max_cost_usd")
        )
        if not bounded:
            raise ConfigurationError(
                "thorough profile requires a trial, time, compute, or cost limit"
            )
    explicit = raw.get("explicit", {}).get("configurations", [])
    if sampler == "explicit" and not explicit:
        raise ConfigurationError("explicit sampler requires [[explicit.configurations]] entries")
    cost = raw.get("cost", {})
    return StudyConfig(
        source_path=source,
        name=str(study.get("name", source.stem)),
        output_dir=Path(study.get("output_dir", "results")),
        mode=mode,
        adapter=adapter,
        profile=profile,
        dataset=dict(raw.get("dataset", {})),
        search=search,
        execution=execution,
        stages=stages,
        parameters={key: dict(value) for key, value in raw.get("parameters", {}).items()},
        explicit_configs=[dict(item) for item in explicit],
        objectives=_parse_objectives(dict(raw.get("objectives", {}))),
        constraints={key: float(value) for key, value in raw.get("constraints", {}).items()},
        cost_rates=CostRates(
            gpu_hour_usd=float(cost.get("gpu_hour_usd", 0.0)),
            cpu_hour_usd=float(cost.get("cpu_hour_usd", 0.0)),
            storage_gb_month_usd=float(cost.get("storage_gb_month_usd", 0.0)),
        ),
        selection=dict(raw.get("selection", {})),
    )
