"""Typed result and configuration primitives shared across the framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Direction = Literal["maximize", "minimize"]
TrialStatus = Literal["completed", "promoted", "pruned", "failed", "rejected", "interrupted"]


@dataclass(frozen=True)
class ResourceBudget:
    """A resource ceiling for one training invocation."""

    kind: Literal["epochs", "steps", "seconds"]
    value: float

    def __post_init__(self) -> None:
        if self.value <= 0:
            raise ValueError("resource budget must be positive")


@dataclass(frozen=True)
class ModelDescription:
    """Architecture telemetry calculated before or immediately after allocation."""

    architecture_id: str
    parameter_count: int
    trainable_parameter_count: int
    flops_per_forward: float | None = None
    macs_per_forward: float | None = None
    activation_memory_mb: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    """Serializable record for one stage, rung, configuration, and seed."""

    study_name: str
    adapter: str
    profile: str
    stage: str
    rung: int | None
    trial_id: str
    status: TrialStatus
    architecture_id: str
    config: dict[str, Any]
    budget: dict[str, Any]
    seed: int
    device: str
    configuration_id: str = ""
    validation_accuracy: float | None = None
    validation_loss: float | None = None
    training_accuracy: float | None = None
    training_loss: float | None = None
    test_accuracy: float | None = None
    test_loss: float | None = None
    best_epoch: int | None = None
    epochs_completed: int = 0
    optimization_steps: int = 0
    examples_processed: int = 0
    elapsed_seconds: float = 0.0
    data_loading_seconds: float = 0.0
    compute_seconds: float = 0.0
    evaluation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0
    throughput_examples_per_second: float = 0.0
    parameter_count: int = 0
    trainable_parameter_count: int = 0
    flops_per_forward: float | None = None
    estimated_training_flops: float | None = None
    peak_memory_mb: float = 0.0
    peak_cpu_memory_mb: float = 0.0
    cpu_hours: float = 0.0
    gpu_hours: float = 0.0
    estimated_cost_usd: float = 0.0
    checkpoint_path: str | None = None
    checkpoint_size_mb: float = 0.0
    failure_reason: str | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    objective_values: dict[str, float] = field(default_factory=dict)
    constraints_satisfied: bool = True
    constraint_violations: list[str] = field(default_factory=list)
    worker_pid: int | None = None
    gpu_indices: list[int] = field(default_factory=list)
    world_size: int = 1
    global_batch_size: int | None = None
    parallel_mode: str = "serial"
    scheduler_wait_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectiveSpec:
    """One Pareto objective and optional post-hoc preference weight."""

    name: str
    direction: Direction
    enabled: bool = True
    weight: float = 1.0


@dataclass(frozen=True)
class CostRates:
    """User-supplied rates; zero means cost estimation is disabled for that resource."""

    gpu_hour_usd: float = 0.0
    cpu_hour_usd: float = 0.0
    storage_gb_month_usd: float = 0.0


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    num_classes: int
    image_size: tuple[int, int]
    channels: int
    train_examples: int
    validation_examples: int
    test_examples: int


@dataclass
class DatasetBundle:
    train: Any
    validation: Any
    test: Any | None
    metadata: DatasetMetadata


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    database: Path
    results_jsonl: Path
    leaderboard_csv: Path
    pareto_csv: Path
    checkpoints: Path
    models: Path
    logs: Path
    plots_png: Path
    plots_svg: Path
    plots_html: Path
