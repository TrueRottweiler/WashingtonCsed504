"""Append-only study records and spreadsheet-friendly exports."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path

from .types import StudyPaths, TrialResult


class StudyStorage:
    def __init__(self, root: Path) -> None:
        self.paths = StudyPaths(
            root=root,
            database=root / "study.db",
            results_jsonl=root / "results.jsonl",
            leaderboard_csv=root / "leaderboard.csv",
            pareto_csv=root / "pareto.csv",
            checkpoints=root / "checkpoints",
            models=root / "models",
            logs=root / "logs",
            plots_png=root / "plots" / "png",
            plots_svg=root / "plots" / "svg",
            plots_html=root / "plots" / "html",
        )
        for directory in (
            self.paths.root,
            self.paths.checkpoints,
            self.paths.models,
            self.paths.logs,
            self.paths.plots_png,
            self.paths.plots_svg,
            self.paths.plots_html,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def append(self, result: TrialResult) -> None:
        with self.paths.results_jsonl.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), sort_keys=True, default=str) + "\n")

    def load(self) -> list[TrialResult]:
        if not self.paths.results_jsonl.exists():
            return []
        allowed = {field.name for field in fields(TrialResult)}
        results = []
        for line in self.paths.results_jsonl.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            results.append(
                TrialResult(**{key: value for key, value in raw.items() if key in allowed})
            )
        return results

    def write_csv(self, path: Path, results: Iterable[TrialResult]) -> None:
        rows = []
        for result in results:
            row = {
                "trial_id": result.trial_id,
                "stage": result.stage,
                "rung": result.rung,
                "status": result.status,
                "architecture_id": result.architecture_id,
                "configuration_id": result.configuration_id,
                "validation_accuracy": result.validation_accuracy,
                "validation_loss": result.validation_loss,
                "test_accuracy": result.test_accuracy,
                "elapsed_seconds": result.elapsed_seconds,
                "parameter_count": result.parameter_count,
                "flops_per_forward": result.flops_per_forward,
                "peak_memory_mb": result.peak_memory_mb,
                "examples_processed": result.examples_processed,
                "gpu_hours": result.gpu_hours,
                "cpu_hours": result.cpu_hours,
                "estimated_cost_usd": result.estimated_cost_usd,
                "checkpoint_size_mb": result.checkpoint_size_mb,
                "seed": result.seed,
                "device": result.device,
                "parallel_mode": result.parallel_mode,
                "worker_pid": result.worker_pid,
                "gpu_indices": json.dumps(result.gpu_indices),
                "world_size": result.world_size,
                "global_batch_size": result.global_batch_size,
                "scheduler_wait_seconds": result.scheduler_wait_seconds,
                "config_json": json.dumps(result.config, sort_keys=True),
                "failure_reason": result.failure_reason,
            }
            rows.append(row)
        if not rows:
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
