"""Markdown and machine-readable summaries."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .objectives import comparable_results, pareto_front, select_trial
from .types import ObjectiveSpec, TrialResult


def _format(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def aggregate_finalists(results: list[TrialResult]) -> list[dict[str, Any]]:
    groups: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        if result.stage == "full" and result.validation_accuracy is not None:
            key = result.configuration_id or result.architecture_id
            groups[key].append(result)
    rows = []
    for configuration_id, group in groups.items():
        accuracies = [float(item.validation_accuracy) for item in group]
        rows.append(
            {
                "configuration_id": configuration_id,
                "architecture_id": group[0].architecture_id,
                "seeds": [item.seed for item in group],
                "mean_validation_accuracy": statistics.mean(accuracies),
                "std_validation_accuracy": statistics.stdev(accuracies)
                if len(accuracies) > 1
                else 0.0,
                "mean_validation_loss": statistics.mean(
                    float(item.validation_loss)
                    for item in group
                    if item.validation_loss is not None
                ),
                "mean_training_seconds": statistics.mean(item.elapsed_seconds for item in group),
                "parameter_count": group[0].parameter_count,
                "flops_per_forward": group[0].flops_per_forward,
                "peak_memory_mb": max(item.peak_memory_mb for item in group),
                "config": group[0].config,
            }
        )
    return sorted(rows, key=lambda row: -row["mean_validation_accuracy"])


def write_reports(
    root: Path,
    results: list[TrialResult],
    objectives: list[ObjectiveSpec],
    selection_policy: str,
    environment: dict[str, Any],
    runtime_estimate: dict[str, Any] | None,
) -> None:
    stages: dict[str, list[TrialResult]] = defaultdict(list)
    for result in results:
        stages[result.stage].append(result)
    stage_lines = ["# Stage summary", ""]
    for stage, stage_results in stages.items():
        completed = [result for result in stage_results if result.validation_accuracy is not None]
        stage_lines.extend(
            [
                f"## {stage}",
                "",
                f"- Recorded executions: {len(stage_results)}",
                f"- Completed with validation metrics: {len(completed)}",
                f"- Failed or rejected: {sum(result.status in {'failed', 'rejected'} for result in stage_results)}",
            ]
        )
        if completed:
            best = max(completed, key=lambda result: result.validation_accuracy or -1)
            stage_lines.append(
                f"- Best validation accuracy: {_format(best.validation_accuracy)} ({best.trial_id})"
            )
        stage_lines.append("")
    (root / "stage_summary.md").write_text("\n".join(stage_lines), encoding="utf-8")

    comparison_pool = comparable_results(results)
    feasible = [result for result in comparison_pool if result.validation_accuracy is not None]
    front = pareto_front(feasible, objectives)
    lines = ["# Final study summary", ""]
    lines.append(f"Total recorded executions: **{len(results)}**")
    lines.append(f"Pareto-optimal executions: **{len(front)}**")
    lines.append("")
    if front:
        winner = select_trial(front, objectives, selection_policy)
        lines.extend(
            [
                "## Selected trial",
                "",
                f"- Trial: `{winner.trial_id}`",
                f"- Architecture: `{winner.architecture_id}`",
                f"- Validation accuracy: {_format(winner.validation_accuracy)}",
                f"- Validation loss: {_format(winner.validation_loss)}",
                f"- Test accuracy: {_format(winner.test_accuracy)}",
                f"- Runtime: {winner.elapsed_seconds:.2f} seconds",
                f"- Parameters: {winner.parameter_count:,}",
                f"- Estimated cost: ${winner.estimated_cost_usd:.6f}",
                "",
            ]
        )
    finalist_rows = aggregate_finalists(results)
    if finalist_rows:
        lines.extend(["## Multi-seed finalists", ""])
        for row in finalist_rows:
            lines.append(
                f"- config `{row['configuration_id']}` / architecture `{row['architecture_id']}`: "
                f"mean={row['mean_validation_accuracy']:.4f}, "
                f"std={row['std_validation_accuracy']:.4f}, seeds={row['seeds']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "Proxy results are approximate screening results. Successive-halving results use unequal resource budgets. Scientific final comparisons should use the from-scratch full-confirmation runs. The test set is isolated until an explicit final evaluation.",
        ]
    )
    (root / "final_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "finalists.json").write_text(json.dumps(finalist_rows, indent=2), encoding="utf-8")
    (root / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if runtime_estimate is not None:
        (root / "runtime_estimate.json").write_text(
            json.dumps(runtime_estimate, indent=2), encoding="utf-8"
        )
