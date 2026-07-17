"""Plots derived only from recorded trial data."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .types import TrialResult


def _save(fig: Any, name: str, png: Path, svg: Path) -> None:
    png.mkdir(parents=True, exist_ok=True)
    svg.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png / f"{name}.png", dpi=160)
    fig.savefig(svg / f"{name}.svg")
    plt.close(fig)


def _scatter(
    results: list[TrialResult], x_field: str, name: str, label: str, png: Path, svg: Path
) -> None:
    points = [
        result
        for result in results
        if result.validation_accuracy is not None and getattr(result, x_field, None) is not None
    ]
    if not points:
        return
    fig, axis = plt.subplots()
    axis.scatter(
        [getattr(result, x_field) for result in points],
        [result.validation_accuracy for result in points],
    )
    axis.set_xlabel(label)
    axis.set_ylabel("Validation accuracy")
    axis.set_title(f"Validation accuracy versus {label.lower()}")
    _save(fig, name, png, svg)


def generate_plots(
    results: list[TrialResult],
    *,
    png_dir: Path,
    svg_dir: Path,
    html_dir: Path,
) -> None:
    completed = [result for result in results if result.validation_accuracy is not None]
    if not completed:
        return
    fig, axis = plt.subplots()
    axis.plot(
        range(1, len(completed) + 1),
        [result.validation_accuracy for result in completed],
        marker="o",
    )
    axis.set_xlabel("Recorded trial execution")
    axis.set_ylabel("Validation accuracy")
    axis.set_title("Validation accuracy by trial")
    _save(fig, "validation_accuracy_by_trial", png_dir, svg_dir)

    best = []
    current = float("-inf")
    for result in completed:
        current = max(current, float(result.validation_accuracy))
        best.append(current)
    fig, axis = plt.subplots()
    axis.plot(range(1, len(best) + 1), best)
    axis.set_xlabel("Recorded trial execution")
    axis.set_ylabel("Best validation accuracy")
    axis.set_title("Best-so-far optimization history")
    _save(fig, "best_so_far", png_dir, svg_dir)

    _scatter(
        completed, "elapsed_seconds", "accuracy_vs_runtime", "Runtime (seconds)", png_dir, svg_dir
    )
    _scatter(
        completed, "parameter_count", "accuracy_vs_parameters", "Parameter count", png_dir, svg_dir
    )
    _scatter(
        completed,
        "flops_per_forward",
        "accuracy_vs_flops",
        "Estimated FLOPs/forward",
        png_dir,
        svg_dir,
    )
    _scatter(
        completed,
        "peak_memory_mb",
        "accuracy_vs_peak_memory",
        "Peak accelerator memory (MB)",
        png_dir,
        svg_dir,
    )

    counts = Counter(result.status for result in results)
    fig, axis = plt.subplots()
    axis.bar(list(counts), list(counts.values()))
    axis.set_ylabel("Trial executions")
    axis.set_title("Trial-status distribution")
    axis.tick_params(axis="x", rotation=30)
    _save(fig, "trial_status_distribution", png_dir, svg_dir)

    rungs: dict[int, list[TrialResult]] = defaultdict(list)
    for result in completed:
        if result.rung is not None:
            rungs[result.rung].append(result)
    if rungs:
        fig, axis = plt.subplots()
        axis.bar([str(key) for key in sorted(rungs)], [len(rungs[key]) for key in sorted(rungs)])
        axis.set_xlabel("Halving rung")
        axis.set_ylabel("Candidates evaluated")
        axis.set_title("Successive-halving survival funnel")
        _save(fig, "halving_survival_funnel", png_dir, svg_dir)

    numeric_parameters: dict[str, list[tuple[float, float]]] = defaultdict(list)
    categorical: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for result in completed:
        for key, value in result.config.items():
            if isinstance(value, bool):
                categorical[key][str(value)].append(float(result.validation_accuracy))
            elif isinstance(value, (int, float)):
                numeric_parameters[key].append((float(value), float(result.validation_accuracy)))
            elif isinstance(value, str):
                categorical[key][value].append(float(result.validation_accuracy))
    preferred = [
        "learning_rate",
        "weight_decay",
        "batch_size",
        "classifier_dropout",
        "embedding_dropout",
        "patch_kernel",
        "patch_stride",
        "embed_dim",
        "depth",
        "heads",
        "mlp_ratio",
        "width_multiplier",
    ]
    for key in preferred:
        points = numeric_parameters.get(key, [])
        if len(points) < 2:
            continue
        fig, axis = plt.subplots()
        axis.scatter([point[0] for point in points], [point[1] for point in points])
        axis.set_xlabel(key)
        axis.set_ylabel("Validation accuracy")
        axis.set_title(f"{key} versus validation accuracy")
        _save(fig, f"parameter_{key}", png_dir, svg_dir)
    for key in ("optimizer", "activation", "normalization", "global_pool", "pooling"):
        groups = categorical.get(key, {})
        if len(groups) < 2:
            continue
        labels = list(groups)
        means = [sum(groups[label]) / len(groups[label]) for label in labels]
        fig, axis = plt.subplots()
        axis.bar(labels, means)
        axis.set_ylabel("Mean validation accuracy")
        axis.set_title(f"Categorical comparison: {key}")
        axis.tick_params(axis="x", rotation=30)
        _save(fig, f"categorical_{key}", png_dir, svg_dir)

    finalists = [result for result in results if result.stage == "full" and result.history]
    if finalists:
        winner = max(finalists, key=lambda result: result.validation_accuracy or -1)
        epochs = [row["epoch"] for row in winner.history]
        fig, axis = plt.subplots()
        axis.plot(epochs, [row["training_loss"] for row in winner.history], label="train")
        axis.plot(epochs, [row["validation_loss"] for row in winner.history], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.legend()
        axis.set_title("Final-candidate loss")
        _save(fig, "final_candidate_loss", png_dir, svg_dir)

    html_dir.mkdir(parents=True, exist_ok=True)
    payload = [result.to_dict() for result in results]
    html = (
        "<html><body><h1>Trial data</h1><pre>"
        + json.dumps(payload, indent=2, default=str)
        + "</pre></body></html>"
    )
    (html_dir / "trial_data.html").write_text(html, encoding="utf-8")
