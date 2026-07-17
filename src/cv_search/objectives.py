"""Pareto objectives, hard constraints, and final-trial selection policies."""

from __future__ import annotations

from collections.abc import Iterable

from .types import ObjectiveSpec, TrialResult

OBJECTIVE_FIELDS: dict[str, str] = {
    "validation_accuracy": "validation_accuracy",
    "validation_loss": "validation_loss",
    "wall_clock_seconds": "elapsed_seconds",
    "compute_hours": "gpu_hours",
    "cpu_hours": "cpu_hours",
    "peak_memory_mb": "peak_memory_mb",
    "examples_processed": "examples_processed",
    "monetary_cost_usd": "estimated_cost_usd",
    "parameter_count": "parameter_count",
    "training_flops": "estimated_training_flops",
    "checkpoint_size_mb": "checkpoint_size_mb",
}


def objective_values(result: TrialResult, objectives: Iterable[ObjectiveSpec]) -> dict[str, float]:
    values: dict[str, float] = {}
    for objective in objectives:
        if not objective.enabled:
            continue
        field = OBJECTIVE_FIELDS.get(objective.name, objective.name)
        value = getattr(result, field, None)
        if value is None:
            value = float("-inf") if objective.direction == "maximize" else float("inf")
        values[objective.name] = float(value)
    return values


def constraint_violations(result: TrialResult, constraints: dict[str, float]) -> list[str]:
    checks = {
        "max_runtime_seconds": result.elapsed_seconds,
        "max_gpu_hours": result.gpu_hours,
        "max_cpu_hours": result.cpu_hours,
        "max_memory_mb": result.peak_memory_mb,
        "max_data_samples": float(result.examples_processed),
        "max_epochs": float(result.epochs_completed),
        "max_cost_usd": result.estimated_cost_usd,
    }
    violations = [
        f"{name}: observed {checks[name]:.6g} > limit {limit:.6g}"
        for name, limit in constraints.items()
        if name in checks and limit > 0 and checks[name] > limit
    ]
    minimum_accuracy = constraints.get("min_validation_accuracy")
    if (
        minimum_accuracy is not None
        and result.validation_accuracy is not None
        and result.validation_accuracy < minimum_accuracy
    ):
        violations.append(
            "min_validation_accuracy: "
            f"observed {result.validation_accuracy:.6g} < target {minimum_accuracy:.6g}"
        )
    return violations


def dominates(
    left: TrialResult,
    right: TrialResult,
    objectives: list[ObjectiveSpec],
) -> bool:
    left_values = objective_values(left, objectives)
    right_values = objective_values(right, objectives)
    no_worse = True
    strictly_better = False
    for objective in objectives:
        if not objective.enabled:
            continue
        lv, rv = left_values[objective.name], right_values[objective.name]
        if objective.direction == "maximize":
            no_worse &= lv >= rv
            strictly_better |= lv > rv
        else:
            no_worse &= lv <= rv
            strictly_better |= lv < rv
    return no_worse and strictly_better


def comparable_results(results: Iterable[TrialResult]) -> list[TrialResult]:
    """Return the highest scientifically comparable stage available.

    Full-confirmation runs are preferred. Without them, only the highest
    successive-halving rung is compared; proxy trials are the final fallback.
    """
    items = list(results)
    full = [result for result in items if result.stage == "full"]
    if full:
        return full
    halving = [result for result in items if result.stage == "halving"]
    if halving:
        highest = max(result.rung if result.rung is not None else -1 for result in halving)
        return [
            result
            for result in halving
            if (result.rung if result.rung is not None else -1) == highest
        ]
    return [result for result in items if result.stage == "proxy"]


def pareto_front(
    results: Iterable[TrialResult], objectives: list[ObjectiveSpec]
) -> list[TrialResult]:
    feasible = [
        result
        for result in results
        if result.status in {"completed", "promoted"} and result.constraints_satisfied
    ]
    return [
        candidate
        for candidate in feasible
        if not any(
            other.trial_id != candidate.trial_id and dominates(other, candidate, objectives)
            for other in feasible
        )
    ]


def _normalized_utilities(results: list[TrialResult], objective: ObjectiveSpec) -> dict[str, float]:
    values = {
        result.trial_id: objective_values(result, [objective])[objective.name] for result in results
    }
    finite = [value for value in values.values() if abs(value) != float("inf")]
    if not finite:
        return {key: 0.0 for key in values}
    low, high = min(finite), max(finite)
    if high == low:
        return {key: 1.0 for key in values}
    utilities = {}
    for key, value in values.items():
        scaled = (value - low) / (high - low)
        utilities[key] = scaled if objective.direction == "maximize" else 1.0 - scaled
    return utilities


def select_trial(
    results: Iterable[TrialResult],
    objectives: list[ObjectiveSpec],
    policy: str = "weighted_pareto",
) -> TrialResult:
    candidates = pareto_front(results, objectives)
    if not candidates:
        raise ValueError("no feasible completed trials are available for selection")
    if policy == "accuracy":
        return max(candidates, key=lambda result: result.validation_accuracy or float("-inf"))
    if policy == "speed":
        return min(candidates, key=lambda result: result.elapsed_seconds)
    if policy == "cost":
        return min(candidates, key=lambda result: result.estimated_cost_usd)
    enabled = [objective for objective in objectives if objective.enabled]
    utility = {result.trial_id: 0.0 for result in candidates}
    total_weight = sum(objective.weight for objective in enabled) or 1.0
    for objective in enabled:
        normalized = _normalized_utilities(candidates, objective)
        for trial_id, score in normalized.items():
            utility[trial_id] += score * objective.weight / total_weight
    return max(candidates, key=lambda result: utility[result.trial_id])
