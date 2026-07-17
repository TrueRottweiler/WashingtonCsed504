"""Pilot-calibrated runtime, compute, memory, data, storage, and cost estimates."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .types import CostRates, TrialResult


@dataclass(frozen=True)
class EstimateRange:
    low: float
    central: float
    high: float


@dataclass(frozen=True)
class RuntimeEstimate:
    calibration_trials: int
    seconds_per_step: EstimateRange
    proxy_seconds: EstimateRange
    halving_seconds: EstimateRange
    confirmation_seconds: EstimateRange
    total_sequential_seconds: EstimateRange
    total_parallel_seconds: EstimateRange
    peak_memory_mb: EstimateRange
    storage_mb: EstimateRange
    estimated_gpu_hours: EstimateRange
    estimated_cpu_hours: EstimateRange
    estimated_cost_usd: EstimateRange
    confidence: str
    uncertainty_factors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def range_from(values: list[float], multiplier: float = 1.0) -> EstimateRange:
    clean = [value * multiplier for value in values if value >= 0]
    if not clean:
        return EstimateRange(0.0, 0.0, 0.0)
    central = statistics.median(clean)
    if len(clean) == 1:
        return EstimateRange(central * 0.7, central, central * 1.5)
    return EstimateRange(min(clean), central, max(clean))


def project_from_pilots(
    pilots: list[TrialResult],
    *,
    proxy_trials: int,
    proxy_steps: int,
    halving_candidate_steps: list[int],
    full_runs: int,
    full_steps: int,
    concurrency: int,
    cost_rates: CostRates,
    parallel_efficiency: float = 0.80,
) -> RuntimeEstimate:
    per_step = [
        pilot.compute_seconds / max(1, pilot.optimization_steps)
        for pilot in pilots
        if pilot.optimization_steps > 0
    ]
    step_range = range_from(per_step)

    def scaled(steps: int) -> EstimateRange:
        return EstimateRange(
            step_range.low * steps,
            step_range.central * steps,
            step_range.high * steps,
        )

    proxy = scaled(proxy_trials * proxy_steps)
    halving = scaled(sum(halving_candidate_steps))
    confirmation = scaled(full_runs * full_steps)
    total = EstimateRange(
        proxy.low + halving.low + confirmation.low,
        proxy.central + halving.central + confirmation.central,
        proxy.high + halving.high + confirmation.high,
    )
    efficiency = min(1.0, max(0.1, parallel_efficiency))
    effective_concurrency = max(1.0, concurrency * efficiency)
    parallel = EstimateRange(
        total.low / effective_concurrency,
        total.central / effective_concurrency,
        total.high / effective_concurrency,
    )
    memory = range_from([pilot.peak_memory_mb for pilot in pilots])
    storage_values = [max(1.0, pilot.parameter_count * 4 * 4 / 2**20) for pilot in pilots]
    storage = range_from(storage_values, multiplier=max(1, full_runs))
    gpu_fraction = 1.0 if any(pilot.device.startswith("cuda") for pilot in pilots) else 0.0
    gpu_hours = EstimateRange(
        total.low / 3600 * gpu_fraction,
        total.central / 3600 * gpu_fraction,
        total.high / 3600 * gpu_fraction,
    )
    cpu_hours = EstimateRange(total.low / 3600, total.central / 3600, total.high / 3600)
    cost = EstimateRange(
        gpu_hours.low * cost_rates.gpu_hour_usd + cpu_hours.low * cost_rates.cpu_hour_usd,
        gpu_hours.central * cost_rates.gpu_hour_usd + cpu_hours.central * cost_rates.cpu_hour_usd,
        gpu_hours.high * cost_rates.gpu_hour_usd + cpu_hours.high * cost_rates.cpu_hour_usd,
    )
    return RuntimeEstimate(
        calibration_trials=len(pilots),
        seconds_per_step=step_range,
        proxy_seconds=proxy,
        halving_seconds=halving,
        confirmation_seconds=confirmation,
        total_sequential_seconds=total,
        total_parallel_seconds=parallel,
        peak_memory_mb=memory,
        storage_mb=storage,
        estimated_gpu_hours=gpu_hours,
        estimated_cpu_hours=cpu_hours,
        estimated_cost_usd=cost,
        confidence="medium"
        if len(pilots) >= 3 and all(p.optimization_steps >= 10 for p in pilots)
        else "low",
        uncertainty_factors=(
            "architecture-dependent throughput",
            "data-loader and augmentation cost",
            "compilation startup overhead",
            "thermal throttling and shared-runtime contention",
            "pruning and early-stopping rates",
            "checkpoint filesystem performance",
            f"assumed parallel scheduler efficiency: {efficiency:.0%}",
        ),
    )


def estimation_errors(
    predicted_seconds: list[float], observed_seconds: list[float]
) -> dict[str, float]:
    pairs = [(p, o) for p, o in zip(predicted_seconds, observed_seconds, strict=False) if o > 0]
    if not pairs:
        return {}
    absolute = [abs(p - o) for p, o in pairs]
    percentages = [abs(p - o) / o * 100 for p, o in pairs]
    return {
        "mean_absolute_error_seconds": statistics.mean(absolute),
        "median_absolute_percentage_error": statistics.median(percentages),
        "underestimation_frequency": sum(p < o for p, o in pairs) / len(pairs),
        "overestimation_frequency": sum(p > o for p, o in pairs) / len(pairs),
    }


def write_estimate(path: Path, estimate: RuntimeEstimate) -> None:
    path.write_text(json.dumps(estimate.to_dict(), indent=2), encoding="utf-8")
