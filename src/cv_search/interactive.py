"""Safe interactive stage gates with automatic behavior in notebooks and pipelines."""

from __future__ import annotations

import sys
from collections.abc import Iterable

from .types import TrialResult


def choose_candidates(
    label: str,
    candidates: Iterable[TrialResult],
    recommended: list[TrialResult],
) -> list[TrialResult]:
    all_candidates = list(candidates)
    if not sys.stdin.isatty():
        return recommended
    print(f"\nInteractive pause: {label}")
    for result in all_candidates:
        print(
            f"  {result.trial_id}: acc={result.validation_accuracy} "
            f"loss={result.validation_loss} time={result.elapsed_seconds:.2f}s "
            f"params={result.parameter_count:,}"
        )
    default_ids = ",".join(result.trial_id for result in recommended)
    answer = input(
        f"Enter trial IDs, 'continue' for recommended [{default_ids}], or 'stop': "
    ).strip()
    if not answer or answer.lower() == "continue":
        return recommended
    if answer.lower() == "stop":
        return []
    selected_ids = {part.strip() for part in answer.split(",") if part.strip()}
    selected = [result for result in all_candidates if result.trial_id in selected_ids]
    return selected or recommended


def confirm(label: str) -> bool:
    if not sys.stdin.isatty():
        return True
    return input(f"{label} [Y/n]: ").strip().lower() not in {"n", "no", "stop"}
