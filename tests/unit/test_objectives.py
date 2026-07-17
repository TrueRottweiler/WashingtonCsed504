from cv_search.objectives import (
    comparable_results,
    constraint_violations,
    pareto_front,
    select_trial,
)
from cv_search.types import ObjectiveSpec, TrialResult


def result(trial_id, accuracy, seconds, cost=0.0):
    return TrialResult(
        study_name="s",
        adapter="fake",
        profile="simple",
        stage="proxy",
        rung=None,
        trial_id=trial_id,
        status="completed",
        architecture_id=trial_id,
        config={},
        budget={"kind": "steps", "value": 1},
        seed=1,
        device="cpu",
        validation_accuracy=accuracy,
        elapsed_seconds=seconds,
        estimated_cost_usd=cost,
    )


def test_pareto_filter_and_policies():
    objectives = [
        ObjectiveSpec("validation_accuracy", "maximize", True, 1.0),
        ObjectiveSpec("wall_clock_seconds", "minimize", True, 0.2),
    ]
    values = [result("a", 0.8, 10), result("b", 0.82, 20), result("c", 0.7, 30)]
    assert {item.trial_id for item in pareto_front(values, objectives)} == {"a", "b"}
    assert select_trial(values, objectives, "accuracy").trial_id == "b"
    assert select_trial(values, objectives, "speed").trial_id == "a"


def test_constraints():
    candidate = result("x", 0.7, 12)
    candidate.peak_memory_mb = 300
    violations = constraint_violations(
        candidate,
        {"max_runtime_seconds": 10, "max_memory_mb": 200, "min_validation_accuracy": 0.8},
    )
    assert len(violations) == 3


def test_full_confirmation_is_preferred_for_final_comparison():
    proxy = result("proxy", 0.99, 1)
    full = result("full", 0.80, 10)
    full.stage = "full"
    assert comparable_results([proxy, full]) == [full]
