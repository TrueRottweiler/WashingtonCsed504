from cv_search.estimation import estimation_errors, project_from_pilots
from cv_search.hardware import inspect_hardware, recommend_execution
from cv_search.types import CostRates, TrialResult


def pilot(seconds, steps, memory=10):
    return TrialResult(
        study_name="s",
        adapter="x",
        profile="simple",
        stage="calibration",
        rung=None,
        trial_id=str(seconds),
        status="completed",
        architecture_id="a",
        config={},
        budget={"kind": "steps", "value": steps},
        seed=1,
        device="cpu",
        compute_seconds=seconds,
        optimization_steps=steps,
        peak_memory_mb=memory,
        parameter_count=100,
    )


def test_hardware_detection_and_defaults():
    info = inspect_hardware()
    assert info["logical_cpu_cores"] >= 1
    defaults = recommend_execution(info)
    assert defaults["num_workers"] <= 8
    assert defaults["trial_concurrency"] >= 1


def test_estimator_and_errors():
    estimate = project_from_pilots(
        [pilot(1, 10), pilot(1.2, 10), pilot(0.8, 10)],
        proxy_trials=2,
        proxy_steps=10,
        halving_candidate_steps=[20],
        full_runs=1,
        full_steps=30,
        concurrency=1,
        cost_rates=CostRates(cpu_hour_usd=1.0),
    )
    assert estimate.total_sequential_seconds.central > 0
    assert estimate.estimated_cost_usd.central > 0
    errors = estimation_errors([9, 11], [10, 10])
    assert errors["median_absolute_percentage_error"] == 10
