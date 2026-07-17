from pathlib import Path

from cv_search.samplers import StudyCoordinator
from cv_search.search_space import SearchSpace
from cv_search.types import ObjectiveSpec


def test_explicit_optuna_candidates_preserve_full_configuration(tmp_path: Path) -> None:
    configurations = [
        {"optimizer": "sgd", "learning_rate": 0.01, "batch_size": 8},
        {"optimizer": "adamw", "learning_rate": 0.001, "batch_size": 16},
    ]
    coordinator = StudyCoordinator(
        database=tmp_path / "study.db",
        study_name="explicit-test",
        sampler_name="explicit",
        objectives=[ObjectiveSpec("validation_accuracy", "maximize")],
        search_space=SearchSpace({}, configurations),
        seed=3,
        explicit_configs=configurations,
    )
    first = coordinator.ask()
    second = coordinator.ask()
    assert first.config == configurations[0]
    assert second.config == configurations[1]
    coordinator.tell(first, [0.5])
    coordinator.tell(second, [0.6])
    assert coordinator.completed_count == 2
