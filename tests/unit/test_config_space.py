from pathlib import Path

import pytest

from cv_search.config import load_config
from cv_search.exceptions import ConfigurationError
from cv_search.search_space import ParameterSpec, SearchSpace, parameter_from_dict


def test_toml_profiles_and_objectives():
    cfg = load_config("configs/searches/resnet_cifar10.toml")
    assert cfg.profile == "simple"
    assert cfg.adapter == "cnn"
    assert [item.direction for item in cfg.objectives] == [
        "maximize",
        "minimize",
        "minimize",
        "minimize",
        "minimize",
    ]


def test_thorough_requires_limit(tmp_path: Path):
    path = tmp_path / "bad.toml"
    path.write_text('[study]\nname="x"\n[model]\nprofile="thorough"\nadapter="cnn"\n')
    with pytest.raises(ConfigurationError):
        load_config(path)


def test_enabled_disabled_frozen_and_conditionals():
    space = SearchSpace(
        {
            "optimizer": ParameterSpec("optimizer", values=("sgd", "adamw")),
            "momentum": ParameterSpec(
                "momentum", "float", low=0.8, high=0.9, condition={"optimizer": ("sgd",)}
            ),
            "frozen": ParameterSpec("frozen", "fixed", enabled=False, fixed=3),
        }
    )
    samples = space.sample_random(40, seed=8)
    assert all(sample["frozen"] == 3 for sample in samples)
    assert all(("momentum" in sample) == (sample["optimizer"] == "sgd") for sample in samples)


def test_size_and_explicit_grid():
    space = SearchSpace(
        {
            "a": ParameterSpec("a", values=(1, 2)),
            "b": ParameterSpec("b", "int", low=1, high=3),
        }
    )
    assert space.finite_size() == 6
    explicit = SearchSpace({}, [{"a": 1}, {"a": 2}])
    assert explicit.grid() == [{"a": 1}, {"a": 2}]


def test_log_step_rejected():
    with pytest.raises(ConfigurationError):
        parameter_from_dict(
            "x", {"type": "float", "low": 1e-4, "high": 1e-2, "log": True, "step": 0.1}
        ).validate()


def test_parallel_execution_validation(tmp_path: Path):
    valid = load_config("configs/searches/resnet_cifar10_multi_gpu.toml")
    assert valid.execution["parallel_mode"] == "hybrid"
    assert valid.stages["full"]["execution"]["gpus_per_trial"] == 2

    path = tmp_path / "bad_parallel.toml"
    path.write_text(
        """
[study]
name = "bad-parallel"
[model]
adapter = "cnn"
profile = "simple"
[execution]
parallel_mode = "parallel_trials"
parallel_efficiency_estimate = 1.5
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="parallel_efficiency_estimate"):
        load_config(path)

    path.write_text(
        """
[study]
name = "bad-backend"
[model]
adapter = "cnn"
profile = "simple"
[execution]
distributed_backend = "invalid"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="distributed_backend"):
        load_config(path)
