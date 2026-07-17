from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cv_search.config import load_config
from cv_search.data import fake_classification
from cv_search.engine import SearchEngine
from cv_search.registry import (
    create_dataset,
    create_model_adapter,
    register_dataset,
    register_model_adapter,
)
from cv_search.search_space import ParameterSpec, SearchSpace
from cv_search.training import safe_run_trial
from cv_search.types import ModelDescription, ResourceBudget


class TinyAdapter:
    name = "tiny"
    baseline = {"num_classes": 2, "image_size": 8, "batch_size": 8}

    def validate_config(self, config):
        if float(config.get("learning_rate", 0.1)) <= 0:
            raise ValueError("learning rate")

    def build_model(self, config):
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, int(config.get("num_classes", 2))))

    def simple_search_space(self):
        return SearchSpace(
            {
                "learning_rate": ParameterSpec("learning_rate", values=(0.01, 0.1)),
                "batch_size": ParameterSpec("batch_size", "fixed", enabled=False, fixed=8),
                "scheduler": ParameterSpec("scheduler", "fixed", enabled=False, fixed="none"),
                "warmup_epochs": ParameterSpec("warmup_epochs", "fixed", enabled=False, fixed=0),
            }
        )

    thorough_search_space = simple_search_space

    def build_optimizer(self, model, config):
        return torch.optim.SGD(model.parameters(), lr=float(config.get("learning_rate", 0.1)))

    def build_scheduler(self, optimizer, config, budget, steps_per_epoch):
        return None

    def describe_model(self, model, config):
        count = sum(p.numel() for p in model.parameters())
        return ModelDescription("tiny", count, count, 100.0, 50.0, 0.1, {})

    def calibration_configs(self):
        return [{"learning_rate": 0.01}, {"learning_rate": 0.05}, {"learning_rate": 0.1}]


def make_config(tmp_path: Path) -> Path:
    path = tmp_path / "study.toml"
    path.write_text(
        f'''[study]\nname="integration"\noutput_dir="{tmp_path.as_posix()}"\n[model]\nadapter="tiny"\nprofile="simple"\n[dataset]\nname="fake"\ntrain_examples=32\nvalidation_examples=16\ntest_examples=16\nimage_size=8\nnum_classes=2\n[search]\nsampler="random"\ntrials=2\nseed=2\n[execution]\ndevice="cpu"\nprecision="fp32"\nnum_workers=0\nintraop_threads=1\ninterop_threads=1\n[objectives.validation_accuracy]\ndirection="maximize"\n[objectives.wall_clock_seconds]\ndirection="minimize"\n[stages.proxy]\ntrials=2\nepochs=1\nmax_train_steps=1\nmax_validation_batches=1\ntop_k=2\n[stages.halving]\nenabled=true\nresource="steps"\nbudgets=[1,2]\nreduction_factor=2\nminimum_candidates=1\ncontinue_checkpoints=true\n[stages.full]\nenabled=true\ntop_k=1\nepochs=1\nseeds=[7]\nevaluate_test=false\n'''
    )
    return path


def test_end_to_end_cpu_search_resume_and_reports(tmp_path):
    adapter = TinyAdapter()
    config = load_config(make_config(tmp_path))
    dataset = fake_classification(config.dataset)
    engine = SearchEngine(adapter, dataset, config)
    engine.preview(skip_calibration=True)
    results = engine.execute()
    assert any(result.stage == "proxy" for result in results)
    assert any(result.status == "pruned" for result in results)
    assert any(result.stage == "full" for result in results)
    assert engine.storage.paths.database.exists()
    assert engine.storage.paths.leaderboard_csv.exists()
    assert engine.storage.paths.pareto_csv.exists()
    summary = engine.storage.paths.root / "final_summary.md"
    assert summary.exists()
    assert "`full-" in summary.read_text()
    # A second execution resumes from JSONL/SQLite and does not duplicate completed stages.
    second = SearchEngine(adapter, dataset, config).execute()
    assert len(second) == len(results)


def test_registry_custom_model_and_dataset():
    register_model_adapter("tiny-test", TinyAdapter, replace=True)
    register_dataset(
        "tiny-data",
        lambda cfg: fake_classification({"image_size": 8, "num_classes": 2}),
        replace=True,
    )
    assert create_model_adapter("tiny-test").name == "tiny"
    assert create_dataset("tiny-data", {}).metadata.num_classes == 2


def test_checkpoint_continuation(tmp_path):
    adapter = TinyAdapter()
    dataset = fake_classification(
        {
            "train_examples": 32,
            "validation_examples": 8,
            "test_examples": 8,
            "image_size": 8,
            "num_classes": 2,
        }
    )
    checkpoint = tmp_path / "c.pt"
    common = dict(
        study_name="s",
        profile="simple",
        stage="halving",
        trial_id="a",
        seed=1,
        device=torch.device("cpu"),
        execution={"precision": "fp32", "num_workers": 0},
        cost_rates=__import__("cv_search.types", fromlist=["CostRates"]).CostRates(
            storage_gb_month_usd=1.0
        ),
        checkpoint_path=checkpoint,
    )
    first = safe_run_trial(
        adapter, {"batch_size": 8}, dataset, ResourceBudget("steps", 1), **common
    )
    common["trial_id"] = "b"
    second = safe_run_trial(
        adapter,
        {"batch_size": 8},
        dataset,
        ResourceBudget("steps", 2),
        resume_checkpoint=checkpoint,
        **common,
    )
    assert second.optimization_steps >= first.optimization_steps
    assert first.configuration_id
    assert first.checkpoint_size_mb > 0
    assert first.estimated_cost_usd > 0


class OOMAdapter(TinyAdapter):
    def build_model(self, config):
        class OOM(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.zeros(()))

            def forward(self, x):
                raise torch.cuda.OutOfMemoryError("mocked")

        return OOM()


def test_mocked_oom_is_recorded(tmp_path):
    dataset = fake_classification(
        {
            "train_examples": 8,
            "validation_examples": 4,
            "test_examples": 4,
            "image_size": 8,
            "num_classes": 2,
        }
    )
    result = safe_run_trial(
        OOMAdapter(),
        {"batch_size": 4},
        dataset,
        ResourceBudget("steps", 1),
        study_name="s",
        profile="simple",
        stage="proxy",
        trial_id="oom",
        seed=1,
        device=torch.device("cpu"),
        execution={"precision": "fp32", "num_workers": 0},
        cost_rates=__import__("cv_search.types", fromlist=["CostRates"]).CostRates(),
        checkpoint_path=tmp_path / "oom.pt",
    )
    assert result.status == "failed"
    assert "out of memory" in result.failure_reason.lower()
