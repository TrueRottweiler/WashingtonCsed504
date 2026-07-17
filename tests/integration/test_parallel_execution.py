from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from cv_search.parallel import ParallelTrialExecutor, TrialTask, launch_ddp_task
from cv_search.types import CostRates


def compact_cnn_config() -> dict[str, object]:
    return {
        "width_multiplier": 0.125,
        "stage_blocks": "1,1,1,1",
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "momentum": 0.9,
        "nesterov": False,
        "batch_size": 8,
        "scheduler": "none",
        "warmup_epochs": 0,
        "gradient_accumulation": 1,
        "gradient_clip": 0.0,
        "early_stopping_patience": 0,
        "label_smoothing": 0.0,
    }


def task(tmp_path: Path, trial_id: str, *, parallel_mode: str) -> TrialTask:
    return TrialTask(
        adapter_name="cnn",
        dataset_name="fake",
        dataset_config={
            "train_examples": 32,
            "validation_examples": 12,
            "test_examples": 8,
            "image_size": 32,
            "num_classes": 10,
            "split_seed": 17,
        },
        model_config=compact_cnn_config(),
        budget_kind="steps",
        budget_value=1,
        study_name="parallel-test",
        profile="custom",
        stage="proxy",
        trial_id=trial_id,
        seed=11,
        execution={
            "device": "cpu",
            "precision": "fp32",
            "parallel_mode": parallel_mode,
            "num_workers": 0,
            "intraop_threads": 1,
            "interop_threads": 1,
            "batch_size_scope": "global",
            "distributed_backend": "gloo",
            "ddp_processes": 2,
        },
        cost_rates=asdict(CostRates()),
        checkpoint_path=str(tmp_path / f"{trial_id}.pt"),
        max_train_steps=1,
        max_validation_batches=1,
    )


def test_two_process_isolated_trials(tmp_path: Path) -> None:
    tasks = [
        task(tmp_path, f"trial-{index}", parallel_mode="parallel_trials") for index in range(4)
    ]
    with ParallelTrialExecutor(mode="parallel_trials", gpu_indices=[], concurrency=2) as executor:
        results = list(executor.run(tasks))
    assert len(results) == 4
    assert len({item.result.worker_pid for item in results}) == 2
    assert all(item.result.status == "completed" for item in results)
    assert all(Path(item.result.checkpoint_path or "").exists() for item in results)


def test_two_process_cpu_ddp_trial(tmp_path: Path) -> None:
    ddp_task = task(tmp_path, "ddp", parallel_mode="ddp")
    dataset_config = dict(ddp_task.dataset_config)
    dataset_config["validation_examples"] = 9
    ddp_task = TrialTask.from_dict(
        ddp_task.to_dict() | {"dataset_config": dataset_config, "max_validation_batches": None}
    )
    result = launch_ddp_task(ddp_task).result
    assert result.status == "completed", result.failure_reason
    assert result.world_size == 2
    assert result.parallel_mode == "ddp"
    assert result.global_batch_size == 8
    assert Path(result.checkpoint_path or "").exists()
