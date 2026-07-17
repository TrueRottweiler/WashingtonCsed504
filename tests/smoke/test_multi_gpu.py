from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from cv_search.parallel import ParallelTrialExecutor, TrialTask, launch_ddp_task
from cv_search.types import CostRates

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="at least two visible CUDA GPUs are required",
)


def _config() -> dict[str, object]:
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


def _task(tmp_path: Path, trial_id: str, mode: str) -> TrialTask:
    return TrialTask(
        adapter_name="cnn",
        dataset_name="fake",
        dataset_config={
            "train_examples": 32,
            "validation_examples": 11,
            "test_examples": 7,
            "image_size": 32,
            "num_classes": 10,
            "split_seed": 17,
        },
        model_config=_config(),
        budget_kind="steps",
        budget_value=1,
        study_name="cuda-parallel-smoke",
        profile="custom",
        stage="proxy",
        trial_id=trial_id,
        seed=11,
        execution={
            "device": "auto",
            "precision": "fp32",
            "parallel_mode": mode,
            "num_workers": 0,
            "intraop_threads": 1,
            "interop_threads": 1,
            "batch_size_scope": "global",
            "distributed_backend": "nccl",
            "distributed_timeout_seconds": 120,
            "worker_timeout_seconds": 180,
        },
        cost_rates=asdict(CostRates()),
        checkpoint_path=str(tmp_path / f"{trial_id}.pt"),
        max_train_steps=1,
        max_validation_batches=2,
    )


def test_two_independent_cuda_trials_use_distinct_physical_gpus(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, f"cuda-{index}", "parallel_trials") for index in range(2)]
    with ParallelTrialExecutor(
        mode="parallel_trials", gpu_indices=[0, 1], concurrency=2
    ) as executor:
        results = list(executor.run(tasks))
    assert len(results) == 2
    assert {tuple(result.result.gpu_indices) for result in results} == {(0,), (1,)}
    assert all(result.result.status == "completed" for result in results)
    assert all(result.result.device.startswith("cuda") for result in results)


def test_two_gpu_nccl_ddp_smoke(tmp_path: Path) -> None:
    task = _task(tmp_path, "cuda-ddp", "ddp")
    task = TrialTask.from_dict(task.to_dict() | {"gpu_indices": (0, 1)})
    result = launch_ddp_task(task).result
    assert result.status == "completed", result.failure_reason
    assert result.world_size == 2
    assert result.gpu_indices == [0, 1]
    assert result.global_batch_size == 8
    assert Path(result.checkpoint_path or "").exists()
