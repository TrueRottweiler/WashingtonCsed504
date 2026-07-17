"""Shared CLI construction and execution."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .. import adapters as _adapters  # noqa: F401
from .. import data as _data  # noqa: F401
from ..config import load_config
from ..engine import SearchEngine
from ..registry import create_dataset, create_model_adapter, registered_models


def parser(default_adapter: str | None = None) -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Hardware-aware proxy → halving → confirmation hyperparameter search"
    )
    command.add_argument("--config", required=True, help="TOML study configuration")
    command.add_argument("--adapter", default=default_adapter, choices=registered_models())
    command.add_argument("--estimate-only", action="store_true")
    command.add_argument("--calibration-steps", type=int, default=25)
    command.add_argument("--skip-calibration", action="store_true")
    command.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override dataset and budgets with a tiny synthetic pipeline",
    )
    command.add_argument(
        "--resume", action="store_true", help="Reuse study.db, results.jsonl, and stage checkpoints"
    )
    command.add_argument("--device", default=None, help="auto, cpu, mps, cuda, or cuda:N")
    command.add_argument(
        "--parallel-mode",
        choices=("auto", "serial", "parallel_trials", "ddp", "hybrid"),
        default=None,
    )
    command.add_argument(
        "--trial-concurrency",
        default=None,
        help="number of concurrent trials or 'auto'",
    )
    command.add_argument(
        "--gpu-indices",
        default=None,
        help="comma-separated physical GPU indices, for example 0,1,3",
    )
    command.add_argument(
        "--gpus-per-trial",
        default=None,
        help="GPUs assigned to each DDP trial or 'all'",
    )
    command.add_argument("--ddp-processes", type=int, default=None)
    return command


def smoke_overrides(config: Any) -> None:
    config.name = f"{config.name}-smoke"
    config.dataset = {
        "name": "fake",
        "train_examples": 64,
        "validation_examples": 24,
        "test_examples": 24,
        "image_size": 32,
        "num_classes": 10,
        "split_seed": 123,
    }
    config.search["sampler"] = "random"
    config.search["trials"] = 2
    config.parameters["warmup_epochs"] = {"enabled": False, "fixed": 0, "type": "fixed"}
    config.parameters["batch_size"] = {"enabled": False, "fixed": 8, "type": "fixed"}
    config.execution.update(
        {
            "device": config.execution.get("device", "cpu"),
            "num_workers": 0,
            "precision": "fp32",
            "compile": False,
            "parallel_mode": "serial",
            "trial_concurrency": 1,
            "gpus_per_trial": 1,
            "ddp_processes": 1,
        }
    )
    config.stages["proxy"] = {
        "enabled": True,
        "trials": 2,
        "epochs": 1,
        "max_train_steps": 1,
        "max_validation_batches": 1,
        "data_fraction": 1.0,
        "top_k": 2,
    }
    config.stages["halving"] = {
        "enabled": True,
        "resource": "steps",
        "budgets": [1],
        "reduction_factor": 2,
        "minimum_candidates": 1,
        "continue_checkpoints": True,
    }
    config.stages["full"] = {
        "enabled": True,
        "top_k": 1,
        "epochs": 1,
        "seeds": [7],
        "evaluate_test": False,
    }


def run(default_adapter: str | None = None, argv: list[str] | None = None) -> int:
    args = parser(default_adapter).parse_args(argv)
    config = load_config(args.config)
    if args.adapter:
        config.adapter = args.adapter
    if args.device:
        config.execution["device"] = args.device
    if args.parallel_mode:
        config.execution["parallel_mode"] = args.parallel_mode
    if args.trial_concurrency:
        config.execution["trial_concurrency"] = (
            args.trial_concurrency
            if args.trial_concurrency.lower() == "auto"
            else int(args.trial_concurrency)
        )
    if args.gpu_indices:
        config.execution["gpu_indices"] = [
            int(value.strip()) for value in args.gpu_indices.split(",") if value.strip()
        ]
    if args.gpus_per_trial:
        config.execution["gpus_per_trial"] = (
            args.gpus_per_trial
            if args.gpus_per_trial.lower() == "all"
            else int(args.gpus_per_trial)
        )
    if args.ddp_processes is not None:
        config.execution["ddp_processes"] = args.ddp_processes
    if args.smoke_test:
        smoke_overrides(config)
    existing_results = config.study_dir / "results.jsonl"
    if existing_results.exists() and not args.resume and not args.estimate_only:
        raise SystemExit(
            f"study results already exist at {config.study_dir}; pass --resume or use a new study.name"
        )
    adapter = create_model_adapter(config.adapter)
    dataset_name = str(config.dataset.get("name", "cifar10"))
    dataset = create_dataset(dataset_name, config.dataset)
    engine = SearchEngine(adapter, dataset, config)
    preview = engine.preview(
        calibration_steps=max(1, args.calibration_steps),
        skip_calibration=args.skip_calibration,
    )
    print(json.dumps(preview, indent=2, default=str))
    if args.estimate_only:
        return 0
    results = engine.execute()
    print(f"Completed {len(results)} recorded trial executions.")
    print(f"Results: {engine.storage.paths.root}")
    return 0
