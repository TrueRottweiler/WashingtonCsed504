"""torchrun entry point for one distributed trial task."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..distributed import destroy_process_group, initialize_from_environment
from ..parallel import TrialTask, _load_components
from ..training import safe_run_trial
from ..types import CostRates, ResourceBudget


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Internal DDP trial worker")
    command.add_argument("--task-file", required=True)
    command.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    task = TrialTask.from_dict(json.loads(Path(args.task_file).read_text(encoding="utf-8")))
    context, device = initialize_from_environment(task.execution)
    try:
        adapter, dataset = _load_components(task)
        result = safe_run_trial(
            adapter,
            dict(task.model_config),
            dataset,
            ResourceBudget(task.budget_kind, task.budget_value),
            study_name=task.study_name,
            profile=task.profile,
            stage=task.stage,
            rung=task.rung,
            trial_id=task.trial_id,
            seed=task.seed,
            device=device,
            execution=task.execution,
            cost_rates=CostRates(**task.cost_rates),
            checkpoint_path=Path(task.checkpoint_path),
            resume_checkpoint=Path(task.resume_checkpoint) if task.resume_checkpoint else None,
            max_train_steps=task.max_train_steps,
            max_validation_batches=task.max_validation_batches,
            evaluate_test=task.evaluate_test,
            distributed=context,
            assigned_gpu_indices=task.gpu_indices,
        )
        if context.is_main:
            if not task.result_path:
                raise RuntimeError("distributed task is missing result_path")
            destination = Path(task.result_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
            os.replace(temporary, destination)
        # Align normal completion so one rank does not tear down the process group
        # while rank zero is still atomically publishing the result.
        context.barrier()
        return 0 if result.status not in {"failed", "interrupted"} else 2
    finally:
        destroy_process_group()


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
