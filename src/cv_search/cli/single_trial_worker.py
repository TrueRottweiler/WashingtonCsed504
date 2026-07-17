"""Internal subprocess entry point for one isolated single-device trial."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..parallel import TrialTask, execute_trial_task


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Internal isolated trial worker")
    command.add_argument("--task-file", required=True)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    task = TrialTask.from_dict(json.loads(Path(args.task_file).read_text(encoding="utf-8")))
    result = execute_trial_task(task).result
    if not task.result_path:
        raise RuntimeError("isolated task is missing result_path")
    destination = Path(task.result_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
    os.replace(temporary, destination)
    return 0 if result.status not in {"failed", "interrupted"} else 2


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
