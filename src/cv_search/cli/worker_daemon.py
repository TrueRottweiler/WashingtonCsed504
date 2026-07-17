"""Long-lived isolated worker used to amortize trial startup and dataset loading."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from ..parallel import TrialTask, execute_trial_task


def _write_result(task: TrialTask) -> None:
    result = execute_trial_task(task).result
    if not task.result_path:
        raise RuntimeError("daemon task is missing result_path")
    destination = Path(task.result_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
    os.replace(temporary, destination)


def main() -> int:
    for line in sys.stdin:
        value = line.strip()
        if not value:
            continue
        if value == "__quit__":
            return 0
        task_path = Path(value)
        task = TrialTask.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
        _write_result(task)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
