"""Keep available GPUs busy with an unattended queue of ImageNet-32 runs.

The scheduler never kills or preempts a process. It starts the next queued job only
when no matching ``train.py --gpu N`` process is visible for that device.

Usage::

    python scheduler.py
    python scheduler.py --plan
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)

# (arguments, tag), in order, one process at a time per GPU.
JOBS: dict[int, list[tuple[list[str], str]]] = {
    0: [
        (["--model", "vit", "--epochs", "40", "--tag", "vit_40ep"], "vit_40ep"),
    ],
    1: [
        (
            [
                "--model",
                "resnet18",
                "--epochs",
                "60",
                "--strong-aug",
                "--tag",
                "resnet18_aug60",
            ],
            "resnet18_aug60",
        ),
        (
            ["--model", "resnet18", "--epochs", "60", "--tag", "resnet18_60ep"],
            "resnet18_60ep",
        ),
    ],
}


def running_on(gpu: int) -> str | None:
    """Return the model/tag currently using ``--gpu gpu``, or ``None`` when idle."""
    for process in psutil.process_iter(["cmdline"]):
        try:
            command = process.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if not command or not any(Path(part).name == "train.py" for part in command):
            continue
        try:
            gpu_index = command.index("--gpu")
            process_gpu = int(command[gpu_index + 1])
        except (ValueError, IndexError):
            continue
        if process_gpu != gpu:
            continue
        for key in ("--tag", "--model"):
            try:
                return command[command.index(key) + 1]
            except (ValueError, IndexError):
                continue
        return "unknown"
    return None


def launch(gpu: int, arguments: list[str], tag: str) -> None:
    """Launch one detached training process and redirect output to its tag log."""
    log = HERE / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "-W",
        "ignore",
        str(HERE / "train.py"),
        *arguments,
        "--gpu",
        str(gpu),
        "--batch",
        "512",
    ]
    print(f"[sched] gpu{gpu}: launching {tag} -> {log}", flush=True)
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log.open("w", encoding="utf-8") as handle:
        subprocess.Popen(  # noqa: S603 - command is a fixed local Python entry point
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=HERE,
            creationflags=creation_flags,
            start_new_session=os.name != "nt",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    if args.plan:
        for gpu, jobs in JOBS.items():
            current = running_on(gpu)
            print(f"gpu{gpu}: currently running {current or '(idle)'}")
            for arguments, tag in jobs:
                print(f"   then: {tag:16s} {' '.join(arguments)}")
        return 0

    queues = {gpu: list(jobs) for gpu, jobs in JOBS.items()}
    print(f"[sched] started; {sum(len(queue) for queue in queues.values())} job(s) queued")
    while any(queues.values()):
        for gpu, queue in queues.items():
            if queue and running_on(gpu) is None:
                arguments, tag = queue.pop(0)
                launch(gpu, arguments, tag)
                time.sleep(min(30, max(1, args.poll_seconds)))
        time.sleep(max(1, args.poll_seconds))
    print("[sched] all queued jobs launched; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
