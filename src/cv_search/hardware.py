"""Cross-platform hardware detection and conservative execution recommendations."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import psutil
import torch


def _nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,utilization.gpu,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            continue
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mb": float(fields[2]),
                "memory_free_mb": float(fields[3]),
                "utilization_percent": float(fields[4]),
                "compute_capability": fields[5],
            }
        )
    return rows


def _nvidia_topology() -> str | None:
    """Return the raw NVIDIA peer-link topology when nvidia-smi exposes it."""

    try:
        completed = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    topology = completed.stdout.strip()
    return topology or None


def inspect_hardware(path: Path | None = None) -> dict[str, Any]:
    gpus = []
    smi = {row["index"]: row for row in _nvidia_smi()}
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        record = {
            "index": index,
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "memory_total_mb": properties.total_memory / 2**20,
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "fp16_supported": True,
        }
        record.update(smi.get(index, {}))
        gpus.append(record)
    try:
        import torchvision

        torchvision_version: str | None = torchvision.__version__
    except ImportError:
        torchvision_version = None
    virtual_memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path.cwd())
    info: dict[str, Any] = {
        "operating_system": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision_version,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "nvidia_topology": _nvidia_topology() if gpus else None,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "cpu_model": platform.processor() or platform.machine(),
        "physical_cpu_cores": psutil.cpu_count(logical=False),
        "logical_cpu_cores": psutil.cpu_count(logical=True),
        "system_ram_total_gb": virtual_memory.total / 2**30,
        "system_ram_available_gb": virtual_memory.available / 2**30,
        "disk_free_gb": disk.free / 2**30,
        "torch_compile_available": hasattr(torch, "compile"),
        "colab": "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules,
        "distributed_world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "distributed_rank": int(os.environ.get("RANK", "0")),
    }
    info["recommended"] = recommend_execution(info)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def recommend_execution(info: dict[str, Any]) -> dict[str, Any]:
    logical = int(info.get("logical_cpu_cores") or 1)
    physical = int(info.get("physical_cpu_cores") or max(1, logical // 2))
    colab = bool(info.get("colab"))
    if info.get("cuda_available"):
        device, precision = (
            "cuda",
            "bf16" if any(g.get("bf16_supported") for g in info["gpus"]) else "fp16",
        )
        pin_memory = True
        workers = min(4 if colab else 8, max(1, physical // 2))
        concurrency = max(1, int(info.get("gpu_count", 1)))
    elif info.get("mps_available"):
        device, precision, pin_memory = "mps", "fp32", False
        workers, concurrency = min(4, max(1, physical // 2)), 1
    else:
        device, precision, pin_memory = "cpu", "fp32", False
        workers, concurrency = min(4, max(0, physical // 2)), 1
    return {
        "device": device,
        "precision": precision,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
        "prefetch_factor": 2 if workers > 0 else None,
        "intraop_threads": min(8, max(1, physical // 2)),
        "interop_threads": min(2, max(1, logical // 8)),
        "trial_concurrency": concurrency,
        "multiprocessing_context": "spawn"
        if info.get("cuda_available") or info.get("system") in {"Windows", "Darwin"}
        else "forkserver",
        "parallel_mode": "parallel_trials" if concurrency > 1 else "serial",
        "gpus_per_trial": 1,
        "batch_size_scope": "global",
        "cpu_core_reserve": 2 if physical >= 4 else 1,
    }


def choose_device(requested: str = "auto", gpu_index: int | None = None) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = f"cuda:{gpu_index or 0}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device
