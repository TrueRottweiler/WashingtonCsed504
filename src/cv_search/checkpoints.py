"""Atomic, resume-safe checkpoints including optimizer, scheduler, scaler, and RNG state."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .distributed import unwrap_model
from .exceptions import CheckpointError


def random_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Return an RNG-state tensor in the format required by PyTorch setters."""
    if isinstance(value, torch.Tensor):
        return (
            value.detach()
            .to(device="cpu", dtype=torch.uint8)
            .contiguous()
        )
    return torch.as_tensor(
        value,
        dtype=torch.uint8,
        device="cpu",
    ).contiguous()


def restore_random_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_cpu_byte_tensor(state["torch"]))
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = [
            _as_cpu_byte_tensor(cuda_state)
            for cuda_state in state["cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    metadata: dict[str, Any],
    *,
    rank_random_states: list[dict[str, Any]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata,
        "random_state": random_state(),
        "rank_random_states": rank_random_states,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
    *,
    map_location: str | torch.device = "cpu",
    random_state_index: int | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise CheckpointError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        unwrap_model(model).load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload.get("scheduler") is not None:
            scheduler.load_state_dict(payload["scheduler"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        states = payload.get("rank_random_states")
        if random_state_index is not None and states and random_state_index < len(states):
            restore_random_state(states[random_state_index])
        else:
            restore_random_state(payload.get("random_state"))
        return dict(payload.get("metadata", {}))
    except (OSError, RuntimeError, TypeError, KeyError, ValueError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
