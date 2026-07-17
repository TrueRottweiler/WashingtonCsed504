"""Distributed-training primitives used by optional DDP trial execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    """Runtime identity for one process participating in a distributed trial."""

    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = "none"

    @property
    def enabled(self) -> bool:
        return self.world_size > 1 and dist.is_available() and dist.is_initialized()

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def all_reduce(
        self, tensor: torch.Tensor, op: dist.ReduceOp = dist.ReduceOp.SUM
    ) -> torch.Tensor:
        if self.enabled:
            dist.all_reduce(tensor, op=op)
        return tensor

    def broadcast_bool(self, value: bool, device: torch.device) -> bool:
        if not self.enabled:
            return value
        payload = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
        dist.broadcast(payload, src=0)
        return bool(payload.item())


def initialize_from_environment(
    execution: dict[str, Any] | None = None,
) -> tuple[DistributedContext, torch.device]:
    """Initialize a torchrun-provided process group and select this rank's device."""

    execution = execution or {}
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested = str(execution.get("distributed_backend", "auto")).lower()
    cuda = torch.cuda.is_available() and str(execution.get("device", "auto")).lower() != "cpu"
    if cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = (
            "nccl"
            if requested == "auto" and dist.is_nccl_available()
            else ("gloo" if requested == "auto" else requested)
        )
    else:
        device = torch.device("cpu")
        backend = "gloo" if requested == "auto" else requested
    if backend == "nccl" and not dist.is_nccl_available():
        raise RuntimeError(
            "NCCL was requested but is unavailable in this PyTorch build; "
            "use distributed_backend='gloo' or install a compatible CUDA/NCCL build"
        )
    if world_size > 1 and not dist.is_initialized():
        timeout_seconds = max(1.0, float(execution.get("distributed_timeout_seconds", 300.0)))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(rank, world_size, local_rank, backend), device


def destroy_process_group() -> None:
    """Destroy the active process group after all ranks have completed."""

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, context: DistributedContext, device: torch.device) -> nn.Module:
    """Wrap a model in DistributedDataParallel using one process per accelerator."""

    if not context.enabled:
        return model
    if device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            find_unused_parameters=False,
        )
    return DistributedDataParallel(model, device_ids=None, find_unused_parameters=False)


def synchronize_module_buffers(
    model: nn.Module, context: DistributedContext, *, source_rank: int = 0
) -> None:
    """Broadcast stateful module buffers before uneven-shard distributed evaluation.

    DDP synchronizes buffers at DDP forward boundaries, but exact evaluation shards may
    contain different batch counts. Evaluation therefore runs through the unwrapped
    module and explicitly aligns buffers such as BatchNorm running statistics first.
    """

    if not context.enabled:
        return
    for buffer in unwrap_model(model).buffers():
        dist.broadcast(buffer, src=source_rank)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying trainable module through DDP/compile wrappers."""

    current = model
    while True:
        if isinstance(current, DistributedDataParallel):
            current = current.module
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module):
            current = original
            continue
        return current
