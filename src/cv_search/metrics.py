"""Streaming classification metrics with optional distributed reduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from .distributed import DistributedContext


@dataclass
class ClassificationAccumulator:
    """Accumulate tensor metrics without synchronizing the accelerator per batch."""

    device: torch.device | None = None

    def __post_init__(self) -> None:
        device = self.device or torch.device("cpu")
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.correct = torch.zeros((), dtype=torch.int64, device=device)
        self.examples = torch.zeros((), dtype=torch.int64, device=device)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> None:
        batch = labels.shape[0]
        self.loss_sum += loss.detach().to(torch.float64) * batch
        self.correct += (logits.detach().argmax(dim=1) == labels).sum().to(torch.int64)
        self.examples += batch

    def synchronize(self, context: DistributedContext | None = None) -> None:
        if context is None or not context.enabled:
            return
        context.all_reduce(self.loss_sum, dist.ReduceOp.SUM)
        context.all_reduce(self.correct, dist.ReduceOp.SUM)
        context.all_reduce(self.examples, dist.ReduceOp.SUM)

    @property
    def mean_loss(self) -> float:
        return float((self.loss_sum / self.examples.clamp_min(1)).item())

    @property
    def accuracy(self) -> float:
        return float((self.correct.to(torch.float64) / self.examples.clamp_min(1)).item())

    @property
    def example_count(self) -> int:
        return int(self.examples.item())
