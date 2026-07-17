"""Model-adapter protocol separating search orchestration from architecture code."""

from __future__ import annotations

from typing import Any, Protocol

import torch
from torch import nn

from ..search_space import SearchSpace
from ..types import ModelDescription, ResourceBudget


class ModelSearchAdapter(Protocol):
    name: str
    baseline: dict[str, Any]

    def resolve_config(self, config: dict[str, Any]) -> dict[str, Any]: ...
    def build_model(self, config: dict[str, Any]) -> nn.Module: ...
    def validate_config(self, config: dict[str, Any]) -> None: ...
    def simple_search_space(self) -> SearchSpace: ...
    def thorough_search_space(self) -> SearchSpace: ...
    def build_optimizer(
        self, model: nn.Module, config: dict[str, Any]
    ) -> torch.optim.Optimizer: ...
    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        budget: ResourceBudget,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None: ...
    def describe_model(self, model: nn.Module, config: dict[str, Any]) -> ModelDescription: ...
    def calibration_configs(self) -> list[dict[str, Any]]: ...
