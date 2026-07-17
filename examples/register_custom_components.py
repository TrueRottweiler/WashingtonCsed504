"""Minimal custom model and dataset registration example."""

from __future__ import annotations

import torch
from torch import nn

from cv_search.data import fake_classification
from cv_search.registry import register_dataset, register_model_adapter
from cv_search.search_space import ParameterSpec, SearchSpace
from cv_search.types import ModelDescription


class TinyImageAdapter:
    name = "tiny-image"
    baseline = {"image_size": 8, "num_classes": 2, "batch_size": 8}

    def validate_config(self, config: dict) -> None:
        if float(config.get("learning_rate", 0.01)) <= 0:
            raise ValueError("learning_rate must be positive")

    def build_model(self, config: dict) -> nn.Module:
        classes = int(config.get("num_classes", 2))
        return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, classes))

    def simple_search_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "learning_rate": ParameterSpec(
                    "learning_rate", "float", low=1e-4, high=1e-1, log=True
                ),
                "batch_size": ParameterSpec("batch_size", "fixed", enabled=False, fixed=8),
            }
        )

    thorough_search_space = simple_search_space

    def build_optimizer(self, model: nn.Module, config: dict) -> torch.optim.Optimizer:
        return torch.optim.SGD(model.parameters(), lr=float(config["learning_rate"]))

    def build_scheduler(self, optimizer, config, budget, steps_per_epoch):
        return None

    def describe_model(self, model: nn.Module, config: dict) -> ModelDescription:
        count = sum(parameter.numel() for parameter in model.parameters())
        return ModelDescription("tiny-image-v1", count, count, None, None, None, {})

    def calibration_configs(self) -> list[dict]:
        return [{"learning_rate": 1e-3}, {"learning_rate": 1e-2}]


def register() -> None:
    register_model_adapter("tiny-image", TinyImageAdapter, replace=True)
    register_dataset(
        "tiny-images",
        lambda cfg: fake_classification({"image_size": 8, "num_classes": 2, **cfg}),
        replace=True,
    )


if __name__ == "__main__":
    register()
    print("Registered tiny-image adapter and tiny-images dataset")
