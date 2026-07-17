"""Reusable layers and architecture utilities."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import nn


def architecture_id(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def activation(name: str) -> nn.Module:
    factories: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU,
    }
    try:
        return factories[name](inplace=True) if name == "relu" else factories[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported activation {name!r}") from exc


def normalization(name: str, channels: int, groups: int = 8) -> nn.Module:
    if name == "batchnorm":
        return nn.BatchNorm2d(channels)
    if name == "groupnorm":
        valid_groups = min(groups, channels)
        while channels % valid_groups:
            valid_groups -= 1
        return nn.GroupNorm(valid_groups, channels)
    if name in {"layernorm", "channel_layernorm"}:
        return nn.GroupNorm(1, channels)
    raise ValueError(f"unsupported CNN normalization {name!r}")


def convolution_output(size: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return math.floor((size + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1)


def resolve_padding(policy: str, kernel: int, dilation: int, explicit: int, stride: int) -> int:
    if policy == "valid":
        return 0
    if policy == "explicit":
        return explicit
    if policy == "same":
        if stride != 1:
            raise ValueError("same padding is only exact for stride=1 in this implementation")
        if kernel % 2 == 0:
            raise ValueError("same padding requires an odd kernel")
        return dilation * (kernel - 1) // 2
    raise ValueError(f"unknown padding policy {policy!r}")


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        self.probability = probability

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.probability == 0:
            return inputs
        keep = 1.0 - self.probability
        shape = (inputs.shape[0],) + (1,) * (inputs.ndim - 1)
        mask = inputs.new_empty(shape).bernoulli_(keep)
        return inputs * mask / keep
