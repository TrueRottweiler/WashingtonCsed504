"""Encoder-only Vision Transformer adapter for 32x32 image classification."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..exceptions import ConfigurationError
from ..search_space import ParameterSpec, SearchSpace
from ..types import ModelDescription, ResourceBudget
from .common import DropPath, activation, architecture_id, convolution_output


def sinusoidal_positions(length: int, dimension: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10000.0) / dimension)
    )
    embeddings = torch.zeros(1, length, dimension)
    embeddings[0, :, 0::2] = torch.sin(positions * frequencies)
    embeddings[0, :, 1::2] = torch.cos(positions * frequencies[: embeddings[0, :, 1::2].shape[-1]])
    return embeddings


class TransformerBlock(nn.Module):
    def __init__(self, config: dict[str, Any], drop_path: float) -> None:
        super().__init__()
        dimension = int(config["embed_dim"])
        heads = int(config["heads"])
        epsilon = float(config["layernorm_eps"])

        def norm_factory() -> nn.LayerNorm:
            return nn.LayerNorm(dimension, eps=epsilon)

        self.pre_norm = bool(config["pre_layernorm"])
        self.norm1 = norm_factory()
        self.norm2 = norm_factory()
        self.attention = nn.MultiheadAttention(
            dimension,
            heads,
            dropout=float(config["attention_dropout"]),
            bias=bool(config["qkv_bias"]),
            batch_first=True,
        )
        hidden = int(config.get("mlp_hidden_dim") or round(dimension * float(config["mlp_ratio"])))
        self.mlp = nn.Sequential(
            nn.Linear(dimension, hidden),
            activation(str(config["activation"])),
            nn.Dropout(float(config["mlp_dropout"])),
            nn.Linear(hidden, dimension),
            nn.Dropout(float(config["mlp_dropout"])),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            normalized = self.norm1(inputs)
            attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
            output = inputs + self.drop_path(attended)
            return output + self.drop_path(self.mlp(self.norm2(output)))
        attended = self.attention(inputs, inputs, inputs, need_weights=False)[0]
        output = self.norm1(inputs + self.drop_path(attended))
        return self.norm2(output + self.drop_path(self.mlp(output)))


class VisionTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        channels = int(config["input_channels"])
        dimension = int(config["embed_dim"])
        self.conv_stem: nn.Module
        if bool(config["conv_stem"]):
            stem_channels = int(config["conv_stem_channels"])
            self.conv_stem = nn.Sequential(
                nn.Conv2d(channels, stem_channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_channels),
                nn.GELU(),
                nn.Conv2d(stem_channels, channels, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            )
        else:
            self.conv_stem = nn.Identity()
        self.patch_projection = nn.Conv2d(
            channels,
            dimension,
            int(config["patch_kernel"]),
            stride=int(config["patch_stride"]),
            padding=int(config["patch_padding"]),
        )
        image_size = int(config["image_size"])
        grid = convolution_output(
            image_size,
            int(config["patch_kernel"]),
            int(config["patch_stride"]),
            int(config["patch_padding"]),
            1,
        )
        self.patch_grid = (grid, grid)
        patch_count = grid * grid
        self.use_cls = bool(config["cls_token"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension)) if self.use_cls else None
        sequence_length = patch_count + int(self.use_cls)
        positional = str(config["positional_embedding"])
        if positional == "learned":
            self.positional_embedding = nn.Parameter(torch.zeros(1, sequence_length, dimension))
            nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        elif positional == "fixed":
            self.register_buffer(
                "positional_embedding",
                sinusoidal_positions(sequence_length, dimension),
                persistent=True,
            )
        else:
            self.positional_embedding = None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.embedding_dropout = nn.Dropout(float(config["embedding_dropout"]))
        depth = int(config["depth"])
        schedule = str(config["drop_path_schedule"])
        blocks = []
        for index in range(depth):
            maximum = float(config["stochastic_depth"])
            probability = maximum if schedule == "uniform" else maximum * index / max(1, depth - 1)
            blocks.append(TransformerBlock(config, probability))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(dimension, eps=float(config["layernorm_eps"]))
        self.pooling = str(config["pooling"])
        head_depth = int(config["head_depth"])
        head_dropout = float(config["head_dropout"])
        head: list[nn.Module] = []
        for _ in range(max(0, head_depth - 1)):
            head.extend([nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(head_dropout)])
        head.extend([nn.Dropout(head_dropout), nn.Linear(dimension, int(config["num_classes"]))])
        self.head = nn.Sequential(*head)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.patch_projection(self.conv_stem(inputs)).flatten(2).transpose(1, 2)
        if self.cls_token is not None:
            output = torch.cat([self.cls_token.expand(output.shape[0], -1, -1), output], dim=1)
        if self.positional_embedding is not None:
            output = output + self.positional_embedding
        output = self.embedding_dropout(output)
        for block in self.blocks:
            output = block(output)
        output = self.final_norm(output)
        if self.pooling == "cls":
            pooled = output[:, 0]
        elif self.cls_token is not None:
            pooled = output[:, 1:].mean(dim=1)
        else:
            pooled = output.mean(dim=1)
        return self.head(pooled)


class VisionTransformerAdapter:
    name = "transformer"
    baseline: dict[str, Any] = {
        "image_size": 32,
        "input_channels": 3,
        "num_classes": 10,
        "patch_kernel": 4,
        "patch_stride": 4,
        "patch_padding": 0,
        "conv_stem": False,
        "conv_stem_channels": 32,
        "embed_dim": 128,
        "depth": 4,
        "heads": 4,
        "head_dim": None,
        "mlp_ratio": 2.0,
        "mlp_hidden_dim": None,
        "qkv_bias": True,
        "embedding_dropout": 0.1,
        "attention_dropout": 0.1,
        "mlp_dropout": 0.1,
        "stochastic_depth": 0.0,
        "drop_path_schedule": "linear",
        "activation": "gelu",
        "normalization": "layernorm",
        "layernorm_eps": 1e-5,
        "pre_layernorm": True,
        "positional_embedding": "learned",
        "cls_token": True,
        "pooling": "cls",
        "head_depth": 1,
        "head_dropout": 0.0,
    }

    def merged(self, config: dict[str, Any]) -> dict[str, Any]:
        return {**self.baseline, **config}

    def resolve_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return the complete resolved configuration stored with every trial."""
        return self.merged(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        cfg = self.merged(config)
        dimension = int(cfg["embed_dim"])
        heads = int(cfg["heads"])
        if dimension <= 0 or heads <= 0 or int(cfg["depth"]) <= 0:
            raise ConfigurationError("embedding dimension, heads, and depth must be positive")
        if dimension % heads:
            raise ConfigurationError("embedding dimension must be divisible by attention heads")
        if cfg.get("head_dim") is not None and int(cfg["head_dim"]) != dimension // heads:
            raise ConfigurationError("explicit head_dim must equal embed_dim // heads")
        grid = convolution_output(
            int(cfg["image_size"]),
            int(cfg["patch_kernel"]),
            int(cfg["patch_stride"]),
            int(cfg["patch_padding"]),
            1,
        )
        if grid <= 0:
            raise ConfigurationError("patch projection produces an invalid token grid")
        if float(cfg["mlp_ratio"]) <= 0:
            raise ConfigurationError("MLP ratio must be positive")
        for key in (
            "embedding_dropout",
            "attention_dropout",
            "mlp_dropout",
            "stochastic_depth",
            "head_dropout",
        ):
            if not 0 <= float(cfg[key]) < 1:
                raise ConfigurationError(f"{key} must be in [0, 1)")
        if cfg["pooling"] == "cls" and not bool(cfg["cls_token"]):
            raise ConfigurationError("CLS pooling requires cls_token=true")
        if cfg["positional_embedding"] not in {"learned", "fixed", "none"}:
            raise ConfigurationError("positional_embedding must be learned, fixed, or none")
        if cfg["drop_path_schedule"] not in {"linear", "uniform"}:
            raise ConfigurationError("drop_path_schedule must be linear or uniform")
        if cfg["normalization"] != "layernorm":
            raise ConfigurationError("this adapter currently supports LayerNorm only")
        if int(cfg["head_depth"]) <= 0:
            raise ConfigurationError("head_depth must be positive")

    def build_model(self, config: dict[str, Any]) -> nn.Module:
        cfg = self.merged(config)
        self.validate_config(cfg)
        return VisionTransformer(cfg)

    def simple_search_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "optimizer": ParameterSpec("optimizer", values=("adamw",)),
                "learning_rate": ParameterSpec(
                    "learning_rate", "float", low=1e-5, high=3e-3, log=True
                ),
                "weight_decay": ParameterSpec(
                    "weight_decay", "float", low=1e-5, high=0.2, log=True
                ),
                "batch_size": ParameterSpec("batch_size", values=(64, 128, 256)),
                "beta1": ParameterSpec("beta1", "float", low=0.85, high=0.95),
                "beta2": ParameterSpec("beta2", values=(0.98, 0.99, 0.999)),
                "scheduler": ParameterSpec("scheduler", values=("cosine", "none")),
                "warmup_epochs": ParameterSpec("warmup_epochs", "int", low=0, high=10),
                "embedding_dropout": ParameterSpec("embedding_dropout", "float", low=0.0, high=0.3),
                "attention_dropout": ParameterSpec("attention_dropout", "float", low=0.0, high=0.3),
                "mlp_dropout": ParameterSpec("mlp_dropout", "float", low=0.0, high=0.3),
                "stochastic_depth": ParameterSpec("stochastic_depth", "float", low=0.0, high=0.3),
                "label_smoothing": ParameterSpec("label_smoothing", "float", low=0.0, high=0.2),
                "gradient_clip": ParameterSpec("gradient_clip", values=(0.0, 1.0, 5.0)),
                "gradient_accumulation": ParameterSpec("gradient_accumulation", values=(1, 2, 4)),
                "early_stopping_patience": ParameterSpec(
                    "early_stopping_patience", values=(0, 5, 10)
                ),
            }
        )

    def thorough_search_space(self) -> SearchSpace:
        params = dict(self.simple_search_space().parameters)
        params.update(
            {
                "patch_kernel": ParameterSpec("patch_kernel", values=(2, 4, 8)),
                "patch_stride": ParameterSpec("patch_stride", values=(2, 4, 8)),
                "patch_padding": ParameterSpec("patch_padding", values=(0, 1)),
                "conv_stem": ParameterSpec("conv_stem", values=(False, True)),
                "embed_dim": ParameterSpec("embed_dim", values=(64, 128, 192, 256, 384)),
                "depth": ParameterSpec("depth", values=(2, 4, 6, 8)),
                "heads": ParameterSpec("heads", values=(2, 4, 6, 8)),
                "mlp_ratio": ParameterSpec("mlp_ratio", values=(2.0, 3.0, 4.0)),
                "qkv_bias": ParameterSpec("qkv_bias", values=(False, True)),
                "drop_path_schedule": ParameterSpec(
                    "drop_path_schedule", values=("linear", "uniform")
                ),
                "activation": ParameterSpec("activation", values=("gelu", "silu", "relu")),
                "pre_layernorm": ParameterSpec("pre_layernorm", values=(False, True)),
                "positional_embedding": ParameterSpec(
                    "positional_embedding", values=("learned", "fixed")
                ),
                "cls_token": ParameterSpec("cls_token", values=(False, True)),
                "pooling": ParameterSpec("pooling", values=("cls", "mean")),
                "head_depth": ParameterSpec("head_depth", values=(1, 2)),
                "head_dropout": ParameterSpec("head_dropout", "float", low=0.0, high=0.5),
            }
        )
        return SearchSpace(params)

    def build_optimizer(self, model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
        optimizer = str(config.get("optimizer", "adamw"))
        kwargs = {
            "lr": float(config.get("learning_rate", 3e-4)),
            "weight_decay": float(config.get("weight_decay", 0.05)),
            "betas": (float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
        }
        if optimizer == "adamw":
            return torch.optim.AdamW(model.parameters(), **kwargs)
        if optimizer == "adam":
            return torch.optim.Adam(model.parameters(), **kwargs)
        raise ConfigurationError(f"unsupported Transformer optimizer {optimizer!r}")

    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        budget: ResourceBudget,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        if str(config.get("scheduler", "cosine")) == "none":
            return None
        total_steps = max(
            1, int(budget.value * steps_per_epoch if budget.kind == "epochs" else budget.value)
        )
        warmup_steps = max(0, int(float(config.get("warmup_epochs", 0)) * steps_per_epoch))
        if warmup_steps >= total_steps:
            raise ConfigurationError("warmup must be shorter than the total training budget")
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps)
        )
        if warmup_steps == 0:
            return cosine
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    def describe_model(self, model: nn.Module, config: dict[str, Any]) -> ModelDescription:
        cfg = self.merged(config)
        grid = model.patch_grid
        patches = grid[0] * grid[1]
        sequence = patches + int(bool(cfg["cls_token"]))
        dimension = int(cfg["embed_dim"])
        depth = int(cfg["depth"])
        hidden = int(cfg.get("mlp_hidden_dim") or round(dimension * float(cfg["mlp_ratio"])))
        attention_macs = depth * (
            4 * sequence * dimension * dimension + 2 * sequence * sequence * dimension
        )
        mlp_macs = depth * 2 * sequence * dimension * hidden
        projection_macs = (
            patches * dimension * int(cfg["input_channels"]) * int(cfg["patch_kernel"]) ** 2
        )
        macs = float(attention_macs + mlp_macs + projection_macs)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        activation_mb = sequence * dimension * max(1, depth) * 4 / 2**20
        return ModelDescription(
            architecture_id=architecture_id({key: cfg[key] for key in self.baseline}),
            parameter_count=parameters,
            trainable_parameter_count=trainable,
            flops_per_forward=2 * macs,
            macs_per_forward=macs,
            activation_memory_mb=activation_mb,
            details={
                "patch_grid": grid,
                "num_patches": patches,
                "sequence_length": sequence,
                "head_dimension": dimension // int(cfg["heads"]),
                "attention_complexity": depth * sequence * sequence * dimension,
            },
        )

    def calibration_configs(self) -> list[dict[str, Any]]:
        return [
            {"embed_dim": 64, "depth": 2, "heads": 2},
            {},
            {"embed_dim": 384, "depth": 6, "heads": 6, "mlp_ratio": 4.0},
        ]
