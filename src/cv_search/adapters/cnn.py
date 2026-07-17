"""Configurable CIFAR-friendly residual CNN adapter."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from ..exceptions import ConfigurationError
from ..search_space import ParameterSpec, SearchSpace
from ..types import ModelDescription, ResourceBudget
from .common import (
    DropPath,
    activation,
    architecture_id,
    convolution_output,
    normalization,
    resolve_padding,
)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = max(1, int(round(channels * ratio)))
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.layers(inputs)


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        config: dict[str, Any],
        drop_path: float,
    ) -> None:
        super().__init__()
        kernel = int(config["block_kernel"])
        dilation = int(config["dilation"])
        padding = resolve_padding(
            str(config["padding_policy"]),
            kernel,
            dilation,
            int(config["explicit_padding"]),
            1,
        )
        conv_mode = str(config["conv_mode"])
        groups = int(config["groups"])
        bias = False

        def convolution(cin: int, cout: int, conv_stride: int) -> nn.Module:
            if conv_mode == "depthwise":
                return nn.Sequential(
                    nn.Conv2d(
                        cin,
                        cin,
                        kernel,
                        stride=conv_stride,
                        padding=padding,
                        dilation=dilation,
                        groups=cin,
                        bias=bias,
                        padding_mode=str(config["padding_mode"]),
                    ),
                    nn.Conv2d(cin, cout, 1, bias=bias),
                )
            actual_groups = 1 if conv_mode == "standard" else groups
            return nn.Conv2d(
                cin,
                cout,
                kernel,
                stride=conv_stride,
                padding=padding,
                dilation=dilation,
                groups=actual_groups,
                bias=bias,
                padding_mode=str(config["padding_mode"]),
            )

        norm_name = str(config["normalization"])
        act_name = str(config["activation"])
        self.conv1 = convolution(in_channels, out_channels, stride)
        self.norm1 = normalization(norm_name, out_channels, int(config["norm_groups"]))
        self.act = activation(act_name)
        self.conv2 = convolution(out_channels, out_channels, 1)
        self.norm2 = normalization(norm_name, out_channels, int(config["norm_groups"]))
        self.spatial_dropout = nn.Dropout2d(float(config["spatial_dropout"]))
        self.se = (
            SqueezeExcitation(out_channels, float(config["se_ratio"]))
            if bool(config["squeeze_excitation"])
            else nn.Identity()
        )
        self.drop_path = DropPath(drop_path)
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                normalization(norm_name, out_channels, int(config["norm_groups"])),
            )
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = self.skip(inputs)
        output = self.act(self.norm1(self.conv1(inputs)))
        output = self.spatial_dropout(output)
        output = self.norm2(self.conv2(output))
        output = self.se(output)
        return self.act(identity + self.drop_path(output))


class ConfigurableResNet(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        stem_channels = int(config["stem_channels"])
        stem_kernel = int(config["stem_kernel"])
        stem_stride = int(config["stem_stride"])
        dilation = int(config["dilation"])
        stem_padding = resolve_padding(
            str(config["padding_policy"]),
            stem_kernel,
            dilation,
            int(config["explicit_padding"]),
            stem_stride,
        )
        self.stem = nn.Sequential(
            nn.Conv2d(
                int(config["input_channels"]),
                stem_channels,
                stem_kernel,
                stride=stem_stride,
                padding=stem_padding,
                dilation=dilation,
                bias=False,
                padding_mode=str(config["padding_mode"]),
            ),
            normalization(str(config["normalization"]), stem_channels, int(config["norm_groups"])),
            activation(str(config["activation"])),
        )
        pool_type = str(config["initial_pool"])
        if pool_type == "none":
            self.initial_pool = nn.Identity()
        else:
            pool_class = nn.MaxPool2d if pool_type == "max" else nn.AvgPool2d
            self.initial_pool = pool_class(
                int(config["pool_kernel"]),
                stride=int(config["pool_stride"]),
                padding=int(config["pool_padding"]),
            )
        stage_channels = [int(value) for value in config["stage_channels"]]
        stage_blocks = [int(value) for value in config["stage_blocks"]]
        stage_strides = [int(value) for value in config["stage_strides"]]
        total_blocks = sum(stage_blocks)
        block_index = 0
        current_channels = stem_channels
        stages: list[nn.Module] = []
        for channels, blocks, stride in zip(
            stage_channels, stage_blocks, stage_strides, strict=True
        ):
            layer: list[nn.Module] = []
            for index in range(blocks):
                probability = (
                    float(config["stochastic_depth"]) * block_index / max(1, total_blocks - 1)
                )
                layer.append(
                    ResidualBlock(
                        current_channels,
                        channels,
                        stride if index == 0 else 1,
                        config,
                        probability,
                    )
                )
                current_channels = channels
                block_index += 1
            stages.append(nn.Sequential(*layer))
        self.stages = nn.Sequential(*stages)
        global_pool = str(config["global_pool"])
        self.global_pool = global_pool
        feature_multiplier = 2 if global_pool == "avgmax" else 1
        self.dropout = nn.Dropout(float(config["classifier_dropout"]))
        self.classifier = nn.Linear(
            current_channels * feature_multiplier, int(config["num_classes"])
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.stages(self.initial_pool(self.stem(inputs)))
        if self.global_pool == "max":
            output = torch.amax(output, dim=(-2, -1))
        elif self.global_pool == "avgmax":
            output = torch.cat(
                [torch.mean(output, dim=(-2, -1)), torch.amax(output, dim=(-2, -1))], dim=1
            )
        else:
            output = torch.mean(output, dim=(-2, -1))
        return self.classifier(self.dropout(output))


class CIFARResNetAdapter:
    name = "cnn"
    baseline: dict[str, Any] = {
        "image_size": 32,
        "input_channels": 3,
        "num_classes": 10,
        "stem_channels": 64,
        "stage_channels": [64, 128, 256, 512],
        "stage_blocks": [2, 2, 2, 2],
        "stage_strides": [1, 2, 2, 2],
        "stem_kernel": 3,
        "stem_stride": 1,
        "block_kernel": 3,
        "padding_policy": "same",
        "explicit_padding": 1,
        "padding_mode": "zeros",
        "dilation": 1,
        "groups": 1,
        "conv_mode": "standard",
        "activation": "relu",
        "normalization": "batchnorm",
        "norm_groups": 8,
        "initial_pool": "none",
        "pool_kernel": 3,
        "pool_stride": 2,
        "pool_padding": 1,
        "global_pool": "avg",
        "classifier_dropout": 0.0,
        "spatial_dropout": 0.0,
        "stochastic_depth": 0.0,
        "squeeze_excitation": False,
        "se_ratio": 0.25,
    }

    def merged(self, config: dict[str, Any]) -> dict[str, Any]:
        merged = {**self.baseline, **config}
        if isinstance(merged.get("stage_blocks"), str):
            merged["stage_blocks"] = [
                int(value) for value in str(merged["stage_blocks"]).split(",")
            ]
        if isinstance(merged.get("stage_channels"), str):
            merged["stage_channels"] = [
                int(value) for value in str(merged["stage_channels"]).split(",")
            ]
        if isinstance(merged.get("stage_strides"), str):
            merged["stage_strides"] = [
                int(value) for value in str(merged["stage_strides"]).split(",")
            ]
        width_multiplier = float(merged.get("width_multiplier", 1.0))
        if "stage_channels" not in config and width_multiplier != 1.0:
            merged["stage_channels"] = [
                max(1, int(round(channel * width_multiplier)))
                for channel in self.baseline["stage_channels"]
            ]
            merged["stem_channels"] = max(
                1, int(round(int(self.baseline["stem_channels"]) * width_multiplier))
            )
        return merged

    def resolve_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Return the complete resolved configuration stored with every trial."""
        return self.merged(config)

    def validate_config(self, config: dict[str, Any]) -> None:
        cfg = self.merged(config)
        channels = [int(value) for value in cfg["stage_channels"]]
        blocks = [int(value) for value in cfg["stage_blocks"]]
        strides = [int(value) for value in cfg["stage_strides"]]
        if not (len(channels) == len(blocks) == len(strides) and channels):
            raise ConfigurationError(
                "stage channels, blocks, and strides must have equal nonzero length"
            )
        if any(value <= 0 for value in channels + blocks + strides):
            raise ConfigurationError("channel, block, and stride counts must be positive")
        if int(cfg["stem_channels"]) <= 0:
            raise ConfigurationError("stem_channels must be positive")
        if cfg["padding_mode"] not in {"zeros", "reflect", "replicate", "circular"}:
            raise ConfigurationError("unsupported padding mode")
        if cfg["conv_mode"] not in {"standard", "grouped", "depthwise"}:
            raise ConfigurationError("conv_mode must be standard, grouped, or depthwise")
        groups = int(cfg["groups"])
        if groups <= 0:
            raise ConfigurationError("groups must be positive")
        if cfg["conv_mode"] == "grouped":
            input_channels = [int(cfg["stem_channels"]), *channels[:-1]]
            for left, right in zip(input_channels, channels, strict=True):
                if left % groups or right % groups:
                    raise ConfigurationError(
                        "group count must divide residual input and output channels"
                    )
        for key in ("classifier_dropout", "spatial_dropout", "stochastic_depth"):
            if not 0 <= float(cfg[key]) < 1:
                raise ConfigurationError(f"{key} must be in [0, 1)")
        image = int(cfg["image_size"])
        stem_padding = resolve_padding(
            str(cfg["padding_policy"]),
            int(cfg["stem_kernel"]),
            int(cfg["dilation"]),
            int(cfg["explicit_padding"]),
            int(cfg["stem_stride"]),
        )
        size = convolution_output(
            image,
            int(cfg["stem_kernel"]),
            int(cfg["stem_stride"]),
            stem_padding,
            int(cfg["dilation"]),
        )
        if cfg["initial_pool"] != "none":
            size = convolution_output(
                size,
                int(cfg["pool_kernel"]),
                int(cfg["pool_stride"]),
                int(cfg["pool_padding"]),
                1,
            )
        for stride in strides:
            size = math.ceil(size / stride)
        if size <= 0:
            raise ConfigurationError(
                "convolution and pooling settings produce an invalid feature map"
            )
        early_reduction = int(cfg["stem_stride"]) * (
            int(cfg["pool_stride"]) if cfg["initial_pool"] != "none" else 1
        )
        if (
            image <= 32
            and early_reduction > 2
            and not bool(cfg.get("allow_aggressive_downsampling", False))
        ):
            raise ConfigurationError(
                "excessive early downsampling for 32x32 input; explicitly enable allow_aggressive_downsampling"
            )

    def build_model(self, config: dict[str, Any]) -> nn.Module:
        cfg = self.merged(config)
        self.validate_config(cfg)
        return ConfigurableResNet(cfg)

    def simple_search_space(self) -> SearchSpace:
        return SearchSpace(
            {
                "optimizer": ParameterSpec("optimizer", values=("sgd", "adamw")),
                "sgd_learning_rate": ParameterSpec(
                    "sgd_learning_rate",
                    "float",
                    low=1e-3,
                    high=0.2,
                    log=True,
                    condition={"optimizer": ("sgd",)},
                ),
                "adamw_learning_rate": ParameterSpec(
                    "adamw_learning_rate",
                    "float",
                    low=1e-5,
                    high=3e-3,
                    log=True,
                    condition={"optimizer": ("adamw",)},
                ),
                "sgd_weight_decay": ParameterSpec(
                    "sgd_weight_decay",
                    "float",
                    low=1e-6,
                    high=5e-3,
                    log=True,
                    condition={"optimizer": ("sgd",)},
                ),
                "adamw_weight_decay": ParameterSpec(
                    "adamw_weight_decay",
                    "float",
                    low=1e-5,
                    high=0.1,
                    log=True,
                    condition={"optimizer": ("adamw",)},
                ),
                "batch_size": ParameterSpec("batch_size", values=(64, 128, 256)),
                "momentum": ParameterSpec(
                    "momentum", "float", low=0.8, high=0.99, condition={"optimizer": ("sgd",)}
                ),
                "nesterov": ParameterSpec(
                    "nesterov", values=(False, True), condition={"optimizer": ("sgd",)}
                ),
                "beta1": ParameterSpec(
                    "beta1", "float", low=0.85, high=0.95, condition={"optimizer": ("adamw",)}
                ),
                "beta2": ParameterSpec(
                    "beta2", values=(0.99, 0.999), condition={"optimizer": ("adamw",)}
                ),
                "scheduler": ParameterSpec("scheduler", values=("cosine", "step", "none")),
                "warmup_epochs": ParameterSpec("warmup_epochs", "int", low=0, high=5),
                "label_smoothing": ParameterSpec("label_smoothing", "float", low=0.0, high=0.2),
                "classifier_dropout": ParameterSpec(
                    "classifier_dropout", "float", low=0.0, high=0.5
                ),
                "stochastic_depth": ParameterSpec("stochastic_depth", "float", low=0.0, high=0.2),
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
                "width_multiplier": ParameterSpec(
                    "width_multiplier", values=(0.5, 0.75, 1.0, 1.25)
                ),
                "stage_blocks": ParameterSpec(
                    "stage_blocks", values=("1,1,1,1", "2,2,2,2", "3,4,6,3")
                ),
                "stem_kernel": ParameterSpec("stem_kernel", values=(3, 5, 7)),
                "stem_stride": ParameterSpec("stem_stride", values=(1, 2)),
                "block_kernel": ParameterSpec("block_kernel", values=(3, 5)),
                "padding_policy": ParameterSpec("padding_policy", values=("same", "valid")),
                "dilation": ParameterSpec("dilation", values=(1, 2)),
                "conv_mode": ParameterSpec(
                    "conv_mode", values=("standard", "grouped", "depthwise")
                ),
                "groups": ParameterSpec(
                    "groups", values=(2, 4, 8), condition={"conv_mode": ("grouped",)}
                ),
                "activation": ParameterSpec(
                    "activation", values=("relu", "gelu", "silu", "leaky_relu")
                ),
                "normalization": ParameterSpec(
                    "normalization", values=("batchnorm", "groupnorm", "layernorm")
                ),
                "initial_pool": ParameterSpec("initial_pool", values=("none", "max", "avg")),
                "global_pool": ParameterSpec("global_pool", values=("avg", "max", "avgmax")),
                "spatial_dropout": ParameterSpec("spatial_dropout", "float", low=0.0, high=0.2),
                "squeeze_excitation": ParameterSpec("squeeze_excitation", values=(False, True)),
            }
        )
        return SearchSpace(params)

    def build_optimizer(self, model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
        optimizer = str(config.get("optimizer", "sgd"))
        if optimizer == "sgd":
            learning_rate = float(config.get("learning_rate", config.get("sgd_learning_rate", 0.1)))
            weight_decay = float(config.get("weight_decay", config.get("sgd_weight_decay", 5e-4)))
            return torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=float(config.get("momentum", 0.9)),
                nesterov=bool(config.get("nesterov", True)),
                weight_decay=weight_decay,
            )
        if optimizer == "adam":
            learning_rate = float(
                config.get("learning_rate", config.get("adamw_learning_rate", 1e-3))
            )
            weight_decay = float(config.get("weight_decay", config.get("adamw_weight_decay", 1e-4)))
            return torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
            )
        if optimizer == "adamw":
            learning_rate = float(
                config.get("learning_rate", config.get("adamw_learning_rate", 3e-4))
            )
            weight_decay = float(config.get("weight_decay", config.get("adamw_weight_decay", 0.01)))
            return torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
            )
        raise ConfigurationError(f"unsupported optimizer {optimizer!r}")

    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        budget: ResourceBudget,
        steps_per_epoch: int,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        scheduler = str(config.get("scheduler", "cosine"))
        total_steps = max(
            1, int(budget.value * steps_per_epoch if budget.kind == "epochs" else budget.value)
        )
        warmup_steps = min(
            total_steps - 1,
            max(0, int(float(config.get("warmup_epochs", 0)) * steps_per_epoch)),
        )
        if scheduler == "none":
            return None
        if scheduler == "step":
            main: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, total_steps // 3), gamma=0.1
            )
        elif scheduler == "cosine":
            main = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps)
            )
        else:
            raise ConfigurationError(f"unsupported scheduler {scheduler!r}")
        if warmup_steps == 0:
            return main
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, main], milestones=[warmup_steps]
        )

    def describe_model(self, model: nn.Module, config: dict[str, Any]) -> ModelDescription:
        cfg = self.merged(config)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        feature_sizes: list[int] = []
        size = int(cfg["image_size"])
        size = math.ceil(size / int(cfg["stem_stride"]))
        if cfg["initial_pool"] != "none":
            size = math.ceil(size / int(cfg["pool_stride"]))
        for stride in cfg["stage_strides"]:
            size = math.ceil(size / int(stride))
            feature_sizes.append(size)
        # A transparent architecture-level approximation, not a profiler measurement.
        macs = 0.0
        cin = int(cfg["stem_channels"])
        current_size = int(cfg["image_size"])
        macs += (
            current_size
            * current_size
            * int(cfg["input_channels"])
            * cin
            * int(cfg["stem_kernel"]) ** 2
        )
        for channels, blocks, fmap in zip(
            cfg["stage_channels"], cfg["stage_blocks"], feature_sizes, strict=True
        ):
            macs += (
                int(blocks) * 2 * fmap * fmap * cin * int(channels) * int(cfg["block_kernel"]) ** 2
            )
            cin = int(channels)
        activation_mb = sum(
            int(channels) * fmap * fmap * 4 / 2**20
            for channels, fmap in zip(cfg["stage_channels"], feature_sizes, strict=True)
        )
        return ModelDescription(
            architecture_id=architecture_id({key: cfg[key] for key in self.baseline}),
            parameter_count=parameters,
            trainable_parameter_count=trainable,
            flops_per_forward=2 * macs,
            macs_per_forward=macs,
            activation_memory_mb=activation_mb,
            details={"feature_map_sizes": feature_sizes, "global_pool": cfg["global_pool"]},
        )

    def calibration_configs(self) -> list[dict[str, Any]]:
        return [
            {"width_multiplier": 0.5, "stage_blocks": [1, 1, 1, 1]},
            {},
            {"width_multiplier": 1.25, "stage_blocks": [2, 2, 2, 2]},
        ]
