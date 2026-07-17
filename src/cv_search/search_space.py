"""Conditional search spaces usable by internal samplers and Optuna."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from random import Random
from typing import Any, Protocol

from .exceptions import ConfigurationError


class SuggestingTrial(Protocol):
    def suggest_categorical(self, name: str, choices: list[Any]) -> Any: ...
    def suggest_int(
        self, name: str, low: int, high: int, *, step: int = 1, log: bool = False
    ) -> int: ...
    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: float | None = None,
        log: bool = False,
    ) -> float: ...


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: str = "categorical"
    enabled: bool = True
    fixed: Any = None
    values: tuple[Any, ...] = ()
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    condition: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

    def active(self, config: Mapping[str, Any]) -> bool:
        return self.enabled and all(
            config.get(key) in values for key, values in self.condition.items()
        )

    def validate(self) -> None:
        if self.kind not in {"categorical", "int", "float", "fixed"}:
            raise ConfigurationError(f"{self.name}: unsupported parameter type {self.kind!r}")
        if not self.enabled or self.kind == "fixed":
            return
        if self.kind == "categorical" and not self.values:
            raise ConfigurationError(f"{self.name}: categorical values cannot be empty")
        if self.kind in {"int", "float"}:
            if self.low is None or self.high is None or self.low > self.high:
                raise ConfigurationError(f"{self.name}: valid low/high bounds are required")
            if self.log and (float(self.low) <= 0 or float(self.high) <= 0):
                raise ConfigurationError(f"{self.name}: logarithmic bounds must be positive")
            if self.log and self.step is not None:
                raise ConfigurationError(f"{self.name}: Optuna does not combine log and step")

    def grid_values(self) -> tuple[Any, ...]:
        if not self.enabled or self.kind == "fixed":
            return (self.fixed,)
        if self.kind == "categorical":
            return self.values
        if self.kind == "int":
            assert self.low is not None and self.high is not None
            return tuple(range(int(self.low), int(self.high) + 1, int(self.step or 1)))
        if self.kind == "float" and self.step is not None:
            assert self.low is not None and self.high is not None
            count = int(math.floor((float(self.high) - float(self.low)) / float(self.step)))
            return tuple(float(self.low) + index * float(self.step) for index in range(count + 1))
        raise ConfigurationError(
            f"{self.name}: a continuous float needs values or step for exhaustive grid search"
        )


@dataclass
class SearchSpace:
    parameters: dict[str, ParameterSpec]
    explicit: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        for spec in self.parameters.values():
            spec.validate()

    def fixed_values(self) -> dict[str, Any]:
        return {
            name: spec.fixed
            for name, spec in self.parameters.items()
            if not spec.enabled or spec.kind == "fixed"
        }

    def sample_random(self, count: int, seed: int = 42) -> list[dict[str, Any]]:
        rng = Random(seed)
        return [self._sample_one(rng) for _ in range(count)]

    def _sample_one(self, rng: Random) -> dict[str, Any]:
        config = self.fixed_values()
        for name, spec in self.parameters.items():
            if name in config or not spec.active(config):
                continue
            if spec.kind == "categorical":
                config[name] = rng.choice(spec.values)
            elif spec.kind == "int":
                assert spec.low is not None and spec.high is not None
                config[name] = rng.randrange(int(spec.low), int(spec.high) + 1, int(spec.step or 1))
            elif spec.kind == "float":
                assert spec.low is not None and spec.high is not None
                if spec.log:
                    config[name] = math.exp(
                        rng.uniform(math.log(float(spec.low)), math.log(float(spec.high)))
                    )
                elif spec.step:
                    values = spec.grid_values()
                    config[name] = rng.choice(values)
                else:
                    config[name] = rng.uniform(float(spec.low), float(spec.high))
        return config

    def suggest(self, trial: SuggestingTrial) -> dict[str, Any]:
        config = self.fixed_values()
        for name, spec in self.parameters.items():
            if name in config or not spec.active(config):
                continue
            if spec.kind == "categorical":
                config[name] = trial.suggest_categorical(name, list(spec.values))
            elif spec.kind == "int":
                assert spec.low is not None and spec.high is not None
                config[name] = trial.suggest_int(
                    name,
                    int(spec.low),
                    int(spec.high),
                    step=int(spec.step or 1),
                    log=spec.log,
                )
            elif spec.kind == "float":
                assert spec.low is not None and spec.high is not None
                config[name] = trial.suggest_float(
                    name,
                    float(spec.low),
                    float(spec.high),
                    step=float(spec.step) if spec.step is not None else None,
                    log=spec.log,
                )
        return config

    def grid(self) -> list[dict[str, Any]]:
        if self.explicit:
            return [dict(config) for config in self.explicit]
        variable = [
            spec for spec in self.parameters.values() if spec.enabled and spec.kind != "fixed"
        ]
        names = [spec.name for spec in variable]
        candidates: list[dict[str, Any]] = []
        for values in product(*(spec.grid_values() for spec in variable)):
            config = self.fixed_values() | dict(zip(names, values, strict=True))
            config = {
                name: value
                for name, value in config.items()
                if name not in self.parameters
                or self.parameters[name].active(config)
                or name in self.fixed_values()
            }
            if config not in candidates:
                candidates.append(config)
        return candidates

    def finite_size(self) -> int | None:
        if self.explicit:
            return len(self.explicit)
        try:
            return len(self.grid())
        except ConfigurationError:
            return None


def parameter_from_dict(name: str, raw: Mapping[str, Any]) -> ParameterSpec:
    values = raw.get("values", ())
    condition = {
        key: tuple(value if isinstance(value, list) else [value])
        for key, value in raw.get("condition", {}).items()
    }
    legacy_parent = raw.get("condition_parameter")
    if legacy_parent is not None:
        legacy_values = raw.get("condition_values", ())
        condition[str(legacy_parent)] = tuple(
            legacy_values if isinstance(legacy_values, list) else [legacy_values]
        )
    enabled = bool(raw.get("enabled", True))
    fixed = raw.get("fixed")
    kind = str(raw.get("type", "fixed" if not enabled else "categorical"))
    return ParameterSpec(
        name=name,
        kind=kind,
        enabled=enabled,
        fixed=fixed,
        values=tuple(values if isinstance(values, list) else [values]) if values else (),
        low=raw.get("low"),
        high=raw.get("high"),
        step=raw.get("step"),
        log=bool(raw.get("log", False)),
        condition=condition,
    )
