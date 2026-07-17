"""Explicit registries for reusable model adapters and dataset factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .exceptions import ConfigurationError

_MODEL_ADAPTERS: dict[str, Callable[[], Any]] = {}
_DATASETS: dict[str, Callable[[dict[str, Any]], Any]] = {}


def register_model_adapter(name: str, factory: Callable[[], Any], *, replace: bool = False) -> None:
    key = name.lower()
    if key in _MODEL_ADAPTERS and not replace:
        raise ConfigurationError(f"model adapter already registered: {key}")
    _MODEL_ADAPTERS[key] = factory


def create_model_adapter(name: str) -> Any:
    try:
        return _MODEL_ADAPTERS[name.lower()]()
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown model adapter {name!r}; registered: {sorted(_MODEL_ADAPTERS)}"
        ) from exc


def register_dataset(
    name: str, factory: Callable[[dict[str, Any]], Any], *, replace: bool = False
) -> None:
    key = name.lower()
    if key in _DATASETS and not replace:
        raise ConfigurationError(f"dataset already registered: {key}")
    _DATASETS[key] = factory


def create_dataset(name: str, config: dict[str, Any]) -> Any:
    try:
        return _DATASETS[name.lower()](config)
    except KeyError as exc:
        raise ConfigurationError(
            f"unknown dataset {name!r}; registered: {sorted(_DATASETS)}"
        ) from exc


def registered_models() -> tuple[str, ...]:
    return tuple(sorted(_MODEL_ADAPTERS))


def registered_datasets() -> tuple[str, ...]:
    return tuple(sorted(_DATASETS))
