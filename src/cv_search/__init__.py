"""Reusable computer-vision hyperparameter optimization framework."""

from . import adapters as _adapters  # noqa: F401
from . import data as _data  # noqa: F401
from .config import StudyConfig, load_config
from .engine import SearchEngine
from .registry import (
    create_dataset,
    create_model_adapter,
    register_dataset,
    register_model_adapter,
)
from .types import ModelDescription, ResourceBudget, TrialResult

__all__ = [
    "ModelDescription",
    "ResourceBudget",
    "SearchEngine",
    "StudyConfig",
    "TrialResult",
    "create_dataset",
    "create_model_adapter",
    "load_config",
    "register_dataset",
    "register_model_adapter",
]
