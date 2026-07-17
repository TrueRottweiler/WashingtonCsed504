"""Built-in model adapters and registry initialization."""

from ..registry import register_model_adapter
from .cnn import CIFARResNetAdapter
from .transformer import VisionTransformerAdapter

register_model_adapter("cnn", CIFARResNetAdapter, replace=True)
register_model_adapter("resnet", CIFARResNetAdapter, replace=True)
register_model_adapter("transformer", VisionTransformerAdapter, replace=True)
register_model_adapter("vit", VisionTransformerAdapter, replace=True)

__all__ = ["CIFARResNetAdapter", "VisionTransformerAdapter"]
