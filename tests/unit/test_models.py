import pytest
import torch

from cv_search.adapters import CIFARResNetAdapter, VisionTransformerAdapter
from cv_search.exceptions import ConfigurationError


def test_cnn_validation_and_forward():
    adapter = CIFARResNetAdapter()
    model = adapter.build_model({"width_multiplier": 0.25, "stage_blocks": "1,1,1,1"})
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    with pytest.raises(ConfigurationError):
        adapter.validate_config({"conv_mode": "grouped", "groups": 3})
    with pytest.raises((ConfigurationError, ValueError)):
        adapter.validate_config({"stem_kernel": 4, "padding_policy": "same"})


def test_cnn_architecture_options_change_shape_metadata():
    adapter = CIFARResNetAdapter()
    avg = adapter.build_model(
        {"global_pool": "avg", "width_multiplier": 0.25, "stage_blocks": "1,1,1,1"}
    )
    avgmax = adapter.build_model(
        {"global_pool": "avgmax", "width_multiplier": 0.25, "stage_blocks": "1,1,1,1"}
    )
    assert avg.classifier.in_features * 2 == avgmax.classifier.in_features


def test_transformer_validation_forward_and_fixed_positions():
    adapter = VisionTransformerAdapter()
    model = adapter.build_model(
        {"embed_dim": 64, "depth": 2, "heads": 4, "positional_embedding": "fixed"}
    )
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)
    assert model.patch_grid == (8, 8)
    with pytest.raises(ConfigurationError):
        adapter.validate_config({"embed_dim": 130, "heads": 8})
    with pytest.raises(ConfigurationError):
        adapter.validate_config({"patch_kernel": 64})
    with pytest.raises(ConfigurationError):
        adapter.validate_config({"pooling": "cls", "cls_token": False})
