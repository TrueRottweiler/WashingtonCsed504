# Extending the framework

## Add another model

Implement the adapter contract in `cv_search.adapters.base.ModelSearchAdapter`:

- `baseline`
- `validate_config`
- `build_model`
- `simple_search_space`
- `thorough_search_space`
- `build_optimizer`
- `build_scheduler`
- `describe_model`
- `calibration_configs`

Register it before constructing the engine:

```python
from cv_search.registry import register_model_adapter
register_model_adapter("swin", SwinAdapter)
```

A hierarchical model such as Swin should expose its own meaningful parameters—window size, per-stage depths and heads, patch merging, and stage widths—rather than pretending the baseline ViT space applies unchanged.

## Add another dataset

A dataset factory accepts the TOML dataset table and returns `DatasetBundle` with train, validation, test, and immutable metadata.

```python
from cv_search.registry import register_dataset
register_dataset("my_images", build_my_dataset)
```

Keep split creation deterministic and ensure the test set is never included in training or selection.

## Custom objective

Add the result field to `TrialResult`, populate it in training/evaluation, and map its public name in `objectives.OBJECTIVE_FIELDS`. Objective direction is configured in TOML.

## Custom constraint

Add the observed result value to `constraint_violations`. Rejected trials remain serialized for auditability.

## Hooks

For project-specific metrics, prefer an adapter or evaluation extension over embedding model-name conditions in the engine. The engine should remain unaware of concrete model classes.

See `examples/register_custom_components.py` for a minimal working example.
