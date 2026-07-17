from __future__ import annotations

import pytest

from cv_search.distributed import DistributedContext
from cv_search.exceptions import ConfigurationError
from cv_search.training import ExactDistributedSampler, _per_device_batch_size


def test_exact_distributed_sampler_has_no_duplicates() -> None:
    shards = [list(ExactDistributedSampler(list(range(11)), rank, 3)) for rank in range(3)]
    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def test_global_batch_size_semantics() -> None:
    context = DistributedContext(rank=0, world_size=4, local_rank=0, backend="gloo")
    per_device, global_batch = _per_device_batch_size(
        {"batch_size": 128, "gradient_accumulation": 2},
        {"batch_size_scope": "global"},
        context,
    )
    assert per_device == 32
    assert global_batch == 256
    with pytest.raises(ConfigurationError):
        _per_device_batch_size({"batch_size": 130}, {"batch_size_scope": "global"}, context)
