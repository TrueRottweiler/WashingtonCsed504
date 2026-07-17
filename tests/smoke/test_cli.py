import os
import subprocess
import sys

import pytest
import torch


def test_help_commands():
    env = os.environ | {"PYTHONPATH": "src", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for script in ("src/a1-cv/search_cnn.py", "src/a1-cv/search_transformer.py"):
        result = subprocess.run(
            [sys.executable, script, "--help"], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "--estimate-only" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cuda_available_when_requested():
    assert torch.zeros(1, device="cuda").is_cuda
