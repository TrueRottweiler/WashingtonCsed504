# Verification record — 2026-07-17

## Environment

- Linux x86_64 container
- Python 3.13.5
- PyTorch 2.10.0+cpu
- Torchvision 0.25.0+cpu
- Optuna 4.8.0
- CUDA and Apple MPS unavailable

## Measured checks

| Check | Command summary | Result |
|---|---|---|
| Ruff lint | `python -m ruff check src/cv_search tests examples` | Passed |
| Ruff formatting | `python -m ruff format --check src/cv_search tests examples` | Passed; 48 files formatted |
| Python syntax | `python -m compileall -q src tests examples` | Passed |
| Shell syntax | `bash -n` on Linux/macOS and ImageNet-32 queue scripts | Passed |
| Unit tests | `pytest -q tests/unit` | **19 passed** |
| Core engine integration tests | `pytest -q tests/integration/test_engine.py` | **4 passed** |
| Parallel and DDP integration tests | `pytest -q tests/integration/test_parallel_execution.py` | **2 passed** |
| Smoke tests | `pytest -q tests/smoke` | **1 passed, 3 CUDA-only skips** |
| Config validation | Load every TOML and construct each registered baseline model | **15 configs passed; 15 models constructed** |
| Notebook validation | Parse the Colab notebook and transform/compile every IPython code cell | **35 cells, including 17 code cells, passed** |
| CLI validation | CNN, Transformer, and generic search CLI `--help` | Passed |
| Wheel build | `pip wheel --no-deps .` | Passed; version 0.3.0 |
| Clean package import | Install wheel into an isolated target and import outside the repository | Passed |
| Registry check | Import installed adapters and datasets | 4 model aliases and 2 datasets registered |
| Personal path and credential scan | Scan text assets outside generated results and caches | Passed; no matches after clearing three stale notebook traceback outputs |
| ImageNet monitor | `monitor.py --once` | Passed on CPU-only Linux |
| ImageNet scheduler | `scheduler.py --plan` | Passed |

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` was used for deterministic pytest execution because the host environment injects unrelated third-party plugins. Test groups were executed independently and terminated normally.

## Measured parallel execution

Two CPU-only synthetic execution paths were run because CUDA hardware was unavailable:

1. **Independent trial scheduling:** four trial executions across two persistent isolated worker interpreters.
2. **Hybrid execution:** two concurrent proxy tasks followed by one two-rank Gloo DDP confirmation task.

Detailed measurements are in `benchmarks/PARALLEL_CPU_SMOKE_2026-07-17.md`. These measurements verify scheduler, persistence, metric-reduction, and checkpoint behavior; they are not GPU-scaling or accuracy benchmarks.

## External `torchrun` compatibility

The distributed trial worker was also launched with:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  -m cv_search.cli.trial_worker --task-file <task.json>
```

The two-rank CPU/Gloo trial completed with synchronized metrics, world size 2, global batch size 8, and rank-zero checkpoint ownership.

## Not available in this environment

CUDA, Apple MPS, multi-GPU NCCL throughput, PCIe/NVLink scaling, Windows runtime, macOS runtime, fresh Colab execution, CIFAR download, long-run accuracy, and paid-cloud cost measurements were not executed. CUDA-specific tests are included and skip cleanly unless at least two visible CUDA devices are present. No GPU speedup or GPU utilization claim is inferred from the CPU verification.
