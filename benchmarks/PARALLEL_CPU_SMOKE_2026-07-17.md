# Parallel and DDP CPU smoke verification — 2026-07-17

These measurements verify execution behavior on the available CPU-only Linux container. They are **not model-quality or GPU-scaling benchmarks**.

## Environment

- Python 3.13.5
- PyTorch 2.10.0+cpu
- Torchvision 0.25.0+cpu
- CUDA unavailable
- Synthetic 32×32 RGB images
- Compact residual CNN (`width_multiplier = 0.125`, one block per stage)
- One optimization step per proxy invocation
- `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`

## Independent trial scheduling

Command:

```bash
PYTHONPATH=src python -m cv_search.cli.search \
  --config configs/searches/parallel_cpu_smoke.toml \
  --skip-calibration
```

Measured result:

| Metric | Value |
|---|---:|
| Completed trial executions | 4 |
| Configured concurrent slots | 2 |
| Distinct persistent worker PIDs | 2 |
| Stage wall time | 6.9872 s |
| Sum of worker training elapsed time | 0.3382 s |
| Measured scheduler efficiency | 2.42% |
| Result-file writer | Parent process only |
| Trial status | All completed |

Each persistent worker executed two configurations, and the recorded results preserve the explicit optimizer, learning-rate, batch-size, and compact-architecture values. Efficiency is intentionally low because Python/PyTorch process startup dominates sub-second CPU training. Long GPU trials should amortize that overhead, but target-hardware measurement is required.

## Hybrid stage policy

Command:

```bash
PYTHONPATH=src python -m cv_search.cli.search \
  --config configs/searches/hybrid_cpu_smoke.toml \
  --skip-calibration
```

The study used independent parallel workers for proxy exploration and a two-rank Gloo DDP group for full confirmation.

| Stage | Mode | Executions | Wall time | Allocated device-seconds | Scheduler efficiency |
|---|---|---:|---:|---:|---:|
| Proxy | `parallel_trials` | 2 | 7.0877 s | 0.2124 s | 1.50% |
| Full confirmation | `ddp` | 1 | 7.8892 s | 0.3933 s | 4.99% |
| **Total/mean** | Hybrid | 3 | **14.9768 s** | **0.6057 s** | **3.24% mean** |

The full result records DDP world size 2, global batch size 8, synchronized validation metrics, and rank-zero checkpoint ownership. The intentionally tiny CPU tasks are dominated by interpreter and process-group startup; these numbers do not predict GPU speedup.

## Two-process DDP integration

Test command:

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q tests/integration/test_parallel_execution.py
```

The DDP test uses two local Gloo ranks, a global batch size of 8, an uneven nine-example validation split, exact nonduplicating evaluation shards, explicit module-buffer synchronization, global metric reductions, and one rank-zero checkpoint. Both integration tests passed.

## GPU boundary

Three CUDA-specific smoke paths skip unless compatible hardware is available:

- the existing single-CUDA smoke test;
- two independent trials assigned to physical GPUs 0 and 1;
- a two-GPU NCCL DDP trial with synchronized metrics and a rank-zero checkpoint.

No CUDA speedup, NCCL throughput, peak GPU memory, PCIe/NVLink efficiency, or GPU utilization was measured in this environment.
