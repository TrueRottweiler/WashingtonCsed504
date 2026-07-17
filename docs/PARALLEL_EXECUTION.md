# Parallel GPU and distributed execution

## Execution modes

The framework supports four concrete execution policies plus automatic selection:

- `serial`: one trial in the parent process.
- `parallel_trials`: one isolated worker subprocess per device slot. This is the preferred search mode when each model fits on one GPU.
- `ddp`: one trial launched as a local rank group, using one process per GPU and the same environment contract as `torchrun`.
- `hybrid`: stage-specific settings select parallel trials for proxy/halving and DDP for confirmation.
- `auto`: selects parallel trials only when multiple CUDA GPUs and concurrency greater than one are available; otherwise serial.

The parent process owns Optuna `ask`/`tell`, JSONL, CSV, plots, and reports. Workers return typed results. This avoids concurrent SQLite and result-file writes.

## One trial per GPU

```toml
[execution]
device = "auto"
parallel_mode = "parallel_trials"
trial_concurrency = "auto"
gpu_indices = [0, 1, 2, 3]
gpus_per_trial = 1
num_workers_total = 16
cpu_core_reserve = 2
```

The scheduler creates one long-lived isolated worker subprocess per independent-trial slot and dynamically gives a free slot the next task. Keeping the interpreter alive amortizes PyTorch import, CUDA-context, and dataset-initialization overhead; the dataset registry cache remains local to that worker. A worker is pinned to its assigned physical CUDA index for that invocation. Proxy candidates are asked lazily, so completed results are reported to Optuna before replacement candidates are sampled.

Successive halving remains rung-synchronous: candidates within one rung run concurrently, all available rung results are ranked, and only promoted candidates enter the next rung. Finalist/seed combinations are independent and also run concurrently.

## DDP for one large trial

```toml
[execution]
parallel_mode = "ddp"
gpu_indices = [0, 1, 2, 3]
gpus_per_trial = 4
trial_concurrency = 1
distributed_backend = "auto"
batch_size_scope = "global"
learning_rate_scaling = "none"
```

The internal single-node launcher restricts `CUDA_VISIBLE_DEVICES` to the selected group, chooses a local rendezvous port, and starts one rank process per selected GPU with `MASTER_ADDR`, `MASTER_PORT`, `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`. This avoids the memory overhead of a separate elastic-agent process while keeping the worker compatible with an external `torchrun` launch. For cluster-managed or multi-node jobs, use `torchrun` directly.

Each rank:

1. initializes from `RANK`, `WORLD_SIZE`, and `LOCAL_RANK`;
2. exclusively selects one GPU;
3. constructs its own model, optimizer, scheduler, dataset, and loader;
4. wraps the model in `DistributedDataParallel`;
5. uses `DistributedSampler` for training;
6. uses an exact nonpadding shard for validation and test data;
7. all-reduces loss sums, correct predictions, and example counts;
8. uses `no_sync()` on non-step accumulation batches;
9. participates in distributed metric and checkpoint synchronization.

Only global rank zero writes the checkpoint and result file. Checkpoints store the unwrapped model, optimizer, scheduler, scaler, counters, history, and one RNG state per rank.

## Global batch semantics

With `batch_size_scope = "global"`:

```text
global batch = configured batch size × gradient accumulation
per-GPU batch = configured batch size ÷ world size
```

The configured batch size must divide evenly by the DDP world size. The framework rejects invalid values instead of silently rounding.

With `batch_size_scope = "per_gpu"`:

```text
global batch = configured batch size × world size × gradient accumulation
```

Changing world size then changes the scientific training configuration. The default is `global`.

Learning-rate scaling is never implicit. The available deliberate policies are:

```toml
learning_rate_scaling = "none"   # default
learning_rate_scaling = "linear"
learning_rate_scaling = "sqrt"
```

The resolved base and scaled learning rates are recorded with the trial.

## Hybrid stages

```toml
[execution]
parallel_mode = "hybrid"
trial_concurrency = "auto"
gpu_indices = [0, 1, 2, 3]

[stages.proxy.execution]
parallel_mode = "parallel_trials"
gpus_per_trial = 1
trial_concurrency = "auto"

[stages.halving.execution]
parallel_mode = "parallel_trials"
gpus_per_trial = 1
trial_concurrency = "auto"

[stages.full.execution]
parallel_mode = "ddp"
gpus_per_trial = 2
trial_concurrency = "auto"
```

This pair-based DDP pattern is the recommended hybrid example for the current CIFAR-scale ResNet and ViT experiments because the built-in batch-size choices are divisible by two. On four GPUs it runs two independent two-GPU confirmation jobs concurrently; broader proxy and halving stages still use one trial per GPU. Set `gpus_per_trial = "all"` only when every candidate's global batch size is divisible by the selected world size.

## CPU and DataLoader partitioning

`num_workers_total` is divided among simultaneously active trial slots. The framework also partitions intra-op CPU threads while reserving `cpu_core_reserve` physical cores for the host. Each slot starts from a fresh Python interpreter and remains isolated from the parent; DataLoader workers use the `spawn` context for CUDA runs to avoid inherited CUDA/NCCL state. Set `persistent_trial_workers = false` to use a fresh interpreter for every trial.

## Persistence

For local parallel trials, only the parent process accesses Optuna, so the existing SQLite study remains safe. DDP ranks never access Optuna directly.

If future multi-node workers independently own Optuna studies, use Optuna JournalStorage on one host or a client/server RDB such as PostgreSQL/MySQL rather than SQLite.

## Result telemetry

Every trial records:

- worker PID;
- physical GPU indices;
- parallel mode;
- DDP world size;
- global batch size;
- elapsed time, GPU-hours, and CPU-hours;
- peak per-device GPU memory;
- rank-zero checkpoint path.

Stage logs include measured wall time, allocated device-seconds, and scheduler efficiency. `parallel_summary.json` aggregates those measured stage records. Runtime previews use an explicit parallel-efficiency assumption rather than dividing by concurrency as if scaling were perfect.

## Commands

Parallel trial search:

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10.toml \
  --parallel-mode parallel_trials \
  --trial-concurrency auto \
  --gpu-indices 0,1,2,3
```

DDP confirmation-oriented run:

```bash
python src/a1-cv/search_transformer.py \
  --config configs/searches/vit_cifar10_multi_gpu.toml \
  --gpu-indices 0,1,2,3
```

Portable CPU process smoke test:

```bash
python -m cv_search.cli.search \
  --config configs/searches/parallel_cpu_smoke.toml \
  --skip-calibration
```

## Verification boundary

CPU spawned-process and two-process Gloo DDP paths are covered by integration tests. CUDA tests are conditionally skipped when fewer than two GPUs are available. Actual GPU scaling must be measured on the target machine; the framework does not claim linear speedup. `distributed_timeout_seconds` and `worker_timeout_seconds` can be set to fail stuck groups rather than waiting indefinitely.
