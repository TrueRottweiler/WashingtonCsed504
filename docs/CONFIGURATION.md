# Configuration guide

Study files use TOML. See `configs/searches/` for complete examples.

## Core sections

```toml
[study]
name = "resnet-cifar10-default"
mode = "continuous"           # or interactive
output_dir = "results"

[model]
adapter = "cnn"              # cnn or transformer
profile = "simple"            # simple, thorough, custom

[dataset]
name = "cifar10"              # cifar10 or fake, plus registered datasets
root = "data"
download = false
validation_fraction = 0.1
split_seed = 42

[search]
sampler = "tpe"               # tpe, random, grid, explicit
trials = 12
seed = 42
concurrency = 1              # legacy fallback; execution.trial_concurrency is preferred
```

## Profiles

- **Simple:** searches high-impact optimizer, learning-rate, batch, schedule, warmup, regularization, clipping, accumulation, and stopping choices while holding the baseline architecture fixed.
- **Thorough:** also searches architecture. It requires a trial, wall-clock, compute, or cost limit.
- **Custom:** starts empty and uses `[parameters.*]` entries supplied by the user.

## Parameter types

```toml
[parameters.learning_rate]
enabled = true
type = "float"
low = 1e-5
high = 1e-2
log = true

[parameters.activation]
enabled = true
type = "categorical"
values = ["relu", "gelu", "silu"]

[parameters.depth]
enabled = true
type = "int"
low = 2
high = 8
step = 2

[parameters.patch_size]
enabled = false
type = "fixed"
fixed = 4

[parameters.momentum]
enabled = true
type = "float"
low = 0.8
high = 0.99
condition_parameter = "optimizer"
condition_values = ["sgd"]
```

Disabled entries use `fixed` when present. Invalid combinations are rejected before model allocation; they are never silently repaired.

## Explicit configurations

```toml
[search]
sampler = "explicit"
trials = 2

[[explicit.configurations]]
optimizer = "sgd"
learning_rate = 0.1
batch_size = 128

[[explicit.configurations]]
optimizer = "adamw"
learning_rate = 0.0005
batch_size = 128
```

## Multi-objective search

```toml
[objectives.validation_accuracy]
enabled = true
direction = "maximize"
weight = 1.0

[objectives.wall_clock_seconds]
enabled = true
direction = "minimize"
weight = 0.25

[objectives.peak_memory_mb]
enabled = true
direction = "minimize"
weight = 0.15
```

Supported built-ins include validation accuracy/loss, wall-clock time, GPU/CPU hours, peak accelerator memory, examples consumed, cost, parameters, and estimated training FLOPs.

Weights are used only by the optional final `weighted_pareto` selection policy, not to replace the Pareto search.

## Constraints and rates

```toml
[constraints]
max_runtime_seconds = 3600
max_gpu_hours = 2
max_cpu_hours = 8
max_memory_mb = 12000
max_data_samples = 1000000
max_epochs = 100
min_validation_accuracy = 0.80
max_cost_usd = 5.00
minimum_free_disk_gb = 2

[cost]
gpu_hour_usd = 0.50
cpu_hour_usd = 0.04
storage_gb_month_usd = 0.02
```

A trial violating a hard constraint is retained for auditability but marked rejected and excluded from the feasible Pareto set.

## Stages

```toml
[stages.proxy]
enabled = true
trials = 40
epochs = 3
max_train_steps = 600
max_validation_batches = 10
data_fraction = 1.0
top_k = 12

[stages.halving]
enabled = true
resource = "epochs"           # epochs, steps, seconds
budgets = [5, 15, 30]
reduction_factor = 2
minimum_candidates = 2
continue_checkpoints = true

[stages.full]
enabled = true
top_k = 2
epochs = 100
seeds = [7, 21, 42]
evaluate_test = false
```

## Execution

```toml
[execution]
device = "auto"               # cpu, mps, cuda, cuda:1
precision = "auto"            # fp32, fp16, bf16, auto
reproducibility = "balanced"  # strict, balanced, fast
compile = false
channels_last = true

parallel_mode = "auto"        # serial, parallel_trials, ddp, hybrid
trial_concurrency = "auto"
gpu_indices = [0, 1, 2, 3]     # omit to use all visible GPUs
gpus_per_trial = 1             # integer or "all" for DDP
distributed_backend = "auto"  # NCCL on compatible CUDA builds; otherwise Gloo
batch_size_scope = "global"   # global or per_gpu
learning_rate_scaling = "none" # none, linear, sqrt

num_workers_total = 16         # partitioned across concurrent trials
cpu_core_reserve = 2
persistent_trial_workers = true
pin_memory = true
persistent_workers = true
prefetch_factor = 2
intraop_threads = 4
interop_threads = 2
```

Hardware settings are execution choices rather than model-quality hyperparameters unless deliberately placed in the search space.

## Stage-specific parallel execution

Any stage may override execution settings:

```toml
[stages.proxy.execution]
parallel_mode = "parallel_trials"
gpus_per_trial = 1
trial_concurrency = "auto"

[stages.full.execution]
parallel_mode = "ddp"
gpus_per_trial = 2
trial_concurrency = "auto"
```

See `PARALLEL_EXECUTION.md` for process isolation, DDP metric reduction, global batch semantics, and checkpoint behavior.

## Parallel execution

```toml
[execution]
parallel_mode = "auto"          # serial, parallel_trials, ddp, hybrid
trial_concurrency = "auto"
gpu_indices = [0, 1, 2, 3]
gpus_per_trial = 1
batch_size_scope = "global"
learning_rate_scaling = "none"
num_workers_total = 16
cpu_core_reserve = 2
distributed_backend = "auto"
distributed_timeout_seconds = 300
worker_timeout_seconds = 0       # zero disables the outer timeout
parallel_efficiency_estimate = 0.80
```

Stage tables may include a nested `[stages.<name>.execution]` table. The provided CIFAR hybrid studies use `parallel_trials` for proxy and halving, then two-GPU DDP groups for full confirmation. Use `gpus_per_trial = "all"` only when the candidate global batch sizes are divisible by the selected world size. The configured batch size is interpreted consistently according to `batch_size_scope`; no learning-rate scaling is applied unless selected explicitly. See `docs/PARALLEL_EXECUTION.md`.
