# Architecture

## Design goal

The framework separates model-specific behavior from experiment orchestration. The same engine drives both user-facing programs:

```text
src/a1-cv/search_cnn.py
src/a1-cv/search_transformer.py
```

They select an adapter and delegate to `cv_search.cli`, `cv_search.engine`, and the shared training stack.

## Main layers

```text
Configuration     config.py, search_space.py
Registration      registry.py
Models            adapters/base.py, adapters/cnn.py, adapters/transformer.py
Data              data.py
Execution         training.py, distributed.py, parallel.py, checkpoints.py, metrics.py
Optimization      samplers.py, engine.py, objectives.py
Estimation        hardware.py, estimation.py
Persistence       storage.py
Analysis          plotting.py, reporting.py
Interfaces        cli/, interactive.py
```

## Three-stage study

1. **Proxy** evaluates many candidates with short step/epoch/time limits and optional data fractions.
2. **Successive halving** increases resource budgets for promoted candidates and normally resumes model, optimizer, scheduler, scaler, counters, metrics, history, and RNG state.
3. **Full confirmation** discards proxy/halving weights and retrains finalists from scratch across configured seeds.

Test evaluation is a separate final action and is disabled by default.

## Parallel execution boundary

The parent process is the control plane. It owns:

- Optuna candidate generation and `ask`/`tell`;
- global budget decisions;
- result and event persistence;
- rung promotion;
- Pareto analysis, plotting, and reporting.

Training workers are the data plane. `parallel.py` launches persistent isolated worker subprocesses for independent trials and fresh local DDP rank groups. Persistent workers amortize framework import, CUDA-context, and dataset construction costs without sharing CUDA state with the parent. Each invocation receives a serializable `TrialTask`, constructs the registered adapter and dataset inside the worker, trains on its assigned device, and returns a `TrialResult`.

For DDP, `parallel.py` launches `cv_search.cli.trial_worker` through `torch.distributed.run`. `distributed.py` initializes the process group, assigns one process per GPU, wraps the model, and supplies collective utilities. Only rank zero writes checkpoints and the final task result.

This separation keeps SQLite and JSONL single-writer while still filling all selected GPUs.

## Adapter boundary

A model adapter owns:

- baseline model/training values;
- simple and thorough spaces;
- validation before allocation;
- model construction;
- optimizer and scheduler construction;
- architecture description and complexity metadata;
- calibration configurations.

The engine contains no ResNet-versus-Transformer branching.

## Candidate generation

- **TPE:** persistent Optuna study in SQLite, coordinated only by the parent. Parallel TPE enables constant-liar sampling while trials are in flight.
- **Random:** reproducible seeded Optuna random sampler.
- **Grid:** finite combinations are counted before execution; oversized grids are rejected rather than truncated.
- **Explicit:** user-provided configurations are enqueued exactly.

Conditional parameters are resolved against sampled and frozen parent values.

## DDP data and metric correctness

Training uses `DistributedSampler` and calls `set_epoch()` each epoch. Evaluation uses an exact rank-strided sampler so examples are not padded or duplicated. Loss sums, correct classifications, and example counts are reduced globally before metrics are computed.

Gradient accumulation uses DDP `no_sync()` for non-step microbatches. Global batch size is explicit and validated. Rank-zero checkpoint writes are surrounded by barriers, and the unwrapped module state is saved for later single-device or DDP restoration.

## Pareto handling

Enabled objectives retain individual directions. The nondominated set is exported to `pareto.csv`; it is not collapsed during search. A final policy may select accuracy-, speed-, cost-, or weighted-Pareto preference after the Pareto set is constructed.

## Persistence and resumption

- Optuna stores parent-owned sampler/study metadata in `study.db`.
- Every execution is appended by the parent to `results.jsonl` and exported to `leaderboard.csv`.
- Stage checkpoints are written atomically.
- Re-running the same study reconstructs completed stages from JSONL and reuses the persistent proxy study.

## Hardware behavior

`hardware.py` records the environment and proposes conservative defaults. Every automatic value can be overridden in TOML. Backend order is CUDA, MPS, then CPU when `device="auto"`.

Multiple CUDA GPUs default to independent trial slots when `trial_concurrency = "auto"`. DDP is opt-in globally or per stage. See `PARALLEL_EXECUTION.md`.
