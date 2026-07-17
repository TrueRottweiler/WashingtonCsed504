# Known limitations

## Verification boundary

Development verification used a CPU-only container with synthetic images. Spawn-isolated parallel trials and two-process Gloo DDP were executed. CUDA, Apple MPS, multi-GPU NCCL, real CIFAR downloads, long-run accuracy, and paid-cloud scaling were not available in this environment.

## Multi-GPU scaling

The implementation schedules one process per selected GPU and supports DDP groups, but no claim of linear scaling is made. Actual speedup depends on model size, input pipeline, PCIe/NVLink topology, NCCL behavior, checkpoint I/O, heterogeneous GPUs, and rung imbalance. CUDA tests skip cleanly when fewer than two GPUs are present.

The current scheduler is single-node. It can launch multiple local DDP groups but does not provision remote hosts, Slurm jobs, Kubernetes pods, or a multi-node rendezvous. Multi-node trial ownership would also require JournalStorage or a client/server Optuna RDB rather than the parent-owned local SQLite design.

## Failure behavior

A failed independent trial does not stop unrelated device slots. In DDP, the local launcher treats the rank group as one job; a fatal rank failure terminates that trial group. Checkpoint recovery is epoch-granular, not minibatch-granular.

## Runtime estimates

Pilot calibration captures representative training steps, but short pilots underrepresent model construction, dataset initialization, distributed launch, collective communication, validation, and checkpoint writes. The estimator applies an explicit parallel-efficiency assumption rather than perfect division by concurrency. Measured stage wall time and scheduler efficiency should replace the initial assumption after real trials run.

## Memory

CUDA peak allocation is measured through PyTorch and reduced as the maximum across DDP ranks. CPU RSS is measured through `psutil`. Apple MPS does not currently have equivalent peak-memory accounting, and CPU/MPS accelerator-memory values may be zero.

The scheduler reads current GPU free memory but does not yet perform predictive bin packing from an adapter's activation-memory estimate. A trial that does not fit is recorded as failed; batch size is never silently changed.

## FLOPs

FLOPs/MACs are analytical approximations from adapter metadata, not profiler-certified operation counts. Different conventions count multiply-add operations differently. Distributed communication FLOPs and bytes are not included in model FLOPs.

## Study resumption

Completed stages and the Optuna proxy study resume from disk. A crash during one epoch resumes from the last completed checkpoint write. Configuration changes under an existing study name can make prior trials scientifically incomparable; use a new study name for changed data, split, preprocessing, metric, world-size-dependent batch semantics, or architecture behavior.

## Interactive mode

Interactive selection supports terminal prompts and safely falls back in noninteractive environments. It does not provide a graphical parameter editor.

## Reports

Core PNG/SVG plots and an HTML data export are generated. Optuna-specific interactive contour and importance plots are not guaranteed for every sampler or parameter type. Confusion matrices and per-class accuracy require prediction collection and are not emitted by the generic engine.

## Test evaluation

Test evaluation is deliberately off by default. Enabling it repeatedly across studies can leak test-set information into model-development decisions even when the code evaluates only the selected finalist.
