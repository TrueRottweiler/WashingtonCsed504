# UW CSED 504 — Summer 2026

Shared repository for the University of Washington **CSED 504** course, Summer 2026.

## Repository Structure

```
WashingtonCsed504/
├── assignments/    # Homework and assignment starter code
├── labs/           # Lab exercises and in-class activities
├── resources/      # Supplementary reading materials and references
└── projects/       # Course project templates and guidelines
```

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
   cd WashingtonCsed504
   ```

2. **Set up a Python virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Stay up to date**
   ```bash
   git pull origin main
   ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting any code or assignments.

## License

This repository is licensed under the [MIT License](LICENSE).
# WashingtonCsed504

Shared environment and starter code for **UW CSED 504** (Computer Vision + NLP).

Each platform has a one-shot setup script that creates a conda environment named
**`uw-csed504`** (Python 3.12) with a matching package set, so everyone in the group
runs the same stack whether they're on Windows, macOS, Linux, or Google Colab.

---

## What's in here

| Path | Purpose |
|------|---------|
| `setup_windows.ps1` | Windows setup (NVIDIA CUDA 12.8) |
| `setup_mac.sh` | macOS setup (Apple MPS / CPU) |
| `setup_linux.sh` | Linux / WSL2 setup (NVIDIA CUDA 12.8, or CPU) |
| `cuda_check.ps1` | Windows: (re)configure GPU visibility any time |
| `src/common/gpu_check.py` | Shared device detection + multi-GPU helpers |
| `src/a1-cv/hello_image.ipynb` | A1 sanity notebook (builds a Vision Transformer) |
| `src/a2-nlp/hello_text.ipynb` | A2 sanity notebook |

If a hello notebook runs top-to-bottom without errors, your environment is ready.

---

## Prerequisites: install Anaconda / Miniconda

The local setups (Windows / macOS / Linux) use **conda**. If you don't already have it,
grab an installer from the official download page — it offers both the full **Anaconda
Distribution** and the smaller **Miniconda** (either works; Miniconda is recommended):

- **Download page (Windows / macOS / Linux):** https://www.anaconda.com/download
- **Direct installer archive (all OSes/versions):** https://repo.anaconda.com/miniconda/

Pick the installer for your OS (Windows `.exe`, macOS `.pkg` / Apple-Silicon, Linux `.sh`).

<details>
<summary>Linux / WSL2 one-liner Miniconda install</summary>

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all      # restart your shell afterward
```
</details>

> **Google Colab needs none of this** — conda and PyTorch are already there. Jump to
> [Google Colab](#google-colab).

---

## Get the code

```bash
git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
cd WashingtonCsed504
```

---

## Setup by platform

### Windows (NVIDIA GPU)

Uses CUDA 12.8 wheels (Blackwell `sm_120`-compatible) and auto-detects all GPUs.

1. Open the **Anaconda Prompt** (Start menu → "Anaconda Prompt"), **not** plain PowerShell —
   the script needs `conda` on the path.
2. `cd` to the repo, then run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
   ```

The script installs everything with pip (so PyTorch is the only OpenMP provider — this is
what avoids **OMP Error #15**), registers the Jupyter kernel, and pins every same-architecture
GPU. Re-run GPU detection any time with `.\cuda_check.ps1`.

### macOS (Apple Silicon or Intel)

Uses Apple **MPS** (Metal) acceleration on Apple-Silicon Macs; CPU otherwise.

```bash
conda activate base
bash setup_mac.sh
```

### Linux / WSL2 (NVIDIA GPU or CPU)

Uses CUDA 12.8 wheels like Windows. CPU-only machines work too (PyTorch falls back to CPU).

```bash
conda activate base
bash setup_linux.sh
```

> Run scripts with `bash setup_*.sh` (no `chmod` needed). WSL2 counts as Linux — use this script.

### Google Colab

No local setup. The hello notebooks self-install their packages and include the clone step.

1. Open the notebook in Colab (e.g. from GitHub: **File → Open notebook → GitHub**, paste the
   repo URL, pick `src/a1-cv/hello_image.ipynb`).
2. **Runtime → Change runtime type → Hardware accelerator: GPU** (T4 is fine).
3. Run all cells. The first cells clone the repo, `%cd` into the notebook's folder, and
   `%pip install` the needed packages.

---

## Using the environment

```bash
conda activate uw-csed504
```

- In **VS Code** or **Jupyter**, select the kernel **"Python (uw-csed504)"**.
- Verify the install and see your device:

  ```bash
  python src/common/gpu_check.py
  ```

  Expected device line by platform:

  | Platform | Output |
  |----------|--------|
  | Windows / Linux + NVIDIA | `Device : cuda [N GPUs visible ...]` |
  | macOS (Apple Silicon) | `Device : MPS - Apple Silicon GPU` |
  | CPU-only / Colab CPU | `Device : CPU (...)` |

Then open `src/a1-cv/hello_image.ipynb` or `src/a2-nlp/hello_text.ipynb`, choose the
`uw-csed504` kernel, and **Run All**. Each ends with "All checks passed."

---

## GPU notes (`src/common/gpu_check.py`)

`get_device()` picks the best device (CUDA → MPS → CPU) and, on multi-GPU NVIDIA machines,
makes all same-architecture GPUs visible. Helpers you can import:

- `get_device()` / `set_seed(42)` — device + reproducibility (used by the notebooks).
- `enable_fast_matmul()` — TF32 + cuDNN autotune; pair with **bf16 autocast** for the biggest
  single-GPU training speedup.
- `get_data_parallel_model(model, DEVICE)` — `nn.DataParallel` across GPUs (helps only when a
  step's compute is large enough to outweigh cross-GPU communication).
- `get_max_memory()` — budget dict for HuggingFace `device_map="auto"` to split a model that's
  too big for one card across multiple GPUs.

---

## Troubleshooting

- **`OMP Error #15` / duplicate `libiomp5md`** — you have a mixed conda+pip install. Re-run the
  setup script for your platform; it installs an all-pip stack so PyTorch is the sole OpenMP
  provider.
- **`conda: command not found`** — on Windows use the **Anaconda Prompt**; on macOS/Linux run
  `conda activate base` first (or `source ~/miniconda3/bin/activate`).
- **`torch.cuda.is_available()` is False** on an NVIDIA box — check the driver with `nvidia-smi`,
  and make sure you're in the `uw-csed504` env (`conda activate uw-csed504`).
- **Permission denied running a `.sh`** — invoke it as `bash setup_linux.sh` (or `setup_mac.sh`);
  no execute bit required.
- **ViT training crashes on macOS with `RuntimeError: view size is not compatible with input
  tensor's size and stride`** — this is a confirmed PyTorch MPS bug (present through at least
  PyTorch 2.5.1). The C++ autograd engine produces non-contiguous gradient tensors during
  `MultiheadAttention` backward, and a subsequent `.view()` call fails. There is no Python-level
  workaround. The `cifar10_train.ipynb` notebook handles this automatically: it detects MPS and
  sets `VIT_DEVICE = cpu` for all ViT *training* runs while keeping the ResNet and all ViT
  *inference* (forward-only) on MPS. The ViT epoch counts are capped at 3 and 5 when on CPU so
  the cells finish in a few minutes (accuracy will be low — this is a pipeline demo only). For
  real ViT results (200 epochs), use Google Colab with a GPU runtime or a Windows/Linux machine
  with CUDA.

---

# New Project Additions: Reusable Hyperparameter Search and GPU Parallelization

This repository now includes a reusable computer-vision experimentation framework built around the existing CSED 504 ResNet and Vision Transformer work. The goal is to keep the original notebooks and course workflow intact while making it easier to run organized, repeatable experiments across different models, datasets, operating systems, CPUs, and GPUs.

Instead of maintaining separate search systems for every model, the project now uses one shared engine with thin CNN and Transformer entry programs. The framework handles candidate generation, training, validation, checkpointing, runtime estimation, Pareto analysis, reporting, and hardware-aware execution. Model-specific behavior stays inside adapters, so another CNN or Transformer can be added without rewriting the entire search process.

## What has changed

The main additions include:

- A three-stage hyperparameter search: proxy screening, successive halving, and from-scratch multi-seed confirmation.
- TPE, random, grid, and explicit candidate generation.
- Simple, thorough, and custom search profiles.
- Configurable model, optimizer, scheduler, regularization, and architecture parameters.
- Persistent Optuna studies, checkpoints, JSONL records, CSV leaderboards, plots, and Markdown summaries.
- Multi-objective Pareto analysis balancing accuracy, runtime, compute, memory, data use, and estimated monetary cost.
- Pilot-calibrated estimates for training time, total search time, throughput, memory, storage, CPU-hours, and GPU-hours.
- Hardware detection for CUDA, Apple MPS, CPU-only systems, Windows, Linux, macOS, WSL2, and Google Colab.
- Independent one-trial-per-GPU scheduling for faster searches across multiple GPUs.
- Optional DistributedDataParallel training when one large trial should use multiple GPUs.
- Hybrid execution, using separate GPUs for search trials and DDP for final confirmation.
- Synthetic smoke tests that verify the pipeline without downloading CIFAR-10.
- A Colab notebook that calls the reusable source modules instead of duplicating the framework in notebook cells.
- Unit, integration, smoke, parallel-execution, and DDP tests.

## New project structure

```text
configs/searches/          CNN, ViT, smoke-test, thorough, and multi-GPU configurations
configs/experiments/       Benchmark and experiment configurations
notebooks/                 Colab hyperparameter-optimization workflow
src/a1-cv/                 Original CV notebooks and thin search launchers
src/a1-imagenet32/         Jummah ImageNet-32 experiments
src/common/                Shared hardware and GPU utilities
src/cv_search/             Reusable search, training, estimation, and reporting framework
tests/                     Unit, integration, smoke, parallel, and DDP tests
benchmarks/                Measured benchmark reports and reproducible commands
docs/                      Architecture, configuration, Colab, migration, and extension guides
examples/                  Custom model and dataset registration examples
```

## Install the new framework

The original setup scripts remain available. For the reusable search package, Python 3.12 or newer is required.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,notebook]'
```

PyTorch and Torchvision should match the hardware being used. On CUDA systems, install the appropriate PyTorch build for the installed driver and GPU before running a long search.

Verify the installation:

```bash
python -m cv_search.cli.search --help
python src/a1-cv/search_cnn.py --help
python src/a1-cv/search_transformer.py --help
```

Run the complete test suite:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

CUDA-only tests skip automatically when compatible GPUs are unavailable.

## Start with a smoke test

I recommend running a smoke test first whenever the repository is cloned onto a new computer or Colab runtime. These commands use generated 32×32 images, so they test the full software pipeline without downloading CIFAR-10 or producing meaningful model accuracy.

### CNN smoke test

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10.toml \
  --smoke-test \
  --skip-calibration
```

### Vision Transformer smoke test

```bash
python src/a1-cv/search_transformer.py \
  --config configs/searches/vit_cifar10.toml \
  --smoke-test \
  --skip-calibration
```

A successful smoke test confirms model construction, training, validation, checkpointing, stage promotion, result storage, and plot generation. Smoke-test accuracy is random and should never be reported as a scientific result.

## Estimate the search before running it

Before committing to a long experiment, run a calibration preview on the actual hardware:

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10.toml \
  --estimate-only \
  --calibration-steps 25
```

The preview reports the detected hardware, representative throughput, memory use, approximate stage duration, search-space size, storage requirements, and estimated CPU/GPU cost. These values are measured projections rather than guarantees. Data loading, compilation, shared Colab resources, pruning behavior, architecture size, and checkpoint overhead can change the real runtime.

## Run the regular searches

In a fresh environment, set `dataset.download = true` in the selected TOML configuration or download CIFAR-10 beforehand.

### ResNet CNN

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10.toml
```

### Vision Transformer

```bash
python src/a1-cv/search_transformer.py \
  --config configs/searches/vit_cifar10.toml
```

### Thorough architecture searches

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/cnn_search_thorough.toml

python src/a1-cv/search_transformer.py \
  --config configs/searches/transformer_search_thorough.toml
```

The thorough profiles search both training and architecture parameters. They can become expensive quickly, so review the search-space size and runtime estimate before execution.

## Multi-GPU search

For the current CIFAR-scale ResNet and ViT experiments, the most efficient multi-GPU strategy is usually one independent trial per GPU. Every GPU trains a different candidate, allowing the framework to explore more of the search space in the same wall-clock time.

### ResNet multi-GPU search

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10_multi_gpu.toml \
  --gpu-indices 0,1,2,3
```

### ViT multi-GPU search

```bash
python src/a1-cv/search_transformer.py \
  --config configs/searches/vit_cifar10_multi_gpu.toml \
  --gpu-indices 0,1,2,3
```

The supplied multi-GPU configurations use a hybrid strategy:

```text
Proxy stage       → one independent trial per GPU
Halving stage     → surviving candidates run independently across GPUs
Full confirmation → finalists use configurable two-GPU DDP groups
```

You can also override the execution mode from the command line:

```bash
python src/a1-cv/search_cnn.py \
  --config configs/searches/resnet_cifar10.toml \
  --parallel-mode parallel_trials \
  --trial-concurrency auto \
  --gpu-indices 0,1,2,3
```

The parent process owns Optuna and all result files. Worker processes only train their assigned configurations and return structured results. This prevents simultaneous workers from corrupting SQLite, JSONL, CSV, or report files.

## Distributed training for one large trial

DistributedDataParallel is available when one model should train across multiple GPUs. This is most useful for larger models, larger input resolutions, or long final-confirmation runs. It is generally not the first choice for small search candidates that already fit on one GPU.

The framework handles:

- One rank process per GPU.
- NCCL on compatible CUDA systems and Gloo fallback where appropriate.
- Distributed training samplers.
- Nonduplicating validation and test shards.
- Global metric reduction.
- Rank-zero checkpoint and report ownership.
- Gradient accumulation with `no_sync()`.
- Model, optimizer, scheduler, scaler, counter, history, and RNG restoration.

The default batch-size interpretation is:

```toml
batch_size_scope = "global"
learning_rate_scaling = "none"
```

This keeps the searched global batch size constant as the number of GPUs changes. The framework rejects nondivisible global batch sizes instead of silently changing the experiment.

For details, see [`docs/PARALLEL_EXECUTION.md`](docs/PARALLEL_EXECUTION.md).

## Google Colab

You do not need to open the original `hello_image.ipynb` or `cifar10_train.ipynb` to run the search framework. You can use the dedicated notebook:

[`notebooks/hyperparameter_optimization_colab.ipynb`](notebooks/hyperparameter_optimization_colab.ipynb)

From a blank Colab notebook, the basic workflow is:

```python
!git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
%cd WashingtonCsed504
!python -m pip install -q -e '.[notebook]'
```

Verify the assigned device:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Run a smoke test:

```python
!python src/a1-cv/search_transformer.py \
    --config configs/searches/vit_cifar10.toml \
    --smoke-test \
    --skip-calibration
```

A standard Colab runtime normally exposes one GPU, so it can run CUDA trials but cannot use multi-GPU DDP unless the runtime actually exposes two or more GPUs. Save important result directories to Google Drive or download them before the Colab runtime resets.

See [`docs/COLAB.md`](docs/COLAB.md) for the complete workflow.

## Search profiles

### Simple

The default profile keeps the baseline architecture fixed and searches the highest-impact training and regularization settings. This is the best place to begin.

### Thorough

The thorough profile includes architecture parameters such as CNN stage width and depth, kernels, groups, normalization, pooling, squeeze-and-excitation, ViT patch size, embedding dimension, attention heads, encoder depth, MLP ratio, positional embeddings, and pooling strategy.

### Custom

A custom TOML profile can enable, disable, freeze, redefine, or condition individual parameters. This allows the same framework to support tightly controlled experiments without editing the engine.

## Understanding the three stages

### 1. Proxy search

The proxy stage screens many candidates using a reduced training budget. Its results are approximate and should be treated as an efficient filter, not the final comparison.

### 2. Successive halving

The strongest proxy candidates receive increasingly larger budgets. Promoted candidates normally continue from their previous halving checkpoints.

### 3. Full confirmation

The final candidates are retrained from scratch across the requested seeds. Proxy and halving weights are not reused for the scientific final comparison.

The test set remains disabled by default. It should only be evaluated after the final configuration has been selected.

## Results and reports

Every study writes its outputs under:

```text
results/<study-name>/
├── study.db
├── resolved_config.toml
├── environment.json
├── hardware.json
├── preview.json
├── runtime_estimate.json
├── results.jsonl
├── leaderboard.csv
├── parallel_summary.json
├── pareto.csv
├── stage_summary.md
├── final_summary.md
├── checkpoints/
└── plots/
    ├── png/
    ├── svg/
    └── html/
```

`leaderboard.csv` and `pareto.csv` can be opened directly in Excel, Google Sheets, LibreOffice, or the group worksheet.

Regenerate reports from an existing study:

```bash
python -m cv_search.cli.report \
  --study results/resnet-cifar10-default \
  --config results/resnet-cifar10-default/resolved_config.toml
```

## Pareto analysis

The framework can optimize more than accuracy alone. Objectives can include:

- Validation performance.
- Wall-clock time.
- GPU-hours and CPU-hours.
- Peak memory.
- Number of training examples consumed.
- Estimated monetary cost.

The complete multi-objective search is preserved as a Pareto frontier instead of being reduced into one weighted score. Preference weights can be used afterward to select a practical final candidate while still comparing it with accuracy-only, speed-only, memory-only, and cost-only alternatives.

## Add another model or dataset

The shared engine is not tied to CIFAR-10, ResNet, or the included ViT. Register another component through the Python API:

```python
from cv_search.registry import register_dataset, register_model_adapter

register_model_adapter("my_model", MyAdapter)
register_dataset("my_dataset", my_dataset_factory)
```

The model adapter defines construction, validation, search spaces, optimization, scheduling, and model-description behavior. The shared engine continues to handle the search stages, persistence, reporting, estimation, and parallel execution.

See:

- [`docs/EXTENDING.md`](docs/EXTENDING.md)
- [`examples/register_custom_components.py`](examples/register_custom_components.py)

## Important scientific-use notes

- Proxy accuracy is approximate.
- Halving candidates are evaluated at different resource budgets.
- Finalists are retrained from scratch.
- Validation accuracy selects candidates; validation loss may break ties.
- The test set remains isolated unless explicitly enabled after selection.
- Synthetic smoke-test results are not model-performance results.
- Adding GPUs must not silently change global batch size, learning rate, preprocessing, augmentation, data split, or evaluation logic.
- Runtime and cost estimates should always be labeled as estimates until replaced by measured results.
- A new study name should be used whenever the dataset, split, preprocessing, augmentation, metric implementation, or architecture meaning changes.

## Additional documentation

- [Framework architecture](docs/ARCHITECTURE.md)
- [Parallel GPU and DDP execution](docs/PARALLEL_EXECUTION.md)
- [Configuration and search profiles](docs/CONFIGURATION.md)
- [Google Colab guide](docs/COLAB.md)
- [Repository comparison](docs/REPOSITORY_COMPARISON.md)
- [Migration and merge decisions](docs/MIGRATION.md)
- [Extending models and datasets](docs/EXTENDING.md)
- [Reference configurations](docs/REFERENCES.md)
- [Verification results](docs/VERIFICATION.md)
- [Known limitations](docs/LIMITATIONS.md)
- [Git branch and commit plan](docs/GIT_PLAN.md)
- [Measured benchmarks](benchmarks/README.md)

## Current limitations

The framework is designed for single-node execution. Independent multi-GPU scheduling and DDP are implemented, but actual scaling depends on the available GPU models, interconnect, dataset pipeline, model size, batch size, and trial duration.

The current implementation does not automatically provision Slurm, Kubernetes, cloud clusters, or multiple machines. FSDP, tensor parallelism, and pipeline parallelism are also outside the current scope. These methods become more relevant when a model cannot fit on one GPU or when DDP communication becomes the limiting factor.

GPU performance should be benchmarked on the target machine before making claims about speedup, utilization, or cost savings.

