# Migration notes

## Retained from Jummah

- Course and assignment folders.
- Current A1 CV notebooks, including CIFAR-100/Hugging Face and run notebooks.
- ImageNet-32 models, preparation, training, monitoring, scheduler, and queue workflows.
- MIT license and GitHub destination.

The platform setup scripts and shared GPU helper were preserved but extended for the new package dependencies and narrower exception handling. The ImageNet-32 scheduler, monitor, and queue scripts were made portable by replacing user-specific Python paths and PowerShell-only process discovery with the active interpreter and cross-platform process inspection.

## Added

- `pyproject.toml` and explicit dependencies.
- `src/cv_search/` reusable framework.
- CNN and Transformer launcher scripts.
- Search, reference, benchmark, and smoke TOML configurations.
- Unit, integration, and smoke tests.
- Colab notebook.
- Benchmark, architecture, configuration, extension, reference, limitation, and Git documentation.

## Refactored or replaced

- The earlier Washington search prototype was not copied wholesale. Its useful design ideas were reimplemented with validated parameters, persistent Optuna studies, multi-objective Pareto handling, measured pilot estimates, result serialization, and from-scratch confirmation.

## Intentionally excluded

- Generated datasets.
- Model checkpoints and result databases from development runs.
- Caches, local environments, and embedded repository metadata.
- Antigravity Electron binaries and runtime assets.

## Compatibility

The original notebooks remain available. The new framework runs independently through Python modules and does not require executing notebook cells. Existing setup scripts remain supported, while editable installation through `pyproject.toml` is now the preferred development workflow.
