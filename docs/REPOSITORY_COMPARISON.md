# Source repository comparison

## Sources inspected

- `WashingtonCsed504-main_jummah_17(3).zip` — authoritative and newest course repository.
- `WashingtonCsed504-main(3)(7).zip` — earlier search-framework draft.
- `Antigravity(1).tar(2).gz` — packaged Antigravity Electron application, not a project source repository.

The original archives were not modified. They were extracted into separate working directories before the final repository was assembled.

## Jummah

### Retained

- Current README, license, setup scripts, GPU utility, and course directories.
- Current computer-vision notebooks.
- CIFAR-100/Hugging Face experiment notebook and GPU-data utility.
- `run_resnet18.ipynb` and `run_vit.ipynb`.
- Complete `src/a1-imagenet32/` workflow.

### Advantages

- Newest assignment behavior and project work.
- Correct destination metadata: `TrueRottweiler/WashingtonCsed504`.
- Broader current experiment coverage.

### Gaps addressed

- No reusable general search package.
- No Pareto framework, persistent study database, calibrated estimator, or integrated testing for the requested workflow.

## Washington draft

### Useful material imported or refactored

- Two thin CNN/Transformer launchers over one shared framework.
- TOML search configurations.
- Adapter-based separation.
- Fake-data smoke concept.
- Stage-oriented output layout.

### Advantages

- Small and easy to inspect.
- Clear initial separation between shared logic and model adapters.
- Good starting point for Colab-safe smoke tests.

### Weaknesses corrected

- Incomplete TPE/resume behavior.
- Several exposed parameters were not fully connected.
- Limited Pareto, estimation, test reporting, and end-to-end coverage.
- Compressed code style and incomplete packaging metadata.

## Antigravity archive

The archive contains a prebuilt Electron application with executables, shared libraries, and runtime resources. It is not a Python source repository for this project. No binary or application code was copied into the merged repository. This avoids unrelated dependencies, oversized artifacts, licensing uncertainty, and executable-supply-chain risk.

## Selection rule

Jummah behavior was preserved unless another implementation was demonstrably more modular, testable, portable, or complete. Overlapping search code was refactored into one `cv_search` package rather than copied into parallel CNN and Transformer systems.
