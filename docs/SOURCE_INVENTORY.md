# Source archive inventory

The archives were extracted into separate temporary working directories. None contained `.git` metadata, so commit dates and remotes could not be read directly from archive history. The authoritative README identifies the intended destination as `https://github.com/TrueRottweiler/WashingtonCsed504.git`.

| Archive | Extracted files | Extracted size | Role |
|---|---:|---:|---|
| Jummah | 36 | 2.0 MB | Authoritative current course repository |
| Washington draft | 48 | 614 KB | Earlier reusable-search draft |
| Antigravity | 362 | 475 MB | Packaged Electron application; excluded |

## Jummah entry points and content

- Platform setup: `setup_windows.ps1`, `setup_mac.sh`, `setup_linux.sh`.
- Shared hardware utility: `src/common/gpu_check.py`.
- Current CV notebooks: CIFAR-10, CIFAR-100/Hugging Face, ResNet and ViT run notebooks.
- Current NLP sanity notebook.
- ImageNet-32 preparation, models, training, monitoring, scheduler, queue scripts, and results notebook.

## Washington draft entry points and content

- `src/a1-cv/search_cnn.py` and `search_transformer.py`.
- Initial `src/cv_search/` package.
- Four search TOML examples.
- Initial tests and search documentation.

## Antigravity contents

- Electron executable and runtime libraries.
- Application resources, locales, shared objects, and binary assets.
- No relevant Python project package, model adapter, dataset implementation, test suite, or configuration source was identified for the requested repository merge.

## Generated or duplicated material

The final archive excludes source-development results, checkpoints, databases, caches, environments, and datasets. Small measured benchmark summaries and plots are retained under `benchmarks/artifacts/`.
