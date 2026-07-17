# Git plan for TrueRottweiler/WashingtonCsed504

The intended remote is:

```text
https://github.com/TrueRottweiler/WashingtonCsed504.git
```

## Apply the completed repository

Prefer starting from a fresh clone so existing Git history is preserved:

```bash
git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
cd WashingtonCsed504
git switch -c feature/parallel-pareto-cv-search
```

Copy the contents of the delivered merged archive into this clone, excluding the archive’s top directory if necessary. Then inspect:

```bash
git status --short
git diff --stat
git diff -- . ':!*.ipynb'
python -m compileall -q src tests
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m ruff check src/cv_search tests
```

## Recommended commits

1. `chore: add packaging and generated-artifact exclusions`
2. `feat: add reusable model data and training interfaces`
3. `feat: add hardware detection and calibrated resource estimates`
4. `feat: add persistent multi-stage Pareto search engine`
5. `feat: add isolated multi-GPU trial scheduling and DDP execution`
6. `feat: add ResNet and ViT search configurations`
7. `test: add unit integration distributed and smoke coverage`
8. `docs: add Colab parallel-execution architecture and migration guides`

Example:

```bash
git add pyproject.toml requirements.txt .gitignore
git commit -m "chore: add packaging and generated-artifact exclusions"

git add src/cv_search src/a1-cv/search_cnn.py src/a1-cv/search_transformer.py
git commit -m "feat: add persistent multi-stage Pareto search engine"

git add src/cv_search/parallel.py src/cv_search/distributed.py \
  src/cv_search/cli/single_trial_worker.py \
  src/cv_search/cli/trial_worker.py src/cv_search/cli/worker_daemon.py
git commit -m "feat: add isolated multi-GPU trial scheduling and DDP execution"

git add configs tests benchmarks examples
git commit -m "test: add model configurations and distributed search coverage"

git add README.md docs notebooks
git commit -m "docs: add Colab workflow architecture and migration guides"
```

Push only after reviewing the diff:

```bash
git push -u origin feature/parallel-pareto-cv-search
```

Open a pull request into the repository’s default branch. No remote push was performed while producing the archive.
