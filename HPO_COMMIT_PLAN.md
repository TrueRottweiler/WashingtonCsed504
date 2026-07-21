# Git Commit Plan

## Extract and copy

```bash
cd ~/Downloads
unzip WashingtonCsed504_hpo_additions.zip -d WashingtonCsed504_hpo_additions
cd ~/Downloads/WashingtonCsed504
git switch -c feature/hpo-framework
rsync -av --exclude 'patches/' \
  ~/Downloads/WashingtonCsed504_hpo_additions/ ./
```

The archive is rooted at repository-relative paths, so the `src/a1-cv/...` files land in place.

## Review before patches

```bash
git status --short
git diff --stat
git diff --check
```

## Apply optional patches

```bash
git apply --check patches/src_a1-cv_train_loop.py.patch
git apply patches/src_a1-cv_train_loop.py.patch

git apply --check patches/.gitignore.patch
git apply patches/.gitignore.patch

git apply --check patches/README.md.patch
git apply patches/README.md.patch
```

## Verify

```bash
python -m pip install -r src/a1-cv/hpo_requirements.txt
python -m pytest -q src/common/test_gpu_check.py
cd src/a1-cv
PYTHONPATH=. python -m pytest -q hpo_tests/unit hpo_tests/integration/test_reporting_and_inputs.py hpo_tests/notebook
PYTHONPATH=. python hpo_tests/integration/run_tiny_suite.py
PYTHONPATH=. python -m hpo.cli --repo-root ../.. validate-space --config hpo_configs/colab/resnet18_cifar10_successive_halving.yaml
```

## Suggested commits

```bash
git add src/a1-cv/hpo/{schemas.py,conditions.py,search_space.py,config.py,exceptions.py}
git commit -m "Add HPO schemas and search-space parsing"

git add src/a1-cv/hpo/{registry.py,adapters.py,trial_runner.py,constraints.py}
git commit -m "Add repository training and dataset adapters"

git add src/a1-cv/hpo/{modes.py,study.py,objectives.py,selection.py,parallel.py}
git commit -m "Add proxy halving full and continuous search modes"

git add src/a1-cv/hpo/{hardware.py,scheduler.py,calibration.py,monitoring.py}
git commit -m "Add hardware-aware scheduling and calibration"

git add src/a1-cv/hpo/{estimation.py,costing.py,persistence.py,reporting.py,benchmark.py,baselines.py}
git commit -m "Add estimation persistence reporting and benchmarks"

git add src/a1-cv/hpo/cli.py src/a1-cv/hpo/notebook_api.py src/a1-cv/hpo/__init__.py
git commit -m "Add HPO CLI and notebook API"

git add src/a1-cv/hpo_configs src/a1-cv/hpo_examples src/a1-cv/hpo_requirements.txt
git commit -m "Add HPO configurations and examples"

git add src/a1-cv/hpo_smoke_test_colab.ipynb
git commit -m "Add HPO Colab smoke notebook"

git add src/a1-cv/hyperparameter_search_colab.ipynb
git commit -m "Add full Colab HPO notebook"

git add src/a1-cv/hpo_tests src/a1-cv/pytest.ini
git commit -m "Add HPO unit integration and notebook tests"

git add src/a1-cv/hpo_docs HPO_*.md patches
git commit -m "Document HPO integration and optional patches"
```

## Push

```bash
git status
git log --oneline --decorate -12
git push -u origin feature/hpo-framework
```

No automatic push is performed by the additions package.
