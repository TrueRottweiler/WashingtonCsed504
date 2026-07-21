# Existing-File Patches

The additions-only framework runs without modifying tracked repository files. Patches are separate and never applied silently.

## 1. `src/a1-cv/train_loop.py`

**Necessity:** Optional for the base framework; required only when searching `gradient_accumulation > 1`.

**Reason:** The existing training loop calls `optimizer.step()` on every batch. An additive adapter cannot correctly implement accumulation without controlling backward/step boundaries inside that loop. The HPO runner detects the function signature and rejects values above one when this patch is absent.

**Unified diff:** `patches/src_a1-cv_train_loop.py.patch`

**Insertion/replacement location:** Replace the `train_one_epoch` signature and the backward/optimizer-step block shown in the patch. Preserve indentation at four spaces inside the function.

**Apply:**

```bash
git apply --check patches/src_a1-cv_train_loop.py.patch
git apply patches/src_a1-cv_train_loop.py.patch
python -m py_compile src/a1-cv/train_loop.py
git diff --check
```

**Verification:**

```bash
PYTHONPATH=src/a1-cv python -m pytest -q src/a1-cv/hpo_tests/unit
```

**Rollback:**

```bash
git restore src/a1-cv/train_loop.py
```

## 2. `.gitignore`

**Necessity:** Optional.

**Reason:** Keeps SQLite side files and runtime HPO outputs out of commits. The framework also supports external output directories, so this is convenience rather than a functional requirement.

**Unified diff:** `patches/.gitignore.patch`

**Apply and verify:**

```bash
git apply --check patches/.gitignore.patch
git apply patches/.gitignore.patch
git diff --check
```

**Rollback:** `git restore .gitignore`

## 3. `README.md`

**Necessity:** Optional.

**Reason:** Adds navigation to the new package, notebooks, and focused documentation without rewriting the existing README.

**Unified diff:** `patches/README.md.patch`

**Apply and verify:**

```bash
git apply --check patches/README.md.patch
git apply patches/README.md.patch
git diff --check
```

**Rollback:** `git restore README.md`
