# Copy-Paste Existing-File Changes

## Optional gradient accumulation

**File:** `src/a1-cv/train_loop.py`

**Function:** `train_one_epoch`

Change the final arguments in the function signature from:

```python
strong_aug=False, clip=None
```

to:

```python
strong_aug=False, clip=None, gradient_accumulation=1
```

Immediately before the training loop, insert:

```python
gradient_accumulation = max(1, int(gradient_accumulation))
optimizer.zero_grad(set_to_none=True)
```

Remove the per-batch `optimizer.zero_grad(set_to_none=True)` at the beginning of the loop.

Replace the existing backward/step block with:

```python
scaler.scale(loss / gradient_accumulation).backward()
should_step = (
    (step + 1) % gradient_accumulation == 0
    or (step + 1) == n_batches
)
if should_step:
    if clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```

Verification:

```bash
python -m py_compile src/a1-cv/train_loop.py
git diff --check
```

The complete context-safe diff remains the recommended method: `patches/src_a1-cv_train_loop.py.patch`.
