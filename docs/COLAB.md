# Google Colab

You do not need to open the original model notebooks. Open the supplied Colab notebook or a blank notebook and run the command-line framework.

## Recommended notebook

Open:

```text
notebooks/hyperparameter_optimization_colab.ipynb
```

It contains clone/install, hardware inspection, smoke testing, runtime preview, ResNet and ViT searches, Pareto inspection, Drive persistence, and export cells.

## Minimal manual workflow

```python
!git clone https://github.com/TrueRottweiler/WashingtonCsed504.git
%cd WashingtonCsed504
!python -m pip install -q -e '.[notebook]'
```

Verify CUDA:

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

Run a software smoke test:

```python
!python src/a1-cv/search_transformer.py \
    --config configs/searches/smoke_vit.toml \
    --smoke-test --skip-calibration
```

For a fresh CIFAR-10 runtime, set `download = true` in the selected configuration. Start with a small copied config rather than the 100-epoch confirmation defaults.

## Persisting work

Colab local storage is temporary. Run the study under `/content` for faster checkpoint I/O, then copy the result directory to Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r results/my-study /content/drive/MyDrive/
```

To resume, copy the complete study directory back to the same configured output path. Preserve `study.db`, `results.jsonl`, and `checkpoints/`.

## Free-tier recommendations

- Use `--smoke-test` first.
- Use 2–4 proxy trials and 1–2 steps during environment validation.
- Keep confirmation to one finalist, one seed, and 5–20 epochs initially.
- Leave `torch.compile` off for very short trials.
- Keep test evaluation disabled.
- Export results before disconnecting.

## Scaling up

Increase one dimension at a time: proxy trials, proxy steps, halving rungs, confirmation epochs, then seed count. Run `--estimate-only` after material changes to architecture ranges or data size.
