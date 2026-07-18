# Workstation results — Toothless (2× RTX PRO 6000 Blackwell, sm_120)

Records added to `results/` by `collect.py` on this machine, plus real-run validation.
CPU is an AMD Threadripper 7000 (Zen4, 24c/48t). See the CIFAR-10 notebooks for the CPU-vs-GPU
story and `README.md` for the estimator itself.

## Perf sweep (CIFAR-100 recipe, `collect.py`)

| model | recipe | t_step | MFU | predicted 40-ep |
|---|---|---|---|---|
| resnet18 | bf16 + channels_last | 14.4 ms | 78% | ~76 s |
| vit | fp16 | 33.5 ms | — | ~2.5 min |
| resnet50 | bf16 + channels_last | 62.6 ms | 31% | ~4.7 min |
| vit_base | fp16 | 179 ms | 46% | ~12.5 min |

MFU rises with model size (bigger model = better arithmetic intensity); resnet18@32² is small
enough to be launch-bound, which is why 78% is about its ceiling here.

## Validation — predicted vs actual (real 40-epoch runs)

| run | best top-1 | wall (train) | estimator error |
|---|---|---|---|
| CIFAR-100 resnet18 | 74.13% | 68.4 s | −13.1% |
| CIFAR-100 vit | 53.86% | 140.1 s | −1.6% |
| ImageNet-32 resnet18 (1.28M imgs) | 36.71% | 27.0 min (31,217 img/s) | — |
| CIFAR-10 resnet18 (GPU) | 91.43% | 73.8 s (20 ep) | — |

CIFAR-100 resnet18 + vit were trained **concurrently**, one model per card. The estimator lands
within ~13% on the launch-bound small net and within ~2% on the compute-bound ViT.

## Recipe finding for sm_120

`bf16 + channels_last` (propagated from the laptop-tuned branch) is a **real ~1.8× win on this GPU
too**, not a regression: resnet18 went from ~18.5k img/s (old fp16 + contiguous) to ~33k. The old
"channels_last is 3× slower" note was **fp16-specific**; under bf16 it is the right call on both
sm_89 and sm_120. `torch.compile` remains unavailable on Windows; CUDA graphs help only at small
batch (launch-bound), not at batch 512.

## CPU (Threadripper 7000) — throughput is the metric, not %util

resnet18@32² CPU training tops out ~60% util = all 24 physical cores busy (SMT siblings can't be
filled by convs). The levers that matter give **2.3×**: `bf16` (AVX-512-BF16) + `channels_last`
(oneDNN NHWC — a CPU win, the reverse of the GPU-fp16 result) + batch 1024 → ~800 img/s vs ~367
naive. GPU is ~16× the CPU on the same code. See `../a1-cv/cifar10_cpu_train.ipynb` and
`cifar10_cpu_vs_gpu.ipynb`.
