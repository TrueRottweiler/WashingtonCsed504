# Reference configurations and primary sources

Accessed 2026-07-17.

## ResNet

- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. “Deep Residual Learning for Image Recognition.” CVPR 2016. https://arxiv.org/abs/1512.03385
- Torchvision ResNet-18 model documentation and implementation. https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

The repository baseline is a **CIFAR adaptation**, not the paper’s ImageNet input pipeline: a 3×3 stride-1 stem is used and the initial ImageNet max pool is disabled. The reference experiment configuration documents this distinction.

## Vision Transformer and data-efficient training

- Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” ICLR 2021. https://arxiv.org/abs/2010.11929
- Hugo Touvron et al. “Training data-efficient image transformers & distillation through attention.” ICML 2021. https://arxiv.org/abs/2012.12877

The included ViT baseline is an encoder-only small-image adaptation. ImageNet-scale patch sizes, schedules, and augmentation are not assumed optimal for 32×32 CIFAR inputs.

## CIFAR

- CIFAR-10 and CIFAR-100 dataset page, University of Toronto. https://www.cs.toronto.edu/~kriz/cifar.html

## Optimization and persistence

- Optuna multi-objective study API. https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html
- Optuna study persistence tutorial. https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html

## PyTorch execution

- Automatic mixed precision examples. https://docs.pytorch.org/docs/stable/notes/amp_examples.html
- Reproducibility notes. https://docs.pytorch.org/docs/stable/notes/randomness.html
- DataLoader documentation. https://docs.pytorch.org/docs/stable/data.html

## Colab

- Google Colab FAQ. https://research.google.com/colaboratory/faq.html

## Interpretation

The files under `configs/experiments/` are strong reference or repository-baseline configurations. They are not claimed to be universal global optima. Reproduction can vary with architecture details, augmentation, dataset split, software versions, hardware, precision, and training budget.
