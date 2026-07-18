"""Shared CIFAR image pipelines — one interface, two backends (CPU DataLoader / GPU-resident).

The point of this module is to let cifar10_cpu_train.ipynb and cifar10_gpu_train.ipynb be nearly
identical notebooks that differ only in `device`, so the CPU-vs-GPU training difference is visible
as a config change, not a rewrite.  Both backends yield the same thing: batches of
`(images_float_on_device, labels_on_device)` already augmented and normalized.

Two backends, because the right pipeline is opposite at each end:

  * **GPU-resident** (`GPUImageLoader`): the whole 32x32 set is tiny (~180 MB), so it lives on the
    GPU and crop/flip/normalize run there in batches.  No DataLoader, no workers, no host->device
    copy — the only way to keep a fast GPU fed at this image size (measured: CPU workers cap ~14k
    img/s, GPU-resident sustains ~30k+).

  * **CPU** (`make_cpu_loader`): a torchvision DataLoader with a few workers.  When the MODEL runs
    on the CPU it is the bottleneck (~850 img/s for resnet18@32 on a 24-core Zen4), which is far
    below what even 2-4 augmentation workers produce, so the CPU recipe is: big batch +
    channels_last (oneDNN's native NHWC) + bf16 autocast (AVX-512-BF16) on the model side, and only
    a handful of loader workers.  Measured 2.3x over a naive fp32/NCHW/small-batch loop.

Measured CPU knobs (resnet18@32, Zen4 24c/48t), best -> 847 img/s at batch 1024 + channels_last +
bf16, vs 367 naive.  channels_last and bf16 are the CPU wins; they do NOT transfer to the GPU here.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F

# Published CIFAR-10 per-channel stats (train split).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


# ─── Data loading ────────────────────────────────────────────────────────────────────────────────

def load_cifar10_arrays(root: str | None = None):
    """torchvision CIFAR-10 -> (train_x, train_y, test_x, test_y) as uint8 NHWC arrays + int64 labels.

    Kept as plain numpy so either backend can take it: the GPU backend ships it to the device once,
    the CPU backend wraps it in a Dataset.  Downloads to `root` (default: ../a1-cv/data) if absent.
    """
    from torchvision.datasets import CIFAR10

    if root is None:
        root = os.path.join(os.path.dirname(__file__), '..', 'a1-cv', 'data')
    tr = CIFAR10(root=root, train=True, download=True)
    te = CIFAR10(root=root, train=False, download=True)
    train_x = tr.data.astype(np.uint8)                       # (50000, 32, 32, 3) NHWC
    test_x = te.data.astype(np.uint8)
    train_y = np.asarray(tr.targets, dtype=np.int64)
    test_y = np.asarray(te.targets, dtype=np.int64)
    return train_x, train_y, test_x, test_y


# ─── GPU-resident backend (data + augmentation on the device) ──────────────────────────────────────

def to_device_uint8(images_nhwc: np.ndarray, labels: np.ndarray, device):
    """Move a uint8 NHWC image array + labels onto `device` as an NCHW uint8 tensor + long tensor."""
    x = torch.from_numpy(np.ascontiguousarray(images_nhwc)).to(device).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.ascontiguousarray(labels)).to(device)
    return x, y


class GPUImageLoader:
    """Yield (augmented_float_batch, labels) with everything on ``images_u8.device``.

    Canonical shared version of the loader first written for the CIFAR-100 notebook.  Augmentation
    (reflect-pad random crop, horizontal flip, random erasing) applies only when ``train=True``;
    normalization always.  ``drop_last`` defaults to ``train`` so training batches share a fixed
    shape (keeps cuDNN autotune happy).  Deterministic given ``seed``.
    """

    def __init__(self, images_u8, labels, batch_size, mean=CIFAR10_MEAN, std=CIFAR10_STD, *,
                 train, crop_pad=4, hflip=True, erasing=False, erase_p=0.25,
                 drop_last=None, seed=None):
        self.x, self.y = images_u8, labels
        self.bs = batch_size
        self.train = train
        self.crop_pad = crop_pad
        self.hflip = hflip
        self.erasing = erasing
        self.erase_p = erase_p
        self.drop_last = train if drop_last is None else drop_last
        dev = images_u8.device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        self._ar = torch.arange(images_u8.size(-1), device=dev)
        self.gen = torch.Generator(device=dev)
        if seed is not None:
            self.gen.manual_seed(seed)

    def __len__(self):
        n = self.x.size(0)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _random_crop(self, x):
        B, C, H, W = x.shape
        x = F.pad(x, (self.crop_pad,) * 4, mode='reflect')
        m = 2 * self.crop_pad + 1
        oy = torch.randint(0, m, (B,), device=x.device, generator=self.gen)
        ox = torch.randint(0, m, (B,), device=x.device, generator=self.gen)
        rows = (oy.view(B, 1) + self._ar).view(B, 1, H, 1)
        cols = (ox.view(B, 1) + self._ar).view(B, 1, 1, W)
        bidx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
        cidx = torch.arange(C, device=x.device).view(1, C, 1, 1)
        return x[bidx, cidx, rows, cols]

    def _random_erase(self, x, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        B, C, H, W = x.shape
        dev = x.device
        do = torch.rand(B, device=dev, generator=self.gen) < self.erase_p
        area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=self.gen) * (H * W)
        logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]),
                                                    generator=self.gen)
        ar = torch.exp(logr)
        h = (area * ar).sqrt().round().clamp(1, H).long()
        w = (area / ar).sqrt().round().clamp(1, W).long()
        top = (torch.rand(B, device=dev, generator=self.gen) * (H - h + 1).float()).long()
        left = (torch.rand(B, device=dev, generator=self.gen) * (W - w + 1).float()).long()
        rows = self._ar.view(1, 1, H, 1)
        cols = self._ar.view(1, 1, 1, W)
        mask = ((rows >= top.view(B, 1, 1, 1)) & (rows < (top + h).view(B, 1, 1, 1)) &
                (cols >= left.view(B, 1, 1, 1)) & (cols < (left + w).view(B, 1, 1, 1)) &
                do.view(B, 1, 1, 1))
        return torch.where(mask, torch.zeros((), device=dev, dtype=x.dtype), x)

    def _augment(self, xb):
        x = xb.float()
        if self.train and self.crop_pad:
            x = self._random_crop(x)
        if self.train and self.hflip:
            flip = torch.rand(x.size(0), device=x.device, generator=self.gen) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)   # sync-free vs boolean-mask write
        x = x.div_(255.0).sub_(self.mean).div_(self.std)
        if self.train and self.erasing:
            x = self._random_erase(x)
        return x

    def __iter__(self):
        n = self.x.size(0)
        idx = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.train
               else torch.arange(n, device=self.x.device))
        last = n - self.bs + 1 if self.drop_last else n
        for s in range(0, last, self.bs):
            sel = idx[s:s + self.bs]
            yield self._augment(self.x[sel]), self.y[sel]


# ─── CPU backend (torchvision DataLoader) ──────────────────────────────────────────────────────────

class _ArrayDataset(torch.utils.data.Dataset):
    """uint8 NHWC array -> (CHW float tensor, label), augmenting per item on the worker."""

    def __init__(self, images_nhwc, labels, transform):
        self.images = images_nhwc
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.images[i])), int(self.labels[i])


def make_cpu_loader(images_nhwc, labels, batch_size, *, train, mean=CIFAR10_MEAN, std=CIFAR10_STD,
                    num_workers=4, erasing=False):
    """A torchvision DataLoader over uint8 NHWC arrays, sized for CPU training.

    Few workers on purpose: with the model on the CPU it is the bottleneck (~850 img/s), well under
    what even 2-4 augmentation workers produce, so more workers just oversubscribe the cores that
    the model's 24 compute threads want.  channels_last + bf16 live on the model side (see the
    notebook / train_engine), not here.
    """
    from torchvision import transforms

    if train:
        tf = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
            + ([transforms.RandomErasing(p=0.25)] if erasing else []))
    else:
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    ds = _ArrayDataset(images_nhwc, labels, tf)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=train, num_workers=num_workers,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=False, drop_last=train)


# ─── Unified factory ───────────────────────────────────────────────────────────────────────────────

# Defaults measured per backend: GPU wants a big batch on-device; CPU wants an even bigger batch +
# channels_last + bf16 (all set here / in train_engine).  The notebooks override as they like.
DEFAULTS = {
    'cuda': dict(batch_size=512, channels_last=False, amp_dtype=torch.bfloat16),
    'cpu':  dict(batch_size=1024, channels_last=True, amp_dtype=torch.bfloat16),
}


def make_loaders(device, train_x, train_y, test_x, test_y, *, batch_size=None,
                 mean=CIFAR10_MEAN, std=CIFAR10_STD, erasing=False, num_workers=4, seed=42):
    """Return (train_iter, test_iter, cfg) appropriate for `device`.

    Both iterators yield (x, y) with x float+normalized and y long.  For CUDA both live on the GPU;
    for CPU they come off the DataLoader on the CPU (the train loop applies channels_last/bf16).
    `cfg` carries the backend-specific defaults (batch_size, channels_last, amp_dtype) so the train
    loop and the notebook stay in sync.
    """
    dt = torch.device(device) if not isinstance(device, torch.device) else device
    cfg = dict(DEFAULTS.get(dt.type, DEFAULTS['cpu']))
    if batch_size is not None:
        cfg['batch_size'] = batch_size
    bs = cfg['batch_size']

    if dt.type == 'cuda':
        xtr, ytr = to_device_uint8(train_x, train_y, dt)
        xte, yte = to_device_uint8(test_x, test_y, dt)
        train_iter = GPUImageLoader(xtr, ytr, bs, mean, std, train=True, erasing=erasing, seed=seed)
        test_iter = GPUImageLoader(xte, yte, 512, mean, std, train=False)
    else:
        train_iter = make_cpu_loader(train_x, train_y, bs, train=True, mean=mean, std=std,
                                     num_workers=num_workers, erasing=erasing)
        test_iter = make_cpu_loader(test_x, test_y, 512, train=False, mean=mean, std=std,
                                    num_workers=num_workers)
    cfg['backend'] = 'gpu-resident' if dt.type == 'cuda' else 'cpu-dataloader'
    return train_iter, test_iter, cfg
