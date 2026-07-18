"""HuggingFace adapter over the shared GPU-resident loader.

The GPU-resident ``GPUImageLoader`` (data + augmentation on the device) now lives in
``../common/cifar_pipeline.py`` as the single source of truth — it is shared by the CIFAR-10
CPU/GPU notebooks and this CIFAR-100 one.  This module keeps only the HuggingFace-specific piece:
decoding a HF image split (PIL) into the uint8 NCHW device tensor the loader consumes.  The public
API (``GPUImageLoader``, ``to_device_uint8``) is unchanged, so ``cifar100_hf_train.ipynb`` imports
exactly as before.

See ``cifar_pipeline.py`` for why GPU-resident beats a DataLoader at 32x32 (CPU workers cap ~14k
img/s; the model is launch-bound at small batch anyway).
"""

import os
import sys

import numpy as np
import torch

# The canonical loader lives in ../common; make this module importable on its own.
_COMMON = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'common'))
if os.path.isdir(_COMMON) and _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)

from cifar_pipeline import GPUImageLoader                       # re-exported: the shared loader
from cifar_pipeline import to_device_uint8 as _arrays_to_device  # generic numpy -> device helper

__all__ = ['GPUImageLoader', 'to_device_uint8']


def to_device_uint8(hf_split, device, img_key='img', label_key='fine_label'):
    """Decode a HuggingFace image split into a device-resident uint8 NCHW tensor + label tensor.

    HF hands us PIL images; we decode the column once into a uint8 NHWC array and hand it to the
    shared array->device helper (the same one the CIFAR-10 notebooks use), so the tensor plumbing
    isn't duplicated here.
    """
    imgs = np.stack([np.asarray(im.convert('RGB'), np.uint8) for im in hf_split[img_key]])
    labels = np.asarray(hf_split[label_key], np.int64)
    return _arrays_to_device(imgs, labels, device)
