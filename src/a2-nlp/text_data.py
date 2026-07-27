"""
The whole token stream, resident in GPU memory, handing back (input, target) windows.

This is the a2 counterpart of imagenet_data.py, and the trick transfers with room to spare:
ImageNet-32 was 3.9 GB against a 96 GB card; WikiText-103 tokenized is about 230 MB. We upload
the stream once at startup and every batch after that is a single gather on data that already
lives in VRAM -- no DataLoader, no workers, no host-to-device copy, no Python in the inner loop.

What replaces augmentation here is windowing. An image model sees a random crop of an image;
a language model sees a random window of the stream. Training draws batch_size random start
offsets per step and gathers seq_len+1 tokens from each: the first seq_len are the input, the
last seq_len are the same tokens shifted one place -- next-token prediction, exactly as in
hello_text.ipynb, just built by slicing one window instead of stacking two.

Validation walks the stream in order with non-overlapping windows instead, so evaluation is
deterministic and every token is scored exactly once.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_stats(dataset: str) -> dict:
    """Read stats.json (vocab size, token counts, tokenizer id) that text_prepare.py wrote.

    Same contract as a1's load_stats: these numbers were measured from this dataset's own
    prepared arrays, and everything downstream reads them instead of hardcoding anything.
    """
    p = os.path.join(DATA_DIR, dataset, 'stats.json')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'{p} not found -- run: python text_prepare.py --dataset {dataset}')
    return json.load(open(p))


class GpuTokens:
    """One split's token stream, resident in GPU memory, serving (x, y) next-token batches.

    The DataLoader-shaped surface (epoch / n_batches / gb) matches a1's GpuImageNet32, so
    train_loop.py drives either one without knowing which modality it is training.
    """

    def __init__(self, device: torch.device, dataset: str, split: str = 'train',
                 seq_len: int = 256, subset: int | None = None):
        stats = load_stats(dataset)
        arr = np.load(os.path.join(DATA_DIR, dataset, f'{split}_tokens.npy'), mmap_mode='r')

        # Smoke tests take a contiguous slice off the front. Unlike ImageNet-32 -- which was
        # sorted by class, making a front slice poison -- a text stream's prefix is ordinary
        # prose, so a contiguous cut is a fair (and reproducible) small sample.
        if subset is not None and subset < len(arr):
            arr = arr[:subset]

        # Upload once, as int16. The prepared files are uint16 (torch has no uint16 tensor), and
        # every vocab in the study is <= 16,384, so the values fit int16 exactly; the guard makes
        # sure that stays true if a bigger vocabulary ever shows up. Embedding lookups need int64
        # indices, but we pay that cast per batch in epoch() -- holding the resident copy at 2
        # bytes per token instead of 8 is the same cheap-form trade as a1 keeping uint8 pixels.
        assert stats['vocab_size'] <= 32_767, 'vocab too large for the int16 resident store'
        self.t = torch.from_numpy(np.ascontiguousarray(arr).astype(np.int16)).to(device)

        self.device = device
        self.seq_len = seq_len
        self.n = len(self.t)                      # tokens in the stream
        self.vocab_size = stats['vocab_size']
        self.tokenizer = stats['tokenizer']
        if self.n < seq_len + 2:
            raise ValueError(f'{dataset}/{split}: only {self.n} tokens, need > seq_len+1')

    def gb(self) -> float:
        """VRAM this split occupies, in GB -- 2 bytes per resident token."""
        return self.t.numel() * 2 / 1e9

    def _gather(self, starts: torch.Tensor):
        """Slice seq_len+1 tokens at each start offset and split into (input, shifted target).

        One broadcasted index builds the whole batch: starts (B, 1) + arange (1, T+1) -> (B, T+1)
        positions, one gather, no loop. The +1 column is what lets x and y come from the same
        window: y is x shifted left by one, which IS the next-token objective.
        """
        pos = starts.view(-1, 1) + torch.arange(self.seq_len + 1, device=self.device)
        w = self.t[pos].long()
        return w[:, :-1], w[:, 1:]

    def epoch(self, batch_size: int, train: bool, generator: torch.Generator | None = None):
        """Yield (x, y) batches for one pass over the split, both (B, seq_len) int64 on-device.

        Train: n_batches steps of random windows. Random offsets see epoch-sized token counts
        without the bookkeeping of a shuffled partition, and window boundaries land differently
        every epoch -- that boundary jitter is the closest thing text has to a random crop.

        Val: fixed non-overlapping windows in stream order, so the same tokens are scored the
        same way every time and two checkpoints' perplexities are exactly comparable.
        """
        if train:
            for _ in range(self.n_batches(batch_size)):
                starts = torch.randint(0, self.n - self.seq_len - 1, (batch_size,),
                                       device=self.device, generator=generator)
                yield self._gather(starts)
        else:
            starts = torch.arange(0, self.n - self.seq_len - 1, self.seq_len,
                                  device=self.device)
            for s in range(0, len(starts), batch_size):
                yield self._gather(starts[s:s + batch_size])

    def n_batches(self, batch_size: int) -> int:
        """Steps per epoch: enough random windows that one epoch sees ~every token once."""
        return max(1, self.n // (batch_size * self.seq_len))
