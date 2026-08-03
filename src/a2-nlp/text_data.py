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

The width of the resident store follows the vocabulary rather than being fixed. Our own rungs
are narrow (65 characters, or a 16k BPE) and fit two bytes per token, but a factory that only
ever holds int16 quietly caps every corpus it will accept at 32,768 types -- which rules out the
multilingual checkpoints a transfer study wants to fine-tune, whose vocabularies run from about
120k to 250k. resolve_store_dtype() picks the narrowest type that holds the ids, --store-dtype
overrides it, and store_bench.py measures what the wider store costs.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Torch has no unsigned integer type below int64, so the resident store has to be signed even
# though token ids never are. That halves the reach of each width relative to the unsigned array
# on disk: int16 tops out at 32,767 where the uint16 file holds 65,535.
STORE_DTYPES = {'int16': torch.int16, 'int32': torch.int32}
MAX_ID = {torch.int16: 32_767, torch.int32: 2_147_483_647}
TORCH_TO_NUMPY = {torch.int16: np.int16, torch.int32: np.int32}


def resolve_store_dtype(vocab_size: int, requested: str = 'auto') -> torch.dtype:
    """Choose the resident integer width for a vocabulary, or check an explicit choice fits.

    'auto' takes the narrowest width that holds every id, which is what all of the Part 2 runs
    used and what keeps WikiText-103 at 0.23 GB rather than 0.46. An explicit 'int32' is always
    accepted -- it costs memory but cannot truncate -- while an explicit 'int16' on a vocabulary
    too large for it is refused. That refusal is the whole point of routing the decision through
    here: a truncated id is not a crash, it is a different valid-looking token, and it would
    poison every perplexity computed downstream without ever announcing itself.
    """
    if requested not in ('auto', *STORE_DTYPES):
        raise ValueError(f"store_dtype must be 'auto', 'int16', or 'int32', got {requested!r}")

    # Ids run 0..vocab_size-1, so it is the largest id that has to fit, not the count.
    largest_id = vocab_size - 1
    if requested == 'auto':
        return torch.int16 if largest_id <= MAX_ID[torch.int16] else torch.int32

    dtype = STORE_DTYPES[requested]
    if largest_id > MAX_ID[dtype]:
        raise ValueError(
            f'vocab of {vocab_size:,} needs ids up to {largest_id:,}, which does not fit '
            f'{requested} (max {MAX_ID[dtype]:,}). Use --store-dtype int32 or auto.')
    return dtype


def load_stats(dataset: str) -> dict:
    """Read stats.json (vocab size, token counts, tokenizer id) that text_prepare.py wrote.

    Same contract as a1's load_stats: these numbers were measured from this dataset's own
    prepared arrays, and everything downstream reads them instead of hardcoding anything.
    """
    p = os.path.join(DATA_DIR, dataset, 'stats.json')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'{p} not found -- run: python text_prepare.py --dataset {dataset}')
    with open(p, encoding='utf-8') as f:
        return json.load(f)


class GpuTokens:
    """One split's token stream, resident in GPU memory, serving (x, y) next-token batches.

    The DataLoader-shaped surface (epoch / n_batches / gb) matches a1's GpuImageNet32, so
    train_loop.py drives either one without knowing which modality it is training.
    """

    def __init__(self, device: torch.device, dataset: str, split: str = 'train',
                 seq_len: int = 256, subset: int | None = None,
                 store_dtype: str = 'auto'):
        stats = load_stats(dataset)
        arr = np.load(os.path.join(DATA_DIR, dataset, f'{split}_tokens.npy'), mmap_mode='r')

        # Smoke tests take a contiguous slice off the front. Unlike ImageNet-32 -- which was
        # sorted by class, making a front slice poison -- a text stream's prefix is ordinary
        # prose, so a contiguous cut is a fair (and reproducible) small sample.
        if subset is not None and subset < len(arr):
            arr = arr[:subset]

        # Upload once, in the narrowest signed type the vocabulary allows (see the module
        # docstring and resolve_store_dtype). Embedding lookups need int64 indices, but we pay
        # that cast per batch in epoch() -- holding the resident copy at 2 or 4 bytes per token
        # instead of 8 is the same cheap-form trade as a1 keeping uint8 pixels.
        self.store_dtype = resolve_store_dtype(stats['vocab_size'], store_dtype)
        self.t = torch.from_numpy(
            np.ascontiguousarray(arr).astype(TORCH_TO_NUMPY[self.store_dtype])).to(device)

        self.device = device
        self.seq_len = seq_len
        self.n = len(self.t)                      # tokens in the stream
        self.vocab_size = stats['vocab_size']
        self.tokenizer = stats['tokenizer']
        if self.n < seq_len + 2:
            raise ValueError(f'{dataset}/{split}: only {self.n} tokens, need > seq_len+1')

    @property
    def bytes_per_token(self) -> int:
        """Resident width in bytes -- 2 for int16, 4 for int32."""
        return self.t.element_size()

    def gb(self) -> float:
        """VRAM this split occupies, in GB, at whichever width the store settled on."""
        return self.t.numel() * self.bytes_per_token / 1e9

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
