"""
train_loop.py -- the training loop, metrics, and checkpointing for a multi-hour LM run.

Duplicated from a1-cv/train_loop.py and adapted from images to tokens. The bones are identical
-- accumulate metrics in on-device tensors, sync to the host once per epoch, checkpoint every
epoch, append one JSONL row per epoch -- because none of that was ever image-specific. What
changes is the scoreboard:

  - top-1/top-5 accuracy  ->  perplexity (exp of the mean per-token cross-entropy), the metric
                              from CSED 503 A2 and A4; lower is better, and the trainer flips
                              its is_best comparison accordingly
  - img/s                 ->  tok/s (each (B, T) batch scores B*T token predictions)
  - mixup/erasing         ->  gone; the windowing in text_data.py is the only "augmentation"

The one-sync-per-epoch discipline carries over untouched: every .item() is a GPU-to-CPU sync,
and at 32x32 that habit alone cost about 7%. Small LMs step even faster than small CNNs, so the
rule matters at least as much here.
"""
from __future__ import annotations

import json
import math
import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm


def ppl_of(mean_loss: float) -> float:
    """Perplexity from a mean cross-entropy, capped so an early wild epoch cannot print inf.

    exp(20) is ~485 million -- any perplexity near the cap means "not a language model yet",
    and the cap only exists so the log line and JSONL stay finite numbers.
    """
    return math.exp(min(20.0, mean_loss))


def train_one_epoch(model, ds, optimizer, criterion, scaler, device, batch_size, epoch, epochs,
                    use_amp=True, amp_dtype=torch.float16, clip=None):
    model.train()

    # The epoch accumulators, on the GPU, synced exactly once at the end -- see the module
    # docstring. n counts scored tokens, not sequences: every position in a (B, T) batch is a
    # prediction, so one batch contributes B*T to the totals.
    loss_sum = torch.zeros((), device=device)
    n = torch.zeros((), device=device)

    n_batches = ds.n_batches(batch_size)
    t0 = time.time()
    bar = tqdm(ds.epoch(batch_size, train=True), total=n_batches,
               desc=f'epoch {epoch:3d}/{epochs} train', leave=False, ncols=110)

    optimizer.zero_grad(set_to_none=True)

    for step, (x, y) in enumerate(bar):

        # Forward under autocast, then cross-entropy over every position at once. The logits
        # come out (B, T, V); flattening to (B*T, V) against (B*T,) targets is the standard
        # LM-loss reshape -- same objective as hello_text.ipynb, batched over the window.
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))

        # The standard AMP dance: scale, backward, (unscale for clipping), step, update.
        # Unlike a1, clipping defaults ON for both families here -- grad clipping is standard
        # LM practice (loss spikes on a bad window otherwise) and applies to LSTM and GPT alike,
        # so it cannot tilt the race the way ResNet-18's clip did in Part 1.
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # Fold the batch into the running totals, still on the GPU, no sync.
        b = y.numel()
        with torch.no_grad():
            loss_sum += loss.detach() * b
            n += b

        # Refresh the tqdm postfix only every 50 steps; each refresh is a host sync.
        if step % 50 == 0:
            done = (step + 1) * batch_size * ds.seq_len
            bar.set_postfix_str(f'loss {loss.item():.3f} '
                                f'{done/(time.time()-t0)/1000:.0f}k tok/s')

    # The one sync we saved up.
    dt = time.time() - t0
    n_f = n.item()
    mean_loss = (loss_sum / n).item()

    # Fail loudly on divergence, same as a1: a NaN run is worthless, do not burn hours on it.
    if mean_loss != mean_loss:
        raise RuntimeError(
            f'loss is NaN at epoch {epoch} -- the run has diverged. '
            f'Almost always the learning rate is too high; try --lr {0.5 * _lr(optimizer):.3g}')

    return {'loss': mean_loss, 'ppl': ppl_of(mean_loss), 'sec': dt, 'tok_s': n_f / dt}


def _lr(optimizer):
    # Current learning rate, straight off the first (and only) param group.
    return optimizer.param_groups[0]['lr']


@torch.no_grad()
def evaluate(model, ds, criterion, device, batch_size=128, use_amp=True,
             amp_dtype=torch.float16):
    # Same accumulate-on-GPU, sync-once discipline as training, over the fixed non-overlapping
    # validation windows. These are the perplexities you actually quote: every token scored
    # exactly once, in order, identically on every call.
    model.eval()
    loss_sum = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    for x, y in ds.epoch(batch_size, train=False):
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.reshape(-1))
        b = y.numel()
        loss_sum += loss * b
        n += b
    mean_loss = (loss_sum / n).item()
    return {'loss': mean_loss, 'ppl': ppl_of(mean_loss)}


# Checkpointing -- byte-for-byte the a1 logic. Nothing in it ever knew about images.

def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history):
    # Everything needed to resume bit-for-bit, written to a temp file then atomically renamed,
    # so a crash mid-write can trash the .tmp but never the good checkpoint underneath.
    tmp = path + '.tmp'
    net = model.module if isinstance(model, nn.DataParallel) else model
    torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                'epoch': epoch, 'best': best, 'history': history}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler, device):
    ck = torch.load(path, map_location=device)
    net = model.module if isinstance(model, nn.DataParallel) else model
    net.load_state_dict(ck['model'])
    optimizer.load_state_dict(ck['optimizer'])
    scheduler.load_state_dict(ck['scheduler'])
    if scaler.is_enabled() and ck.get('scaler'):
        scaler.load_state_dict(ck['scaler'])
    return ck['epoch'], ck['best'], ck['history']


# Logging: one human line plus one JSONL row per epoch. The JSONL is the contract with the
# dashboard and the analysis notebooks, same as a1 -- only the keys inside train/val changed.

def fmt_time(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec//60}m{sec%60:02d}s'
    return f'{sec//3600}h{(sec%3600)//60:02d}m'


def log_epoch(tag, epoch, epochs, tr, va, lr, elapsed, device, is_best, jsonl_path):
    mem = torch.cuda.max_memory_allocated(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    remaining = (epochs - epoch) * (tr['sec'] + 3)
    star = ' *' if is_best else '  '

    print(f'[{tag}] epoch {epoch:3d}/{epochs}{star}| '
          f'train loss {tr["loss"]:.3f} ppl {tr["ppl"]:9.2f} | '
          f'val ppl {va["ppl"]:9.2f} | '
          f'lr {lr:.5f} | {tr["sec"]:5.1f}s {tr["tok_s"]/1000:6.0f}k tok/s | '
          f'mem {mem:4.1f}/{total:.0f}GB | '
          f'elapsed {fmt_time(elapsed)} ETA {fmt_time(remaining)}', flush=True)

    with open(jsonl_path, 'a') as f:
        f.write(json.dumps({'epoch': epoch, 'lr': lr, 'elapsed': elapsed, 'is_best': is_best,
                            'train': tr, 'val': va}) + '\n')
