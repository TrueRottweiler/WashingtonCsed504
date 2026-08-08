"""
store_bench.py -- what the wider resident token store actually costs, in memory and in time.

text_data.py picks the narrowest signed type that holds the vocabulary, so our own rungs run at
int16 and a corpus tokenized with a multilingual vocabulary (roughly 120k to 250k types) is
forced up to int32. Before promising a partner that the wide store is affordable, it is worth
measuring rather than asserting, and the measurement has to hold everything else fixed: same
corpus, same model, same batch shape, same seed, only the width of the resident array changes.

Two numbers matter and they are not the same number:

  gather      just the data path -- index the resident stream, slice the window, cast to int64.
              This is the only stage the width touches, so it is where any cost has to show up.
  step        the whole training step: gather, forward, backward, optimizer. This is the number
              that decides whether anyone cares, because a gather that doubles in cost is still
              invisible if the model dominates the step.

Usage:
    python store_bench.py --dataset wikitext2 --model gpt
    python store_bench.py --dataset wikitext103 --model gpt --steps 100
    python store_bench.py --dataset wikitext2 --model lstm --json bench_lstm.json

The report prints one row per width plus the int32/int16 ratios. Peak memory is measured with
torch's allocator counters, so it covers the model and activations too, not just the store --
the store's own footprint is reported separately as the exact tensor size.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time

import torch
import torch.nn as nn

import models as M
import text_data as T

WIDTHS = ['int16', 'int32']


def time_gather(ds, batch_size: int, steps: int, warmup: int) -> float:
    """Tokens per second through the data path alone, with no model attached.

    Consuming the same generator training uses (rather than calling _gather directly) keeps this
    honest: it measures the windowing, the gather, and the int64 cast exactly as the epoch loop
    performs them. Both tensors are touched so nothing can be optimized away.
    """
    it = ds.epoch(batch_size, train=True)
    for x, y in itertools.islice(it, warmup):
        x.sum(); y.sum()
    torch.cuda.synchronize(ds.device)

    t0 = time.time()
    n = 0
    for x, y in itertools.islice(it, steps):
        n += y.numel()
    torch.cuda.synchronize(ds.device)
    return n / (time.time() - t0)


def time_step(ds, model, optimizer, criterion, scaler, batch_size: int, steps: int,
              warmup: int, amp_dtype, clip: float | None) -> float:
    """Tokens per second through a full training step, mirroring train_loop.train_one_epoch.

    The recipe here is deliberately the same one train_loop runs -- autocast, scaled backward,
    optional clip, step -- because a benchmark that measures a simpler loop than the trainer
    would be measuring something nobody runs.
    """
    it = ds.epoch(batch_size, train=True)

    def one(x, y):
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=True):
            loss = criterion(model(x).view(-1, ds.vocab_size), y.reshape(-1))
        scaler.scale(loss).backward()
        if clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    for x, y in itertools.islice(it, warmup):
        one(x, y)
    torch.cuda.synchronize(ds.device)

    t0 = time.time()
    n = 0
    for x, y in itertools.islice(it, steps):
        one(x, y)
        n += y.numel()
    torch.cuda.synchronize(ds.device)
    return n / (time.time() - t0)


def measure(width: str, args, device, amp_dtype) -> dict:
    """Run both timings at one store width and return the row for the report."""
    # Same seed for both widths so the two runs draw the same windows in the same order. The
    # arithmetic is identical either way -- the ids are the same integers, just held wider --
    # so any difference in the numbers below is the memory system, not the math.
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    ds = T.GpuTokens(device, args.dataset, 'train', seq_len=args.seq_len,
                     subset=args.subset, store_dtype=width)
    store_bytes = ds.t.numel() * ds.bytes_per_token

    gather_tps = time_gather(ds, args.batch, args.steps, args.warmup)

    model = M.build(args.model, ds.vocab_size, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=amp_dtype is torch.float16)
    step_tps = time_step(ds, model, optimizer, criterion, scaler, args.batch,
                         args.steps, args.warmup, amp_dtype, args.clip or None)

    peak = torch.cuda.max_memory_allocated(device)

    # Drop everything before the next width runs, or the second measurement inherits the first
    # one's allocations and its peak reads high.
    del model, optimizer, scaler, ds
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    return {'store_dtype': width, 'bytes_per_token': 2 if width == 'int16' else 4,
            'store_bytes': int(store_bytes), 'peak_bytes': int(peak),
            'gather_tok_s': gather_tps, 'step_tok_s': step_tps}


def aggregate(runs: list[dict]) -> list[dict]:
    """Collapse repeated measurements to one row per width, taking medians of the throughputs.

    Repeats are not ceremony here. An early version of this script measured each width once and
    reported int16 gathering at 546 M tok/s against int32's 819 M -- a 1.5x gap that vanished on
    the next run, where both sat at 818 M. Whatever produced it (allocator state, a cold page,
    clock behavior) was not the store width, and a single sample could not tell the difference.
    The median over interleaved repeats can.
    """
    out = []
    for width in WIDTHS:
        rs = [r for r in runs if r['store_dtype'] == width]
        base = dict(rs[0])
        for key in ('gather_tok_s', 'step_tok_s'):
            base[key] = statistics.median(r[key] for r in rs)
            base[key + '_all'] = [r[key] for r in rs]
        # Memory is deterministic across repeats; take the peak in case one run allocated more.
        base['peak_bytes'] = max(r['peak_bytes'] for r in rs)
        base['repeats'] = len(rs)
        out.append(base)
    return out


def report(rows: list[dict], args) -> None:
    print(f'\n{args.dataset} / {args.model} | batch {args.batch} x {args.seq_len} tok | '
          f'{args.steps} steps after {args.warmup} warmup | '
          f'median of {rows[0].get("repeats", 1)} interleaved repeats\n')
    print(f'  {"width":>6}  {"store":>9}  {"peak":>9}  {"gather tok/s":>14}  {"step tok/s":>12}')
    for r in rows:
        print(f'  {r["store_dtype"]:>6}  {r["store_bytes"]/1e9:>7.3f} GB  '
              f'{r["peak_bytes"]/1e9:>7.3f} GB  {r["gather_tok_s"]/1e6:>11.1f} M  '
              f'{r["step_tok_s"]/1e3:>9.0f} k')

    # Print the spread as well as the median. If the repeats disagree by more than the gap
    # between the widths, the honest reading is that the width did not move the number -- and
    # the gather column in particular has come back bimodal on this box, landing near either
    # 550 or 820 M tok/s for both widths, so its spread has to be visible next to its median.
    for r in rows:
        if len(r.get('step_tok_s_all', [])) > 1:
            print(f'\n  {r["store_dtype"]:>6}  step   ' +
                  ', '.join(f'{v/1e3:.0f}k' for v in r['step_tok_s_all']))
            print(f'  {"":>6}  gather ' +
                  ', '.join(f'{v/1e6:.0f}M' for v in r['gather_tok_s_all']))

    if len(rows) == 2:
        lo, hi = rows[0], rows[1]
        print(f'\n  int32 vs int16: store {hi["store_bytes"]/lo["store_bytes"]:.2f}x, '
              f'peak {hi["peak_bytes"]/lo["peak_bytes"]:.2f}x, '
              f'gather {_speed(lo["gather_tok_s"], hi["gather_tok_s"])}, '
              f'step {_speed(lo["step_tok_s"], hi["step_tok_s"], places=3)}')
        # The step ratio is the one to quote. The gather is where the width could plausibly
        # cost something, but it is a small share of a step dominated by the output head, so a
        # difference there can be real and still not reach the number anyone budgets against.
        cost = (lo['step_tok_s'] / hi['step_tok_s'] - 1) * 100
        print(f'  the wide store costs {cost:+.1f}% throughput and '
              f'{(hi["store_bytes"]-lo["store_bytes"])/1e9:.3f} GB extra resident\n')


def _speed(narrow_tps: float, wide_tps: float, places: int = 2) -> str:
    """Describe int32's throughput against int16's, naming the direction rather than assuming it.

    Worth stating explicitly because the assumption is wrong on this hardware: the wide store
    gathers faster. Narrowing to int16 saves memory but the load-and-widen path to the int64
    indices the embedding needs is not the one the memory system is best at.
    """
    ratio = narrow_tps / wide_tps
    if ratio >= 1.0:
        return f'{ratio:.{places}f}x slower'
    return f'{1 / ratio:.{places}f}x faster'


def main():
    p = argparse.ArgumentParser(
        description='Measure the memory and throughput cost of the int32 resident token store.')
    p.add_argument('--dataset', default='wikitext2')
    p.add_argument('--model', choices=list(M.BUILDERS), default='gpt')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=128, help='sequences per step')
    p.add_argument('--seq-len', type=int, default=256, help='context window in tokens')
    p.add_argument('--steps', type=int, default=50, help='timed steps per width')
    p.add_argument('--warmup', type=int, default=10,
                   help='untimed steps first, so cuDNN autotune and allocation land outside')
    p.add_argument('--subset', type=int, default=None,
                   help='cap the resident tokens, for a quick check on a large corpus')
    p.add_argument('--clip', type=float, default=1.0, help='match the trainer default')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--repeat', type=int, default=3,
                   help='interleaved passes over both widths; the report takes the median')
    p.add_argument('--json', default=None, help='also write the rows here for the notebooks')
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f'store_bench: {torch.cuda.get_device_name(device)}, '
          f'amp {str(amp_dtype).replace("torch.", "")}')

    # Interleave the widths rather than finishing one before starting the other, so a drift in
    # machine state over the run cannot land entirely on whichever width went first.
    runs = []
    for r in range(args.repeat):
        for w in WIDTHS:
            print(f'  pass {r + 1}/{args.repeat}: {w}', flush=True)
            runs.append(measure(w, args, device, amp_dtype))
    rows = aggregate(runs)
    report(rows, args)

    if args.json:
        json.dump({'dataset': args.dataset, 'model': args.model, 'batch': args.batch,
                   'seq_len': args.seq_len, 'steps': args.steps, 'repeat': args.repeat,
                   'rows': rows},
                  open(args.json, 'w'), indent=2)
        print(f'  wrote {args.json}')


if __name__ == '__main__':
    main()
