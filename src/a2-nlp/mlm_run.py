"""
mlm_run.py -- one process, one GPU, one cell of the (data x compute) grid.

The counterpart of train_run.py for masked language models. Everything long-running belongs
here rather than in a notebook: a kernel that dies at hour three takes its checkpoints with it,
and the group's real study is a grid of multi-hour runs.

Usage:
    python mlm_run.py --corpus yor --tokens 32000000 --steps 24000 --gpu 0
    python mlm_run.py --corpus yor --tokens 2000000  --steps 6000  --gpu 1 --seed 1
    python mlm_run.py --corpus yor --smoke                      # wiring check, ~a minute
    python mlm_run.py --corpus yor --random-init                # the no-pretraining control
    python mlm_run.py --corpus yor --estimate --steps 24000     # predict, do not train

Watch a fleet of these with:  python dashboard.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

import mlm_data as D
import mlm_train as M

RUNS = M.RUNS
LOGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')


class _Tee:
    """Write console output to logs/<tag>.log as well as the terminal.

    The runner owns its own log rather than relying on the fleet to redirect it. Without this a
    run started by hand left no log at all, so the dashboards -- which discover runs and read
    their headers from logs/ -- could not see it, show its progress, or estimate its finish.
    """

    def __init__(self, stream, path):
        self.stream = stream
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, 'a', encoding='utf-8', buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def main():
    p = argparse.ArgumentParser(description='Pretrain one MLM grid cell on one GPU.')
    p.add_argument('--corpus', required=True, help='a corpus prepared by mlm_data.py')
    p.add_argument('--tokens', type=int, default=32_000_000, help='unique tokens (the data axis)')
    p.add_argument('--steps', type=int, default=24_000, help='optimizer steps (the compute axis)')
    p.add_argument('--preset', choices=list(M.SIZE_PRESETS), default='poc')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch', type=int, default=128,
                   help='sequences per step; 128 measured 1.33x the throughput of 64')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--lr', type=float, default=None,
                   help='peak LR; default is per-preset (see PRESET_LR) because 5e-4 collapses the afriberta preset')
    p.add_argument('--mlm-prob', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--clip', type=float, default=1.0)
    p.add_argument('--warmup', type=float, default=None,
                   help='warmup fraction of the run; default is per-length (see '
                        'pretrain). Set explicitly to sweep the schedule itself.')
    p.add_argument('--accum', type=int, default=1,
                   help='micro-batches per optimizer step; effective batch is batch x accum')
    p.add_argument('--no-resume', action='store_true',
                   help='start fresh even if an interrupted run left state behind')
    p.add_argument('--store-dtype', choices=['auto', 'int16', 'int32'], default='auto')
    p.add_argument('--tag', default=None)
    p.add_argument('--smoke', action='store_true',
                   help='2M tokens, 200 steps -- proves the wiring, then exits')
    p.add_argument('--random-init', action='store_true',
                   help='write the untrained control checkpoint and exit')
    p.add_argument('--estimate', action='store_true',
                   help='measure throughput, print the predicted hours, and exit')
    args = p.parse_args()

    # Fail here with a sentence someone can act on, rather than a CUDA assertion thrown from
    # inside the first forward pass. A one-GPU machine asked for --gpu 1 is the common case.
    # Test the count, not is_available(): with CUDA_VISIBLE_DEVICES empty, is_available() can
    # still report True while there are no devices to use, which produced the nonsense message
    # "this machine has 0 (numbered 0..-1)".
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemExit('no CUDA device found. Pretraining needs a GPU; on Colab, choose a '
                         'GPU runtime under Runtime > Change runtime type.')
    if args.gpu >= n_gpu:
        raise SystemExit(f'--gpu {args.gpu} was requested but this machine has {n_gpu} '
                         f'(numbered 0..{n_gpu - 1}).')

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    tokenizer = D.load_tokenizer(args.corpus)

    # The control needs no data and no training, so it short-circuits before anything loads.
    if args.random_init:
        stats = D.T.load_stats(args.corpus)
        out = os.path.join(RUNS, f'{args.corpus}_random_init')
        M.save_random_init(stats['vocab_size'], tokenizer, args.preset, args.seq_len, out)
        print(f'random-init control written -> {out}')
        return

    tokens = 2_000_000 if args.smoke else args.tokens
    steps = 200 if args.smoke else args.steps

    ds = D.MlmTokens(device, args.corpus, 'train', seq_len=args.seq_len, subset=tokens,
                     store_dtype=args.store_dtype)
    tag = args.tag or M.cell_tag(args.corpus, ds.n, steps, args.seed, args.preset)
    if args.smoke:
        # Name the smoke run after the cell it stands in for, not after the shrunken shape it
        # actually ran. Every cell collapses to the same 2M x 200, so a tag built from that is
        # the same string for every cell -- and a fleet smoke test then points two cards at one
        # run directory, where they race on train_state.pt and one dies with WinError 32. The
        # requested shape is what makes the cells distinct, so it is what the tag has to carry.
        tag = 'smoke-' + M.cell_tag(args.corpus, args.tokens, args.steps,
                                    args.seed, args.preset)

    # From here on, everything printed also lands in logs/<tag>.log.
    sys.stdout = _Tee(sys.stdout, os.path.join(LOGS, f'{tag}.log'))

    # The header dashboard.py parses: it looks for "device cuda:N" and "dataset <name> |" to
    # label a run and predict its finish time. Without them an MLM run showed up unlabelled --
    # and the absent dataset took the whole dashboard down with a TypeError rather than simply
    # going unpredicted. Keep the shape of these lines if you edit them.
    print(f'[{tag}] device {device} ({torch.cuda.get_device_name(device)})')
    print(f'[{tag}] dataset {args.corpus} | seq_len {args.seq_len} | batch {args.batch} | '
          f'preset {args.preset}')
    print(f'[{tag}] {ds.n:,} tokens resident as '
          f'{str(ds.store_dtype).replace("torch.", "")} ({ds.gb():.3f} GB), '
          f'vocab {ds.vocab_size:,}')

    if args.estimate:
        tok_s = M.measure_throughput(ds, tokenizer, args.preset, args.batch)
        est = M.estimate_hours(tok_s, [(ds.n, steps)], args.batch, args.seq_len, args.preset)
        cell = est['cells'][0]
        print(f'\nmeasured {tok_s/1e3:.0f}k tok/s at preset {args.preset!r}')
        print(f'{steps:,} steps = {cell["tokens_seen"]:,} tokens seen '
              f'({cell["passes"]:.1f} passes) -> {cell["hours"]:.2f} h\n')
        return

    val = D.MlmTokens(device, args.corpus, 'val', seq_len=args.seq_len,
                      store_dtype=args.store_dtype)
    record = M.pretrain(ds, tokenizer, tag, steps, preset=args.preset, batch=args.batch,
                        lr=args.lr, mlm_prob=args.mlm_prob, seed=args.seed, clip=args.clip,
                        accum=args.accum, resume=not args.no_resume, warmup=args.warmup,
                        val_batches=val.fixed_val_batches(mlm_prob=args.mlm_prob))
    print(json.dumps({k: v for k, v in record.items() if k != 'history'}, indent=2))


if __name__ == '__main__':
    main()

    # Leave immediately rather than through interpreter shutdown. A finished run has already
    # written everything that matters -- the result JSON, the checkpoint, and the log, which is
    # line-buffered -- so there is nothing left to flush but the console stream.
    #
    # This is not defensive tidying. multi_ind returned from main() and then sat for thirty
    # hours with one core pegged at 100% and its GPU memory still held, while four identical
    # runs exited cleanly, so it is a race somewhere in CUDA or thread teardown rather than a
    # path we can find and fix. The cost of leaving it alone is worse than a hung process: the
    # fleet advances a card only when its child exits, so one run that never returns parks that
    # card for the rest of the queue.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
