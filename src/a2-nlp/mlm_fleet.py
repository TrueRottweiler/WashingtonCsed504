"""
mlm_fleet.py -- run the whole (data x compute) grid across both cards.

The counterpart of train_fleet.py, and the piece that helps the group most for the least new
code: their POC runs the grid one cell after another inside a notebook, so a 2x2 grid takes the
sum of four runs on one card while the second card sits idle. The scheduler here is the same one
the causal study used -- a queue of specs, one child process per card, refill a card the moment
it frees -- because nothing in it was ever specific to the objective being trained.

Usage:
    python mlm_fleet.py --corpus yor --queue poc          # the POC's 2x2 grid
    python mlm_fleet.py --corpus yor --queue poc --smoke  # prove the wiring first
    python mlm_fleet.py --corpus yor --queue full         # the 3-rung study grid
    python mlm_fleet.py --corpus yor --queue seeds        # the headline cell, three seeds
    python mlm_fleet.py --corpus yor --data 2000000 32000000 --update-tokens 49152000

Watch it with:  python dashboard.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import mlm_train

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(HERE, 'logs')

# Each spec is (data_tokens, update_tokens, seed) or (data_tokens, update_tokens, seed, preset).
#
# The fourth field exists because model size is a thing a queue may want to VARY rather than fix.
# With --preset applying to the whole fleet, a study that compares two model sizes had to be run
# as two fleets, and the scheduler could only balance within each -- which matters here, since the
# 86M preset measures 2.17x slower than 33.8M and a fleet of one size finishes long before a fleet
# of the other. Specs that omit it inherit --preset, so every existing queue is unchanged.
#
# The compute axis is a count of TOKENS OF UPDATES, not a count of steps, and that distinction
# is the whole reason this is worth stating. Steps are not comparable across batch sizes: the
# POC's 6,000 steps at batch 64 and 6,000 steps at batch 128 differ by a factor of two in work
# done, so "same steps, bigger batch" is not a faster run of the same experiment -- it is a
# different experiment that happens to score better. Fixing the token budget instead means the
# batch size is free to change for throughput reasons without moving the science, and steps
# fall out of it: steps = update_tokens / (batch * seq_len).
#
# Every queue is ordered LONGEST COMPUTE BUDGET FIRST, which is the same lesson a1-cv's
# train_fleet learned: a short job started last finishes early and leaves its card idle, but a
# long job started last sets the finish time by itself. Measured on the poc grid, the natural
# reading order (49M, 197M, 49M, 197M) put both short cells and one long cell on card 0 while
# card 1 ran a single long cell and then sat idle for four minutes -- 12.2 min wall-clock for
# 10.1 min of work. Longest-first balances the two cards at ~246M tokens of updates each.
#
#   poc    the notebook's 2x2 grid: two data rungs against two compute budgets. 49M and 197M
#          are exactly the POC's 6k and 24k steps at its batch 64 x seq 128.
#   full   the three-rung ladder Gate 7 extrapolates to, at 12 passes over each rung.
#   seeds  one cell repeated, to size the seed spread. The POC's own warning applies: a gain
#          smaller than this spread is not a gain.
QUEUES = {
    'poc': [(2_000_000, 196_608_000, 0), (32_000_000, 196_608_000, 0),
            (2_000_000, 49_152_000, 0), (32_000_000, 49_152_000, 0)],

    'full': [(64_000_000, 768_000_000, 0), (16_000_000, 192_000_000, 0),
             (4_000_000, 48_000_000, 0)],

    'seeds': [(32_000_000, 196_608_000, 0), (32_000_000, 196_608_000, 1),
              (32_000_000, 196_608_000, 2)],

    # The English ladder. Two things the Yoruba ladder cannot answer, for the same reason: all of
    # FineWeb-2 Yoruba is 69.1M tokens, so its 64M rung already uses 93% of the language.
    #
    # Compute is held FIXED at 1.024B tokens of updates across every rung, which is the Yoruba
    # ladder's design at 5.3x the budget. Holding it fixed is the point: scaling compute along
    # with data would confound the two axes, and we would not be able to say which one moved the
    # loss. The Yoruba version found data worth 0.075 nats against 0.049 of seed noise -- but
    # every rung there was compute-bound, so the data axis was never really under test. At 1.024B
    # updates neither model is starved, and the rungs span 256x instead of 16x.
    #
    # Both presets at every rung, because the 86M model losing at every Yoruba rung reads as
    # undertraining rather than a verdict on capacity. If there is a crossover this finds it.
    'engladder': [(d, 1_024_000_000, 0, p)
                  for p in ('afriberta', 'poc')          # slowest preset first; see below
                  for d in (1_024_000_000, 256_000_000, 64_000_000, 16_000_000, 4_000_000)],
}


def steps_for(update_tokens: int, batch: int, seq_len: int) -> int:
    """How many optimizer steps this token budget buys at this batch shape."""
    return max(1, round(update_tokens / (batch * seq_len)))


def unpack(spec, default_preset):
    """(tokens, update_tokens, seed, preset), filling in the fleet-wide preset when omitted."""
    tokens, update_tokens, seed = spec[0], spec[1], spec[2]
    preset = spec[3] if len(spec) > 3 else default_preset
    return tokens, update_tokens, seed, preset


def spec_tag(corpus, spec, preset, batch, seq_len):
    # The one definition lives in mlm_train, so the fleet, the single-cell runner, and the
    # notebook API cannot drift into naming the same run three different things. The tag carries
    # the STEP count, so a cell keeps its identity across batch sizes only if the token budget
    # is what was held fixed -- which is exactly what QUEUES now does.
    tokens, update_tokens, seed, preset = unpack(spec, preset)
    return mlm_train.cell_tag(corpus, tokens,
                              steps_for(update_tokens, batch, seq_len), seed, preset)


def resolve_queue(args):
    if args.data and args.update_tokens:
        return [(d, u, seed) for d in args.data for u in args.update_tokens
                for seed in args.seeds]
    if args.queue:
        return list(QUEUES[args.queue])
    return None


def launch(spec, gpu, args):
    """Start one mlm_run.py on one card and hand back a live handle."""
    tokens, update_tokens, seed, preset = unpack(spec, args.preset)
    steps = steps_for(update_tokens, args.batch, args.seq_len)
    base = spec_tag(args.corpus, spec, args.preset, args.batch, args.seq_len)

    stale = os.path.join(LOGS, f'{base}.log')
    if os.path.exists(stale):
        os.remove(stale)          # mlm_run appends; a fresh cell starts a fresh log

    cmd = [PY, '-u', '-W', 'ignore', os.path.join(HERE, 'mlm_run.py'),
           '--corpus', args.corpus, '--tokens', str(tokens), '--steps', str(steps),
           '--preset', preset, '--gpu', str(gpu), '--seed', str(seed),
           '--batch', str(args.batch), '--seq-len', str(args.seq_len)]
    if args.smoke:
        cmd.append('--smoke')
        base = f'smoke-{base}'

    # mlm_run.py writes logs/<tag>.log itself now, so the fleet must NOT also redirect into
    # that file -- two writers on one path interleave and corrupt it. We keep stderr so a child
    # that dies before it can log still reports why.
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f'{base}.stderr.log'), 'w', encoding='utf-8')
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=log, cwd=HERE,
                            creationflags=flags)
    # Same compact formatting the tags use, so the console line and the run name agree. The
    # raw //10**6 version printed "0M x 0k" for every cell below a million tokens.
    print(f'  [gpu {gpu}] start  {base:34s} '
          f'{mlm_train.compact(tokens):>6} data x {mlm_train.compact(update_tokens):>6} upd '
          f'= {mlm_train.compact(steps):>5} steps  {preset:9s} (pid {proc.pid})  '
          f'logs/{base}.log', flush=True)
    return {'base': base, 'gpu': gpu, 'proc': proc, 'log': log, 't0': time.time()}


def run_fleet(args):
    queue = resolve_queue(args)
    print(f'\nFleet: {len(queue)} cells across {args.n_gpu} cards'
          f'{" (SMOKE)" if args.smoke else ""}')
    print(f'  batch {args.batch} x seq {args.seq_len} = {args.batch*args.seq_len:,} tok/step')
    for spec in queue:
        print(f'  {spec_tag(args.corpus, spec, args.preset, args.batch, args.seq_len)}')
    print()

    slots = [None] * args.n_gpu
    for g in range(args.n_gpu):
        if queue:
            slots[g] = launch(queue.pop(0), g, args)

    print('\n  cards are busy. Watch:  python dashboard.py\n')
    try:
        while any(slots) or queue:
            for g in range(args.n_gpu):
                job = slots[g]
                if job is None:
                    continue
                ret = job['proc'].poll()
                if ret is None:
                    continue
                job['log'].close()
                dt = time.time() - job['t0']
                ok = 'done ' if ret == 0 else f'FAILED({ret})'
                print(f'  [gpu {g}] {ok} {job["base"]:34s} in {dt/60:.1f} min', flush=True)
                slots[g] = launch(queue.pop(0), g, args) if queue else None
            time.sleep(1.0)
    except KeyboardInterrupt:
        print('\nCtrl+C received, stopping all runs...')
        for job in slots:
            if job:
                job['proc'].terminate()
                job['log'].close()

    print('\nFleet done. Records in runs/*_result.json ; curves in runs/*.jsonl')
    print('Next: the random-init control, if it does not exist yet:')
    print(f'  python mlm_run.py --corpus {args.corpus} --random-init')


def main():
    p = argparse.ArgumentParser(
        description='Train the (data x compute) MLM grid across every GPU.')
    p.add_argument('--corpus', required=True)
    p.add_argument('--queue', choices=list(QUEUES))
    p.add_argument('--data', type=int, nargs='+', default=None, help='custom data rungs')
    p.add_argument('--update-tokens', type=int, nargs='+', default=None,
                   help='custom compute budgets, in tokens of updates (not steps)')
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    p.add_argument('--preset', choices=['poc', 'afriberta'], default='poc')
    p.add_argument('--batch', type=int, default=128,
                   help='sequences per step; 128 measured 1.33x the throughput of 64 and the '
                        'token budget above keeps the work constant when this changes')
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--n-gpu', type=int, default=None,
                   help='cards to spread across (default: however many this machine has)')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()

    if not args.queue and not (args.data and args.update_tokens):
        p.error('nothing to train -- pick a preset (--queue poc) or give both '
                '--data and --update-tokens.')

    # Count the cards rather than assuming two. This workstation has two; a Colab session has
    # one, and asking for a second there fails deep inside CUDA with an unhelpful message after
    # the first cell has already started.
    if args.n_gpu is None:
        import torch
        args.n_gpu = max(1, torch.cuda.device_count())
    if args.n_gpu == 1:
        print('one GPU: cells will run one after another. The queue order still matters '
              'for nothing here, but the records and dashboard work the same.')

    run_fleet(args)


if __name__ == '__main__':
    main()
