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
import json
import os
import subprocess
import sys
import time

import mlm_train

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(HERE, 'logs')
RUNS_DIR = os.path.join(HERE, 'runs')

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

    # Seeds at the budget the ladder actually ran at. Reports 04 and 05 measure everything
    # against a spread of 0.049, and that number came from the 33.8M model on Yoruba at 196.6M
    # updates -- a different corpus, a different model budget, and 5x less compute than any rung
    # it is used to judge. Two claims rest on it directly: that the data axis is flat past 64M
    # (+0.007 and +0.017, "inside the spread") and that Yoruba and English decelerate alike
    # (0.025 apart, "half the spread"). Both are unfalsifiable until the spread is measured here.
    #
    # The 86M rungs are the other half: seeding two of its cells gave sd 1.327, so the 256M and
    # 1024M cells are single draws from a distribution nobody has characterised.
    # Split into two queues, cheapest and most valuable first, because the scheduler orders by
    # cost and would otherwise run four 1.6-hour 86M cells before the four 40-minute 33.8M ones.
    # The 33.8M spread is what reports 04 and 05 actually lean on; the 86M column is already
    # labelled anecdotal, so it is the part worth having late rather than early.
    'seedcheck': [(d, 1_024_000_000, s, 'poc')
                  for d in (256_000_000, 64_000_000) for s in (1, 2)],

    'seedcheck86': [(d, 1_024_000_000, s, 'afriberta')
                    for d in (1_024_000_000, 256_000_000) for s in (1, 2)],
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
    if getattr(args, 'tag_prefix', ''):
        base = f'{args.tag_prefix}_{base}'

    stale = os.path.join(LOGS, f'{base}.log')
    if os.path.exists(stale):
        os.remove(stale)          # mlm_run appends; a fresh cell starts a fresh log

    cmd = [PY, '-u', '-W', 'ignore', os.path.join(HERE, 'mlm_run.py'),
           '--corpus', args.corpus, '--tokens', str(tokens), '--steps', str(steps),
           '--preset', preset, '--gpu', str(gpu), '--seed', str(seed),
           '--batch', str(args.batch), '--seq-len', str(args.seq_len)]
    if getattr(args, 'tag_prefix', ''):
        cmd += ['--tag', base]
    if getattr(args, 'warmup', None) is not None:
        cmd += ['--warmup', str(args.warmup)]
    if getattr(args, 'lr', None) is not None:
        cmd += ['--lr', str(args.lr)]
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


def write_plan(queue, args):
    """Publish the whole queue so the dashboard can show what has not run yet.

    Without this the queue exists only in this process's memory, so a watcher sees two runs in
    flight and no way to tell whether that is the whole study or the first tenth of it. Only the
    PLAN is written -- each cell's status is derived live by the dashboard from result files and
    running processes, so this file never goes stale and never needs updating as cells finish.
    """
    cells = []
    for spec in queue:
        tokens, update_tokens, seed, preset = unpack(spec, args.preset)
        cells.append({
            'tag': (f"{args.tag_prefix}_" if getattr(args, 'tag_prefix', '') else '')
                   + spec_tag(args.corpus, spec, args.preset, args.batch, args.seq_len),
            'tokens': tokens, 'update_tokens': update_tokens,
            'steps': steps_for(update_tokens, args.batch, args.seq_len),
            'preset': preset, 'seed': seed,
        })
    # A driver script that runs several fleets one after another used to be invisible past its
    # current step: each fleet overwrote this file, so the dashboard showed three cells of nine
    # and predicted a finish time for the third of them. UW_FLEET_QUEUE lets the driver declare
    # the whole queue up front; each fleet then contributes its own cells to it and leaves the
    # others alone.
    # Either an environment variable, for a driver launched with one, or a sentinel file --
    # which is the only channel available to a driver that is ALREADY RUNNING, since each fleet
    # is a fresh process that re-reads this module but inherits the driver's frozen environment.
    name = os.environ.get('UW_FLEET_QUEUE') or ''
    if not name:
        try:
            with open(os.path.join(RUNS_DIR, '_fleet_queue'), encoding='utf-8') as f:
                name = f.read().strip()
        except OSError:
            pass
    plan = {'corpus': args.corpus, 'queue': name or args.queue, 'batch': args.batch,
            'seq_len': args.seq_len, 'n_gpu': args.n_gpu, 'started': time.time(),
            'cells': cells}
    if name:
        try:
            with open(os.path.join(RUNS_DIR, '_fleet_plan.json'), encoding='utf-8') as f:
                prev = json.load(f)
            if prev.get('queue') == name:
                # Same queue, later fleet: keep the cells already declared and add ours, so the
                # panel counts the whole night rather than restarting at each stage.
                have = {c['tag'] for c in cells}
                plan['cells'] = [c for c in prev.get('cells', []) if c['tag'] not in have] + cells
                plan['started'] = prev.get('started', plan['started'])
        except (OSError, ValueError, KeyError):
            pass
    os.makedirs(RUNS_DIR, exist_ok=True)
    tmp = os.path.join(RUNS_DIR, '_fleet_plan.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, os.path.join(RUNS_DIR, '_fleet_plan.json'))


def run_fleet(args):
    queue = resolve_queue(args)
    if not args.smoke:
        write_plan(queue, args)
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
    p.add_argument('--tag-prefix', default='',
                   help='namespace every cell in this fleet, e.g. --tag-prefix warm15. '
                        'Required when sweeping anything the cell tag does not carry: '
                        'the tag is (corpus, tokens, steps, seed, preset), so two fleets '
                        'differing only in learning rate or warmup would name their cells '
                        'identically and the second would overwrite the first.')
    p.add_argument('--warmup', type=float, default=None,
                   help='warmup fraction, passed through to every cell')
    p.add_argument('--lr', type=float, default=None,
                   help='peak learning rate, passed through to every cell')
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
