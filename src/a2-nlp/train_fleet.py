"""Keep both RTX PRO 6000s busy training language models, one run per card.

Duplicated from a1-cv/train_fleet.py; the scheduler itself is unchanged -- a queue of specs,
one train_run.py per card, refill a card the moment it frees up -- because nothing in it was
ever image-specific. Only the presets changed, to the A2 study's runs.

Usage (pick a preset queue, or a custom batch):
    python train_fleet.py --queue shakespeare          # both baselines on the smoke rung (~min)
    python train_fleet.py --queue wikitext2            # all four models on the small rung
    python train_fleet.py --queue seeds                # both headline models on wikitext103, x3 seeds
    python train_fleet.py --queue overnight            # seeds + both capacity controls
    python train_fleet.py --queue wikitext2 --smoke    # prove the wiring first, then it exits
    python train_fleet.py --dataset wikitext103 --models lstm gpt --epochs 10   # a custom batch

Watch it from two other terminals:
    python dashboard.py                       # the live dashboard: both cards, curves, ETA
    nvidia-smi dmon -s u                      # the raw truth; the sm column is the compute engine

Same why-a-fleet story as Part 1: two cards buy two experiments, not one faster one. The LSTM
and the GPT are the pair we always want side by side anyway, so the fleet's natural schedule --
one family per card -- is also the experiment's.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

# HERE is this folder (src/a2-nlp); train_run.py, dashboard.py, runs/ and logs/ all live here.
# PY is the exact Python running this script, so the children run in the same conda env.
HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LOGS = os.path.join(HERE, 'logs')

# The preset queues. Each entry is (dataset, model, epochs[, seed]); epochs=None means "use the
# --epochs default". Within each list the first two entries land one per card.
#
#   shakespeare  both baselines on the char-level rung. This is the "does anything learn" rung:
#                minutes end to end, and the perplexities anchor against CSED 503 A2/A4.
#   wikitext2    all four models on the 2M-token rung. The small models will overfit hard here
#                -- that is data, not a bug; best-so-far val ppl is what gets recorded, and how
#                early each family peaks IS a finding.
#   seeds        the two headline models on wikitext103 at a matched budget, three seeds each,
#                interleaved lstm, gpt so the two cards stay evenly loaded as the queue drains.
#                Three seeds is the Part 1 lesson: a single run's gap can be pure noise.
#   overnight    seeds plus both capacity controls at the same budget, longest jobs first so
#                the big models cannot end up starting last and setting the finish time alone.
QUEUES = {
    'shakespeare': [('shakespeare', 'lstm', 30), ('shakespeare', 'gpt', 30)],

    'wikitext2':   [('wikitext2', 'gpt', 30), ('wikitext2', 'lstm', 30),
                    ('wikitext2', 'gpt_medium', 30), ('wikitext2', 'lstm_large', 30)],

    'seeds':       [('wikitext103', 'lstm', 20, 1), ('wikitext103', 'gpt', 20, 1),
                    ('wikitext103', 'lstm', 20, 2), ('wikitext103', 'gpt', 20, 2),
                    ('wikitext103', 'lstm', 20, 3), ('wikitext103', 'gpt', 20, 3)],

    'overnight':   [('wikitext103', 'gpt_medium', 20, 1), ('wikitext103', 'lstm_large', 20, 1),
                    ('wikitext103', 'lstm', 20, 1), ('wikitext103', 'gpt', 20, 1),
                    ('wikitext103', 'lstm', 20, 2), ('wikitext103', 'gpt', 20, 2),
                    ('wikitext103', 'lstm', 20, 3), ('wikitext103', 'gpt', 20, 3)],
}


def tag_base(dataset, model):
    """The run's name stem, matching train_run.py: always <dataset>_<model>."""
    return f'{dataset}_{model}'


def spec_name(spec):
    """The run name for one queue spec, with the _sN suffix when the spec carries a seed."""
    base = tag_base(spec[0], spec[1])
    return f'{base}_s{spec[3]}' if len(spec) > 3 else base


def resolve_queue(args):
    """Turn the CLI options into the list of specs to run. A custom --models list wins."""
    if args.models:
        return [(args.dataset, m, None) for m in args.models]
    if args.queue:
        return list(QUEUES[args.queue])
    return None


def launch(spec, gpu, args):
    """Start one train_run.py on one card and return a live handle we can poll later."""
    dataset, model, epochs = spec[0], spec[1], spec[2]
    epochs = args.epochs if epochs is None else epochs
    seed = spec[3] if len(spec) > 3 else None
    base = spec_name(spec)

    # -u keeps the child's stdout unbuffered so the dashboard sees each tqdm line immediately.
    cmd = [PY, '-u', '-W', 'ignore', os.path.join(HERE, 'train_run.py'),
           '--dataset', dataset, '--model', model, '--gpu', str(gpu), '--epochs', str(epochs)]

    if seed is not None:
        cmd += ['--seed', str(seed), '--tag', base]

    if args.smoke:
        cmd.append('--smoke-test')

    # Each run's console output goes to logs/<base>.log: the dashboard scrapes the live tqdm
    # tail from it, and an hours-long run's output must survive us closing this terminal.
    os.makedirs(LOGS, exist_ok=True)
    log = open(os.path.join(LOGS, f'{base}.log'), 'w', encoding='utf-8')

    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE,
                            creationflags=flags)

    label = 'SMOKE' if args.smoke else f'{epochs}ep'
    print(f'  [gpu {gpu}] start  {base:22s} {label:>6s}  (pid {proc.pid})  '
          f'logging to logs/{base}.log', flush=True)
    return {'base': base, 'gpu': gpu, 'proc': proc, 'log': log, 't0': time.time()}


def run_fleet(args):
    """The scheduler: keep every card busy until the queue is empty."""
    queue = resolve_queue(args)
    names = [spec_name(s) for s in queue]
    print(f'\nFleet: {len(queue)} runs {names} across {args.n_gpu} cards'
          f'{" (SMOKE)" if args.smoke else ""}\n')

    slots = [None] * args.n_gpu

    # Prime the pump: one job per card so both start immediately.
    for g in range(args.n_gpu):
        if queue:
            slots[g] = launch(queue.pop(0), g, args)

    print('\n  both cards are now training. Watch:  python dashboard.py\n')

    # Poll each card about once a second; when a run finishes, start the next queued job there.
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
                print(f'  [gpu {g}] {ok} {job["base"]:22s}  in {dt/60:.1f} min', flush=True)
                slots[g] = launch(queue.pop(0), g, args) if queue else None
            time.sleep(1.0)
    except KeyboardInterrupt:
        # Ctrl+C tears the whole fleet down cleanly rather than orphaning the trainers.
        print('\nCtrl+C received, stopping all runs...')
        for job in slots:
            if job:
                job['proc'].terminate()
                job['log'].close()

    print('\nFleet done. Results in runs/*_result.json ; curves in runs/*.jsonl')


def main():
    p = argparse.ArgumentParser(
        description='Train a batch of language models across both GPUs, one run per card.')
    p.add_argument('--queue', choices=list(QUEUES),
                   help="which preset batch to train (see the QUEUES table at the top)")
    p.add_argument('--models', nargs='+', default=None,
                   help='train a custom list of models instead of a preset')
    p.add_argument('--dataset', default='wikitext2',
                   help='dataset for a custom --models list (presets carry their own)')
    p.add_argument('--epochs', type=int, default=30,
                   help='epochs per run when a spec sets none of its own')
    p.add_argument('--n-gpu', type=int, default=2, help='how many GPUs to spread across')
    p.add_argument('--smoke', action='store_true',
                   help='run each job as a quick --smoke-test to prove the wiring, then exit')
    args = p.parse_args()

    if not args.queue and not args.models:
        p.error('nothing to train -- choose a preset (--queue wikitext2) or a custom batch '
                '(--models lstm gpt --dataset wikitext103).')

    run_fleet(args)


if __name__ == '__main__':
    main()
