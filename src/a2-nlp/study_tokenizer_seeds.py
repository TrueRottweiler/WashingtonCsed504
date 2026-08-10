"""Three more seeds per arm, so the tokenizer penalty is either established or dropped.

claims_audit.py found that the headline of report 08 -- a 0.144 bits-per-character penalty for
XLM-R's vocabulary at matched compute -- does not survive its own null. Welch t = 1.68, p = 0.22,
at three seeds against three.

The number is not wrong and the direction is consistent across every seed. What was wrong is the
bar it cleared. This project's rule is "bigger than the cell's own seed spread", and 0.144 against
a pooled 0.105 is 1.4x, so it passed. That rule is weaker than a significance test: it correctly
rejects differences SMALLER than the noise and says nothing useful about differences slightly
larger, which is exactly where this one sits. At n = 3 you need roughly 2.5x the sd.

Rather than hedge the sentence, buy the power. At the observed effect size:

    n = 3   t = 1.68   p = 0.168
    n = 4   t = 1.94   p = 0.100
    n = 5   t = 2.17   p = 0.061
    n = 6   t = 2.38   p = 0.039   <- clears

So three more seeds per arm, six runs, about four GPU-hours. Note what this is NOT: it is not
running seeds until the p-value cooperates. Six is chosen in advance from the power calculation
above, it is written down here before the runs start, and if the penalty fails at n = 6 the
report says the penalty is not established. Deciding the sample size after seeing the result is
the failure mode this whole board is about.

    bash src/a2-nlp/py.sh study_tokenizer_seeds.py --dry-run
    bash src/a2-nlp/py.sh study_tokenizer_seeds.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import time

from scipy import stats

import fleet_plan
import mlm_api as f
import mlm_train as _train

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'tokenizer_seeds.json')

# Both arms exactly as the existing three seeds were run. Anything that differs here makes the
# new seeds a different cell rather than more of the same one.
ARMS = [
    {'name': '250k vocabulary', 'corpus': 'yor_xlmr', 'tokens': 121_339_416,
     'steps': 12_000, 'prefix': 'swap_yor_xlmr'},
    {'name': '16k vocabulary', 'corpus': 'yor', 'tokens': 69_096_452,
     'steps': 62_500, 'prefix': 'swap62k'},
]
NEW_SEEDS = [3, 4, 5]
LR, BATCH, PRESET, CLIP = 5e-4, 128, 'poc', 1.0


def bpc(rec, corpus):
    return rec['val_loss'] / math.log(2) / f.corpus_info(corpus)['chars_per_token']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    cells = [(arm, s) for arm in ARMS for s in NEW_SEEDS]
    mins = sum(41 for _ in cells)
    print(f'{len(cells)} runs, ~41 min each => about {mins/60:.1f} GPU-hours')
    for arm in ARMS:
        have = sorted(r['seed'] for r in f.results(f"{arm['prefix']}_*")
                      if r.get('corpus') == arm['corpus'])
        print(f"  {arm['name']:<16} corpus {arm['corpus']:<9} {arm['steps']:>6,} steps  "
              f"have seeds {have}, adding {NEW_SEEDS}")
    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return

    fleet_plan.announce('tokenizer penalty: three more seeds per arm',
                        [fleet_plan.cell(
                            f"{arm['prefix']}_{_train.cell_tag(arm['corpus'], arm['tokens'], arm['steps'], s, PRESET)}",
                            f"{arm['name']}  seed {s}", corpus=arm['corpus'],
                            steps=arm['steps'], eta_s=2460,
                            update_tokens=arm['steps'] * 128 * 128)
                         for arm, s in cells],
                        replace_prefix='swap', owner='Patrick')

    rows, t0 = [], time.time()
    for i, (arm, seed) in enumerate(cells, 1):
        # The existing tags were produced by cell_tag with a prefix, so match that shape exactly
        # or the new seeds will not be picked up by the same glob the report reads.
        tag = f"{arm['prefix']}_{_train.cell_tag(arm['corpus'], arm['tokens'], arm['steps'], seed, PRESET)}"
        try:
            rec = f.pretrain(arm['corpus'], tokens=arm['tokens'], steps=arm['steps'], seed=seed,
                             preset=PRESET, lr=LR, batch=BATCH, clip=CLIP, gpu=a.gpu,
                             tag=tag, reuse=True)
            rows.append({'arm': arm['name'], 'corpus': arm['corpus'], 'seed': seed, 'tag': tag,
                         'val_loss': rec['val_loss'], 'bpc': bpc(rec, arm['corpus'])})
        except Exception as e:                     # noqa: BLE001 -- one cell must not stop the rest
            print(f'  FAILED {tag}: {repr(e)[:120]}', flush=True)
            rows.append({'arm': arm['name'], 'seed': seed, 'tag': tag, 'error': repr(e)[:200]})
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(cells)}] {tag}  ({(time.time()-t0)/60:.0f} min elapsed)', flush=True)

    report()


def report():
    """Re-test the penalty over every seed now on disk, old and new together."""
    print('\n' + '=' * 74)
    print('IS THE TOKENIZER PENALTY ESTABLISHED AT SIX SEEDS?')
    print('=' * 74)
    arms = {}
    for arm in ARMS:
        v = sorted(bpc(r, arm['corpus']) for r in f.results(f"{arm['prefix']}_*")
                   if r.get('corpus') == arm['corpus'])
        arms[arm['name']] = v
        print(f"\n  {arm['name']:<16} n={len(v)}  " + ' '.join(f'{x:.3f}' for x in v))
        print(f"  {'':<16} mean {st.mean(v):.3f}  sd {st.stdev(v):.3f}")

    b, c = arms['250k vocabulary'], arms['16k vocabulary']
    d = st.mean(b) - st.mean(c)
    t, p = stats.ttest_ind(b, c, equal_var=False)
    print(f'\n  penalty = {d:.3f} bits per character')
    print(f'  Welch t = {t:.2f}, p = {p:.4f}, n = {len(b)} vs {len(c)}')
    print()
    if p < 0.05:
        print('  ESTABLISHED. Report 08 can state the penalty without hedging, and should say')
        print('  it took six seeds rather than pretending three were enough.')
    else:
        print('  STILL NOT ESTABLISHED at the sample size fixed in advance. The honest write-up')
        print('  is that the penalty is consistent in direction across every seed and does not')
        print('  clear significance -- not a quiet extension to more seeds until it does.')


if __name__ == '__main__':
    main()
