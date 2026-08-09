"""Does tighter gradient clipping PREVENT failures, or only tidy up the runs that succeed?

Panel 13 pulled apart a claim this project had been making as one sentence and found it was two:

  * ESTABLISHED -- clipping at 0.5 tightens the spread of the runs that work. At the matched
    256M cell the spread goes 0.224 -> 0.052, and report 07 rests on that.
  * UNTESTED -- whether it stops runs failing at all. On the records we have, the failure rate
    is 20% at clip 0.5 against 17% at the default, on ten runs versus thirty-six. That is no
    difference, measured at a sample size that could not have found one.

Panel 13's recommendation was to stop trying to detect failures -- the arithmetic there cannot
close -- and go and find out whether they can be prevented instead. This is that experiment.

Two outcomes come out of the same runs and they need very different sample sizes. A FAILURE RATE
is a proportion and proportions are expensive; a SPREAD is a continuous quantity and cheap. The
design below is sized for the first and reports the second as a bonus, because sizing an
experiment for the outcome it can actually resolve is the whole lesson of section 9.

The design is shaped by what it is allowed to cost, and at a 48-hour budget it can be honest.
An earlier version truncated every run to 35,000 steps to fit a night, which measured only
whether a run leaves its plateau and could say nothing about final quality. That caveat is gone:
these are full 62,500-step runs, so the same experiment answers all three questions -- does
clipping prevent failures, does it tighten the spread, and does it cost anything at the end.

How many seeds, and the honest answer about what they buy. Computed rather than guessed:

    if the true failure rate is 20%, P(seeing zero failures)
        n=3    51.2%      n=8    16.8%      n=16    2.8%
        n=5    32.8%      n=12    6.9%      n=20    1.2%

    comparing the two arms, Fisher exact
        n=5:  1/5 vs 0/5   p=1.000        n=12: 2/12 vs 0/12  p=0.478
        n=8:  2/8 vs 0/8   p=0.467        n=20: 4/20 vs 0/20  p=0.106

    detecting the spread difference already observed (SD ratio 4.3x), F-test
        n=3  needs 6.24x -- missed        n=8  needs 2.23x -- detected
        n=5  needs 3.10x -- detected      n=12 needs 1.86x -- detected

Read those three blocks together and the design writes itself, but not in the direction you would
guess. The SPREAD question is settled at five seeds; twelve buys almost nothing there. The RATE
question is not settled at twelve, or at twenty, or at any n this hardware can afford -- 85
minutes a run means a usable rate bound would cost something like 113 GPU-hours for one cell.

So twelve is not chosen to make the rate significant. It cannot be. It is chosen because at n=12
a clean sweep means something on its own terms: seeing 0/12 when the rate was 20% happens 6.9% of
the time, which is a bound worth having even though the two-arm comparison stays p=0.478. The
write-up has to say that, and the report() function below prints it so it cannot be forgotten.

Split across the two cards BY SEED, never by clip value. Splitting by treatment would confound
the thing being tested with whichever card ran it -- different thermals, different neighbours on
the box -- and that is exactly the class of mistake section 9 is a list of.

    bash src/a2-nlp/py.sh study_clip_prevention.py --dry-run
    bash src/a2-nlp/py.sh study_clip_prevention.py --gpu 0 --seeds 0,1,2,3,4,5
    bash src/a2-nlp/py.sh study_clip_prevention.py --gpu 1 --seeds 6,7,8,9,10,11
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st
import time

import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'clip_prevention.json')

CORPUS = 'eng_1b'
PRESET = 'afriberta'          # the 98M model -- the only one that fails
# One rung, not two, and the arithmetic below is why. The scarce thing here is SEEDS, not data
# sizes: a rare-event rate needs replication, and spending half the budget on a second rung would
# halve the seeds on both. 1024M is the worst cell we have -- spread 5.003 across five seeds --
# so it is where failures are most likely to be observable at all.
TOKENS = [1_024_000_000]
CLIPS = [1.0, 0.5]
SEEDS = list(range(12))
STEPS = 62_500                # full length: no truncation, so final quality is measurable too
LR = 3e-4                     # what every afriberta run in the study used


def cells():
    return [(n, c, s) for n in TOKENS for c in CLIPS for s in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--seeds', default=None,
                    help='comma-separated seeds for this card. Split the study by SEED, not by '
                         'clip -- splitting by treatment confounds it with the card')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    todo = cells()
    if a.seeds:
        want = {int(s) for s in a.seeds.split(',')}
        todo = [c for c in todo if c[2] in want]
        print(f'this card runs seeds {sorted(want)}: {len(todo)} of {len(cells())} cells\n')
    per_min = STEPS * 128 * 128 / 201_000 / 60          # measured 98M throughput
    print(f'{len(TOKENS)} data rungs x {len(CLIPS)} clip values x {len(SEEDS)} seeds '
          f'= {len(todo)} runs')
    print(f'{STEPS:,} steps each (~{per_min:.0f} min) => about '
          f'{len(todo)*per_min/60:.1f} GPU-hours\n')
    print('Full-length runs, so failure rate, spread and final quality all come out of the')
    print('same cells. Twelve seeds settles the spread; it only gestures at the rate.')
    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return

    rows, t0 = [], time.time()
    for i, (n, clip, seed) in enumerate(todo, 1):
        tag = f'clipprev_{f.compact(n)}_c{clip:g}_s{seed}'
        try:
            rec = f.pretrain(CORPUS, tokens=n, steps=STEPS, seed=seed, preset=PRESET,
                             lr=LR, clip=clip, gpu=a.gpu, tag=tag, reuse=True)
            rows.append({'tokens': n, 'clip': clip, 'seed': seed, 'tag': tag,
                         'val_loss': rec['val_loss'], 'random_loss': rec.get('random_loss'),
                         'seconds': rec['seconds']})
        except Exception as e:                     # noqa: BLE001 -- one cell must not stop the night
            print(f'  FAILED {tag}: {repr(e)[:120]}', flush=True)
            rows.append({'tokens': n, 'clip': clip, 'seed': seed, 'tag': tag,
                         'error': repr(e)[:200]})
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(todo)}] {tag}  ({(time.time()-t0)/60:.0f} min elapsed)', flush=True)

    report(rows)


def report(rows=None):
    rows = [r for r in (rows or json.load(open(OUT, encoding='utf-8'))) if 'val_loss' in r]
    print('\n' + '=' * 74)
    print('DOES TIGHTER CLIPPING PREVENT FAILURES?')
    print('=' * 74)

    # Same rule as panel 13: a run failed if it ended far worse than the best any run of this
    # exact cell reached. Not an absolute threshold -- that is what made the first version of
    # panel 13 come out backwards on data-starved runs.
    best = {}
    for r in rows:
        best[r['tokens']] = min(best.get(r['tokens'], 9e9), r['val_loss'])

    print(f"\n{'tokens':>13}{'clip':>7}{'n':>4}{'failed':>9}{'spread':>9}   values")
    summary = collections.defaultdict(dict)
    for n in TOKENS:
        for c in CLIPS:
            g = [r for r in rows if r['tokens'] == n and r['clip'] == c]
            if not g:
                continue
            v = sorted(r['val_loss'] for r in g)
            fails = [x for x in v if x > best[n] + 1.5]
            spread = max(v) - min(v)
            summary[n][c] = {'n': len(v), 'failed': len(fails), 'spread': spread}
            print(f'{n:>13,}{c:>7.1f}{len(v):>4}{len(fails):>9}{spread:>9.3f}   '
                  + ' '.join(f'{x:.2f}' for x in v))

    print('\n  Read the two columns against each other, per rung:')
    for n in TOKENS:
        if len(summary[n]) == 2:
            a_, b_ = summary[n][1.0], summary[n][0.5]
            print(f'   {n:>13,}  failures {a_["failed"]}/{a_["n"]} at clip 1.0 '
                  f'vs {b_["failed"]}/{b_["n"]} at 0.5; '
                  f'spread {a_["spread"]:.3f} vs {b_["spread"]:.3f}')
    print('\n  The SPREAD comparison is the one twelve seeds can carry. The failure counts are')
    print('  a secondary observation: 0/12 against 3/12 is p~0.22, which is suggestive and not')
    print('  a result, and should be written up as suggestive.')


if __name__ == '__main__':
    main()
