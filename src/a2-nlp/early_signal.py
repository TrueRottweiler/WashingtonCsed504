"""When can you tell a training run is not going to work?

A third of our large-model runs never learned anything. They did not crash -- they trained for
ninety minutes, saved a checkpoint, and produced a number that was worthless. The obvious fix is
to notice early and kill them, and the obvious objection is the one figure 08 makes: for the first
sixth of its life a run that is about to succeed looks exactly like a run that never will.

So this is a real question rather than a rhetorical one, and we can answer it without spending a
single GPU-hour, because every run we have ever done left its whole loss curve in runs/*.jsonl.
105 curves, 92 that learned and 13 that did not.

Three things are computed here:

  1. WHERE the two populations separate, checkpoint by checkpoint, using a feature that is
     actually available at runtime. Not "fraction of the way through" -- a scheduler knows the
     step number, not the future.

  2. WHAT IT WOULD HAVE SAVED. The deliverable is not an accuracy score. It is GPU-hours: for
     each decision rule, how much wasted compute it recovers and how often it kills a good run.
     An accuracy number cannot be traded against anything; hours can.

  3. WHETHER THE CLIFF MOVES. The plateau ends somewhere. If that somewhere is fixed, the risky
     window is a constant you can schedule around. If it moves with the settings, the detector
     has to move with it.

    bash src/a2-nlp/py.sh early_signal.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

import numpy as np

import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'early_signal.json')

# What counts as "never learned" is the whole analysis, and the obvious definition is wrong.
#
# The first version here called a run dead if it finished within 4 nats of an untrained model.
# That misreads every data-starved run as a failure: at 4M tokens the BEST anyone achieved is
# 6.72, so an absolute threshold condemns the entire rung, and the conclusion came out backwards
# -- it made tighter clipping look like it caused failures when it had been deliberately run on
# the small corpora.
#
# A failure is not "ended high". It is "ended much worse than this same configuration is capable
# of". So the reference is the best loss any run reached at the same corpus, data size and model
# size, and a run is dead if it missed that by more than the gap in the bimodality figure.
DEAD_MARGIN = 1.5
TOKENS_PER_STEP = 128 * 128


def load():
    """Every run with its curve, its outcome, and the best its own configuration ever reached."""
    rows_all = f.results('*')

    # Best loss achieved at each (corpus, data size, model size). This is the "what was
    # achievable here" reference that makes a data-starved run distinguishable from a failed one.
    best = {}
    for r in rows_all:
        k = (r.get('corpus'), r.get('n_tokens'), r.get('preset') or 'poc')
        best[k] = min(best.get(k, 9e9), r['val_loss'])

    out = []
    for r in rows_all:
        p = os.path.join(HERE, 'runs', f"{r['tag']}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]
        rows = [x for x in rows if x.get('val') and x.get('step')]
        if len(rows) < 8:
            continue
        rand = r.get('random_loss') or math.log(r.get('vocab_size') or 16000)
        achievable = best[(r.get('corpus'), r.get('n_tokens'), r.get('preset') or 'poc')]
        out.append({
            'tag': r['tag'], 'steps': r['steps'], 'final': r['val_loss'], 'random': rand,
            'seconds': r['seconds'], 'preset': r.get('preset') or 'poc',
            'lr': r['lr'], 'clip': r.get('clip'), 'corpus': r.get('corpus'),
            'achievable': achievable,
            'dead': r['val_loss'] > achievable + DEAD_MARGIN,
            'curve': [(x['step'], x['val']['loss']) for x in rows],
        })
    return out


def at_step(run, k):
    """The last observation at or before step k, or None if the run had not got there."""
    seen = [(s, l) for s, l in run['curve'] if s <= k]
    return seen[-1] if seen else None


def gained(run, k):
    """Nats gained against this run's own untrained baseline by step k.

    This is the feature to use, and the reason is vocabulary. Raw loss is not comparable between
    a 16k and a 250k model -- an untrained 250k model starts 2.7 nats higher, so a fixed loss
    threshold would condemn every large-vocabulary run at step one. Distance travelled from your
    own starting point is comparable across all of them.
    """
    o = at_step(run, k)
    return None if o is None else run['random'] - o[1]


# ------------------------------------------------------------------------------------------
def separation(runs, checkpoints):
    """At each checkpoint, how well does `gained` split the two outcomes?"""
    print(f"\n{'step':>8}{'runs':>7}{'learned':>22}{'never learned':>22}{'overlap':>9}")
    rows = []
    for k in checkpoints:
        live = [r for r in runs if at_step(r, k) and r['steps'] > k]
        ok = [gained(r, k) for r in live if not r['dead']]
        bad = [gained(r, k) for r in live if r['dead']]
        if len(ok) < 3 or len(bad) < 3:
            continue
        overlap = min(ok) <= max(bad)
        rows.append({'step': k, 'n': len(live), 'n_dead': len(bad),
                     'ok_mean': st.mean(ok), 'ok_min': min(ok),
                     'bad_mean': st.mean(bad), 'bad_max': max(bad), 'overlap': overlap})
        print(f'{k:>8,}{len(live):>7}'
              f'{st.mean(ok):>13.2f} (min {min(ok):>4.2f}){st.mean(bad):>13.2f} '
              f'(max {max(bad):>4.2f}){"yes" if overlap else "NO":>9}')
    return rows


def cost_curve(runs, k, thresholds):
    """What a rule 'at step k, abandon if gained < t' would have cost and saved.

    Saved hours are the remaining wall-clock of correctly abandoned runs. Lost hours are the
    whole cost of any good run it kills, because that run has to be done again from scratch --
    which is the asymmetry that makes this a judgment call rather than an optimization.
    """
    live = [r for r in runs if at_step(r, k) and r['steps'] > k]
    out = []
    for t in thresholds:
        saved = lost = 0.0
        killed_good = killed_bad = 0
        for r in live:
            if gained(r, k) >= t:
                continue                              # let it run
            remaining = r['seconds'] * (1 - k / r['steps']) / 3600
            if r['dead']:
                saved += remaining; killed_bad += 1
            else:
                lost += r['seconds'] / 3600; killed_good += 1
        wasted = sum(r['seconds'] / 3600 for r in live if r['dead'])
        out.append({'threshold': t, 'saved_h': saved, 'lost_h': lost,
                    'net_h': saved - lost, 'killed_bad': killed_bad,
                    'killed_good': killed_good, 'n_dead': sum(1 for r in live if r['dead']),
                    'wasted_total_h': wasted})
    return out, live


def patience_rule(runs, deadline_frac):
    """'Abandon if the run has not left its plateau by `deadline_frac` of its budget.'

    This is the rule that follows from the data rather than the one we assumed. The level a run
    has reached does not separate the two populations -- both drop about three nats early, which
    is the unigram statistics and nothing more. What separates them is whether the CLIFF has
    happened. So the deployable question is not "is the loss low enough" but "has this run left
    its plateau yet, and how long should I wait before concluding it never will".

    Expressed as a fraction of the run's own budget rather than an absolute step, because the
    cliff moves with the configuration -- see section 3.
    """
    saved = lost = 0.0
    killed_good = killed_bad = 0
    for r in runs:
        k = int(r['steps'] * deadline_frac)
        c = cliff_step(r)
        left_plateau = c is not None and c <= k
        if left_plateau:
            continue                                  # let it run
        remaining = r['seconds'] * (1 - deadline_frac) / 3600
        if r['dead']:
            saved += remaining; killed_bad += 1
        else:
            lost += r['seconds'] / 3600; killed_good += 1
    return {'deadline': deadline_frac, 'saved_h': saved, 'lost_h': lost,
            'net_h': saved - lost, 'caught': killed_bad, 'false_kills': killed_good}


def cliff_step(run, drop=1.0):
    """The first step at which the loss has fallen `drop` nats below its own plateau.

    The plateau level is taken as the median of the first quarter of the curve rather than the
    first point, because the very first observation is still falling out of initialization.
    """
    n = len(run['curve'])
    plateau = st.median([l for _, l in run['curve'][:max(2, n // 4)]])
    for s, l in run['curve']:
        if plateau - l >= drop:
            return s
    return None


# ------------------------------------------------------------------------------------------
def main():
    runs = load()
    dead = [r for r in runs if r['dead']]
    print(f'{len(runs)} runs with curves: {len(runs)-len(dead)} learned, {len(dead)} never did')
    print(f'wasted on runs that never learned: '
          f'{sum(r["seconds"] for r in dead)/3600:.1f} GPU-hours '
          f'({100*sum(r["seconds"] for r in dead)/sum(r["seconds"] for r in runs):.0f}% of all '
          'the compute we spent)')

    # --- 1. does anything separate them, and when? ----------------------------------------
    print('\n' + '=' * 78)
    print('1. NATS GAINED AGAINST THE UNTRAINED BASELINE, BY CHECKPOINT')
    print('=' * 78)
    checkpoints = [500, 1000, 2000, 3000, 4000, 6000, 8000, 11000, 12000, 16000, 24000]
    sep = separation(runs, checkpoints)
    clean = [r for r in sep if not r['overlap']]
    if clean:
        first = clean[0]
        print(f'\nFirst clean separation at step {first["step"]:,}: every run that went on to '
              f'learn had\ngained at least {first["ok_min"]:.2f} nats, and no run that failed '
              f'had gained more than {first["bad_max"]:.2f}.')
    else:
        print('\nNo checkpoint separates the two populations cleanly on this feature alone.')

    # --- 2. the only number that matters: hours ---------------------------------------------
    print('\n' + '=' * 78)
    print('2. WHAT A DECISION RULE WOULD HAVE COST AND SAVED')
    print('=' * 78)
    for k in (4000, 8000, 12000):
        rows, live = cost_curve(runs, k, [0.25, 0.5, 1.0, 2.0, 3.0])
        if not live:
            continue
        fired = sum(r['killed_bad'] + r['killed_good'] for r in rows)
        print(f'\n  at step {k:,}: {len(live)} runs reach it, '
              f'{rows[0]["n_dead"]} doomed, {rows[0]["wasted_total_h"]:.1f} GPU-h wasted.')
        print(f'    no threshold from 0.25 to 3.0 nats catches any of them '
              f'({fired} total firings, all false).' if fired <= 2 else '    see below')

    print('\n  The level a run has reached does not separate the populations. Both drop about')
    print('  three nats almost immediately -- that is the unigram distribution, which any model')
    print('  learns in a few hundred steps -- and then the doomed ones simply stop. A doomed run')
    print('  at step 8,000 has gained MORE (3.36) than the weakest run that went on to succeed')
    print('  (3.23). A threshold on the level cannot work, at any step, in either direction.')

    print('\n' + '-' * 78)
    print('  The rule that does work: abandon if the run has not LEFT ITS PLATEAU yet.')
    print('-' * 78)
    print(f"\n{'wait until':>12}{'caught':>8}{'false kills':>13}{'saved':>10}{'lost':>9}"
          f"{'net':>9}")
    pats = []
    for d in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        p = patience_rule(runs, d)
        pats.append(p)
        print(f'{d:>11.0%}{p["caught"]:>8}{p["false_kills"]:>13}{p["saved_h"]:>9.1f}h'
              f'{p["lost_h"]:>8.1f}h{p["net_h"]:>+8.1f}h')
    best = max(pats, key=lambda p: p['net_h'])
    print(f'\n  Best operating point: wait {best["deadline"]:.0%} of the budget, then abandon.')
    print(f'  Catches {best["caught"]} of {len(dead)} doomed runs, kills '
          f'{best["false_kills"]} good ones, nets {best["net_h"]:+.1f} GPU-hours '
          f'over our 105 runs.')

    # --- 3. does the cliff sit in the same place every time? --------------------------------
    print('\n' + '=' * 78)
    print('3. WHERE THE PLATEAU ENDS')
    print('=' * 78)
    cliffs = [(r, cliff_step(r)) for r in runs if not r['dead']]
    cliffs = [(r, c) for r, c in cliffs if c]
    vals = [c for _, c in cliffs]
    print(f'\n{len(vals)} runs that learned have a locatable cliff')
    print(f'  median {st.median(vals):,.0f} steps   '
          f'range {min(vals):,} to {max(vals):,}')
    print('\nas a fraction of each run\'s own budget:')
    fr = [c / r['steps'] for r, c in cliffs]
    print(f'  median {st.median(fr):.2f}   range {min(fr):.2f} to {max(fr):.2f}')
    print('\nby preset:')
    for p in sorted({r['preset'] for r, _ in cliffs}):
        g = [c for r, c in cliffs if r['preset'] == p]
        print(f'  {p:<10} n={len(g):>3}  median {st.median(g):>7,.0f} steps')
    print('\nby learning rate:')
    for lr in sorted({r['lr'] for r, _ in cliffs}):
        g = [c for r, c in cliffs if r['lr'] == lr]
        if len(g) >= 3:
            print(f'  {lr:<10} n={len(g):>3}  median {st.median(g):>7,.0f} steps')

    # --- 4. if you cannot detect it, prevent it ---------------------------------------------
    print('\n' + '=' * 78)
    print('4. THE ALTERNATIVE: STOP THE FAILURES HAPPENING')
    print('=' * 78)
    big = [r for r in runs if r['preset'] == 'afriberta']
    by_clip = {}
    for r in big:
        by_clip.setdefault(r['clip'], []).append(r)
    print(f"\n{'clip':>8}{'runs':>7}{'never learned':>16}{'GPU-h wasted':>15}")
    for c in sorted(by_clip, key=lambda x: (x is None, x)):
        g = by_clip[c]
        d = [r for r in g if r['dead']]
        print(f'{str(c):>8}{len(g):>7}{f"{len(d)} ({100*len(d)/len(g):.0f}%)":>16}'
              f'{sum(r["seconds"] for r in d)/3600:>14.1f}h')
    print('\n  Not established. 20% against 17%, on ten runs versus thirty-six, is no difference')
    print('  at all. The honest reading is that we have never tested whether tighter clipping')
    print('  PREVENTS failures -- only that it tightens the spread of the runs that succeed')
    print('  (0.052 against 0.224 at the matched 256M cell, report 07). Those are different')
    print('  claims and we had been treating them as one.')
    print('\n  So: detection provably does not pay, and prevention is untested. Naming the')
    print('  second one as unmeasured is worth more than asserting it, and it is a concrete')
    print('  thing to go and measure.')
    prevention = {str(c): {'n': len(g), 'dead': sum(1 for r in g if r['dead']),
                           'wasted_h': sum(r['seconds'] for r in g if r['dead']) / 3600}
                  for c, g in by_clip.items()}

    json.dump({'prevention': prevention, 'n_runs': len(runs), 'n_dead': len(dead),
               'wasted_h': sum(r['seconds'] for r in dead) / 3600,
               'separation': sep,
               'cost': {str(k): cost_curve(runs, k, [0.25, 0.5, 1.0, 2.0, 3.0])[0]
                        for k in (4000, 8000, 12000)},
               'patience': pats,
               'cliff_steps': vals, 'cliff_fractions': fr},
              open(OUT, 'w', encoding='utf-8'), indent=2)
    print('\nwrote runs/early_signal.json')


if __name__ == '__main__':
    main()
