"""Luck, skill, and search budget, priced from records that already exist.

Week 8 of the bottom board asks whether a tuned setting transfers between languages. The answer
is no, and it was already written up. This script asks the question underneath it, which is the
one a student with one graphics card actually faces:

    Given a fixed budget, is it better to train one model longer, or several models and keep the
    best? And how much of a "tuned" result is knowing what you are doing rather than having run
    enough things that one of them came out well?

Nothing new is trained here. The 60-run learning-rate grid varies the seed at every rate, and the
Yoruba cell was run at two step budgets with several seeds at each, so both arms of the trade were
already paid for. That is the whole reason the question can be asked after the fact instead of
being re-run: the records kept enough to answer a question nobody had when they were written.

Five sections, matching the five claims in the build sheet:

    A  the seed lottery in catastrophic failure
    B  how much of the outcome is the rate, and how much is the seed
    C  whether the winning rate is distinguishable from the runner-up at all
    D  what it would cost to actually identify it
    E  longer, or more seeds

    bash src/a2-nlp/py.sh study_budget.py
    bash src/a2-nlp/py.sh study_budget.py --json runs/budget.json
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import statistics as st
from collections import defaultdict
from itertools import combinations

import mlm_api as factory

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(HERE, 'runs', 'lr_transfer.json')

# A run that never left its initialization sits near the uniform-prediction loss. In this grid the
# trained runs top out at 4.08 and the collapsed ones start at 5.17, so any cut inside that gap
# gives the same answer and the exact value is not load-bearing. It is stated as a constant rather
# than buried in a comparison because it decides which rows every section below is computed over,
# and an earlier draft of this analysis used 5.5, which let two barely-alive runs into the trained
# pool and inflated section B's headline from 1.2x to 3.1x.
COLLAPSE = 4.5

# The Yoruba cell that was run at two step budgets. Everything except `steps` is held.
LADDER = dict(corpus='yor', n_tokens=69096452, preset='poc', lr=0.0005, batch=128)


def grid():
    with io.open(GRID_PATH, encoding='utf-8') as f:
        return json.load(f)


def by_rate(rows, lang, trained_only=True):
    """The grid for one language, keyed by learning rate."""
    out = defaultdict(list)
    for r in rows:
        if r['lang'] != lang:
            continue
        if trained_only and r['val_loss'] >= COLLAPSE:
            continue
        out[r['lr']].append(r['val_loss'])
    return out


def pair_sd(pair):
    """The sd implied by two observations: |a - b| / sqrt(2), which is the unbiased estimate."""
    return abs(pair[0] - pair[1]) / math.sqrt(2)


def section_a(rows):
    """Collapse is partly a seed event, and the split cells are how you can tell."""
    cells = defaultdict(dict)
    for r in rows:
        cells[(r['lang'], r['lr'])][r['seed']] = r['val_loss']

    split, dead, alive = [], [], []
    for (lang, lr), seeds in sorted(cells.items()):
        n_dead = sum(v >= COLLAPSE for v in seeds.values())
        if n_dead == len(seeds):
            dead.append((lang, lr))
        elif n_dead:
            split.append({'lang': lang, 'lr': lr,
                          'losses': {s: round(v, 4) for s, v in sorted(seeds.items())}})
        else:
            alive.append((lang, lr))

    print('A. the seed lottery in catastrophic failure')
    print(f'   {len(cells)} cells, two seeds each')
    print(f'   both seeds collapsed : {len(dead)}')
    print(f'   both seeds trained   : {len(alive)}')
    print(f'   one of each          : {len(split)}')
    for s in split:
        losses = '  '.join(f's{k}={v}' for k, v in s['losses'].items())
        print(f'      {s["lang"]} at {s["lr"]}:  {losses}')
    at_risk = len(dead) + len(split)
    print(f'   of the {at_risk} cells where collapse happened at all, {len(split)} collapsed for '
          f'one seed and not the other')
    return {'cells': len(cells), 'both_dead': len(dead), 'both_alive': len(alive),
            'split': split, 'cells_with_collapse': at_risk}


def section_b(rows):
    """Inside the usable band, is the rate worth more than the seed?"""
    print('\nB. how much of the outcome is the rate, and how much is the seed')
    print(f'   {"lang":6} {"between-rate sd":>16} {"within-rate sd":>15} {"ratio":>7}')
    out, ratios = {}, []
    for lang in sorted({r['lang'] for r in rows}):
        live = {lr: v for lr, v in by_rate(rows, lang).items() if len(v) == 2}
        if len(live) < 2:
            continue
        between = st.stdev([st.mean(v) for v in live.values()])
        within = st.mean([pair_sd(v) for v in live.values()])
        ratios.append(between / within)
        out[lang] = {'between': between, 'within': within, 'ratio': between / within,
                     'rates_alive': len(live)}
        print(f'   {lang:6} {between:16.3f} {within:15.3f} {between / within:6.1f}x')
    med = st.median(ratios)
    print(f'   median {med:.1f}x — below about 2x, choosing the rate inside the band is worth')
    print('   roughly what choosing the seed is worth. The rate\'s effect is the cliff, not the slope.')
    return {'per_language': out, 'median_ratio': med}


def section_c(rows):
    """Is the argmin distinguishable from the runner-up?"""
    print('\nC. is the winning rate distinguishable from the runner-up')
    out = {}
    for lang in sorted({r['lang'] for r in rows}):
        live = {lr: v for lr, v in by_rate(rows, lang).items() if len(v) == 2}
        ranked = sorted(live.items(), key=lambda kv: st.mean(kv[1]))
        if len(ranked) < 2:
            continue
        (lr1, v1), (lr2, v2) = ranked[0], ranked[1]
        gap = st.mean(v2) - st.mean(v1)
        sd = st.mean([pair_sd(v) for v in live.values()])
        out[lang] = {'best_lr': lr1, 'best': st.mean(v1), 'runner_up_lr': lr2,
                     'runner_up': st.mean(v2), 'gap': gap, 'seed_sd': sd,
                     'identified': gap > sd}
        print(f'   {lang:6} best {lr1:<8}{st.mean(v1):.3f}   next {lr2:<8}{st.mean(v2):.3f}   '
              f'gap {gap:.3f}   seed sd {sd:.3f}   '
              f'{"identified" if gap > sd else "NOT identified"}')
    return out


def section_d(rows, ranked):
    """What identifying it would cost, in runs and in hours."""
    print('\nD. what identifying the winner would cost')
    seconds = st.mean([r['seconds'] for r in rows])
    n_rates = len({r['lr'] for r in rows if r['lang'] == rows[0]['lang']})
    total = 0
    out = {}
    for lang, d in ranked.items():
        effect = d['gap'] / d['seed_sd']
        # Seeds a side to clear a two-sided 0.05 test for an effect of this many sd, in the
        # large-n limit where t* -> 1.96. Generous: at any finite n the true t* is larger, so
        # this understates the cost.
        n = math.ceil(2 * (1.96 / effect) ** 2)
        runs = n * n_rates
        total += runs
        out[lang] = {'effect_sd': effect, 'seeds_per_rate': n, 'runs': runs}
        print(f'   {lang:6} {effect:5.2f} sd  ->  {n:>5} seeds/rate  =  {runs:>6} runs')
    hours = total * seconds / 3600
    print(f'   all five: {total:,} runs at {seconds / 60:.1f} min each = {hours:,.0f} GPU-hours')
    return {'per_language': out, 'total_runs': total, 'gpu_hours': hours,
            'seconds_per_run': seconds}


def exp_min(vals, k):
    """E[min of k draws without replacement] from the observed sample, computed exactly.

    Enumerating the combinations rather than assuming a distribution: with four observations the
    exact answer is cheap, and a normal-theory expected-minimum would be asserting the shape of a
    tail measured from four points.
    """
    return st.mean([min(c) for c in combinations(vals, k)])


def section_e():
    """Longer, or more seeds?"""
    print('\nE. longer, or more seeds')
    cell = [r for r in factory.results('*')
            if all(r.get(k) == v for k, v in LADDER.items()) and r.get('val_loss') is not None]
    budgets = sorted({r['steps'] for r in cell})
    if len(budgets) < 2:
        print('   the two-budget Yoruba cell is not on disk; nothing to compare')
        return None
    lo, hi = budgets[0], budgets[-1]
    short = sorted(r['val_loss'] for r in cell if r['steps'] == lo)
    long_ = sorted(r['val_loss'] for r in cell if r['steps'] == hi)
    sec_lo = st.mean([r['seconds'] for r in cell if r['steps'] == lo])
    sec_hi = st.mean([r['seconds'] for r in cell if r['steps'] == hi])
    cost = sec_hi / sec_lo

    print(f'   {lo:,} steps, n={len(short)}: mean {st.mean(short):.4f}  best {min(short):.4f}')
    print(f'   {hi:,} steps, n={len(long_)}: mean {st.mean(long_):.4f}  worst {max(long_):.4f}')
    print(f'   one long run costs {cost:.1f}x one short run')
    rows = {}
    for k in range(1, len(short) + 1):
        rows[k] = exp_min(short, k)
        print(f'      best of {k} short: {rows[k]:.4f}')
    print(f'      one long run   : {st.mean(long_):.4f}')

    gain_len = st.mean(short) - st.mean(long_)
    gain_seed = st.mean(short) - rows[len(short)]
    print(f'   length buys {gain_len:.4f} nats; best-of-{len(short)} buys {gain_seed:.4f} '
          f'-> {gain_len / gain_seed:.0f}x')
    print(f'   the worst long run ({max(long_):.4f}) beats the best short run ({min(short):.4f}) '
          f'by {min(short) - max(long_):.4f}')
    return {'short_steps': lo, 'long_steps': hi, 'short': short, 'long': long_,
            'cost_ratio': cost, 'best_of_k': rows,
            'gain_length': gain_len, 'gain_seeds': gain_seed,
            'ratio': gain_len / gain_seed}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', help='write the numbers to this path as well as printing them')
    args = ap.parse_args()

    rows = grid()
    out = {'collapse_threshold': COLLAPSE, 'grid_runs': len(rows)}
    out['a_seed_lottery'] = section_a(rows)
    out['b_rate_vs_seed'] = section_b(rows)
    out['c_identified'] = section_c(rows)
    out['d_cost_to_identify'] = section_d(rows, out['c_identified'])
    out['e_longer_or_more'] = section_e()

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(HERE, args.json)
        with io.open(path, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=1, default=float)
        print(f'\nwrote {path}')


if __name__ == '__main__':
    main()
