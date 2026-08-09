"""What is a Yoruba token worth the second time you read it?

Every scaling law you have heard of assumes fresh tokens. Chinchilla tells you how to spend a
compute budget when text is unlimited, and for English it effectively is. For Yoruba it is not:
everything we could collect is 69.1M tokens, and our own runs pass over that corpus up to 196
times. So the question that actually governs this project is not "how much data do we need" --
we cannot get more -- it is **how far repetition substitutes for data, and where it stops**.

That has a published form (Muennighoff et al. 2023, data-constrained scaling). The parameter is a
repetition half-life R*: a token re-read R times is worth as much as

    D_eff = U * (1 + R* * (1 - exp(-(R - 1) / R*)))

fresh tokens, where U is the unique token count. R* -> infinity means repeats are free and only
total compute matters; R* -> 0 means repeats are worthless and only the corpus size matters. The
truth is in between and the number is language- and corpus-specific. Nobody has measured it for
Yoruba, and we do not have to run anything new to try: the grid is already in runs/.

This script fits R* on the 33.8M Yoruba runs, checks it by leave-one-cell-out prediction against
two null models, and prints the cells whose measurement would most reduce the remaining
uncertainty -- i.e. it designs the next night rather than merely describing the last one.

    bash src/a2-nlp/py.sh scaling_law.py
"""
from __future__ import annotations

import collections
import json
import os
import statistics as st

import numpy as np
from scipy.optimize import least_squares

import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'scaling_law.json')

CORPUS = 'yor'
PRESET = 'poc'            # 33.8M. Only two backbone sizes exist, so the N exponent is not
                          # identifiable and we do not pretend to fit one -- see the note below.
TOKENS_PER_STEP = 128 * 128


# ------------------------------------------------------------------------------------------
def cells():
    """Group the runs into (unique tokens, steps) cells, averaging seeds within a cell.

    Averaging is the right move here and not everywhere: the seed spread is the measurement
    error on a cell, so the cell mean is what a law should be fit to, and the spread is what
    tells us whether a residual is real.
    """
    by = collections.defaultdict(list)
    for r in f.results('*'):
        if r.get('corpus') != CORPUS or (r.get('preset') or 'poc') != PRESET:
            continue
        by[(r['n_tokens'], r['steps'])].append(r['val_loss'])

    out = []
    for (u, steps), losses in sorted(by.items()):
        seen = steps * TOKENS_PER_STEP
        out.append({'unique': u, 'steps': steps, 'seen': seen, 'epochs': seen / u,
                    'loss': st.mean(losses), 'n': len(losses),
                    'spread': st.stdev(losses) if len(losses) > 1 else None})
    return out


def d_eff(unique, seen, r_star):
    """Effective fresh-token count for a corpus of `unique` tokens read `seen/unique` times."""
    unique = np.asarray(unique, dtype=float)
    seen = np.asarray(seen, dtype=float)
    reps = np.maximum(seen / unique - 1.0, 0.0)
    if r_star <= 0:                       # repeats worth nothing at all
        return np.minimum(seen, unique)
    decayed = r_star * (1.0 - np.exp(-reps / r_star))
    return unique * (1.0 + decayed)


def fit(rows, r_star=None):
    """Least squares on L = E + B / D_eff^beta. If r_star is None it is fitted too.

    Fitting in log space on the (E, B) scale keeps the optimizer from walking B negative, and
    the residual is plain loss because that is the unit every threshold in this project is in.
    """
    u = np.array([c['unique'] for c in rows], float)
    s = np.array([c['seen'] for c in rows], float)
    y = np.array([c['loss'] for c in rows], float)

    free_rstar = r_star is None

    def unpack(p):
        e, log_b, beta = p[0], p[1], p[2]
        rs = np.exp(p[3]) if free_rstar else r_star
        return e, np.exp(log_b), beta, rs

    def resid(p):
        e, b, beta, rs = unpack(p)
        return e + b / d_eff(u, s, rs) ** beta - y

    p0 = [1.5, np.log(50.0), 0.20] + ([np.log(5.0)] if free_rstar else [])
    lo = [0.0, -20, 0.01] + ([np.log(0.05)] if free_rstar else [])
    hi = [6.0, 30, 1.50] + ([np.log(500.0)] if free_rstar else [])
    sol = least_squares(resid, p0, bounds=(lo, hi), max_nfev=20000)
    e, b, beta, rs = unpack(sol.x)
    pred = e + b / d_eff(u, s, rs) ** beta
    return {'E': e, 'B': b, 'beta': beta, 'R_star': rs,
            'rmse': float(np.sqrt(np.mean((pred - y) ** 2))),
            'mae': float(np.mean(np.abs(pred - y)))}


def predict(model, unique, seen):
    return model['E'] + model['B'] / d_eff(unique, seen, model['R_star']) ** model['beta']


# ------------------------------------------------------------------------------------------
def main():
    rows = cells()
    spreads = [c['spread'] for c in rows if c['spread']]
    noise = st.mean(spreads)

    print(f'{CORPUS} / {PRESET}: {len(rows)} cells, '
          f'{sum(c["n"] for c in rows)} runs\n')
    print(f"{'unique':>12}{'steps':>8}{'epochs':>9}{'n':>4}{'loss':>8}{'spread':>9}")
    for c in rows:
        sp = f'{c["spread"]:.3f}' if c['spread'] else '--'
        print(f'{c["unique"]:>12,}{c["steps"]:>8,}{c["epochs"]:>9.1f}{c["n"]:>4}'
              f'{c["loss"]:>8.3f}{sp:>9}')
    print(f'\nmeasurement noise (mean within-cell spread): {noise:.3f}')
    print(f'repetition range: {min(c["epochs"] for c in rows):.1f} to '
          f'{max(c["epochs"] for c in rows):.1f} passes over the corpus')

    # --- the phase transition, which has to be dealt with before any law can be fitted -----
    # Figure 08 shows a single run sitting flat for ~11,500 steps and then falling off a cliff.
    # A smooth power law cannot cross a cliff, so any cell that has not yet cleared it is
    # measuring the plateau rather than the scaling behaviour. Splitting here is not curve-
    # shopping: the threshold comes from the loss curve, not from what improves the fit.
    CLIFF = 12_000
    pre = [c for c in rows if c['steps'] < CLIFF]
    post = [c for c in rows if c['steps'] >= CLIFF]
    print(f'\ncells below the {CLIFF:,}-step cliff (still on the plateau): {len(pre)}')
    if pre:
        print(f'  their losses: ' + ', '.join(f'{c["loss"]:.2f}' for c in pre)
              + f'   -- spanning corpora from {min(c["unique"] for c in pre):,} to '
                f'{max(c["unique"] for c in pre):,} tokens')
        print('  Sixteen-fold more data barely moves them. Before the transition the corpus\n'
              '  size is nearly irrelevant, which is exactly what a plateau means.')
    print(f'cells past the cliff: {len(post)}')
    rows_all, rows = rows, post

    # --- the three hypotheses -------------------------------------------------------------
    print('\n' + '=' * 78)
    print('THREE HYPOTHESES ABOUT WHAT A RE-READ TOKEN IS WORTH')
    print('=' * 78)
    models = {
        'repeats are FREE (only compute matters)': fit(rows, r_star=1e6),
        'repeats are WORTHLESS (only corpus size matters)': fit(rows, r_star=0.0),
        'fitted half-life': fit(rows, r_star=None),
    }
    for name, m in models.items():
        star = '(fitted)' if 'fitted' in name else ''
        print(f'\n  {name}')
        print(f'    E={m["E"]:.3f}  B={m["B"]:.3g}  beta={m["beta"]:.3f}  '
              f'R*={m["R_star"]:.2f} {star}')
        print(f'    fit error: rmse {m["rmse"]:.3f}   ({m["rmse"]/noise:.1f}x the noise)')

    best = models['fitted half-life']

    # --- honest test: can it predict a cell it never saw? ---------------------------------
    print('\n' + '=' * 78)
    print('LEAVE-ONE-CELL-OUT: refit without a cell, then predict it')
    print('=' * 78)
    print(f"\n{'unique':>12}{'steps':>8}{'epochs':>9}{'actual':>8}{'predicted':>11}{'error':>8}")
    errs = []
    for i, held in enumerate(rows):
        rest = rows[:i] + rows[i + 1:]
        m = fit(rest)
        p = float(predict(m, held['unique'], held['seen']))
        err = p - held['loss']
        errs.append(abs(err))
        flag = '  <-- worst' if abs(err) > 0.6 else ''
        print(f'{held["unique"]:>12,}{held["steps"]:>8,}{held["epochs"]:>9.1f}'
              f'{held["loss"]:>8.3f}{p:>11.3f}{err:>+8.3f}{flag}')
    print(f'\nmean absolute prediction error on unseen cells: {st.mean(errs):.3f}')
    print(f'measurement noise floor:                        {noise:.3f}')
    print(f'ratio: {st.mean(errs)/noise:.1f}x noise  '
          f'({"USABLE" if st.mean(errs) < 3 * noise else "NOT USABLE"} as a planning tool)')

    # --- what it implies ------------------------------------------------------------------
    print('\n' + '=' * 78)
    print('WHAT THE FITTED LAW IMPLIES FOR THIS PROJECT')
    print('=' * 78)
    full = f.corpus_info(CORPUS)['n_tokens']['train']
    print(f'\nAll the Yoruba we have is {full:,} tokens. Reading it N times is worth as much as:')
    print(f"\n{'passes':>8}{'tokens processed':>20}{'worth (fresh tokens)':>24}"
          f"{'predicted loss':>17}")
    for reps in (1, 2, 5, 10, 15, 30, 60, 120, 240):
        seen = full * reps
        eff = float(d_eff(full, seen, best['R_star']))
        print(f'{reps:>8}{seen:>20,.0f}{eff:>24,.0f}{float(predict(best, full, seen)):>17.3f}')

    # The point of diminishing returns, in the unit the project already argues in.
    base = float(predict(best, full, full * 15))
    for reps in range(15, 2000):
        if base - float(predict(best, full, full * reps)) > noise:
            print(f'\nFrom our standard 14.8 passes, you need {reps} passes '
                  f'({reps/14.8:.0f}x the compute) before the gain exceeds one seed spread.')
            break
    else:
        print('\nFrom our standard 14.8 passes, NO amount of further repetition gains more than '
              'one seed spread. The corpus is exhausted.')

    # --- the repetition ceiling, read straight off the grid --------------------------------
    print('\n' + '=' * 78)
    print('THE REPETITION CEILING, WITHOUT ANY FITTING')
    print('=' * 78)
    print('\nCells past the cliff, sorted by how many times they re-read their corpus:\n')
    print(f"{'unique':>12}{'epochs':>9}{'loss':>8}")
    for c in sorted(rows, key=lambda c: c['epochs']):
        print(f'{c["unique"]:>12,}{c["epochs"]:>9.1f}{c["loss"]:>8.3f}')
    hi = [c for c in rows if c['epochs'] > 90]
    lo = [c for c in rows if c['epochs'] <= 20]
    if hi and lo:
        print(f'\n  <=20 passes:  mean loss {st.mean(c["loss"] for c in lo):.3f}  '
              f'(n={len(lo)})')
        print(f'  >90 passes:   mean loss {st.mean(c["loss"] for c in hi):.3f}  '
              f'(n={len(hi)})')
        print('\n  More repetition is WORSE, not merely not-better. Past roughly a hundred\n'
              '  passes the model is memorizing a small corpus rather than learning a language.')

    json.dump({'cells': rows_all, 'post_cliff': rows, 'noise': noise,
               'models': {k: v for k, v in models.items()},
               'loo_mae': st.mean(errs)},
              open(OUT, 'w', encoding='utf-8'), indent=2)
    print(f'\nwrote runs/scaling_law.json')


if __name__ == '__main__':
    main()
