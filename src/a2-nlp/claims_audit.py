"""Test every falsifiable claim the poster makes, against the null that says it is nothing.

The staleness check in poster_bottom.ipynb verifies NUMBERS -- that a figure in the prose still
matches the records. Every mistake this project has shipped was in the CLAIM wrapped around the
number, not in the number:

    "the sign flips"          r = +0.303, which is t = 1.19 -- absence, not inversion
    "the floors explain it"   57% against 52%, which explains nothing
    "hardware moved it"       a rule that printed that 56.6% of the time under the null

Each was falsifiable in under a minute with data already on disk, and each survived because it
read well. Fluency was doing the work verification should have done.

So this is the missing gate. For each comparative claim: state the null, compute the statistic
that would refute it, and print a verdict the writeup has to match. Not the effect size -- the
test. An effect size cannot tell you it is indistinguishable from nothing; that is the whole
category of error above.

    bash src/a2-nlp/py.sh claims_audit.py
"""
from __future__ import annotations

import json
import math
import os
import statistics as st

from scipy import stats

import ft_api
import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05
results = []


def verdict(name, claim, supported, detail, note=''):
    # scipy hands back numpy.bool_, and `numpy.bool_(True) is True` is False. The first version
    # of this counted those as neither supported nor refuted, so the summary said "8 claims:
    # 2 supported, 1 not, 2 underpowered" -- five of eight -- and then listed six as failing.
    # A verification tool whose own arithmetic does not add up is worse than none, so: coerce.
    if supported is not None:
        supported = bool(supported)
    results.append({'claim': name, 'supported': supported})
    mark = 'SUPPORTED    ' if supported is True else (
        'NOT SUPPORTED' if supported is False else 'UNDERPOWERED ')
    print(f'\n{mark}  {name}')
    print(f'    the poster says : {claim}')
    print(f'    the test        : {detail}')
    if note:
        print(f'    note            : {note}')


def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    for p, i in enumerate(o):
        out[i] = float(p)
    return out


# ------------------------------------------------------------------------------------------
def main():
    corr = json.load(open(os.path.join(HERE, 'runs', 'downstream_correlation.json'),
                          encoding='utf-8'))
    down = ft_api.results('*')
    pre = f.results('*')
    CUT = 3.1

    print('=' * 86)
    print('CLAIMS AUDIT -- every comparative claim on the board, against its null')
    print('=' * 86)

    # --- 1 & 2. the correlations ---------------------------------------------------------
    for task, label, expect in (('sib200', 'SIB-200', True), ('masakhaner', 'MasakhaNER', False)):
        g = [r for r in corr if r['task'] == task and r['val_loss'] < CUT]
        x = [r['val_loss'] for r in g]
        y = [r['mean'] for r in g]
        r, p = stats.pearsonr(x, y)
        sig = p < ALPHA
        verdict(f'validation loss predicts {label} score',
                'strong on SIB, absent on NER',
                sig if expect else (not sig),
                f'r = {r:+.3f}, n = {len(g)}, p = {p:.3f} against H0: r = 0',
                '' if expect else 'absence is what we claim; do not read the sign')

    # --- 3. the band-width difference, which is the corrected explanation ------------------
    bands = {}
    for task in ('sib200', 'masakhaner'):
        y = [r['mean'] for r in corr if r['task'] == task and r['val_loss'] < CUT]
        bands[task] = y
    s1, s2 = st.stdev(bands['sib200']), st.stdev(bands['masakhaner'])
    n1, n2 = len(bands['sib200']), len(bands['masakhaner'])
    F = (s1 ** 2) / (s2 ** 2)
    p = 2 * min(stats.f.cdf(F, n1 - 1, n2 - 1), 1 - stats.f.cdf(F, n1 - 1, n2 - 1))
    verdict('SIB scores vary more between models than NER scores do',
            'the band is 0.143 against 0.044 -- which is why loss predicts one and not the other',
            p < ALPHA,
            f'sd {s1:.4f} vs {s2:.4f}, F = {F:.1f} on ({n1-1}, {n2-1}) df, p = {p:.4f}')

    # --- 4. the floors, which is the explanation we RETRACTED -----------------------------
    shares = []
    for task, steps in (('sib200', 1056), ('masakhaner', 2150)):
        floor = max((r for r in down if r.get('task') == task and r.get('steps') == steps
                     and 'yor_random_init' in r['model']), key=lambda r: r['mean'])['mean']
        shares.append(floor / max(bands[task]))
    verdict('the floors explain the task divergence',
            'RETRACTED -- was in an email and a figure before it was checked',
            False,
            f'floor is {shares[0]:.0%} of best on SIB and {shares[1]:.0%} on NER',
            'five points apart. Kept in the audit so the retraction stays visible')

    # --- 5. from-scratch ahead of mmBERT on SIB -------------------------------------------
    def best(frag, task, steps):
        c = [r for r in down if frag in r['model'] and r.get('task') == task
             and r.get('steps') == steps]
        return max(c, key=lambda r: r['mean']) if c else None

    ours, mmb = best('yor_64M', 'sib200', 1056), best('mmBERT', 'sib200', 1056)
    overlap = ours['ci'][0] < mmb['ci'][1]
    verdict('from-scratch is ahead of mmBERT on topic classification',
            f"{ours['mean']:.3f} vs {mmb['mean']:.3f}, a margin of {ours['mean']-mmb['mean']:.3f}",
            None if overlap else True,
            f"CIs [{ours['ci'][0]:.3f},{ours['ci'][1]:.3f}] and "
            f"[{mmb['ci'][0]:.3f},{mmb['ci'][1]:.3f}] -- "
            f"{'OVERLAP' if overlap else 'disjoint'}",
            'both arms also selected on the same 204 test items they are scored on')

    # --- 6. the tokenizer penalty against the noise it must beat --------------------------
    def arm(pat, corpus):
        rr = [r for r in f.results(pat) if r.get('corpus') == corpus]
        cpt = f.corpus_info(corpus)['chars_per_token']
        return [r['val_loss'] / math.log(2) / cpt for r in rr]

    a, c = arm('swap_yor_xlmr_*', 'yor_xlmr'), arm('swap62k_*', 'yor')
    t, p = stats.ttest_ind(a, c, equal_var=False)
    verdict('the tokenizer penalty is real at matched compute',
            f'{st.mean(a)-st.mean(c):.3f} bits per character',
            p < ALPHA,
            f'Welch t = {t:.2f}, p = {p:.4f}, n = {len(a)} vs {len(c)}')

    # --- 7. does the best learning rate transfer? -----------------------------------------
    lr_rows = [r for r in json.load(open(os.path.join(HERE, 'runs', 'lr_transfer.json'),
                                         encoding='utf-8')) if 'val_loss' in r]
    if lr_rows:
        langs = sorted({r['lang'] for r in lr_rows})
        argmin, gaps = {}, []
        for lang in langs:
            cells = {}
            for lr in sorted({r['lr'] for r in lr_rows}):
                v = [r['val_loss'] for r in lr_rows if r['lang'] == lang and r['lr'] == lr]
                if v:
                    cells[lr] = (st.mean(v), v)
            if len(cells) < 2:
                continue
            order = sorted(cells.items(), key=lambda kv: kv[1][0])
            argmin[lang] = order[0][0]
            gaps.append((lang, order[0][1][0], order[1][1][0], order[0][1][1], order[1][1][1]))
        differs = len(set(argmin.values())) > 1
        # A best rate only "differs" if the winner beats the runner-up by more than the seeds do.
        resolved = [g for g in gaps if len(g[3]) > 1 and len(g[4]) > 1
                    and abs(g[1] - g[2]) > max(abs(g[3][0] - g[3][1]), abs(g[4][0] - g[4][1]))]
        verdict('the best learning rate is not the same across languages',
                f'argmins {sorted(set(argmin.values()))}',
                None if len(resolved) < len(gaps) else differs,
                f'{len(resolved)} of {len(gaps)} languages separate their top two rates by more '
                f'than their own seed gap',
                'at two seeds a per-language argmin is barely resolved; extension pending')

    # --- 8. more text stops helping -------------------------------------------------------
    ladder = {}
    for r in pre:
        if (r.get('corpus') == 'eng_1b' and (r.get('preset') or 'poc') == 'poc'
                and r['steps'] == 62500):
            ladder.setdefault(r['n_tokens'], []).append(r['val_loss'])
    xs = sorted(ladder)
    if len(xs) >= 3:
        lo, hi = ladder[xs[-3]], ladder[xs[-1]]
        gain = st.mean(lo) - st.mean(hi)
        spread = st.mean(st.stdev(v) for v in ladder.values() if len(v) > 1)
        verdict('past 64M tokens, more text buys nothing measurable',
                f'16x more text moves the loss {gain:+.3f}',
                abs(gain) < spread,
                f'|{gain:.3f}| against a run-to-run spread of {spread:.3f}',
                'claim is ABSENCE of an effect, so it survives by being smaller than the noise')

    # --- summary ---------------------------------------------------------------------------
    print('\n' + '=' * 86)
    ok = sum(1 for r in results if r['supported'] is True)
    no = sum(1 for r in results if r['supported'] is False)
    weak = sum(1 for r in results if r['supported'] is None)
    print(f'{len(results)} claims: {ok} supported, {no} NOT supported, {weak} underpowered')
    if no or weak:
        print('\nanything not SUPPORTED must be hedged in the prose or removed from the board:')
        for r in results:
            if r['supported'] is not True:
                print(f"  - {r['claim']}")
    print('\nTesting eight claims at alpha=0.05 means roughly one false positive is expected by')
    print('chance alone. These are not independent hypotheses from a pre-registered plan, so')
    print('read them as a checklist against overstatement, not as a significance ceremony.')
    json.dump(results, open(os.path.join(HERE, 'runs', 'claims_audit.json'), 'w',
                            encoding='utf-8'), indent=2)


if __name__ == '__main__':
    main()
