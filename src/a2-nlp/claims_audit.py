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


def exact_p(a, b):
    """Two-sided permutation p over every way of splitting a+b, and the smallest p it can return.

    Worth reporting next to a t-test, because at these sample sizes the t-test quotes precision
    the data does not contain. Five against five cannot go below 2/C(10,5) = 0.0079 no matter how
    far apart the groups are; three against three cannot go below 0.10. A Welch p of 0.0002 on
    five-a-side is a parametric extrapolation about forty times into a tail that ten observations
    do not reach, and quoting it invites exactly the objection this file exists to pre-empt.
    """
    r = stats.permutation_test((list(a), list(b)),
                               lambda x, y: st.mean(x) - st.mean(y),
                               permutation_type='independent', alternative='two-sided',
                               n_resamples=100_000, random_state=0)
    floor = 2.0 / math.comb(len(a) + len(b), len(a))
    return float(r.pvalue), floor


def separation(a_rec, b_rec, label_a='A', label_b='B'):
    """Every test the saved data supports, because no single one answers the whole question.

    Three different uncertainties are in play and they are not nested:

      seeds   would a rerun land somewhere else?      -- uses `scores`, blind to the test set
      items   do 204 items place the number?          -- uses `ci`,     blind to the seeds
      paired  do the two arms differ on THESE items?  -- needs predictions, cancels item difficulty

    The project used to decide separation by asking whether the two intervals overlap. That is
    not a weaker version of the right test, it is a different one: two 95% intervals miss each
    other only when the margin clears 1.96*(SE_a + SE_b), which is algebraically the assumption
    that the arms' per-item errors are perfectly ANTI-correlated. Its effective alpha is 0.0056.
    It is not safe in the other direction either -- the interval cannot see seed spread, so a
    high-variance arm gets a narrow one and the gate calls it separated when the seeds cannot.

    So print all of them and let the disagreements show.
    """
    a, b = a_rec['scores'], b_rec['scores']
    d = st.mean(a) - st.mean(b)
    out = {'difference': d, 'n_a': len(a), 'n_b': len(b)}

    t, p_welch = stats.ttest_ind(a, b, equal_var=False)
    out['welch'] = (float(t), float(p_welch))
    out['exact'], out['exact_floor'] = exact_p(a, b)

    # The seed-spread rule this project quotes, fed the SAMPLE sd it was derived for.
    sd_a, sd_b = ft_api.sample_sd(a), ft_api.sample_sd(b)
    pooled = math.sqrt((sd_a ** 2 + sd_b ** 2) / 2)
    dfree = len(a) + len(b) - 2
    bar = stats.t.ppf(1 - ALPHA / 2, dfree) * math.sqrt(1 / len(a) + 1 / len(b))
    out['spread_ratio'] = abs(d) / pooled if pooled else float('inf')
    out['spread_bar'] = bar

    # Item-level, from the intervals already stored. Treating each as +-1.96 SE and testing the
    # DIFFERENCE rather than the overlap is the same data, asked the right question.
    se_a = (a_rec['ci'][1] - a_rec['ci'][0]) / (2 * 1.959964)
    se_b = (b_rec['ci'][1] - b_rec['ci'][0]) / (2 * 1.959964)
    se_item = math.sqrt(se_a ** 2 + se_b ** 2)
    out['item'] = (d / se_item, 2 * (1 - stats.norm.cdf(abs(d) / se_item)))
    out['overlap'] = not (a_rec['ci'][0] > b_rec['ci'][1] or b_rec['ci'][0] > a_rec['ci'][1])

    # Both sources, unpaired -- the conservative combination, since pairing only helps.
    se_seed = math.sqrt(sd_a ** 2 / len(a) + sd_b ** 2 / len(b))
    se_both = math.sqrt(se_item ** 2 + se_seed ** 2)
    out['combined'] = (d / se_both, 2 * (1 - stats.norm.cdf(abs(d) / se_both)))

    # And the one that settles it, if the runs were new enough to have saved predictions.
    out['paired'] = ft_api.paired_bootstrap(a_rec['tag'], b_rec['tag'])

    lines = [f"{label_a} {st.mean(a):.4f} vs {label_b} {st.mean(b):.4f}, margin {d:+.4f}",
             f"seeds  exact p = {out['exact']:.4f} (floor {out['exact_floor']:.4f} at "
             f"{len(a)}v{len(b)}), Welch p = {p_welch:.4g}",
             f"spread {out['spread_ratio']:.2f}x pooled sample sd, bar is {bar:.2f}x at "
             f"{dfree} df",
             f"items  z = {out['item'][0]:.2f}, p = {out['item'][1]:.4f}"
             f"  (intervals {'overlap' if out['overlap'] else 'disjoint'})",
             f"both   z = {out['combined'][0]:.2f}, p = {out['combined'][1]:.4f}"]
    if out['paired']:
        pb = out['paired']
        lines.append(f"paired {pb['difference']:+.4f} over {pb['n_items']} shared items, "
                     f"p = {pb['p']:.4f}")
    else:
        lines.append('paired unavailable -- these runs predate saved predictions')
    out['text'] = '\n                      '.join(lines)
    # Supported only if the two tests that use DIFFERENT uncertainties both clear alpha. That is
    # deliberately stricter than either alone, and it is the honest reading when the arms have
    # not been compared item by item.
    out['supported'] = bool(out['exact'] < ALPHA and out['combined'][1] < ALPHA)
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
            # Difference the ROUNDED shares, not the raw ones. Printing 61%, 78% and "18 points
            # apart" in the same breath is the kind of one-point inconsistency that makes a
            # reader stop and recheck the whole line, and this file exists to be trusted.
            f'{abs(round(shares[0]*100)-round(shares[1]*100)):.0f} points apart, and the note '
            f'here used to say "five" '
            f'from when it was.\n                      The floors moved when the NER control was '
            f'finally swept; the retraction did not, because the\n                      band '
            f'widths are what explain the divergence either way. Kept visible on purpose.')

    # --- 5. from-scratch ahead of mmBERT on SIB -------------------------------------------
    # Both fixes here are Patrick's, and both are the same bugs he had just removed from
    # fig_headline. Arms were matched with `frag in r['model']` -- a substring of a filesystem
    # path, so 'yor_64M' also caught s1, s2 and the 46.9k model, and the field itself differs
    # between machines, which is what silently dropped three of five arms from a table in his
    # sweep notebook. And the rate was taken as max(mean) over TEST cells, which re-selects on
    # the very items being reported. model_slug fixes the first; reading the rate off the dev
    # cells fixes the second. They happen to land on the same two cells today, which is exactly
    # why this needed fixing before it drifted rather than after.
    dev = ft_api.results(eval_split='validation')

    def dev_pick(slug, task, steps):
        """The cell report 11 would report: rate chosen on dev, number read off test."""
        cand = [r for r in dev if r.get('model_slug') == slug and r.get('task') == task
                and r.get('steps') == steps]
        if not cand:
            return None
        b = max(cand, key=lambda r: r['mean'])
        hit = [r for r in down if r.get('model_slug') == slug and r.get('task') == task
               and r.get('steps') == steps and abs(r['lr'] - b['lr']) < 1e-12]
        return hit[0] if hit else None

    ours = dev_pick('yor-64M-62.5k-s0', 'sib200', 1056)
    mmb = dev_pick('mmBERT-base', 'sib200', 1056)
    if ours and mmb:
        s = separation(ours, mmb, 'ours', 'mmBERT')
        verdict('from-scratch is ahead of mmBERT on topic classification',
                f"{ours['mean']:.3f} vs {mmb['mean']:.3f}, a margin of "
                f"{s['difference']:.3f}",
                s['supported'],
                s['text'],
                f"rate chosen on the 99 dev items (ours {ours['lr']:g}, mmBERT {mmb['lr']:g}), "
                f"then scored on the 204 test items -- so this is no longer selected on what it "
                f"reports.\n                      Three of the five reported seeds are the three "
                f"the dev sweep used, so a residual selection channel remains.")

    # --- 6. the tokenizer penalty against the noise it must beat --------------------------
    def arm(pat, corpus):
        rr = [r for r in f.results(pat) if r.get('corpus') == corpus]
        cpt = f.corpus_info(corpus)['chars_per_token']
        return [r['val_loss'] / math.log(2) / cpt for r in rr]

    a, c = arm('swap_yor_xlmr_*', 'yor_xlmr'), arm('swap62k_*', 'yor')
    t, p = stats.ttest_ind(a, c, equal_var=False)
    pe, floor = exact_p(a, c)
    F = st.stdev(a) ** 2 / st.stdev(c) ** 2
    pF = 2 * min(stats.f.cdf(F, len(a) - 1, len(c) - 1),
                 1 - stats.f.cdf(F, len(a) - 1, len(c) - 1))
    inside = sum(1 for x in a if min(c) <= x <= max(c))
    verdict('the tokenizer penalty is real at matched compute',
            f'{st.mean(a)-st.mean(c):.3f} bits per character',
            p < ALPHA and pe < ALPHA,
            f'Welch t = {t:.2f}, p = {p:.4f}, n = {len(a)} vs {len(c)}\n'
            f'                      exact permutation p = {pe:.3f} (floor {floor:.4f} at '
            f'{len(a)}v{len(c)})\n'
            f'                      the arms INTERLEAVE: {inside} of {len(a)} large-vocabulary '
            f'runs land inside the small-vocabulary range',
            f'Six seeds a side, fixed in advance. The penalty did not grow into significance, it '
            f'SHRANK --\n                      0.144 bits/char at three seeds, '
            f'{st.mean(a)-st.mean(c):.3f} at six. There is no direction left to report.')

    # The six seeds did not merely fail to establish the penalty; they moved the effect from the
    # mean to the variance. Testing only location would have recorded a null and thrown away the
    # finding, which is the mirror image of the mistake this file was built to catch.
    verdict('the large vocabulary makes the RESULT less predictable',
            f'seed spread {st.stdev(a):.3f} against {st.stdev(c):.3f}, '
            f'{st.stdev(a)/st.stdev(c):.1f}x wider',
            pF < ALPHA,
            f'F = {F:.1f} on ({len(a)-1}, {len(c)-1}) df, p = {pF:.4f}, '
            f'Levene p = {stats.levene(a, c)[1]:.4f}',
            'Same shape downstream, where the arms are 4.0x wider on topic and 7.7x on entities. '
            'A large\n                      vocabulary is not reliably more expensive per '
            'character -- it is a lottery, landing both\n                      better and much '
            'worse than the small one depending on the seed.')

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
    # Derived, not typed. This line said "eight" for a while after the ninth claim was added,
    # which is a small thing to get wrong in the file whose subject is numbers that stopped
    # matching their sentence.
    print(f'\nTesting {len(results)} claims at alpha={ALPHA} means roughly '
          f'{len(results)*ALPHA:.1f} false positives are expected by chance alone. These are not')
    print('independent hypotheses from a pre-registered plan, so read them as a checklist')
    print('against overstatement, not as a significance ceremony.')
    json.dump(results, open(os.path.join(HERE, 'runs', 'claims_audit.json'), 'w',
                            encoding='utf-8'), indent=2)


if __name__ == '__main__':
    main()
