"""Poster figures, generated from the run records so they cannot drift from the reports.

Writes both SVG and 300-dpi PNG into reports/figures/. The SVG is what belongs on the poster --
it stays sharp at A0 -- and the PNG is what the markdown and the notebook display.

Design follows one rule that is worth stating because it is easy to get wrong: color is assigned
to the ENTITY, in fixed slot order, never to its rank. A chart that recolors its bars when a
filter changes the ordering teaches the reader the wrong thing. Text stays in ink colors; a
colored mark beside a label carries the identity.

    bash src/a2-nlp/py.sh poster_figures.py
"""
from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ft_api
import mlm_api as f
import study_label_quantity as label_quantity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'reports', 'figures')

# The validated categorical palette, light mode, in fixed slot order.
C1, C2, C3, C4, C5 = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#8a8985'
GRID, SURFACE = '#e6e5e1', '#fcfcfb'

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'axes.titlecolor': INK,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.9, 'axes.axisbelow': True,
    'axes.spines.top': False, 'axes.spines.right': False,
    # Poster type: readable from two meters away, so everything is larger than a paper figure.
    'font.size': 13, 'axes.titlesize': 16, 'axes.labelsize': 13, 'legend.fontsize': 12,
    'axes.titleweight': 'bold', 'legend.frameon': False,
    'lines.linewidth': 2.4, 'lines.markersize': 9,
    # Deterministic element ids -- see save().
    'svg.hashsalt': 'csed504-a2-nlp-poster',
})


def save(fig, name):
    """Write both formats, byte-identical across runs when nothing has changed.

    Matplotlib stamps an SVG with the current time and with element ids derived from object
    addresses, so re-rendering an unchanged figure produces a different file and eight of these
    turn up in `git status` every time the notebook runs. Noise like that is how people stop
    reading `git status` at all. Fixing the hashsalt and dropping the date makes a real change
    the only thing that shows up in a diff.
    """
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (('svg', {'metadata': {'Date': None}}), ('png', {'dpi': 300})):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), bbox_inches='tight', **kw)
    plt.close(fig)
    print(f'  wrote figures/{name}.svg and .png')


# --------------------------------------------------------------------------------------------
def fig_headline():
    """The result: a small from-scratch model against two much larger multilingual ones.

    Grouped bars because the job is comparing magnitudes within two separate tasks, and the
    untrained floor is drawn as a rule rather than a bar -- it is a reference level, not a
    competitor, and drawing it as a bar invites reading it as one.

    Every bar carries its per-seed scores as dots, which is not decoration. XLM-R's topic cell is
    four seeds that trained and one that collapsed below chance, so its mean of 0.358 describes no
    run that happened. A bar hides that. So does the confidence interval, which resamples test
    items with predictions pooled across seeds and cannot see between-seed spread at all.
    """
    test = ft_api.results()                        # test-scored cells only; that is the default
    dev = ft_api.results(eval_split='validation')

    def pick(slug, task, steps):
        """The cell this figure reports, chosen the way the study chose it.

        Where dev cells exist the learning rate is taken from THEM and only then looked up on
        test -- what report 11 did. Taking max() over test cells instead re-selects on the items
        being reported, which is the practice report 11 removed; feeding this figure the new rows
        without changing this would leave it doing the old thing while looking updated, because
        the superseded grids are still on disk.

        Arms are matched on model_slug. r['model'] is the directory a run was launched from and
        differs between machines for the same weights, and a substring like 'yor_64M' also matches
        the s1, s2 and 46.9k checkpoints -- either way "best" can wander onto a different model.
        """
        t = [r for r in test
             if r['model_slug'] == slug and r.get('task') == task and r.get('steps') == steps]
        if not t:
            return None
        d = [r for r in dev
             if r['model_slug'] == slug and r.get('task') == task and r.get('steps') == steps]
        if d:
            lr = max(d, key=lambda r: r['mean'])['lr']
            on_test = [r for r in t if r['lr'] == lr]
            if on_test:
                return dict(on_test[0], chosen_on='dev')
        return dict(max(t, key=lambda r: r['mean']), chosen_on='test')

    # Two short lines rather than one long one: at three bars across a half-width panel, a
    # single line of "33.8M - Yoruba only" runs into its neighbor.
    models = [('from-scratch', '33.8M\nYoruba only', 'yor-64M-62.5k-s0', C1),
              ('mmBERT', '246M\n1,800 languages', 'mmBERT-base', C2),
              ('XLM-R', '277M\n100 languages', 'xlm-roberta-base', C3)]
    tasks = [('Topic classification', 'needs meaning', 'sib200', 1056),
             ('Entity recognition', 'needs surface form', 'masakhaner', 2150)]
    FLOOR = 'yor-random-init'

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    floors, chosen_on_test, mixed = [], [], []
    for ax, (tlabel, tsub, task, steps) in zip(axes, tasks):
        recs, colors, names = [], [], []
        for short, detail, slug, color in models:
            r = pick(slug, task, steps)
            if r is None:
                continue
            recs.append(r)
            colors.append(color)
            names.append((short, detail))
            if r['chosen_on'] == 'test':
                chosen_on_test.append(tlabel.lower())

        ax.bar(range(len(recs)), [r['mean'] for r in recs], color=colors, width=0.58)

        # The seeds themselves, over the bar they average to.
        for i, r in enumerate(recs):
            ys = r.get('scores') or []
            if ys:
                ax.scatter([i] * len(ys), ys, s=20, zorder=3, color=INK, alpha=0.5, linewidths=0)
            # The value goes above the bar AND above its dots. bar_label only knows the bar, so
            # with five seeds drawn it puts the number inside the cluster on every arm whose best
            # seed beats its mean -- which is most of them.
            top = max([r['mean']] + list(ys))
            ax.text(i, top + 0.022, f'{r["mean"]:.3f}', ha='center', va='bottom',
                    fontsize=14, color=INK, fontweight='bold')
            chance = r.get('chance')
            if chance and 0 < sum(y < chance for y in ys) < len(ys):
                ok = sum(y >= chance for y in ys)
                # Say which count this is. "(4 of 5)" after the words "below chance" reads as
                # four seeds having failed, when four is the number that worked.
                mixed.append(f'{names[i][0]} on {tlabel.lower()} trained in {ok} of {len(ys)} '
                             f'seeds, {len(ys) - ok} collapsing below chance')

        f = pick(FLOOR, task, steps)
        if f:
            ax.axhline(f['mean'], ls=(0, (5, 4)), lw=1.8, color=MUTED)
            floors.append(f'{f["mean"]:.3f} on {tlabel.lower()}')

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n[0] for n in names], fontsize=13.5, color=INK)
        # The size/coverage detail sits below the name as a second, quieter row rather than as
        # part of the tick label, which is what made them run into each other.
        for i, (_, detail) in enumerate(names):
            ax.text(i, -0.08, detail, ha='center', va='top', fontsize=11, color=MUTED,
                    linespacing=1.35, transform=ax.get_xaxis_transform())

        ax.set_title(f'{tlabel}\n{tsub}', pad=12, fontsize=15)
        # One scale across both panels: a reader WILL compare them, and different scales would
        # make 0.688 and 0.837 look like the same height.
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('score  (higher is better)')

    fig.suptitle('A 33.8M model trained only on Yoruba, against two much larger multilingual models',
                 fontsize=17, fontweight='bold', color=INK, y=1.02)

    # Everything the bars cannot say, said beneath them. Each line is generated from the records,
    # so none of it can drift the way a typed caption does.
    notes = ['The dashed line is what the same architecture scores with no pretraining at all: '
             + ', '.join(floors) + '.',
             'Dots are individual seeds.']
    if mixed:
        # No .capitalize() here: it lowercases the rest of the string and turns XLM-R into
        # Xlm-r. The arm names already start these clauses and are already cased correctly.
        notes.append('; '.join(mixed) + ' -- a bar like that is a mixture, not a mean.')
    if chosen_on_test:
        notes.append('Learning rates for ' + ', '.join(sorted(set(chosen_on_test)))
                     + ' were chosen on the same test items they are scored on, which inflates '
                       'them; the other panel was chosen on a held-out split.')
    for i, line in enumerate(notes):
        fig.text(0.5, 0.075 - i * 0.028, line, ha='center', color=INK2, fontsize=11.5)
    fig.subplots_adjust(bottom=0.28, wspace=0.22)
    save(fig, '01-headline')


# --------------------------------------------------------------------------------------------
def fig_gradient():
    """What XLM-R's vocabulary costs, per language, sorted.

    A dot plot rather than bars: the quantity is a ratio around 1.0, so a bar from zero wastes
    most of its length and exaggerates small differences. Color carries the one thing being
    tested -- whether XLM-R was trained on the language.

    The caption is generated, and it was added on 12 August because there was none. The two means
    panel 3 leads with -- 1.150 against 1.593 -- appeared nowhere on the figure, so the chart showed
    a gradient and left the reader to eyeball the summary it is evidence for.

    The title changed at the same time and that is a retraction rather than a rewording. It read
    "A multilingual vocabulary costs nothing -- until the language is left out", which overclaims
    twice: covered languages average 1.150x, which is not nothing, and Wolof is UNCOVERED at 1.31x
    yet sits below covered Xhosa at 1.42x, so there is no clean split to be "until" the far side of.
    Report 07 and CLAUDE.md both call this "a gradient with one exception"; the figure was the only
    place asserting otherwise, and it is the place a poster reader looks first.
    """
    rows = [r for r in json.load(open(os.path.join(HERE, 'runs', 'gradient_table.json'),
                                     encoding='utf-8')) if r['corpus'] != 'eng_1b']
    NAME = {'cmn': 'Mandarin', 'ind': 'Indonesian', 'eng': 'English', 'fra': 'French',
            'afr': 'Afrikaans', 'swh': 'Swahili', 'som': 'Somali', 'hau': 'Hausa',
            'amh': 'Amharic', 'xho': 'Xhosa', 'wol': 'Wolof', 'lug': 'Luganda',
            'nya': 'Chichewa', 'sna': 'Shona', 'kin': 'Kinyarwanda', 'yor': 'Yoruba',
            'ibo': 'Igbo'}
    rows.sort(key=lambda r: r['penalty'])

    fig, ax = plt.subplots(figsize=(9.5, 8))
    for i, r in enumerate(rows):
        covered = r['in_xlmr'] is True
        color = C3 if covered else C2
        ax.plot([1.0, r['penalty']], [i, i], color=color, lw=2.2, alpha=0.45, zorder=1)
        ax.plot(r['penalty'], i, 'o', color=color, zorder=2,
                markersize=11 if r['corpus'] == 'yor' else 9)
        # The one language below 1.0 gets its number on the other side, so it doesn't cross
        # the reference rule it sits just left of.
        side = 1 if r['penalty'] >= 1.0 else -1
        ax.text(r['penalty'] + 0.022 * side, i, f"{r['penalty']:.2f}", va='center',
                ha='left' if side > 0 else 'right', fontsize=11, color=INK2)
    ax.axvline(1.0, color=MUTED, lw=1.6)
    ax.text(1.0, len(rows) - 0.2, '  no penalty', color=INK2, fontsize=11.5, va='top')

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([NAME.get(r['corpus'], r['corpus']) for r in rows], fontsize=12.5,
                       color=INK)
    ax.set_xlabel("tokens XLM-R needs per token of a purpose-built vocabulary")
    ax.set_xlim(0.83, 1.95)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.grid(axis='y', visible=False)
    ax.set_title('The penalty tracks coverage — not script, and not region', pad=14)
    ax.plot([], [], 'o', color=C3, label='XLM-R was trained on this language')
    ax.plot([], [], 'o', color=C2, label='XLM-R was not')
    ax.legend(loc='lower right', fontsize=12)

    # -- the caption, generated ---------------------------------------------------------------
    # The African-only control is report 07 section 4's, and the set it needs is not on the
    # records -- there is no region field -- so it is named here. The two sets must PARTITION the
    # table: a language added later then fails this assertion instead of being silently counted as
    # African, which is the failure this project keeps paying for in other forms.
    AFRICAN = {'afr', 'swh', 'som', 'hau', 'amh', 'xho', 'wol', 'lug', 'nya', 'sna', 'kin',
               'yor', 'ibo'}
    ELSEWHERE = {'cmn', 'ind', 'eng', 'fra'}
    seen = {r['corpus'] for r in rows}
    assert seen == AFRICAN | ELSEWHERE, f'unclassified languages: {seen ^ (AFRICAN | ELSEWHERE)}'

    cov = [r for r in rows if r['in_xlmr'] is True]
    unc = [r for r in rows if r['in_xlmr'] is not True]
    def mean(group):
        return st.mean(r['penalty'] for r in group)

    top_cov = max(cov, key=lambda r: r['penalty'])
    above = [r for r in unc if r['penalty'] > top_cov['penalty']]
    exc = [r for r in unc if r['penalty'] < top_cov['penalty']]
    yor = next(r for r in rows if r['corpus'] == 'yor')
    rank = sorted((r['penalty'] for r in rows), reverse=True).index(yor['penalty']) + 1

    notes = [f"XLM-R's vocabulary costs the {len(cov)} languages it covers {mean(cov):.3f}× on "
             f'average, and the {len(unc)} it does not {mean(unc):.3f}×.',
             f'Restricted to African languages on both sides — which rules out script and region '
             f'as the explanation — {mean([r for r in cov if r["corpus"] in AFRICAN]):.3f}× '
             f'against {mean([r for r in unc if r["corpus"] in AFRICAN]):.3f}×.',
             f'{len(above)} of the {len(unc)} uncovered languages sit above every covered one. '
             + ('The exception is not smoothed: '
                + ', '.join(f'{NAME[r["corpus"]]} at {r["penalty"]:.2f}×' for r in exc)
                + f' is uncovered and below {NAME[top_cov["corpus"]]} at '
                  f'{top_cov["penalty"]:.2f}×.' if exc else 'The separation is clean.'),
             f'Yoruba, the target language, ranks {rank} of {len(rows)} at {yor["penalty"]:.2f}× — '
             f'so the original figure was not a quirk of one language.']
    for i, line in enumerate(notes):
        fig.text(0.5, -0.015 - i * 0.027, line, ha='center', color=INK2, fontsize=11.5)
    save(fig, '02-tokenizer-gradient')


# --------------------------------------------------------------------------------------------
def fig_matched():
    """The same three experiments, read two ways, giving opposite answers.

    The bars are the measurement; the annotation is the argument. Wall-clock is printed on each
    bar because it is the variable the naive reading ignores, and the whole point is that it was
    not held fixed.
    """
    def arm(pat, corpus):
        rr = [r for r in f.results(pat) if r.get('corpus') == corpus]
        cpt = f.corpus_info(corpus)['chars_per_token']
        v = [r['val_loss'] / math.log(2) / cpt for r in rr]
        return st.mean(v), st.stdev(v), st.mean(r['seconds'] for r in rr) / 60

    a = arm('swap_yor_*', 'yor')            # 16k vocabulary, 12k steps
    b = arm('swap_yor_xlmr_*', 'yor_xlmr')  # 250k vocabulary, 12k steps
    c = arm('swap62k_*', 'yor')             # 16k vocabulary, matched wall clock

    labels = ['purpose-built\n16k vocabulary\n12,000 steps',
              "XLM-R's\n250k vocabulary\n12,000 steps",
              'purpose-built\n16k vocabulary\n62,500 steps']
    vals = [a[0], b[0], c[0]]
    errs = [a[1], b[1], c[1]]
    mins = [a[2], b[2], c[2]]
    colors = [C1, C2, C1]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    bars = ax.bar(range(3), vals, yerr=errs, color=colors, width=0.58,
                  error_kw=dict(ecolor=INK2, capsize=6, lw=1.6))
    for i, (v, m) in enumerate(zip(vals, mins)):
        ax.text(i, 0.06, f'{m:.0f} min\nof GPU time', ha='center', va='bottom',
                color=SURFACE, fontsize=12, fontweight='bold')
        ax.text(i, v + errs[i] + 0.05, f'{v:.3f}', ha='center', color=INK,
                fontsize=14, fontweight='bold')

    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=11.5, color=INK2)
    ax.set_ylabel('bits per character  (lower is better)')
    ax.set_ylim(0, 1.70)
    ax.set_title('Same three experiments. Two ways to read them. Opposite conclusions.', pad=14)

    # The two readings, drawn as brackets so the comparison being made is explicit.
    def bracket(x0, x1, y, text, color):
        ax.plot([x0, x0, x1, x1], [y - 0.03, y, y, y - 0.03], color=color, lw=1.8)
        ax.text((x0 + x1) / 2, y + 0.015, text, ha='center', va='bottom',
                fontsize=12, color=color, fontweight='bold')

    # Both brackets clear the tallest bar label (1.135 + its error bar) with room to spare;
    # at 1.28 the lower one ran through the "1.056" printed above the middle bar.
    bracket(0, 1, 1.40, 'same STEPS  →  looks identical', MUTED)
    bracket(1, 2, 1.56, 'same GPU TIME  →  purpose-built wins by 0.144', C1)
    save(fig, '03-matched-steps-vs-compute')


# --------------------------------------------------------------------------------------------
def fig_bimodal():
    """Thirteen runs of the larger model. Not a spread around a mean -- two outcomes.

    A one-dimensional strip, because the shape of the distribution IS the finding and any
    summary statistic destroys it.
    """
    v = sorted(r['val_loss'] for r in f.results('eng_1b_*')
               if (r.get('preset') or 'poc') == 'afriberta' and r.get('clip') in (None, 1.0))
    fig, ax = plt.subplots(figsize=(11, 3.2))
    for x in v:
        broke = x < 4.5
        # Surface-colored ring: eight of the thirteen runs land close enough together that
        # without it the cluster reads as one blob and the count is lost.
        ax.plot(x, 0, 'o', color=C1 if broke else C2, markersize=15,
                markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvspan(3.8, 5.3, color=GRID, alpha=0.7, zorder=0)
    ax.text(4.55, 0.38, 'nothing lands here', ha='center', color=INK2, fontsize=12.5)
    ax.text(3.0, -0.42, 'learned the language', ha='center', color=C1, fontsize=13,
            fontweight='bold')
    ax.text(6.4, -0.42, 'never got started', ha='center', color=C2, fontsize=13,
            fontweight='bold')
    ax.set_yticks([]); ax.set_ylim(-0.75, 0.75)
    ax.set_xlabel('validation loss  (lower is better)')
    ax.grid(axis='y', visible=False)
    ax.spines['left'].set_visible(False)
    ax.set_title('The same model, same settings, thirteen times — the average describes none of them',
                 pad=14)
    save(fig, '04-two-outcomes')


# --------------------------------------------------------------------------------------------
def fig_saturation():
    """More text stops helping. A line, because the x axis is a continuum.

    The shaded band is the run-to-run spread. Without it a reader sees a curve that still rises;
    with it they see that everything past 64M is inside the noise.
    """
    cells = {}
    for r in f.results('eng_1b_*'):
        if (r.get('preset') or 'poc') == 'poc' and r['steps'] == 62500:
            cells.setdefault(r['n_tokens'], []).append(r['val_loss'])
    xs = sorted(cells)
    ys = [st.mean(cells[n]) for n in xs]
    sds = [st.stdev(cells[n]) if len(cells[n]) > 1 else 0 for n in xs]
    spread = st.mean(s for s in sds if s)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.fill_between(xs, [y - s for y, s in zip(ys, sds)], [y + s for y, s in zip(ys, sds)],
                    color=C1, alpha=0.16, lw=0)
    ax.plot(xs, ys, 'o-', color=C1)
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.09, f'{y:.2f}', ha='center', color=INK, fontsize=12)

    ax.axvspan(64e6, xs[-1], color=GRID, alpha=0.55, zorder=0)
    ax.text(2.6e8, max(ys) * 0.86,
            f'sixteen times more text\nchanges the score by 0.08\n(run-to-run noise is {spread:.2f})',
            ha='center', color=INK2, fontsize=12.5)

    ax.set_xscale('log')
    ax.set_xticks(xs)
    ax.set_xticklabels([f.compact(x) for x in xs], fontsize=12)
    ax.set_xlabel('words of English the model was trained on')
    ax.set_ylabel('validation loss  (lower is better)')
    ax.set_title('More data stops helping, and stops early', pad=14)
    save(fig, '05-data-saturation')


# --------------------------------------------------------------------------------------------
def fig_cost():
    """What the project cost, three ways.

    Dots on a log axis, not bars. The values span four orders of magnitude, so the axis has to
    be logarithmic -- but a bar encodes its value as a LENGTH from a baseline, and on a log axis
    that length is a lie: $7 against $24,000 is 3,400x and the bars looked about 4x apart.
    A dot makes no promise about length; only its position carries the number.

    Every number here was hand-typed until 14 August, and every one of them had gone stale.
    The figure said 83 GPU-hours, $4.30, $250 to rent and 105 trained models. The project is
    148.0 GPU-hours, ~$7, $107 to rent and 197 models -- so the board's cost cell was quoting
    "148 GPU-hours, 71 kWh, $7, $107 on an A100" in type directly above a figure that said 83,
    $4.30 and $250. One cell, two answers, and the same 83.3 constant bench_portable.py already
    carries a paragraph about.

    So it computes. Hours and model count come from the live records; the rent comes from
    runs/hardware.json by the same arithmetic fig_hardware uses. Only the two physical constants
    below are typed, and they are the two nobody can derive from a run record.
    """
    # Measured at the wall over the project, not from card TDP: 71 kWh across 148.0 GPU-hours is
    # 0.48 kW, which includes cooling and the rest of the machine. Report 09 carries the full
    # accounting. These are the only hand-entered numbers left in this figure, and they are
    # properties of a building rather than of a run.
    WALL_KW, USD_PER_KWH = 71.0 / 148.0, 7.0 / 71.0

    hours = sum(_workstation_hours_by_preset().values())
    models = len([r for r in f.results('*') if r.get('seconds')])
    power = hours * WALL_KW * USD_PER_KWH
    items = [(f'Electricity\nfor {hours:.0f} GPU-hours', power, C3),
             ('Renting the same\ntime in the cloud', _rent_the_project(), C1),
             ('Buying the\nworkstation', 24000, C2)]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    for i, (label, v, color) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([1, v], [y, y], color=color, lw=2.2, alpha=0.4, zorder=1)
        ax.plot(v, y, 'o', color=color, markersize=15, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=2)
        ax.text(v * 1.45, y, f'${v:,.0f}' if v >= 100 else f'${v:.2f}', va='center',
                color=INK, fontsize=15, fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlim(1, 3e5)
    # Dollar amounts rather than 10^4: this panel is read by people who should not have to
    # translate scientific notation to know what the workstation cost.
    ax.set_xticks([1, 10, 100, 1_000, 10_000, 100_000])
    ax.set_xticklabels([f'${v:,}' for v in (1, 10, 100, 1_000, 10_000, 100_000)], fontsize=12)
    ax.minorticks_off()
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([i[0] for i in reversed(items)], fontsize=12.5, color=INK2)
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel('each gridline is ten times the one before it')
    ax.grid(axis='y', visible=False)
    ax.set_title(f'What {models} trained models cost', pad=26, loc='left')
    ax.text(0, 1.03, f'Renting wins below 9,300 GPU-hours of work. This project used '
                     f'{hours:.0f}.',
            transform=ax.transAxes, ha='left', va='bottom', color=INK2, fontsize=12.5)
    save(fig, '06-what-it-cost')




# --------------------------------------------------------------------------------------------
def fig_why_long():
    """Why a run has to be long, told by the one run whose curve makes it obvious.

    Two panels on a shared x axis: the learning-rate schedule the run actually used, and its
    validation loss. What matters is the flat stretch. For eleven thousand steps -- seven minutes,
    a sixth of the run -- the loss barely moves, and then it falls off a cliff. A run that is
    going to fail looks exactly like a run that is about to succeed, for as long as it takes to
    make the wrong call about it.

    The tail is the honest counterweight and is annotated too: the last third is worth about 0.10,
    which is inside the run-to-run noise. That part could have been cut. The head could not.
    """
    c = f.curve('yor_64M_62.5k_s0')
    steps = [r['step'] for r in c]
    lrs = [r['lr'] for r in c]
    vals = [r['val']['loss'] for r in c]
    mins = [r['elapsed'] / 60 for r in c]
    final = vals[-1]

    PLATEAU_END = 11_500
    pi = min(range(len(steps)), key=lambda i: abs(steps[i] - PLATEAU_END))
    cut = min(range(len(steps)), key=lambda i: abs(steps[i] - 41_667))
    remaining = vals[cut] - final

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                                 gridspec_kw={'height_ratios': [1, 1.7]})
    a1.plot(steps, lrs, color=C4)
    a1.set_ylabel('learning rate')
    a1.set_title('The schedule warms up, then anneals to zero', fontsize=14, loc='left', pad=8)

    a2.plot(steps, vals, color=C1)
    a2.set_ylabel('validation loss\n(lower is better)')
    a2.set_xlabel('optimizer steps')
    a2.set_title('The loss does nothing at all, and then it does everything',
                 fontsize=14, loc='left', pad=8)

    for ax in (a1, a2):
        ax.axvspan(0, PLATEAU_END, color=GRID, alpha=0.7, zorder=0)
    a2.annotate(f'{PLATEAU_END:,} steps — {mins[pi]:.0f} minutes — where nothing visibly happens.\n'
                f'A run that never learns looks exactly like this, forever.',
                xy=(PLATEAU_END, vals[pi]), xytext=(15_500, 5.7),
                color=INK2, fontsize=12, va='top',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.6))
    a2.annotate(f'the last third is worth {remaining:.2f} —\ninside the run-to-run noise',
                xy=(steps[-1], final), xytext=(43_000, 3.5),
                color=INK2, fontsize=12,
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.6))

    fig.suptitle('Why a run cannot be judged early', fontsize=17, fontweight='bold',
                 color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, '08-why-not-shorter')


# --------------------------------------------------------------------------------------------
def fig_scaling():
    """More cards finish a queue sooner. They never make one run shorter.

    The distinction matters because "buy more GPUs" is the reflex answer to "training is slow",
    and at this model size it is only half true. The curve is the wall-clock to clear one real
    twenty-job queue against the number of cards; the flat rule is the longest single job in it,
    which no amount of hardware moves.
    """
    # The twenty longest completed runs at the standard budget -- a realistic distribution of
    # job lengths rather than one particular evening. Worth being precise about, because the
    # first version of this called itself "a real twenty-job night" and then quietly started
    # including runs from a study that was still landing while the figure was being drawn. A
    # generated figure SHOULD move when the data moves; a caption that names a specific night
    # must not.
    jobs = sorted((r['seconds'] / 60 for r in f.results('*') if r['steps'] == 62_500),
                  reverse=True)[:20]
    longest = jobs[0]

    def makespan(n):
        """Longest-first onto n cards -- the ordering the scheduler actually uses."""
        cards = [0.0] * n
        for j in jobs:
            i = min(range(n), key=lambda k: cards[k])
            cards[i] += j
        return max(cards)

    ns = list(range(1, 25))
    span = [makespan(n) / 60 for n in ns]

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    ax.plot(ns, span, 'o-', color=C1, markersize=7)
    ax.axhline(longest / 60, color=C2, lw=2.2, ls=(0, (5, 4)))
    # Above the rule, not below it: below, the text sat on the dashes and both became unreadable.
    ax.text(4.5, longest / 60 + 0.3, 'the longest single job — no number of cards goes below this',
            ha='left', va='bottom', color=C2, fontsize=12, fontweight='bold')

    for n in (2, 10, 20):
        y = span[n - 1]
        ax.plot([n], [y], 'o', color=C1, markersize=13, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=3)
        ax.annotate(f'{n} cards\n{y:.1f} h', xy=(n, y), xytext=(n + 0.7, y + 0.9),
                    color=INK, fontsize=12, fontweight='bold')

    ax.set_xlabel('graphics cards working the queue')
    ax.set_ylabel('hours to clear the queue')
    ax.set_xticks([1, 5, 10, 15, 20, 24])
    ax.set_ylim(0, max(span) * 1.12)
    ax.set_title('Ten cards would not have made this ten times faster', pad=14)
    save(fig, '09-scaling-with-cards')


# --------------------------------------------------------------------------------------------
def fig_early_signal():
    """Why you cannot safely abandon a training run early, priced in GPU-hours.

    Two panels. Left: the feature everyone reaches for -- how far the loss has fallen -- with
    both outcomes overlaid. They overlap at every checkpoint, and the doomed runs are not even
    on the low side. Right: the cost of acting anyway. Every operating point is net negative,
    and the reason is structural rather than a tuning failure: only 12% of runs are doomed, a
    doomed run wastes at most its remaining time, and a false kill wastes a whole run.

    Drawn as a cost curve rather than an ROC because hours can be traded against a decision and
    an AUC cannot.
    """
    d = json.load(open(os.path.join(HERE, 'runs', 'early_signal.json'), encoding='utf-8'))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- left: the two populations, checkpoint by checkpoint --------------------------------
    sep = d['separation']
    steps = [s['step'] for s in sep]
    a1.fill_between(steps, [s['ok_min'] for s in sep], [s['ok_mean'] for s in sep],
                    color=C1, alpha=0.18, lw=0)
    a1.plot(steps, [s['ok_mean'] for s in sep], 'o-', color=C1, label='went on to learn')
    a1.plot(steps, [s['ok_min'] for s in sep], '--', color=C1, lw=1.6, alpha=0.8)
    a1.fill_between(steps, [s['bad_mean'] for s in sep], [s['bad_max'] for s in sep],
                    color=C2, alpha=0.18, lw=0)
    a1.plot(steps, [s['bad_mean'] for s in sep], 'o-', color=C2, label='never learned')
    a1.plot(steps, [s['bad_max'] for s in sep], '--', color=C2, lw=1.6, alpha=0.8)

    a1.set_xlabel('checkpoint (optimizer steps)')
    a1.set_ylabel('nats gained against an untrained model')
    a1.set_title('The obvious signal does not separate them', fontsize=14, pad=10)
    a1.legend(loc='upper left')
    # Low and centre-right: the only quarter of this panel no line passes through.
    a1.annotate('dashed = the worst survivor and the best failure.\nThey cross, at every '
                'checkpoint.',
                xy=(0.34, 0.06), xycoords='axes fraction', va='bottom',
                color=INK2, fontsize=11.5)

    # --- right: what acting on it would have cost -------------------------------------------
    pats = d['patience']
    x = [p['deadline'] * 100 for p in pats]
    a2.axhline(0, color=MUTED, lw=1.6)
    a2.plot(x, [p['saved_h'] for p in pats], 'o-', color=C3, label='saved (doomed runs cut short)')
    a2.plot(x, [-p['lost_h'] for p in pats], 'o-', color=C2, label='lost (good runs killed)')
    a2.plot(x, [p['net_h'] for p in pats], 'o-', color=C1, lw=3.2, label='net')
    best = max(pats, key=lambda p: p['net_h'])
    a2.annotate(f'best case is still\n{best["net_h"]:+.0f} GPU-hours',
                xy=(best['deadline'] * 100, best['net_h']),
                xytext=(best['deadline'] * 100 - 14, best['net_h'] - 24),
                ha='center', color=INK, fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.6))

    a2.set_xlabel('wait this % of the budget, then abandon if still flat')
    a2.set_ylabel('GPU-hours across our 105 runs')
    a2.set_title('Every operating point loses money', fontsize=14, pad=10)
    # Headroom above the zero line so the legend has somewhere to sit that no line reaches.
    a2.set_ylim(-78, 58)
    a2.legend(loc='upper left', fontsize=11)

    fig.suptitle('Early stopping does not pay at this scale — and here is the price',
                 fontsize=17, fontweight='bold', color=INK, y=1.01)
    fig.tight_layout()
    save(fig, '10-early-signal')




# --------------------------------------------------------------------------------------------
def fig_metric_validity():
    """Does the number we spent a term minimizing predict the number we care about?

    Two panels, one per task, pretraining loss on x and downstream score on y. The same 19
    checkpoints in both. The finding is the contrast: on topic classification the relationship is
    strong and orderly; on entity recognition it is entirely carried by three under-trained models
    at the right, and among the sixteen that actually worked it is flat.

    The shaded band is the region those three occupy. Drawing it is the whole point -- an
    aggregate correlation of -0.935 looks like the tightest result in the study until you can see
    that three points are holding it up.
    """
    rows = json.load(open(os.path.join(HERE, 'runs', 'downstream_correlation.json'),
                          encoding='utf-8'))
    CUT = 3.1                     # the gap between "trained" and "under-trained" in this sample

    def stats(g):
        x = [r['val_loss'] for r in g]
        y = [r['mean'] for r in g]
        return st.correlation(x, y) if len(g) > 2 else float('nan')

    panels = [('Topic classification', 'sib200', C1),
              ('Entity recognition', 'masakhaner', C2)]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for ax, (label, task, color) in zip(axes, panels):
        g = [r for r in rows if r['task'] == task]
        well = [r for r in g if r['val_loss'] < CUT]
        poor = [r for r in g if r['val_loss'] >= CUT]

        ax.axvspan(CUT, max(r['val_loss'] for r in g) + 0.3, color=GRID, alpha=0.75, zorder=0)
        ax.plot([r['val_loss'] for r in well], [r['mean'] for r in well], 'o',
                color=color, markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.6,
                label=f'trained ({len(well)})')
        ax.plot([r['val_loss'] for r in poor], [r['mean'] for r in poor], 'o',
                color=MUTED, markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.6,
                label=f'under-trained ({len(poor)})')

        r_all, r_well = stats(g), stats(well)
        ax.set_title(f'{label}\nall {len(g)}: r = {r_all:+.3f}      '
                     f'the {len(well)} trained: r = {r_well:+.3f}',
                     fontsize=14, pad=10)
        ax.set_xlabel('pretraining loss  (lower is better)')
        ax.set_ylabel('downstream score  (higher is better)')
        ax.legend(loc='lower left', fontsize=11)
        ax.set_ylim(0.42, 0.86)

    # "Flat", not "inverted". r = +0.303 at n = 16 is t = 1.19, p ~ 0.25 -- indistinguishable from
    # zero. Calling it a sign flip would assert an inversion the data cannot carry, which is
    # exactly the failure the panel beside this one is about. Patrick's catch.
    axes[1].annotate('these three carry the whole correlation.\nTake them out and it is flat.',
                     xy=(4.6, 0.55), xytext=(3.15, 0.70), color=INK, fontsize=12,
                     fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.8))

    fig.suptitle('Validation loss tells you a model is broken. On entity recognition it does '
                 'not tell you which working model is better.',
                 fontsize=16, fontweight='bold', color=INK, y=1.02)
    fig.tight_layout()
    save(fig, '11-metric-validity')




# --------------------------------------------------------------------------------------------
def fig_floors():
    """Why one task can be predicted from pretraining loss and the other cannot.

    Cell 5 of the board. It replaces figure 01 there, which went to Patrick -- that comparison is
    his and its selection rule is his sweep's to fix. This is the part that is ours.

    An earlier version of this figure drew the floor as a share of the achievable score and
    claimed the difference between those shares explained the task divergence. It does not: the
    shares are 57% and 52%, near enough identical, and the claim was wrong in the writeup, in this
    figure and in an email before anybody checked it.

    What actually separates the tasks is the VARIABILITY of the gain, not its size. Entity
    recognition hands every working model between 0.340 and 0.384 -- a large benefit that is
    nearly constant, varying by 13% of the smallest gain. Topic classification hands them 0.159 to
    0.301, varying by 90%. A near-constant benefit cannot be predicted from anything, which is why
    validation loss tracks one task and not the other. So the figure draws the BAND the trained
    models occupy rather than a single best.
    """
    corr = json.load(open(os.path.join(HERE, 'runs', 'downstream_correlation.json'),
                          encoding='utf-8'))
    rows = ft_api.results('*')
    out = []
    for label, task, steps in (('Topic classification\n(needs meaning)', 'sib200', 1056),
                               ('Entity recognition\n(needs surface form)', 'masakhaner', 2150)):
        # n_train_requested is None means the FULL split, and that filter is load-bearing now.
        # The label-quantity experiment runs the same untrained control at 701 and 2,000 labels,
        # and without this the figure picked up the 2,000-label cell -- drawing the NER floor as
        # 0.465 instead of 0.626 under a caption about the full-split band. A subsampled control
        # is a different experiment wearing the same model name.
        floor = max((r for r in rows if r.get('task') == task and r.get('steps') == steps
                     and 'yor_random_init' in r['model']
                     and r.get('n_train_requested') is None),
                    key=lambda r: r['mean'])['mean']
        # The 16 that actually trained -- same cut as figure 11, for the same reason.
        y = [r['mean'] for r in corr if r['task'] == task and r['val_loss'] < 3.1]
        out.append((label, floor, min(y), max(y)))

    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    for i, (label, floor, lo, hi) in enumerate(out):
        ax.bar(i, floor, color=MUTED, width=0.46)
        ax.bar(i, hi - lo, bottom=lo, color=C1, width=0.46)
        # The dead space between the floor and the worst trained model: benefit every model gets.
        ax.bar(i, lo - floor, bottom=floor, color=C1, width=0.46, alpha=0.28)

        ax.text(i, floor / 2, f'floor\n{floor:.3f}', ha='center', va='center',
                color=SURFACE, fontsize=13, fontweight='bold')
        ax.text(i, floor + (lo - floor) / 2, f'+{lo-floor:.3f}\nevery model gets this',
                ha='center', va='center', color=INK2, fontsize=11.5)
        ax.text(i, lo + (hi - lo) / 2 + 0.005, f'{hi-lo:.3f}', ha='center', va='center',
                color=SURFACE, fontsize=13, fontweight='bold')
        ax.annotate(f'{lo:.3f} – {hi:.3f}', xy=(i + 0.26, (lo + hi) / 2),
                    xytext=(i + 0.34, (lo + hi) / 2), va='center', color=INK, fontsize=12,
                    fontweight='bold')

    ax.plot([], [], 's', color=MUTED, markersize=11, label='an untrained model already scores')
    ax.plot([], [], 's', color=C1, alpha=0.28, markersize=11,
            label='the gain EVERY trained model gets')
    ax.plot([], [], 's', color=C1, markersize=11,
            label='the band 16 trained models actually span')
    ax.set_xticks(range(len(out)))
    ax.set_xticklabels([o[0] for o in out], fontsize=13, color=INK)
    ax.set_ylabel('score  (higher is better)')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.5, 1.75)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(axis='x', visible=False)
    ax.set_title('A benefit that every model gets equally cannot be predicted', pad=14)
    fig.text(0.5, -0.04,
             'Entity recognition hands every working model roughly the same large gain — the band '
             'is 0.044 wide.\nTopic classification spreads them over 0.143. That, not the floor, '
             'is why loss predicts one task and not the other.',
             ha='center', color=INK2, fontsize=12)
    save(fig, '12-floors')




# --------------------------------------------------------------------------------------------
def fig_how_many_seeds():
    """The bar our own rule was set below, and by how much.

    Week 4 of the board. The project's working rule was "treat a difference smaller than the
    cell's own seed spread as no difference" -- a threshold of 1.0x the spread. That rule is
    sound in one direction and silent in the other, and we had been reading it as symmetric.

    The curve is what a two-sample t-test actually demands at alpha = 0.05: at three seeds per
    arm a difference has to be 2.27x the spread before it is distinguishable from nothing. Our
    rule sat at 1.0, which is the shaded band -- every difference that lands there is one we
    would have called real and could not have.

    The points are our own claims. The tokenizer penalty sits squarely in the band, which is how
    claims_audit.py found it, and is why six more runs are queued rather than a hedge written.
    """
    from scipy.stats import t as tdist

    ns = list(range(2, 21))
    need = [tdist.ppf(0.975, 2 * n - 2) * math.sqrt(2 / n) for n in ns]

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.fill_between(ns, 1.0, need, color=C2, alpha=0.16, lw=0)
    ax.plot(ns, need, 'o-', color=C2, markersize=7,
            label='what a t-test actually needs (α = 0.05)')
    ax.axhline(1.0, color=C1, lw=2.6, ls=(0, (5, 4)),
               label='our rule: "bigger than the seed spread"')

    # Upper middle: the only region neither the curve, the rule line, nor a claim label crosses.
    ax.annotate('everything in this band we would\nhave called real, and could not have',
                xy=(12.5, 2.9), ha='center', color=C2, fontsize=12.5, fontweight='bold')

    # Our own claims, placed where they actually sit.
    # Both labels sit to the RIGHT of their point and never below it: the lower one ran off the
    # axis and collided with the tick labels.
    pts = [(3, 0.144 / 0.105,
            'tokenizer penalty, 0.144 bits/char —\ninside the band, which is how\nthe audit caught it', 0.60),
           (3, 0.080 / 0.185,
            '16× more text: an ABSENCE claim,\nso below the bar is the point', 0.28)]
    for n, ratio, label, dy in pts:
        ax.plot(n, ratio, 'o', color=INK, markersize=13, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(label, xy=(n, ratio), xytext=(n + 2.2, ratio + dy),
                    color=INK, fontsize=11.5, fontweight='bold', va='center',
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.6))

    ax.set_xlabel('seeds per arm')
    ax.set_ylabel('difference, as a multiple of the seed spread')
    ax.set_xticks([2, 3, 4, 5, 6, 8, 10, 12, 16, 20])
    ax.set_ylim(0, 4.6)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_title('Three seeds tells you what is definitely noise, not what is definitely real',
                 pad=14)
    fig.text(0.5, -0.03,
             'The rule is sound in one direction and silent in the other. Below the spread is '
             'reliably nothing;\nslightly above it is not reliably something — and that is where '
             'our own headline number sat.',
             ha='center', color=INK2, fontsize=12)
    save(fig, '13-how-many-seeds')




# --------------------------------------------------------------------------------------------
def fig_speedup():
    """Where the 2.07x came from, and why the two halves are different kinds of win.

    Panel 1. A waterfall rather than a before/after pair, because the point is that the speedup
    decomposes into two independent changes with different characters: a batch size that was
    simply too small, and a second card the notebook had never used. The first is efficiency --
    less GPU-time for the same work. The second is only wall-clock: the same GPU-minutes, spent
    in parallel. Conflating them is how a project claims 2x efficiency when it bought 1.3x.

    Numbers from report 03's A/B, four matched cells, same recipe both sides.
    """
    stages = [('the notebook\nbatch 64, one card', 25.2, MUTED),
              ('batch 128\n1.32× less GPU-time', 19.1, C1),
              ('both cards\nsame GPU-time', 12.2, C3)]

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    xs = range(len(stages))
    ax.bar(xs, [s[1] for s in stages], color=[s[2] for s in stages], width=0.55)
    for i, (label, v, _) in enumerate(stages):
        ax.text(i, v + 0.5, f'{v:.1f} min', ha='center', color=INK, fontsize=15,
                fontweight='bold')

    # The two arrows are the story: one shrinks the work, the other splits it.
    ax.annotate('', xy=(0.72, 19.1), xytext=(0.28, 25.2),
                arrowprops=dict(arrowstyle='->', color=INK2, lw=2.2))
    ax.text(0.5, 23.4, '−1.32×\nefficiency', ha='center', color=INK2, fontsize=12,
            fontweight='bold')
    ax.annotate('', xy=(1.72, 12.2), xytext=(1.28, 19.1),
                arrowprops=dict(arrowstyle='->', color=INK2, lw=2.2))
    ax.text(1.5, 17.0, '−1.57×\nparallelism', ha='center', color=INK2, fontsize=12,
            fontweight='bold')

    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in stages], fontsize=12, color=INK, linespacing=1.4)
    ax.set_ylabel('wall-clock for the same four cells (minutes)')
    ax.set_ylim(0, 29)
    ax.grid(axis='x', visible=False)
    ax.set_title('2.07× — and only 1.32× of it is efficiency', pad=14)
    fig.text(0.5, -0.06,
             'Both changes were available on day one and neither is clever. The batch was simply '
             'too small, and the\nsecond card had never been used. Utilization went from one card '
             'busy and one idle to 91% and 93%.',
             ha='center', color=INK2, fontsize=12)
    save(fig, '14-where-the-speedup-came-from')


# --------------------------------------------------------------------------------------------
def fig_pipeline():
    """What a language-model run is actually made of, in wall-clock.

    Panel 2. Students picture "training a model" as one activity. It is a pipeline, and the
    proportions are so lopsided that they decide the shape of the whole factory: preparation is
    well under a minute and training is an hour and a half, which is exactly why everything cheap
    belongs in a notebook and everything expensive belongs in a queue.

    Dots on a log axis, not bars -- the same correction figure 06 needed. The stages differ by
    more than two orders of magnitude so the axis has to be logarithmic, and a bar encodes its
    value as a length from a baseline, which on a log axis is meaningless. The first version drew
    bars and clipped the two shortest stages out of the plot entirely.

    Measured on the real Yoruba corpus by pipeline_bench.py: 80,000 documents, 260M characters.
    The two CPU stages are reported for the FULL corpus rather than the timed sample, which is
    what "a run" actually costs.
    """
    b = json.load(open(os.path.join(HERE, 'runs', 'pipeline_bench.json'), encoding='utf-8'))
    by = {s['name']: s for s in b['stages']}

    def find(prefix):
        return next(s for k, s in by.items() if k.startswith(prefix))

    tok = find('2.')['seconds']
    enc = find('3.').get('extrapolated_full_s') or find('3.')['seconds']
    load = find('5.')['seconds']
    small = find('6. train step, 33.8M')['hours_for_62500_steps'] * 3600
    big = find('6. train step, 98M')['hours_for_62500_steps'] * 3600

    stages = [('train the tokenizer', tok, C4, 'CPU'),
              ('encode the whole corpus', enc, C4, 'CPU'),
              ('load it onto the card', load, C4, 'once'),
              ('PRETRAIN, 33.8M model', small, C1, '62,500 steps'),
              ('PRETRAIN, 98M model', big, C2, '62,500 steps')]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for i, (label, s, color, note) in enumerate(stages):
        y = len(stages) - 1 - i
        ax.plot([0.5, s], [y, y], color=color, lw=2.4, alpha=0.4, zorder=1)
        ax.plot(s, y, 'o', color=color, markersize=15, markeredgecolor=SURFACE,
                markeredgewidth=2, zorder=2)
        txt = f'{s:.0f} s' if s < 90 else f'{s/60:.0f} min'
        ax.text(s * 1.35, y, txt, va='center', color=INK, fontsize=14, fontweight='bold')
        ax.text(s * 1.35, y - 0.30, note, va='center', color=MUTED, fontsize=10.5)

    ax.set_xscale('log')
    ax.set_xlim(0.5, big * 12)
    ax.set_xticks([1, 10, 60, 600, 3600])
    ax.set_xticklabels(['1 s', '10 s', '1 min', '10 min', '1 hour'], fontsize=12)
    ax.minorticks_off()
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s[0] for s in reversed(stages)], fontsize=12.5, color=INK)
    ax.set_ylim(-0.7, len(stages) - 0.3)
    ax.grid(axis='y', visible=False)
    ax.set_title('Preparation is under a minute. Training is an hour and a half.',
                 pad=34, loc='left')
    ax.text(0, 1.04, 'every stage of one run, measured end to end on the real corpus',
            transform=ax.transAxes, color=INK2, fontsize=12.5, va='bottom')
    fig.text(0.5, -0.05,
             'Which is the whole reason the factory is shaped the way it is: everything cheap '
             'happens interactively in a\nnotebook, and everything expensive goes into a queue '
             'that runs while nobody is watching.',
             ha='center', color=INK2, fontsize=12)
    save(fig, '15-what-a-run-is-made-of')


def fig_tokenizer_lottery():
    """Is the large vocabulary a cost, or a coin flip?

    Cell 7 of the board, and the figure that replaced a bar chart. The bar chart was honest about
    the number it drew and dishonest about the finding, because the finding is not a difference
    between two heights.

    Report 08 carried a 0.144 bits-per-character penalty from three seeds a side. Six seeds --
    fixed in advance from the power calculation and written into the script before the runs
    started -- did not confirm it. The gap SHRANK to 0.059 with p = 0.37, and the two arms
    interleave: three of the six large-vocabulary runs land below the small vocabulary's median.
    A mean and an error bar would show two overlapping blobs and invite the reader to conclude
    "no difference", which is also wrong.

    What is there is a difference in SPREAD -- 0.145 against 0.037, F = 15.1, p = 0.0098 -- and
    the large-vocabulary arm is not merely wider, it is in two clusters. So the figure draws all
    twelve runs as individual points. That is the only presentation where the reader sees the
    thing that is actually true: choosing the large vocabulary does not cost you a fixed amount,
    it decides how much of a gamble the run is.

    The same shape holds downstream on ENTITIES and not on topic, which is noted on the figure
    rather than drawn, and computed by _downstream_spread() rather than stated here. This docstring
    used to assert "4.0x wider on topic classification, 7.7x on entities": both were three-seed
    values, the topic one was never significant (p = 0.115) and it reversed outright at four seeds.
    The caption was fixed to compute them and this paragraph was not, which is the same defect one
    layer out -- a figure whose prose remembers what its code has stopped believing.
    """
    # Selection comes from the study that pre-registered it, not from a glob written here. A glob
    # over `swap_yor_xlmr_*` picks up seven runs for a six-seed arm, because two studies named the
    # same cell two ways and a third built a fourth seed it needed for its own comparison. That
    # turned n=6 into n=7 and moved this figure's headline from 0.059 to 0.037 -- a pre-registered
    # sample size quietly growing, which is the one thing pre-registration is meant to prevent.
    import study_tokenizer_seeds as tk

    def arm(spec):
        recs, _ = tk.arm_records(spec)
        return sorted(tk.bpc(r, spec['corpus']) for r in recs.values())

    big = arm(next(a for a in tk.ARMS if a['corpus'] == 'yor_xlmr'))
    small = arm(next(a for a in tk.ARMS if a['corpus'] == 'yor'))
    _, p_loc = stats_ttest(big, small)
    F = (st.stdev(big) ** 2) / (st.stdev(small) ** 2)
    # Every number in the caption is computed, including the seed count and both p-values. An
    # earlier version hard-coded "Six seeds a side" and "p = 0.0098" while computing the means
    # from whatever records were on disk, so on a checkout without the six-seed data it drew
    # three seeds a side under a caption asserting the six-seed conclusion. A figure that
    # recomputes its data and quotes its caption from memory is worse than one that hard-codes
    # both, because it looks live.
    from scipy import stats as _s
    p_var = 2 * min(_s.f.cdf(F, len(big) - 1, len(small) - 1),
                    1 - _s.f.cdf(F, len(big) - 1, len(small) - 1))
    down = _downstream_spread()

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    for i, (label, vals, color) in enumerate((('250k vocabulary\n(XLM-R’s)', big, C2),
                                              ('16k vocabulary\n(ours)', small, C1))):
        # A little horizontal jitter would hide that these are exact values, so points are laid
        # out on a fixed lattice instead -- deterministic, and it keeps the SVG stable.
        for j, v in enumerate(vals):
            ax.plot([i + (j - (len(vals) - 1) / 2) * 0.055], [v], 'o', color=color,
                    markersize=13, zorder=3)
        ax.plot([i - 0.20, i + 0.20], [st.mean(vals)] * 2, '-', color=INK2, lw=2.0, zorder=4)
        ax.text(i - 0.24, st.mean(vals), f'mean {st.mean(vals):.3f}', va='center', ha='right',
                fontsize=12, color=INK2)
        # sd goes ABOVE its column, inside the axes. Below the axis it landed on the tick label.
        ax.text(i - 0.30, 0.99, f'sd {st.stdev(vals):.3f}', ha='right', va='top', fontsize=14,
                color=color, fontweight='bold', transform=ax.get_xaxis_transform())

    # The band the small arm occupies: the large arm straddles it rather than sitting above it.
    # Label on the LOWER edge -- the upper edge sits a hair under the large arm's mean line.
    ax.axhspan(min(small), max(small), color=C1, alpha=0.10, zorder=1)
    ax.text(-0.45, min(small), ' the 16k range', va='bottom', ha='left',
            fontsize=11.5, color=MUTED)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['250k vocabulary\n(XLM-R’s)', '16k vocabulary\n(ours)'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('bits per character  (lower is better)')
    ax.set_title('Not a penalty — a lottery', pad=14)
    below = sum(1 for x in big if x < st.median(small))
    fig.text(0.5, -0.10,
             f'{len(big)} seeds a side, the sample size fixed in advance. The MEANS do not '
             f'separate: {st.mean(big)-st.mean(small):+.3f} bits/char, p = {p_loc:.2f}, and '
             f'{below} of the {len(big)}\n250k runs land below the 16k median. The SPREADS do: '
             f'F = {F:.1f}, p = {p_var:.4f}. {down}',
             ha='center', fontsize=11.5, color=INK2)
    save(fig, '17-tokenizer-lottery')


def residual_permutation(a, b):
    """Two-sided exact permutation test for a difference in SPREAD, on centered residuals.

    Panel 11 of the top board needs this and it existed only in an email, which is the one rule
    this project has about where numbers live. It is here so it regenerates and so a test can pin
    it.

    Two design points, both of which were got wrong once already and are the whole reason this is
    a named function rather than four lines inline.

    CENTER FIRST. Permuting raw scores between two arms whose MEANS differ inflates the spread of
    every reshuffled group, so the real grouping looks unusually tight and a location difference
    is reported as a variance finding. On topic that returns p = 0.057 against the F-test's 0.553
    -- a significant variance difference conjured out of a 0.144 gap, on the one task where the
    point is that consistency does not differ. Subtracting each arm's own mean removes it. The
    F-test has that property built in; a permutation test has to be given it.

    The statistic must be symmetric under swapping the arms. The first version here used
    |sd(g1)/sd(g2) - 1|, which is not: a ratio of 0.7 scores 0.30 while its reciprocal 1.43 scores
    0.43, so the test is quietly directional. It returned 1/70 on entities -- and 1/70 is
    unreachable for any two-sided test at four a side, because every split is enumerated alongside
    its complement and both always count, which floors a symmetric statistic at 2/70. Patrick
    identified the sidedness from that number alone. |log(sd ratio)| is symmetric on the ratio
    scale and is what a variance ratio deserves, being the scale the F distribution lives on.

    Returns (p, hits, total, floor). The floor is 2/C(2n, n) and is worth printing beside p for
    the same reason it is printed beside the top row's 0.029: at four seeds a side this test
    cannot go below 0.029 however cleanly the spreads separate.
    """
    ra = [x - st.mean(a) for x in a]
    rb = [x - st.mean(b) for x in b]
    pool = ra + rb
    n = len(ra)

    def stat(g1, g2):
        return abs(math.log(st.stdev(g1) / st.stdev(g2)))

    obs = stat(ra, rb)
    hits = total = 0
    for idx in itertools.combinations(range(len(pool)), n):
        g1 = [pool[i] for i in idx]
        g2 = [pool[i] for i in range(len(pool)) if i not in idx]
        total += 1
        # The tolerance matters: the observed split and its complement are both enumerated and
        # must both count, and in floating point they are equal only to within rounding.
        if stat(g1, g2) >= obs - 1e-12:
            hits += 1
    return hits / total, hits, total, 2 / math.comb(len(pool), n)


def _hardware_rate(rows):
    """The one rate a pile of sittings for one machine and preset reduces to.

    The order of these three rules is the whole function. A realistic-loop row beats a bare-step
    one and they are never blended, because they time different work -- the bare step omits
    batch-building, clipping and the host sync, and reads 2.7% high here and 0.7% high on a Mac.
    A timed row beats a burst. Only then, the median across comparable sittings.

    Written once and called from three places on purpose. fig_cost and fig_hardware each had
    their own copy for a day, and the two disagreed by $1.08 immediately -- one medianed all four
    of the A100's rows, the other picked the best method first. Two numbers for one machine, from
    one file, three inches apart on the same board.
    """
    rank = {'realistic-loop': 0, 'bare-step': 1}
    method = min((r.get('method', 'bare-step') for r in rows), key=lambda m: rank.get(m, 2))
    use = [r for r in rows if r.get('method', 'bare-step') == method]
    use = [r for r in use if r.get('timed_seconds')] or use
    # A run that ended faster than it started never settled, and its mean is not a rate.
    #
    # throttle is first third over last third, so below 1.0 means the machine sped up while being
    # measured. Small values are noise -- the A100 and the L4 sit at 0.98-1.00 -- but the Surface
    # laptop's first 98M sitting came back at 0.87: its last third was 15% faster than its first,
    # still climbing when the window closed. Averaged whole, it read 25,038 against 28,027 from
    # the sitting immediately after it, and the two would have medianed to a number neither run
    # measured. The cause is mundane and worth knowing: that was the first run of a cold session,
    # so the fans had not spun up yet. On a laptop the FIRST reading is the bad one, and not
    # because the card is hot.
    #
    # Dropping it is not cherry-picking, because the criterion is stated in advance and the data
    # corroborates it: with the unsettled row out, the battery penalty comes to 15.4% on one
    # preset and 14.9% on the other. With it in, they disagree -- 15.4% against 10.1%.
    settled = [r for r in use if (r.get('throttle') or 1.0) >= 0.95]
    use = settled or use
    rates = sorted(r['tokens_per_s'] for r in use)
    return dict(rate=st.median(rates), method=method, sittings=len(use),
                lo=rates[0], hi=rates[-1],
                near=min(use, key=lambda r: abs(r['tokens_per_s'] - st.median(rates))))


def _rent_the_project():
    """What the cheapest measured Colab tier would charge for this project's whole term of work.

    Rescales the project's own card time by each tier's throughput against the workstation's,
    separately per model shape because the 98M half is the expensive half and the tiers do not
    scale identically across the two. A tier whose rows carry no billing fields is skipped rather
    than guessed at.

    The same arithmetic fig_hardware's cost table runs, and now the same code: both reduce their
    sittings through _hardware_rate(). test_board_numbers still asserts they agree, because the
    guarantee is worth keeping even once the duplication is gone.
    """
    rows = json.load(open(os.path.join(HERE, 'runs', 'hardware.json'), encoding='utf-8'))
    ws_hours = _workstation_hours_by_preset()
    best = {}
    for r in rows:
        if r.get('compute_units_per_hour') and r.get('usd_per_compute_unit'):
            best.setdefault(r['device'], {}).setdefault(r['preset'], []).append(r)
    ws = {r['preset']: r for r in rows
          if 'PRO 6000' in r['device'] and r.get('method') == 'realistic-loop'}
    quotes = []
    for presets in best.values():
        if not all(p in presets for p in ws_hours):
            continue
        rate = {p: _hardware_rate(presets[p])['rate'] for p in ws_hours}
        hrs = sum(ws_hours[p] * ws[p]['tokens_per_s'] / rate[p] for p in ws_hours)
        one = presets[next(iter(ws_hours))][0]
        quotes.append(hrs * one['compute_units_per_hour'] * one['usd_per_compute_unit'])
    return min(quotes)


def _downstream_spread():
    """One sentence on whether the same variance gap appears downstream, computed not recalled.

    It appears on entities and not on topic, and the caption used to assert both. The topic ratio
    was 4.0x at three pretraining seeds -- never significant, p = 0.115 -- and reversed outright
    at four when our own arm produced a weak seed. Quoting a ratio without testing it is the
    mistake this whole board is about, so the figure now runs the test.
    """
    from scipy import stats as _s
    p = os.path.join(HERE, 'runs', 'swap_downstream.json')
    if not os.path.exists(p):
        return 'Downstream comparison not on disk.'
    rows = json.load(open(p, encoding='utf-8'))
    out = []
    for task, stage in (('topic', 'test'), ('entities', 'ner')):
        a = [x['mean'] for x in rows if x.get('stage') == stage
             and x['arm'] == "XLM-R's vocabulary"]
        b = [x['mean'] for x in rows if x.get('stage') == stage and x['arm'] == 'our vocabulary']
        if len(a) < 3 or len(b) < 3:
            continue
        F = st.stdev(a) ** 2 / st.stdev(b) ** 2
        pv = 2 * min(_s.f.cdf(F, len(a) - 1, len(b) - 1), 1 - _s.f.cdf(F, len(a) - 1, len(b) - 1))
        out.append(f'{task} {st.stdev(a)/st.stdev(b):.1f}× (p = {pv:.3f})')
    return 'Downstream, over ' + str(len(a)) + ' pretraining seeds: ' + ', '.join(out) + '.' \
        if out else 'Downstream comparison unavailable.'


def stats_ttest(a, b):
    """Welch, kept separate so the figure module does not import scipy at the top level."""
    from scipy import stats as _s
    return _s.ttest_ind(a, b, equal_var=False)


def fig_lr_transfer():
    """Does a learning rate tuned on one language transfer to another?

    Cell 8 of the board, and the figure that cell had been waiting on. The obvious way to draw
    this is five curves on one axis, and it is the wrong way: the reader compares heights, and the
    heights are not comparable. Each language has its own corpus, its own vocabulary and its own
    intrinsic difficulty, so Nyanja sitting below Hausa says nothing about learning rates. Small
    multiples put the comparison where it belongs -- the SHAPE of each curve, and where its
    minimum falls.

    The finding is stronger than "the best rate differs", which would only mean somebody leaves a
    little performance on the table. Three languages have their best rate at 7e-4. At that exact
    rate Igbo does not merely do worse, it collapses: 2.889 at 5e-4 against 5.638 at 7e-4, and it
    stays collapsed at every higher rate. So the risk of transferring a tuned setting is not a
    slightly worse model, it is a wasted night that looks like a result.

    Failed cells are drawn as hollow markers, not as tall points on the line. A collapsed run and
    a merely-poor run are different kinds of thing, and joining them with a single line invites
    reading the collapse as the top of a smooth curve.
    """
    rows = [r for r in json.load(open(os.path.join(HERE, 'runs', 'lr_transfer.json'),
                                      encoding='utf-8')) if 'val_loss' in r]
    by = {}
    for r in rows:
        by.setdefault(r['lang'], {}).setdefault(r['lr'], []).append(r['val_loss'])

    # A cell counts as failed when it lands more than 1.5 nats above what the SAME language
    # reaches at its own best rate -- a per-language test, for the same reason the seed-spread
    # rule is a per-cell property rather than a constant. 1.5 nats is not a close call at this
    # scale: the whole usable range within a language spans about 0.4.
    #
    # The first version of this figure used a threshold halfway to random-guessing loss, and it
    # was too lenient to catch the thing the figure exists to show -- Igbo's collapsed cells came
    # out as filled markers joined by a line, which is precisely what the docstring above says
    # not to do. Worth leaving the reason here: the failures are far from THIS language's best
    # and nowhere near random, so a threshold anchored on random cannot see them.
    FAIL_MARGIN = 1.5
    NAMES = {'hau': 'Hausa', 'ibo': 'Igbo', 'nya': 'Nyanja', 'swh': 'Swahili', 'yor': 'Yoruba'}
    langs = sorted(by, key=lambda l: (l != 'ibo', l))     # Igbo first: it is the point
    fig, axes = plt.subplots(1, len(langs), figsize=(17.5, 5.0), sharey=True)

    for ax, lang in zip(axes, langs):
        seeds = {lr: sorted(v) for lr, v in sorted(by[lang].items())}
        cells = {lr: st.mean(v) for lr, v in seeds.items()}
        best_lr = min(cells, key=cells.get)
        best = cells[best_lr]
        cutoff = best + FAIL_MARGIN
        color = C2 if lang == 'ibo' else C1

        # Three cells in this grid are SPLIT -- one seed trains and the other collapses, so their
        # mean describes no run that happened. Yoruba at 1e-3 is 3.326 and 5.540; drawing 4.433
        # as a point on a curve is the same error this project corrected in the clipping ladder
        # and in XLM-R's topic cell. So every seed is drawn, the line joins only cells where all
        # seeds trained, and a split cell gets a vertical tie between its two outcomes and no
        # mean marker at all.
        kind = {}
        for lr, v in seeds.items():
            n_ok = sum(1 for x in v if x < cutoff)
            kind[lr] = 'ok' if n_ok == len(v) else ('bad' if n_ok == 0 else 'split')

        line = [(lr, cells[lr]) for lr in seeds if kind[lr] == 'ok']
        if line:
            ax.plot([p[0] for p in line], [p[1] for p in line], '-', color=color, zorder=3)

        for lr, v in seeds.items():
            if kind[lr] == 'split':
                ax.plot([lr, lr], [min(v), max(v)], '-', color=color, lw=1.4, alpha=0.55,
                        zorder=2)
            for x in v:
                filled = x < cutoff
                ax.plot([lr], [x], 'o', markersize=7.5, zorder=3,
                        color=color if filled else SURFACE,
                        mec=color, mew=0 if filled else 2.0)

        ax.plot([best_lr], [best], 'o', color=color, markersize=14, zorder=4)
        n_split = sum(1 for k in kind.values() if k == 'split')
        if n_split:
            ax.text(0.96, 0.955, f'{n_split} cell{"s" if n_split > 1 else ""} split\nby seed',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10.5, color=INK2)

        # Fixed corner, so it can never land on a tick label or on the curve.
        ax.text(0.04, 0.04, f'best {best:.2f}\nat {best_lr:g}', transform=ax.transAxes,
                ha='left', va='bottom', fontsize=11.5, color=INK)

        ax.set_xscale('log')
        ax.set_title(NAMES.get(lang, lang), color=C2 if lang == 'ibo' else INK)
        ax.set_xlabel('learning rate  ($\\times 10^{-4}$)')
        # Ticks at the rates actually swept. Matplotlib's log minor ticks put five overlapping
        # labels under each panel here, which is unreadable at any size.
        rates = sorted(cells)
        ax.set_xticks(rates, minor=False)
        ax.set_xticklabels([f'{r*1e4:g}' for r in rates], fontsize=11)
        ax.set_xticks([], minor=True)
        # The rate three of the five settle on, so the eye can find it in every panel.
        ax.axvline(7e-4, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=1)

    axes[0].set_ylabel('validation loss')
    axes[0].set_ylim(2.3, 7.0)
    fig.suptitle('A rate tuned on one language is not a rate for the next', y=1.04,
                 fontsize=17, fontweight='bold', color=INK)
    fig.text(0.5, -0.13,
             'Every seed is drawn. Dashed line: 7e-4 — the best rate for Hausa, Nyanja and '
             'Swahili, and the rate at which Igbo collapses,\nas it does at every rate above it. '
             'Hollow markers are runs that failed rather than runs that merely scored poorly; a '
             'vertical tie\nmarks a cell whose two seeds did different things, where the mean '
             'describes neither. 12,000 steps, 16M tokens.',
             ha='center', fontsize=11.5, color=INK2)
    save(fig, '16-lr-transfer')


# --------------------------------------------------------------------------------------------
def fig_label_quantity():
    """The decisive experiment: is NER flat because of the task, or because it has ten times
    the labels? Panel 9 of the top board, and the one panel that had no figure.

    A slope chart, because the question is not what any model scores -- it is whether the models
    move APART as labels are removed. Sixteen lines, three label counts, and the eye reads the
    fanning directly. It also makes the study's central trap structural rather than a caveat: a
    slope chart cannot be drawn over a model that is missing a level, so the matched set is forced
    by the form of the chart. A range grows with the number of models in it, and the printed
    verdict of this experiment once reversed because five models joined the set.

    Neither the band membership nor the verdict is re-derived here. Both come from
    study_label_quantity -- band_models() for the set, spread() for the statistics, reading() for
    the decision -- so this figure cannot claim something the study's own rule does not. A figure
    holding its own copy of a model set is exactly the defect that reversed the verdict once.

    The right panel plots between-model sd rather than the range, for the reason the study's
    docstring opens with: SIB-200's figure is over sixteen models, and only a statistic that does
    not grow with set size may be held against an outside constant.
    """
    S = label_quantity
    with open(os.path.join(HERE, 'runs', 'label_quantity.json'), encoding='utf-8') as fh:
        rows = [r for r in json.load(fh) if 'mean' in r]

    band_all, _dropped = S.band_models()
    trained = {slug for slug, _, ok in band_all if ok}

    # Band cells only. mmBERT and the untrained floor are context arms, not points on this axis.
    by_level = {}
    for r in rows:
        if r.get('kind') == 'band' and r['model_slug'] in trained:
            by_level.setdefault(r['n_train_requested'], {})[r['model_slug']] = r
    if not by_level:
        raise SystemExit('no band cells in runs/label_quantity.json')

    # The models carrying EVERY level. A seventeenth has a 701 cell and no 2,000 cell, so the
    # dose-response -- which compares the levels with each other -- only means anything here.
    matched = set.intersection(*(set(v) for v in by_level.values()))
    full = S.full_data_band(matched)

    # The full split's label count, read off the records rather than typed. It is the one number
    # on this chart that is a property of the dataset rather than of the experiment.
    n_full = max(r['n_train'] for r in ft_api.results('*', task=S.TASK, lang=S.LANG)
                 if r.get('steps') == S.STEPS and r.get('lr') == S.BAND_LR
                 and not r.get('n_train_requested') and r['model_slug'] in matched)

    levels = [(n_full, full)]
    for n in sorted(by_level, reverse=True):
        levels.append((n, S.spread([by_level[n][slug] for slug in matched])))

    xs = list(range(len(levels)))
    # The board's own model, so a reader can find it inside the band. Identity, not rank -- the
    # highlight does not move if some other checkpoint happens to come top.
    HERO = 'yor-64M-62.5k-s0'

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.4),
                             gridspec_kw={'width_ratios': [1.5, 1.0]})
    ax = axes[0]
    for slug in sorted(matched):
        ys = [stats['models'][slug] for _, stats in levels]
        hero = slug == HERO
        ax.plot(xs, ys, '-o',
                color=C2 if hero else C1,
                alpha=1.0 if hero else 0.38,
                lw=2.8 if hero else 1.3,
                markersize=8 if hero else 4,
                zorder=4 if hero else 2)

    # The band itself, drawn once per level beside the cloud rather than inferred from it.
    for x, (_, stats) in zip(xs, levels):
        ax.plot([x + 0.19] * 2, [stats['lo'], stats['hi']], '-', color=INK2, lw=2.4, zorder=3)
        # At the TOP of the rule, not its midpoint. The midpoint of the first two rules lands
        # inside the descending cloud, and a bold number over sixteen thin lines is unreadable.
        ax.text(x + 0.19, stats['hi'] + 0.0018, f'{stats["range"]:.3f}',
                va='bottom', ha='center', fontsize=11.5, color=INK, fontweight='bold')

    # Explicit, with headroom for those labels: autoscale fits the DATA, and matplotlib does not
    # know a text annotation sits above the highest point.
    seen_vals = [v for _, stats in levels for v in stats['models'].values()]
    span = max(seen_vals) - min(seen_vals)
    ax.set_ylim(min(seen_vals) - 0.07 * span, max(seen_vals) + 0.11 * span)

    ax.plot([], [], '-o', color=C2, lw=2.8, markersize=8, label='our 33.8M model')
    ax.plot([], [], '-o', color=C1, alpha=0.38, lw=1.3, markersize=4,
            label=f'the other {len(matched) - 1} from-scratch models')
    ax.plot([], [], '-', color=INK2, lw=2.4, label='the band at that label count')
    ax.legend(loc='lower left', fontsize=11)

    ax.set_xticks(xs)
    ax.set_xticklabels([f'{n:,}' for n, _ in levels], fontsize=13.5, color=INK)
    for x, (n, _) in zip(xs, levels):
        sub = 'full split' if n == n_full else ('= SIB-200\'s count' if n == min(
            lv[0] for lv in levels) else 'one rung between')
        ax.text(x, -0.075, sub, ha='center', va='top', fontsize=11, color=MUTED,
                transform=ax.get_xaxis_transform())
    ax.set_xlim(-0.35, len(levels) - 0.5)
    ax.set_xlabel('MasakhaNER training sentences', labelpad=26)
    ax.set_ylabel('entity F1  (higher is better)')
    ax.set_title('Every model loses ground. Only at 701 do they separate.', pad=12, fontsize=15)

    # -- the statistic the rule actually decides on ------------------------------------------
    ax2 = axes[1]
    sds = [stats['between_sd'] for _, stats in levels]
    ax2.bar(xs, sds, color=C1, width=0.56)
    ax2.axhline(S.SIB_BETWEEN_SD, ls=(0, (5, 4)), lw=1.9, color=C2, zorder=3)
    ax2.text(len(levels) - 0.5, S.SIB_BETWEEN_SD - 0.0016,
             f'topic classification: {S.SIB_BETWEEN_SD:.4f}',
             ha='right', va='top', fontsize=11.5, color=C2, fontweight='bold')

    base_sd = levels[0][1]['between_sd']
    for x, (n, stats) in zip(xs, levels):
        ax2.text(x, stats['between_sd'] + 0.0012, f'{stats["between_sd"]:.4f}', ha='center',
                 va='bottom', fontsize=13, color=INK, fontweight='bold')
        # Above the bar, in ink. Inside it these were SURFACE-on-C1, and "vs full split" is wider
        # than a 0.56-wide bar at this size -- so both ends of the string ran off the bar and
        # became near-white text on the near-white page, leaving a clipped fragment behind. A
        # label that only fits inside its bar at some values does not fit inside its bar.
        ax2.text(x, stats['between_sd'] + 0.0052,
                 'baseline' if not x else f'×{stats["between_sd"] / base_sd:.2f}\nvs full split',
                 ha='center', va='bottom', fontsize=11, color=INK2, linespacing=1.35)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f'{n:,}' for n, _ in levels], fontsize=13.5, color=INK)
    # Explicit, because the reference line's label is right-aligned to the axis edge and the
    # autoscaled limit from three bars leaves it a hair outside.
    ax2.set_xlim(-0.6, len(levels) - 0.4)
    ax2.set_ylim(0, S.SIB_BETWEEN_SD * 1.32)
    ax2.set_ylabel('between-model sd')
    ax2.set_title('How far the labels take it\ntoward topic classification', pad=12, fontsize=15)

    # -- the reading, from the study's own decision function ---------------------------------
    smallest, at = levels[-1]
    verdict = S.reading(at, levels[0][1])[0].split('->')[1].split('.')[0].strip()
    fig.suptitle(f'Cutting the labels threefold changes nothing. Cutting them tenfold spreads the '
                 f'models {at["between_sd"] / base_sd:.1f}× further apart.',
                 fontsize=17, fontweight='bold', color=INK, y=1.02)

    notes = [f'{len(matched)} from-scratch models — the ones carrying every label count. A range '
             f'grows with the number of models in it, so a slope chart is the honest form: it '
             f'cannot be drawn over a model missing a level.',
             f'Every cell at lr {S.BAND_LR:g} with the step budget fixed at {S.STEPS:,} updates, '
             f'so a subsample does not silently become a compute cut.',
             f'The rule, written before the runs: {verdict} — between-model sd '
             f'{at["between_sd"]:.4f} at {smallest} labels against {base_sd:.4f} on the full '
             f'split, {at["between_sd"] / S.SIB_BETWEEN_SD:.0%} of the way to topic '
             f'classification.',
             'mmBERT and the untrained floor ran at each label count as context and are not part '
             'of the band.']
    for i, line in enumerate(notes):
        fig.text(0.5, 0.072 - i * 0.031, line, ha='center', color=INK2, fontsize=11.5)
    fig.subplots_adjust(bottom=0.30, wspace=0.26)
    save(fig, '18-label-quantity')


def _api_surface():
    """The published interface and the code behind it, read off the files rather than recalled.

    Every quantity this figure draws is counted at render time on purpose. The build sheet quoted
    the folder at 12,861 lines for two days, then it was 14,409, and it is 16,102 now -- checks, a
    benchmark and a second study were added in between, none of which changed the interface at
    all. A figure whose whole subject is "a small surface on a large body" cannot carry a literal
    for the body.

    Counting at render time is necessary and not sufficient: this figure sat in git drawing 15,175
    for a day because nothing re-rendered it, while the prose beside it said "fourteen thousand"
    and the folder had passed sixteen. The number was computed and still stale, which is why the
    print gate's last step is to regenerate every figure rather than the ones that look affected.
    """
    def count(path):
        with open(os.path.join(HERE, path), encoding='utf-8') as fh:
            return len(fh.read().splitlines())

    import ast
    import glob as _glob

    with open(os.path.join(HERE, 'mlm_api.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    sigs = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name.startswith('_'):
            continue
        pos = [a.arg for a in node.args.args]
        required = pos[:len(pos) - len(node.args.defaults)]
        # Required arguments in full, optionals as an ellipsis. The real pretrain() takes fifteen
        # and printing them all would make the one line nobody can read the widest thing on the
        # panel -- which would say the opposite of what the panel is for.
        #
        # The join has to handle "no required arguments at all": results() and bits_per_char()
        # take only optionals, and the obvious f-string renders them as `results(, …)`. Caught by
        # looking at the rendered figure, which is the only way that class of defect is ever
        # caught.
        inner = ', '.join(required + (['…'] if len(pos) > len(required) else []))
        sigs.append(f'{node.name}({inner})')

    return {
        'signatures': sigs,
        'api': count('mlm_api.py'),
        'factory': sum(count(p) for p in
                       ('mlm_api.py', 'mlm_data.py', 'mlm_train.py', 'text_data.py')),
        'folder': sum(count(os.path.basename(p))
                      for p in sorted(_glob.glob(os.path.join(HERE, '*.py')))),
        'files': len(_glob.glob(os.path.join(HERE, '*.py'))),
    }


def fig_api():
    """What somebody else has to be able to call. Week 5 of the bottom board.

    NOT A CHART, and that is the design decision worth recording. The panel's content is nine
    function signatures, which are type; the only quantity is one ratio, and it is roughly 1:40
    between the file you import and the folder you do not. A bar chart of that is one visible bar
    and one invisible one.

    So: emphasis rather than categorical. One accent for the surface a caller touches, the
    de-emphasis gray for everything behind it, and a single stacked track in ONE unit -- lines of
    Python -- so the nesting is legible without a second axis. The function count is a hero
    number set beside the track rather than a segment in it, because nine functions and 14,409
    lines are not the same quantity and stacking them would be exactly the dual-axis mistake.

    The caption is the panel's actual finding, and it is a failure rather than a number: Leon
    cloned the repository, read the documentation and asked whether there was an interface he was
    supposed to be using. There was.
    """
    d = _api_surface()
    fig = plt.figure(figsize=(13.0, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.58, 1.0], wspace=0.16)

    # --- left: the interface itself, which is the content ------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 0.965, 'T H E   W H O L E   I N T E R F A C E', color=MUTED, fontsize=12,
            fontweight='bold')
    ax.text(0, 0.905, 'import mlm_api as factory', color=INK2, fontsize=14.5,
            family='DejaVu Sans Mono', style='italic')
    for i, sig in enumerate(d['signatures']):
        y = 0.815 - i * 0.092
        ax.plot([0.004], [y + 0.018], marker='s', markersize=7, color=C1, clip_on=False)
        ax.text(0.035, y, sig, color=INK, fontsize=14.5, family='DejaVu Sans Mono', va='bottom')

    # --- right: the one ratio, in one unit ----------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax2.text(0.5, 0.98, f'{len(d["signatures"])}', color=C1, fontsize=104,
             fontweight='bold', ha='center', va='top')
    ax2.text(0.5, 0.675, 'functions to import', color=INK, fontsize=15.5, ha='center')
    ax2.text(0.5, 0.620, 'nothing else in the folder', color=MUTED, fontsize=13, ha='center')

    # The track: three nested quantities, all lines of Python, drawn to scale. The segments are
    # keyed below rather than labelled with leader lines -- the first version used annotate() and
    # the 1,634 leader landed on top of the subtitle, which a rendered look caught and no amount
    # of reading would have.
    x0, x1, y, h = 0.06, 0.94, 0.435, 0.075
    span = x1 - x0
    for frac, color, alpha in ((1.0, GRID, 1.0),
                               (d['factory'] / d['folder'], C1, 0.32),
                               (d['api'] / d['folder'], C1, 1.0)):
        ax2.add_patch(plt.Rectangle((x0, y), span * frac, h, facecolor=color, alpha=alpha,
                                    edgecolor='none'))

    ax2.text(x0, y + h + 0.045, 'LINES OF PYTHON, TO SCALE', color=MUTED, fontsize=11)

    key = ((C1, 1.0, f'{d["api"]:,}', 'you import — mlm_api.py', INK),
           (C1, 0.32, f'{d["factory"]:,}', 'of factory behind it', INK2),
           (GRID, 1.0, f'{d["folder"]:,}', f'in the folder, {d["files"]} files', MUTED))
    for i, (color, alpha, num, label, ink) in enumerate(key):
        yy = y - 0.105 - i * 0.105
        ax2.add_patch(plt.Rectangle((x0, yy), 0.055, 0.045, facecolor=color, alpha=alpha,
                                    edgecolor='none', clip_on=False))
        ax2.text(x0 + 0.085, yy + 0.004, num, color=ink, fontsize=15, fontweight='bold')
        ax2.text(x0 + 0.325, yy + 0.006, label, color=MUTED, fontsize=12.5)

    fig.suptitle('A tool nobody can find does not exist', x=0.5, y=1.015,
                 fontsize=19, fontweight='bold', color=INK)
    fig.text(0.5, -0.055,
             'Leon cloned the repository, read the documentation, and asked whether there was an '
             'interface he was supposed to be using.\nThere was. The folder’s front page was '
             'titled with a different study and mlm_api first appeared on line 25, one row of a '
             'second table.\nEvery word of it accurate. Still unfindable — and that is a failure '
             'of the interface, not of the reader.',
             ha='center', color=INK2, fontsize=12.5, linespacing=1.7)
    save(fig, '19-the-interface')


def fig_board_layout():
    """The bottom board drawn to scale on the real UW template, so what fits is visible.

    Every dimension is read out of ResearchPoster_Template_Vertical_2023.pptx rather than assumed:
    24 x 36 inches, three columns at x = 1.50 / 8.75 / 16.13 and 6.35 wide, a header band 8.5 deep,
    body type 18 pt, section headers 40 pt, the title 115 pt.

    The build sheet had the board at 36 x 48 with 24 pt body -- 2.25x the area -- so every number
    downstream of that was sized for a board nobody is printing. This figure exists because the
    error was invisible in prose and obvious the moment anything was drawn to scale.

    Not generated from run records like the other nineteen, because its subject is a PowerPoint
    template rather than an experiment. The dimensions are pinned as constants and the template is
    the citation.
    """
    from matplotlib.patches import Rectangle
    PURPLE = '#4b2e83'
    W, H = 24.0, 36.0
    COLX = [1.50, 8.75, 16.13]
    COLW = 6.35
    HEADER_H, BODY_TOP, BODY_BOT = 8.50, 9.25, 34.60

    ROW_H, GUTTER = 6.70, 0.25
    ROWY = [BODY_TOP + i * (ROW_H + GUTTER) for i in range(3)]
    STRIP_Y = ROWY[2] + ROW_H + 0.30
    STRIP_H = BODY_BOT - STRIP_Y

    # number, title, big number, figure (None = type only), words of body
    CELLS = [
        ('1', 'What does a run cost,\nand in what unit?', '62,500 steps\n= 1.024B tokens',
         'fig 21', 55),
        # `1.32× real`, not `only 1.32× is efficiency`. This mockup exists to prove what fits, and
        # it was drawing the 24-character version that the board text two paragraphs above uses as
        # its worked example of a big number that OVERFLOWS the column. A scale drawing arguing
        # against its own caption.
        ('2', 'Why optimize before\nanything needs it?', '2.07×\n1.32× real',
         'fig 14', 55),
        ('3', 'Notebook,\nor queue?', '53 s vs 85 min\n96×', 'fig 15', 55),
        ('4', 'What makes a record\nsurvive you?', 'fingerprint\n15abd33de5af', 'fig 07', 55),
        ('5', 'What must someone\nelse be able to call?', '9\nfunctions', 'fig 19', 55),
        ('6', 'Is this\ndifference real?', '2.27×\nnot 1.0×', 'fig 13', 55),
        ('7', 'Which of your units\nare not units?', '5.1×\nat "matched" steps', None, 110),
        ('8', 'Does a tuned\nsetting transfer?', '7e-4\nfatal to a fourth', 'fig 16', 55),
        ('9', 'Detect the failure,\nor prevent it?', '0 of 11\ncheckpoints', 'fig 10', 55),
    ]

    # The three rubric items that have no cell: cost, ethics, next steps + sources + AI.
    STRIP = [
        # Written from the same records fig_06 draws, not typed beside them. The typed version
        # said $120 and 143 GPU-hours and 69 kWh while the cell it mocks up said $107 and 148
        # and 71 -- a layout mockup disagreeing with the layout it is a mockup of.
        ('WHAT IT COST', C3,
         # Escaped, for the third time in this file. A PAIR of dollar signs is a mathtext span to
         # matplotlib, so "$24k / $110 / free" rendered as italic "24k/110 / free" -- both prices
         # silently eaten on the one panel whose subject is cost.
         f'fig 06 · \\$24k / \\${_rent_the_project():.0f} / free\n'
         f'{sum(_workstation_hours_by_preset().values()):.0f} GPU-hours, 71 kWh, ~$7\n'
         f'the workstation bought\nlatency, not access'),
        ('WHAT WE DO NOT CLAIM', C2,
         'ETHICS — the methods half\n2 of 9 claims not supported\nand printed anyway; nobody\n'
         'here reads Yoruba'),
        ('NEXT · SOURCES · AI', C4,
         # 96 of 168 as of 14 Aug. Counted by hand and therefore pinned by nothing, which is
         # why it was 24 commits and a full ratio out of date. Squash-merging is why the
         # numerator can fall: four trailered commits on a branch arrive as one.
         'a wall meter on the box\n19 references, shortened\n96 of 168 commits carry\n'
         'a Co-Authored-By trailer'),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 13.4))
    ax.set_xlim(-0.6, W + 4.1)
    ax.set_ylim(H + 1.1, -1.1)                       # inverted: y grows downward, as in the template
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), W, H, facecolor=SURFACE, edgecolor=INK, lw=1.6))
    ax.add_patch(Rectangle((0, 0), W, HEADER_H, facecolor=PURPLE, edgecolor='none'))
    ax.text(1.5, 3.1, 'CSED 505: BUILDING A MODEL FACTORY', color='white', fontsize=10.5,
            fontweight='bold', va='center')
    ax.text(1.5, 4.5, 'the course that would come after 504', color='#d9d0e8', fontsize=7,
            style='italic', va='center')
    ax.text(1.5, 7.3, 'Jeffrey Stall  ·  A2-NLP  ·  the upper board is the experiment this served',
            color='#d9d0e8', fontsize=6.2, va='center')
    ax.text(W - 1.5, 3.1, '115 pt', color='#b9a8d4', fontsize=6.4, ha='right', va='center')
    ax.text(W - 1.5, 7.3, '24 pt', color='#b9a8d4', fontsize=6.4, ha='right', va='center')

    for i, (num, title, big, figname, words) in enumerate(CELLS):
        cx, cy = COLX[i % 3], ROWY[i // 3]
        blocked = figname is not None and 'BLOCK' in figname
        ax.add_patch(Rectangle((cx, cy), COLW, ROW_H, facecolor='white',
                               edgecolor=C2 if blocked else GRID, lw=2.0 if blocked else 1.1))
        ax.text(cx + 0.22, cy + 0.60, num, color=C1, fontsize=13, fontweight='bold', va='center')
        ax.text(cx + 0.92, cy + 0.58, title, color=INK, fontsize=7.0, fontweight='bold',
                va='center', linespacing=1.35)
        ax.text(cx + COLW / 2, cy + 2.15, big, color=C1, fontsize=9.0, fontweight='bold',
                ha='center', va='center', linespacing=1.3)

        if figname:
            fy, fh = cy + 2.95, 2.30
            ax.add_patch(Rectangle((cx + 0.3, fy), COLW - 0.6, fh,
                                   facecolor='#fbe3d8' if blocked else GRID, edgecolor='none'))
            ax.text(cx + COLW / 2, fy + fh / 2, figname, color=C2 if blocked else MUTED,
                    fontsize=7, ha='center', va='center',
                    fontweight='bold' if blocked else 'normal')
        else:
            ax.text(cx + COLW / 2, cy + 4.1, 'the two readings,\nset as type — no figure',
                    color=MUTED, fontsize=7, ha='center', va='center', linespacing=1.4)
        ax.text(cx + COLW / 2, cy + ROW_H - 0.42, f'{words} words at 18 pt',
                color=MUTED, fontsize=6.8, ha='center', va='center')

    for i, (head, col, body) in enumerate(STRIP):
        cx = COLX[i]
        ax.add_patch(Rectangle((cx, STRIP_Y), COLW, STRIP_H, facecolor='white',
                               edgecolor=col, lw=1.6))
        ax.text(cx + 0.22, STRIP_Y + 0.42, head, color=col, fontsize=6.8, fontweight='bold',
                va='center')
        ax.text(cx + 0.22, STRIP_Y + 1.95, body, color=INK2, fontsize=6.3, va='center',
                linespacing=1.55)
        ax.text(cx + COLW - 0.22, STRIP_Y + STRIP_H - 0.28, '100 words',
                color=MUTED, fontsize=6.2, ha='right', va='center')


    def brace(y0, y1, label):
        x = W + 0.5
        ax.plot([x, x], [y0, y1], color=MUTED, lw=1.0)
        for yy in (y0, y1):
            ax.plot([x - 0.16, x], [yy, yy], color=MUTED, lw=1.0)
        ax.text(x + 0.24, (y0 + y1) / 2, label, color=INK2, fontsize=6.6, va='center')


    brace(0, HEADER_H, 'header\n8.5 in')
    brace(ROWY[0], ROWY[0] + ROW_H, f'each row\n{ROW_H} in')
    brace(STRIP_Y, BODY_BOT, f'strip\n{STRIP_H:.2f} in')
    ax.annotate('', xy=(COLX[0], -0.45), xytext=(COLX[0] + COLW, -0.45),
                arrowprops=dict(arrowstyle='<->', color=MUTED, lw=1.0))
    ax.text(COLX[0] + COLW / 2, -0.95, 'column 6.35 in', color=INK2, fontsize=6.6, ha='center')
    ax.text(W / 2, H + 0.75, '24 × 36 in — the UW vertical template, measured, not assumed',
            color=INK, fontsize=8.5, ha='center', fontweight='bold')

    save(fig, '20-board-layout')


def _workstation_hours_by_preset():
    """The project's card time split by model shape, because the two shapes are not interchangeable.

    afriberta is 63% of the hours and runs at 48% of poc's rate, so projecting the whole project
    onto another card using either preset's ratio alone is wrong by about 5% in opposite
    directions. Fine-tuning is counted with poc, which is the shape those runs are.
    """
    hours = {}
    for r in f.results('*'):
        if r.get('seconds'):
            key = r.get('preset') or 'poc'
            hours[key] = hours.get(key, 0.0) + r['seconds'] / 3600
    ft_h = sum((r.get('seconds_per_seed') or 0) * len(r.get('scores') or [])
               for r in ft_api.results('*', eval_split=None)) / 3600
    hours['poc'] = hours.get('poc', 0.0) + ft_h
    return hours


def fig_hardware():
    """What one run costs on each machine, and whether you need the workstation.

    Panel 1's figure, and for weeks the last blocked cell on the bottom board -- blocked because
    it is the only number in the project that cannot be produced from this machine. It needed
    somebody to run bench_portable.py somewhere else, seven times.

    Seven machines: the workstation, a Colab A100, L4 and free T4, a MacBook Pro, and the 8 GB
    laptop measured three ways -- plugged in, on battery, and with --cpu on its own processor,
    which is the triple that answers "is a small mobile GPU even worth using, and does it need
    the wall". More rows drop in without touching this function.

    Every row except the workstation's is a three-minute timed run; the workstation's stays the
    median of 127 and 70 real training runs until one can be taken on an idle card. Where a
    device has both a burst and a sustained row, the sustained one is later in the file and wins.

    The Mac needed two things nothing else did: it is not a CUDA row, so "does it have a compute
    capability" was the wrong question to ask about whether it is a GPU at all (it was being
    filed as a CPU baseline and dropped), and at 45.6 h it is far enough off the other bars to
    need the axis capped. Both are handled below and both are commented where they happen.

    The first T4 reading was 33x and said no. It was bf16 running in software on a card whose
    tensor cores do not have it; measured in fp16 the same card is 5.9x on a burst and 7.1x held
    for three minutes -- the only tier where those two disagree, because a free runtime shares a
    host. Both numbers looked equally plausible on the page, which is why the row carries its
    dtype, and the same reason the bars now come from timed runs. The laptop's first
    reading failed the same way in a different direction: Windows spilled the 98M model into
    system RAM and reported 5,075 tok/s of PCIe traffic as if it were the GPU. Its row carries
    the gradient-accumulation configuration that actually fits for the same reason.

    CPU rows are kept off the bar panel on purpose: 551 hours next to 1.5 would flatten every
    GPU bar to nothing on a linear axis, and the number a reader needs from the CPU is not its
    bar but its ratio to the card sitting in the same chassis.
    """
    rows = json.load(open(os.path.join(HERE, 'runs', 'hardware.json'), encoding='utf-8'))
    WS_HOURS = _workstation_hours_by_preset()
    # Recovered from the rows rather than restated as 62,500 x 128 x 128 here. A second copy of
    # a constant is how this project's worst numbers happened: bench_portable.py's own
    # PROJECT_GPU_HOURS sat at 83.3 for weeks after the project reached 148. If the benchmark
    # ever changes what a full run means, these bars follow it instead of quietly disagreeing.
    TOKENS_PER_RUN = rows[0]['full_run_hours'] * 3600 * rows[0]['tokens_per_s']

    # Every sitting for a machine gets collected first and reduced second, because last-row-wins
    # is not a choice -- it is whatever order the file happens to be in. Two rules follow.
    #
    # A timed row beats a burst row, always, and they are never averaged. They measure different
    # things, and blending them would bury the one finding the method change produced: the free
    # T4 reads 19-22% high on a 40-step burst while nothing else moves by 2%. So a machine with
    # any --seconds row drops its burst rows entirely.
    #
    # Among comparable sittings, the median. The laptop has been measured plugged in twice, 1.5%
    # apart, and quoting whichever landed last in the file means appending a row silently redraws
    # the figure. Hours come from that median rather than off one row, so the bar and the table
    # beside it cannot disagree.
    # Three kinds of row live here now, and medianing across them would be the bug this
    # figure keeps finding in other people's numbers.
    #
    #   realistic-loop   the step mlm_train.pretrain() runs. What every row should eventually be.
    #   bare-step        the old step-only loop. Reads high, by 3% on the workstation and by an
    #                    amount nobody has measured on any other machine.
    #   real-run median  not a benchmark: the median over 127 and 70 completed training runs.
    #
    # A machine takes the best method it has and DISCARDS the others -- never an average. The
    # real-run column is keyed apart on purpose rather than competing, because Jeffrey asked for
    # both workstation columns on the figure: the benchmark says what an idle card does, the real
    # runs say what a term of work actually delivered, and the gap between them is the point.
    buckets, order = {}, []
    for r in rows:
        # A battery sitting is the same silicon telling a different story, so it is keyed
        # apart -- otherwise the plugged laptop and the battery laptop average into a machine
        # that does not exist, at a rate neither of them ran at.
        key = r['device'] + (' (battery)' if 'battery' in (r.get('note') or '').lower() else '')
        if r.get('method') == 'real-run median':
            key += ' (real runs)'
        if key not in buckets:
            buckets[key] = {}
            order.append(key)
        buckets[key].setdefault(r['preset'], []).append(r)

    machines = {}
    for key, presets in buckets.items():
        machines[key] = {}
        for preset, got in presets.items():
            # _hardware_rate carries the selection rules; the rest of the record comes from the
            # sitting nearest the median it picked, so dtype, peak memory and the micro-batch the
            # row records all still describe one real run rather than a composite of several.
            r = _hardware_rate(got)
            machines[key][preset] = dict(r['near'], tokens_per_s=round(r['rate']),
                                         sittings=r['sittings'], method=r['method'],
                                         lo=r['lo'], hi=r['hi'],
                                         full_run_hours=TOKENS_PER_RUN / r['rate'] / 3600)
    # An accelerator row, or the bare processor? `compute_capability` answers that for CUDA and
    # for nothing else -- Apple's MPS rows carry a null there, so keying off it alone filed the
    # MacBook as a CPU baseline and dropped it off the bar panel without saying so. The row a
    # student is most likely to look for, silently missing, because a NVIDIA-shaped field was
    # read as "is this a GPU".
    def is_accelerator(rec):
        return bool(rec.get('compute_capability')) or '(MPS)' in rec['device']

    cpu_only = [k for k in order if not is_accelerator(machines[k]['poc'])]
    order = [k for k in order if is_accelerator(machines[k]['poc'])]
    # Fastest first, so the workstation anchors the left and the question "how far down can you
    # go" reads left to right.
    order.sort(key=lambda k: -machines[k]['poc']['tokens_per_s'])

    # Seven columns' worth of two-line tick labels need more rail than five did -- at 17.0 the
    # battery column made "Colab A100, Pro" and "the workstation" set as one run-on word.
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(23.5, 6.0),
                                  gridspec_kw={'width_ratios': [2.15, 1.0]})

    # Two short lines each. The first line names the machine, the second carries the generation
    # and the dtype -- which is the pair that tells a reader whether two bars are the same
    # computation. Eight columns of these collided at the old widths, so the machine names lost
    # their "Pro"/"laptop" suffixes rather than the technical line losing anything.
    WS = 'NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition'
    short = {WS: 'the workstation\nsm_120 · bf16',
             # Not a benchmark. 127 and 70 completed training runs on the same card, which is the
             # only column here that knows what a term of work actually costs rather than what an
             # idle machine can do. The gap to the column beside it is 0.86x at 33.8M and 0.98x
             # at 98M -- dispersion, not bias: 9-minute runs span 1.73x from p10 to p90 while
             # 93-minute runs span 1.11x.
             WS + ' (real runs)': 'the same box,\n197 real runs',
             'NVIDIA A100-SXM4-80GB': 'Colab A100\nsm_80 · bf16',
             'NVIDIA L4': 'Colab L4\nsm_89 · bf16',
             'Tesla T4': 'Colab T4, free\nsm_75 · fp16',
             'NVIDIA RTX 2000 Ada Generation Laptop GPU': 'RTX 2000 Ada\nsm_89 · 8 GB *',
             'NVIDIA RTX 2000 Ada Generation Laptop GPU (battery)': 'same laptop\non battery',
             # The dtype is on every tick label for a reason, and this is the row where it earns
             # its place: fp32 against everyone else's bf16 or fp16, because MPS has no autocast
             # path we would trust to be the same computation. The bar is honest and not exactly
             # comparable, and the label has to say so where the bar is read.
             'Apple arm64 (MPS)': 'MacBook Pro\nM4 Pro · fp32 † *',
             'Intel64 Family 6 Model 186 Stepping 2, GenuineIntel': 'the same laptop,\nCPU only'}

    # A CAPPED AXIS, because the MacBook is 30x the A100 and the bars have to stay linear.
    #
    # This is the same problem the CPU rows have, one order of magnitude smaller, and it does not
    # have the same answer. The CPU is kept off this panel because nobody is deciding whether to
    # train on a CPU; a Mac is the machine a large share of this poster's readers actually own,
    # so its bar belongs here even when the bar is bad news. But drawn to full height, 45.6 h
    # puts the workstation and the A100 at 1.5% of the axis and destroys the comparison the panel
    # exists to make -- the one between the tiers a student might rent.
    #
    # So the axis is capped just above the tallest bar that is not the Mac's 98M run, and that
    # one bar is drawn clipped, hatched at the cut, and labelled with its real value. A reader
    # loses nothing: the number is on the bar either way, and every bar that fits stays honestly
    # proportional to every other. What they gain is being able to see that the T4 and the L4 are
    # the same order of magnitude, which is the whole argument of the left panel.
    vals_by_preset = {p: [machines[k][p]['full_run_hours'] for k in order]
                      for p in ('poc', 'afriberta')}
    tallest = max(v for vs in vals_by_preset.values() for v in vs)
    fits = [v for vs in vals_by_preset.values() for v in vs if v < tallest]
    cap = max(fits) * 1.30

    xs = range(len(order))
    for i, preset, color, label in ((0, 'poc', C1, '33.8M model'),
                                    (1, 'afriberta', C2, '98M model')):
        vals = vals_by_preset[preset]
        pos = [x + (i - 0.5) * 0.36 for x in xs]
        ax.bar(pos, [min(v, cap) for v in vals], 0.34, color=color, label=label, zorder=3)
        for x, v, k in zip(pos, vals, order):
            star = '*' if 'micro_batch' in machines[k][preset] else ''
            if v > cap:
                # The break: a hatched band across the top of the bar, so the eye is told the
                # bar is cut rather than left to read the cap as the value.
                ax.bar([x], [cap * 0.055], 0.34, bottom=cap * 0.945, color=color,
                       hatch='///', edgecolor='white', linewidth=0, zorder=4)
                # Set INSIDE the bar and low, rather than above it: the neighbouring 33.8M bar on
                # this machine is nearly as tall as the cap, and its label was landing on top of
                # this one -- "45.6 h" reading as "5.6 h", which is worse than not labelling it.
                ax.text(x, cap * 0.45, f'{v:.1f} h{star}', ha='center', va='center',
                        color='white', fontsize=12, fontweight='bold', zorder=5)
                ax.text(x, cap * 1.005, 'off the scale', ha='center', color=INK2, fontsize=10.5)
            else:
                ax.text(x, v + cap * 0.014, f'{v:.1f} h{star}', ha='center',
                        color=INK, fontsize=12, fontweight='bold')
    ax.set_xticks(list(xs))
    # A column measured with the old step-only loop is marked, because the alternative is a
    # figure that looks finished while half its bars were taken a different way. The mark comes
    # off the row's own `method` field rather than a list kept here, so it clears itself the
    # moment a machine is re-measured and nobody has to remember to delete it.
    stale = [k for k in order if machines[k]['poc'].get('method') == 'bare-step']
    ax.set_xticklabels([short.get(k, k) + (' ‡' if k in stale else '') for k in order],
                       fontsize=10.0)
    ax.set_ylabel('hours for one 62,500-step run')
    ax.set_ylim(0, cap * 1.06)
    ax.legend(loc='upper left')
    ax.set_title('One run, on each machine', pad=30, loc='left')
    ax.text(0, 1.04, 'same experiment, same token budget', transform=ax.transAxes,
            color=INK2, fontsize=12, va='bottom')

    # The right panel is the claim the panel has to earn, in the unit a student actually has.
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    # The cost table, and why it leads with a ratio rather than a price. Colab bills in compute
    # units whose rate moves with tier pricing and demand, so the dollars are one reading on one
    # day rather than a property of the machine -- Jeffrey flagged that before it bit us, which
    # is the discipline pointing the right way for once. The DURABLE finding is the ratio: the
    # A100 costs 4.40x the L4 per hour and returns about 4.8x the work on this workload -- 4.45x
    # on the 33.8M model and 5.06x on the 98M, which is why the blend beats the price ratio. So
    # the faster tier is also, slightly, the cheaper one: $100 against $110, five times sooner.
    # You buy latency, not access, which is the workstation argument arrived at from a completely
    # different direction. On burst readings this came out "within 5% of the same price"; the
    # sustained rows moved it to 10% and did not change the direction.
    # The table's rates come from the same rows as the bars, so the two halves of the figure
    # cannot disagree. The first version hardcoded burst readings here -- a hand-typed count on
    # the one board that forbids them -- and the day the sustained rows landed, the bars moved
    # and the table did not. Billing fields ride on the rows, read off the usage page the day
    # of the sitting; a paid tier whose rows lack them shows an em dash rather than a guess.
    def tier(name, device, fixed=None):
        # Named lookup with a named failure. This was a bare machines[device], and #88 renamed
        # the workstation's device string across every row in runs/hardware.json to what torch
        # actually reports -- so the two PRs were individually fine and jointly a KeyError on
        # the cost table, which the Mac caught while checking whether they could co-exist. A
        # missing machine now says which one and what is available.
        if device not in machines:
            raise KeyError(f'fig_hardware: no rows for {device!r}. runs/hardware.json has '
                           f'{sorted(machines)}')
        m = machines[device]
        return {'name': name, 'fixed': fixed, 'device': device,
                'rate': {p: m[p]['tokens_per_s'] for p in ('poc', 'afriberta')},
                'hours': m['poc']['full_run_hours'],
                'uph': m['poc'].get('compute_units_per_hour'),
                'usd': m['poc'].get('usd_per_compute_unit')}

    tiers = [tier('Colab T4, free', 'Tesla T4', fixed='$0'),
             tier('Colab L4, Pro', 'NVIDIA L4'),
             tier('Colab A100, Pro', 'NVIDIA A100-SXM4-80GB'),
             tier('the workstation', WS, fixed='$24,000')]
    ws_rate = tiers[-1]['rate']

    ax2.text(0, 0.95, 'CAN YOU DO THIS WITHOUT THE WORKSTATION?', color=MUTED, fontsize=11.5,
             fontweight='bold', va='top')
    ax2.text(0, 0.865, 'Yes — and the tier you pick changes', color=INK, fontsize=14.5, va='top')
    ax2.text(0, 0.805, 'the wait, not the bill.', color=C3, fontsize=19, fontweight='bold',
             va='top')

    cols = (0.0, 0.54, 0.75, 0.99)
    for lbl, x in (('one run', cols[1]), ('project', cols[2]), ('to rent', cols[3])):
        ax2.text(x, 0.700, lbl, color=MUTED, fontsize=10.5, ha='right', fontweight='bold')
    ax2.plot([0, 1.0], [0.670, 0.670], color=GRID, lw=1.2)

    costs = {}
    for n, t in enumerate(tiers):
        # Lifted and tightened to buy room underneath. Four hand-placed prose blocks now sit
        # below this table and the lowest of them, the provisional warning, only exists while
        # some machine is still on the old loop -- so the panel has to hold its worst case.
        y = 0.595 - n * 0.079
        hrs = t['project_hours'] = sum(WS_HOURS[k] * ws_rate[k] / t['rate'][k] for k in WS_HOURS)
        if not t['fixed'] and t['uph'] and t['usd']:
            costs[t['name']] = hrs * t['uph'] * t['usd']
        cost = t['fixed'] or (f"${costs[t['name']]:,.0f}" if t['name'] in costs else '—')
        strong = t['name'].startswith('Colab A100')
        col = C3 if strong else INK
        ax2.text(cols[0], y, t['name'], color=col, fontsize=12.5,
                 fontweight='bold' if strong else 'normal')
        # Quote the row's own full_run_hours, exactly as the bars do. Re-deriving hours from
        # the rate here put 5.2 h on the T4's bar and 5.3 h in the table beside it -- the same
        # machine, the same measurement, split by a second rounding.
        ax2.text(cols[1], y, f"{t['hours']:.1f} h", color=col, fontsize=12.5, ha='right')
        ax2.text(cols[2], y, f'{hrs / 24:.0f} d', color=col, fontsize=12.5, ha='right')
        ax2.text(cols[3], y, cost, color=col, fontsize=12.5, ha='right',
                 fontweight='bold' if strong else 'normal')

    # Computed, not quoted: this sentence read "4.4x / 4.5x / within 5%" as prose while the
    # numbers lived in the rows, and prose does not notice when its numbers move.
    l4 = next(t for t in tiers if t['name'] == 'Colab L4, Pro')
    a100 = next(t for t in tiers if t['name'] == 'Colab A100, Pro')
    if l4['name'] in costs and a100['name'] in costs:
        # The work ratio is the BLEND across both model shapes, not the 33.8M model's alone. The
        # A100 is 4.4x the L4 on the small model and 5.1x on the 98M, and the project is mostly
        # the 98M half -- so a poc-only ratio (4.4x) sat exactly on the hourly price ratio and
        # made the two tiers look like a wash. They are not: the faster tier is also the cheaper
        # one, which is a sharper claim and the one the dollars beside it actually show.
        work = l4['project_hours'] / a100['project_hours']
        # Raised from 0.085 to clear the provisional warning below it, which is two lines the
        # panel did not budget for. Everything in ax2 is hand-placed in axes coordinates, so a
        # block that grows silently overwrites its neighbour rather than pushing it.
        # Both dollar signs are escaped. Matplotlib reads a PAIR of $ as a mathtext span, so
        # "$100 against $110" rendered as italic "100against110" -- the money vanished and the
        # sentence still looked like a sentence. The single-$ strings elsewhere in this panel
        # survive by accident of being odd-numbered; these do not.
        ax2.text(0, 0.20,
                 f"The A100 costs {a100['uph'] / l4['uph']:.1f}× the L4 per hour and returns "
                 f"{work:.1f}× the work, so the whole\n"
                 f"project is \\${costs[a100['name']]:,.0f} against "
                 f"\\${costs[l4['name']]:,.0f} — you buy {work:.1f}× the speed, not the bill.",
                 color=INK, fontsize=11.5, va='bottom', linespacing=1.6)
    # The provisional warning has to sit HERE, next to the dollars, not only in the caption.
    # Every tier in this table is still a bare-step reading divided by a realistic-loop
    # workstation, which is the same mixing-of-methods this figure was just corrected for --
    # pointing the other way, and flattering the tiers. It clears itself when `stale` empties.
    # Names the tiers that are actually stale rather than saying "the Colab rows". It said the
    # latter while the T4 and the L4 were already re-measured, which is a warning that has
    # started lying in the safe direction -- still the wrong direction for a warning.
    tiers_stale = [t for t in tiers if machines[t['device']]['poc'].get('method') == 'bare-step'
                   and not t['fixed']]
    if tiers_stale:
        who = ' and '.join(t['name'].replace('Colab ', '').replace(', Pro', '')
                           for t in tiers_stale)
        ax2.text(0, 0.095,
                 f'PROVISIONAL — the {who} row is still the old step-only loop, measured\n'
                 f'against a re-measured workstation. Re-run before trusting its dollars.',
                 color=C2, fontsize=10, va='bottom', linespacing=1.6)
    ax2.text(0, 0.0,
             'Compute-unit rates move with pricing and demand — read 13 Aug 2026 at $9.99\n'
             'per 100 units. One reading, not a constant. The ratio is the durable part.',
             color=MUTED, fontsize=10, va='bottom', linespacing=1.6)

    # The 8 GB laptop measured against its own processor: the ratio that tells a student whether
    # the GPU they already own is worth plugging in for. Leon's line, kept.
    lap = machines.get('NVIDIA RTX 2000 Ada Generation Laptop GPU')
    cpu = next((machines[k] for k in machines if 'Intel' in k or 'AMD' in k), None)
    if lap and cpu:
        ax2.text(0, 0.30,
                 f"The same laptop with its GPU ignored is "
                 f"{lap['poc']['tokens_per_s'] / cpu['poc']['tokens_per_s']:.0f}× slower again.",
                 color=INK2, fontsize=11.5, va='bottom')


    # The ‡ sentence explains a mark. With no column carrying one it is dead text, and
    # dead text on a poster is a reader hunting for something that is not there -- so it
    # clears itself the same way the marks do. The measured spread survives either way,
    # because that is the finding rather than the bookkeeping.
    marked = ("‡ marks a column still on the old step-only loop, which omits all three and\n"
              "reads high. How high is a property of the machine and not predictable from its\n"
              "speed: "
              if stale else
              "Every column times that loop. The step-only version it replaced omits all\n"
              "three and reads high by an amount that is a property of the machine and not\n"
              "predictable from its speed: ")
    fig.text(0.5, -0.40,
             "Measured by bench_portable.py, three minutes per model per machine, timing the "
             "step mlm_train.pretrain() actually runs — batches built and masked on-device, "
             "gradients\nclipped, the loss read back every step. " + marked +
             "2.6% on the workstation, 5.0% on "
             "an A100 whose step is dearer, 0.8% on an L4, 0.7% on a Mac 26× slower again. "
             "Machines measured more than once "
             "show the median.\nRead every bar as a CEILING. Against the 184 of this project's own "
             "training runs taken at this shape, the benchmark matches a run that gets the machine "
             "to itself — 0.995 of the 98M preset's p90 — while\nthe median 33.8M run reached 0.86 of it. That is "
             "dispersion, not bias: 9-minute runs span 1.76× from p10 to p90, 93-minute runs "
             "1.11×. The 'real runs' column is that median.\nThe T4's first reading was 33× and "
             "said no: torch.cuda.is_bf16_supported() defaults to including_emulation=True, so a "
             "card without bf16 tensor cores ran it in software —\nhence the dtype on every "
             "label. * The 98M model does not fit whole in 8 GB, or inside the 17.8 GB Metal "
             "recommends on a 24 GB Mac; both run the same 16,384-token step\nvia gradient "
             "accumulation. † The Mac holds fp32 where every CUDA row is bf16 or fp16, so it is "
             "directional rather than exactly comparable — measured anyway, because silently\n"
             "changing precision between machines compares two different computations.",
             ha='center', color=INK2, fontsize=11, linespacing=1.7)
    save(fig, '21-hardware')


# ==============================================================================================
# COLUMN-WIDTH FIGURES -- the ones that go on the printed board.
# ==============================================================================================
#
# The figures above are 8-16 in wide, which is right for a report and wrong for a poster cell.
# A cell on a 24 x 36 in board gives a figure 5.59 x 2.40 in, so a 13 in figure is scaled to
# 0.24x and its 13 pt type prints at 3.1 pt, against 18 pt body text beside it. Measured across
# the seven the top board uses, the range was 2.5-3.9 pt. `poster_figures` sets 13 pt with the
# comment "readable from two meters away"; the placement was undoing it.
#
# Scale cannot be fixed by giving the cell more room. The binding constraint is source width
# against a 6.35 in column: even starving the prose entirely, the headline figure caps at 0.43x.
# So these are drawn AT the size they print, one idea each, and the multi-panel versions stay in
# the reports where the argument is made in full.
#
# Everything here reads the same records as its wide counterpart. A column figure that re-derived
# its numbers would be a second copy of a result.

COLUMN_W, COLUMN_H = 5.59, 2.40

# SIB-200 topic between-model sd over the same sixteen -- the scale bar cell 7 is measured against.
SIB_SD = 0.0457
COLUMN_RC = {
    'font.size': 9.5, 'axes.titlesize': 10.5, 'axes.labelsize': 9,
    'legend.fontsize': 8.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 9,
    'lines.linewidth': 1.9, 'lines.markersize': 6,
}


@contextlib.contextmanager
def column_type():
    """Poster-cell type for the duration of one figure."""
    with plt.rc_context(COLUMN_RC):
        yield


def _column_fig(h=COLUMN_H):
    return plt.subplots(figsize=(COLUMN_W, h))


def _pick_cell(slug, task, steps):
    """One reportable cell, with its learning rate chosen on dev where dev cells exist.

    The same rule `fig_headline` uses, lifted out so the column figures cannot quietly select
    differently from the wide ones. Taking max() over test cells re-selects on the items being
    reported, which is what report 11 removed.
    """
    test = [r for r in ft_api.results()
            if r['model_slug'] == slug and r.get('task') == task and r.get('steps') == steps]
    if not test:
        return None
    dev = [r for r in ft_api.results(eval_split='validation')
           if r['model_slug'] == slug and r.get('task') == task and r.get('steps') == steps]
    if dev:
        lr = max(dev, key=lambda r: r['mean'])['lr']
        on_test = [r for r in test if r['lr'] == lr]
        if on_test:
            return on_test[0]
    return max(test, key=lambda r: r['mean'])


def _hbars(ax, rows, floor=None, floor_label=None, xlim=None):
    """Horizontal bars with per-seed dots -- the shape that fits a wide, short cell.

    Horizontal because the arm names are words: vertical bars in a 5.59 in box either rotate the
    labels or truncate them, and a poster reader will not tilt their head.
    """
    ys = range(len(rows))
    right = (xlim or (0, 1))[1] * 0.985
    ax.barh(list(ys), [r['mean'] for r in rows],
            color=[r['color'] for r in rows], height=0.62, zorder=2)
    for y, r in zip(ys, rows):
        for s in r.get('scores') or []:
            ax.plot(s, y, 'o', color=SURFACE, markersize=4.2,
                    markeredgecolor=INK2, markeredgewidth=0.8, zorder=4)
        ax.text(right, y, f'{r["mean"]:.3f}', va='center', ha='right',
                fontsize=9.5, color=INK, fontweight='bold', zorder=5)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r['label'] for r in rows], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis='y', visible=False)
    if floor is not None:
        ax.axvline(floor, ls=(0, (4, 3)), lw=1.5, color=INK2, zorder=3)
        # Rotated along the rule. Set horizontally it collides with the title above the top bar
        # and with the x tick labels below the bottom one -- there is no clear horizontal band in
        # a cell this short.
        ax.text(floor, (len(rows) - 1) / 2, f'  {floor_label}', rotation=90, ha='left',
                va='center', fontsize=8.8, color=INK2, fontweight='bold', zorder=5)
    if xlim:
        ax.set_xlim(*xlim)


def fig_headline_column():
    """Cell 5. One task, four arms: does a small from-scratch model win on topic?

    The wide figure carries both tasks side by side; the cell that uses this one asks only about
    topic, and entities get their own cell with their own floor. Splitting them is what buys the
    type size back.
    """
    arms = [('from-scratch 33.8M', 'yor-64M-62.5k-s0', C1),
            ('mmBERT base 246M', 'mmBERT-base', C2),
            ('untrained, our arch', 'yor-random-init', MUTED),
            ('XLM-R base 277M', 'xlm-roberta-base', C3)]
    rows = []
    for label, slug, color in arms:
        r = _pick_cell(slug, 'sib200', 1056)
        if r:
            rows.append({'label': label, 'mean': r['mean'],
                         'scores': r.get('scores'), 'color': color})
    with column_type():
        fig, ax = _column_fig()
        _hbars(ax, rows, floor=1 / 7, floor_label='chance', xlim=(0, 0.80))
        ax.set_xlabel('SIB-200 topic, macro-F1 (dev-selected rate, 5 seeds)')
        ax.set_title('64M Yoruba tokens, ahead of 3 trillion across 1,800 languages')
        fig.tight_layout()
        save(fig, '01-headline-column')


def fig_floors_column():
    """Cell 6. The same four arms on entities, with the untrained floor drawn in.

    The floor is the point of the cell, so it is a rule across the bars rather than a bar of its
    own -- it is a reference level, not a competitor.
    """
    arms = [('mmBERT base', 'mmBERT-base', C2),
            ('XLM-R base', 'xlm-roberta-base', C3),
            ('from-scratch 33.8M', 'yor-64M-62.5k-s0', C1)]
    rows = []
    for label, slug, color in arms:
        r = _pick_cell(slug, 'masakhaner', 2150)
        if r:
            rows.append({'label': label, 'mean': r['mean'],
                         'scores': r.get('scores'), 'color': color})
    sweep = json.load(open(os.path.join(HERE, 'runs', 'ner_control_sweep.json'), encoding='utf-8'))
    ours = [r for r in sweep if 'random_init' in r.get('model', '') and 'xlm' not in r.get('arm', '')]
    floor = max(r['mean'] for r in (ours or sweep))
    with column_type():
        fig, ax = _column_fig()
        _hbars(ax, rows, floor=floor, floor_label=f'untrained {floor:.3f}', xlim=(0, 1.0))
        ax.set_xlabel('MasakhaNER entity F1')
        ax.set_title('Most of every bar is capitalisation and name shape')
        fig.tight_layout()
        save(fig, '12-floors-column')


def fig_gradient_column():
    """Cell 3. Seventeen languages as one strip, covered against not.

    The wide figure is a ranked bar per language, which needs 17 rows and cannot be read at 2.4 in.
    Two strips of dots carry the finding -- the separation, the two means, and Wolof sitting inside
    the covered range -- in a tenth of the height. The exception is drawn rather than described,
    because a gradient with one exception is what the data is.
    """
    rows = [r for r in json.load(open(os.path.join(HERE, 'runs', 'gradient_table.json'),
                                      encoding='utf-8')) if r['corpus'] != 'eng_1b']
    cov = [r for r in rows if r['in_xlmr'] is True]
    unc = [r for r in rows if r['in_xlmr'] is not True]
    with column_type():
        fig, ax = _column_fig(2.15)
        for y, (grp, color, name) in enumerate([(unc, C2, 'not covered'), (cov, C3, 'covered')]):
            ax.plot([r['penalty'] for r in grp], [y] * len(grp), 'o', color=color,
                    markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.4,
                    zorder=3, clip_on=False)
            m = sum(r['penalty'] for r in grp) / len(grp)
            ax.plot([m], [y], '|', color=INK, markersize=22, markeredgewidth=2.2, zorder=4)
            ax.text(m, y + 0.30, f'mean {m:.3f}', ha='center', fontsize=9,
                    color=INK, fontweight='bold')
            ax.text(0.90, y, name, ha='right', va='center', fontsize=9.5, color=INK2)
        yor = next(r for r in rows if r['corpus'] == 'yor')
        ax.annotate('Yoruba', (yor['penalty'], 0), textcoords='offset points', xytext=(0, -20),
                    ha='center', fontsize=9, color=INK, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=INK2, lw=1.2))
        wol = next(r for r in rows if r['corpus'] == 'wol')
        ax.annotate('Wolof, the exception', (wol['penalty'], 0), textcoords='offset points',
                    xytext=(-2, 24), ha='center', fontsize=8.5, color=MUTED,
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=1.1))
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([])
        ax.set_xlim(0.88, 1.95)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel("XLM-R tokens per word, against each language's own 16k BPE")
        ax.set_title('The penalty tracks coverage; learnability does not')
        fig.tight_layout()
        save(fig, '02-tokenizer-gradient-column')


def fig_saturation_column():
    """Cell 2. The English ladder: more text stops paying at or before 64M tokens."""
    rows = [r for r in f.results('eng_1b_*')
            if r.get('steps') and r.get('val_loss')]
    by = {}
    for r in rows:
        by.setdefault(r.get('data_tokens') or r.get('n_tokens'), []).append(r['val_loss'])
    pts = sorted((k, v) for k, v in by.items() if k)
    xs = [p[0] for p in pts]
    means = [sum(v) / len(v) for _, v in pts]
    los = [min(v) for _, v in pts]
    his = [max(v) for _, v in pts]
    with column_type():
        fig, ax = _column_fig(2.15)
        ax.axvspan(64e6, xs[-1] * 1.15, color=GRID, alpha=0.65, zorder=0)
        ax.fill_between(xs, los, his, color=C1, alpha=0.18, zorder=1)
        ax.plot(xs, means, 'o-', color=C1, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3)
        ax.set_xscale('log')
        ax.set_xlabel('training tokens (English, fixed compute, 3 seeds a rung)')
        ax.set_ylabel('val loss')
        ax.set_title('Past ~64M tokens, more text buys nothing measurable')
        ax.text(2.6e8, max(means) - 0.05, 'sixteen times the text,\nloss moves −0.080',
                fontsize=8.5, color=INK2, va='top')
        fig.tight_layout()
        save(fig, '05-data-saturation-column')


def fig_swap_column():
    """Cell 8. Swap the vocabulary and nothing else; every seed of ours wins on both tasks."""
    rows = json.load(open(os.path.join(HERE, 'runs', 'swap_downstream.json'), encoding='utf-8'))
    # Topic lands under stage 'test' and entities under stage 'ner'; the arm is a phrase, not a
    # slug. Read both off the file rather than assuming a shape.
    tasks = [('sib200', 'Topic'), ('masakhaner', 'Entities')]
    OURS = 'our vocabulary'
    with column_type():
        fig, ax = _column_fig(2.2)
        for y, (task, tlabel) in enumerate(tasks):
            for arm, color, off in ((OURS, C1, -0.17), ("XLM-R's vocabulary", C2, 0.17)):
                pts = [r['mean'] for r in rows
                       if r.get('arm') == arm and r.get('task') == task
                       and r.get('stage') in ('test', 'ner')]
                if not pts:
                    continue
                ax.plot(pts, [y + off] * len(pts), 'o', color=color, markersize=7,
                        markeredgecolor=SURFACE, markeredgewidth=1.3, zorder=3)
                m = sum(pts) / len(pts)
                ax.plot([m], [y + off], '|', color=INK, markersize=15,
                        markeredgewidth=2.0, zorder=4)
                ax.text(m, y + off - 0.31, f'{m:.3f}', ha='center', fontsize=8.6,
                        color=color, fontweight='bold')
            ax.text(0.425, y, tlabel, ha='right', va='center', fontsize=9.5, color=INK2)
        ax.set_yticks([])
        ax.set_ylim(-0.62, 1.62)
        ax.set_xlim(0.42, 0.86)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('score, four pretraining seeds a side, both arms dev-swept')
        ax.set_title('Same text, same compute — only the vocabulary differs')
        ax.plot([], [], 'o', color=C1, label='our 16k vocabulary')
        ax.plot([], [], 'o', color=C2, label="XLM-R's 250k")
        ax.legend(loc='lower right', ncol=2, frameon=False)
        fig.tight_layout()
        save(fig, '03-matched-steps-vs-compute-column')


def fig_lottery_column():
    """Cell 9. Twelve runs as points: the means do not separate, the spreads do."""
    rows = json.load(open(os.path.join(HERE, 'runs', 'tokenizer_seeds.json'), encoding='utf-8'))
    arms = {}
    for r in rows:
        bpc = r.get('bpc')
        if bpc is None and r.get('val_loss'):
            cpt = 2.1328 if r['arm'] != 'ours' else 3.7339
            bpc = r['val_loss'] / math.log(2) / cpt
        arms.setdefault(r['arm'], []).append(bpc)
    order = [(k, v) for k, v in arms.items()]
    order.sort(key=lambda kv: -st.pstdev(kv[1]))
    with column_type():
        fig, ax = _column_fig(2.15)
        for y, (arm, vals) in enumerate(order):
            color = C2 if y == 0 else C1
            ax.plot(vals, [y] * len(vals), 'o', color=color, markersize=8,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3)
            m = sum(vals) / len(vals)
            ax.plot([m], [y], '|', color=INK, markersize=24, markeredgewidth=2.2, zorder=4)
            ax.text(m, y + 0.32, f'sd {st.stdev(vals):.3f}', ha='center',
                    fontsize=9, color=INK, fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["XLM-R's 250k", 'our 16k'], fontsize=9.5)
        ax.set_ylim(-0.6, 1.7)
        ax.grid(axis='y', visible=False)
        ax.set_xlabel('bits per character, six pre-registered seeds a side')
        ax.set_title('Not a cost — a lottery. F = 15.1, p = 0.0098')
        fig.tight_layout()
        save(fig, '17-tokenizer-lottery-column')


def fig_label_quantity_column():
    """Cell 7. Between-model spread as labels are removed, against topic's for scale.

    The wide figure is a sixteen-line slope chart; at 2.4 in the lines are indistinguishable, so
    this draws the statistic the study's rule actually decides on -- between-model sd, which does
    not grow with set size -- at the three label counts.

    The set and the statistics come from , exactly as the wide figure takes
    them. runs/label_quantity.json holds only the SUBSAMPLED levels; the full split lives in the
    ordinary MasakhaNER records, and the matched sixteen are the models carrying every level. A
    first draft read the json alone and drew two points instead of three, with 701 at 0.0190 --
    the seventeen-model value -- against the board's 0.0195. Both are correct over their own set,
    which is the trap this experiment is about.
    """
    S = label_quantity
    with open(os.path.join(HERE, 'runs', 'label_quantity.json'), encoding='utf-8') as fh:
        rows = [r for r in json.load(fh) if 'mean' in r]
    band_all, _dropped = S.band_models()
    trained = {slug for slug, _, ok in band_all if ok}
    by_level = {}
    for r in rows:
        if r.get('kind') == 'band' and r['model_slug'] in trained:
            by_level.setdefault(r['n_train_requested'], {})[r['model_slug']] = r
    matched = set.intersection(*(set(v) for v in by_level.values()))
    full = S.full_data_band(matched)
    n_full = max(r['n_train'] for r in ft_api.results('*', task=S.TASK, lang=S.LANG)
                 if r.get('steps') == S.STEPS and r.get('lr') == S.BAND_LR
                 and not r.get('n_train_requested') and r['model_slug'] in matched)
    levels = [(n_full, full)]
    for n in sorted(by_level, reverse=True):
        levels.append((n, S.spread([by_level[n][slug] for slug in matched])))

    xs = list(range(len(levels)))
    sds = [stats['between_sd'] for _, stats in levels]
    with column_type():
        fig, ax = _column_fig(2.15)
        ax.plot(xs, sds, 'o-', color=C1, markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3)
        for x, sd in zip(xs, sds):
            ax.text(x, sd + 0.0020, f'{sd:.4f}', ha='center', fontsize=9,
                    color=INK, fontweight='bold')
        ax.axhline(SIB_SD, ls=(0, (4, 3)), lw=1.6, color=C2, zorder=2)
        ax.text(xs[-1], SIB_SD - 0.0024, f'SIB-200 topic, {SIB_SD:.4f}', ha='right', va='top',
                fontsize=8.8, color=C2, fontweight='bold')
        ax.set_xticks(xs)
        ax.set_xticklabels([f'{n:,}' for n, _ in levels])
        ax.set_xlim(-0.32, len(levels) - 0.68)
        ax.set_ylim(0, SIB_SD * 1.22)
        ax.set_xlabel(f'NER training labels ({len(matched)} from-scratch models, all levels)')
        ax.set_ylabel('between-model sd')
        ax.set_title('Threefold does nothing; tenfold gets 43% of the way')
        fig.tight_layout()
        save(fig, '18-label-quantity-column')


if __name__ == '__main__':
    import sys

    ALL = (fig_headline, fig_gradient, fig_matched, fig_bimodal, fig_saturation, fig_cost,
           fig_why_long, fig_scaling, fig_early_signal, fig_metric_validity, fig_floors,
           fig_how_many_seeds, fig_speedup, fig_pipeline, fig_lr_transfer,
           fig_tokenizer_lottery, fig_label_quantity, fig_api, fig_board_layout, fig_hardware,
           fig_headline_column, fig_floors_column, fig_gradient_column,
           fig_saturation_column, fig_swap_column, fig_lottery_column,
           fig_label_quantity_column)

    # No argument regenerates everything, which is the right default. Naming one or more figures
    # renders only those, and that matters when the machine at hand is not the one the rest were
    # rendered on: a different matplotlib can shift layout by a pixel and churn all fourteen files
    # for no reason, which is the noise save()'s fixed hashsalt exists to keep out of git status.
    # Patrick's, from #53, kept verbatim -- he rendered figure 01 on a Colab runtime and this is
    # what stopped it touching the other nine.
    want = [w.lower() for w in sys.argv[1:]]
    fns = [fn for fn in ALL if not want or any(w in fn.__name__.lower() for w in want)]
    if want and not fns:
        raise SystemExit('no figure matches ' + repr(want) + ' -- have: '
                         + ', '.join(fn.__name__ for fn in ALL))

    for fn in fns:
        try:
            fn()
        except Exception as e:                       # noqa: BLE001 -- one figure must not stop the rest
            print('  ' + fn.__name__ + ' FAILED: ' + repr(e)[:140])
