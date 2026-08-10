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

import json
import math
import os
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ft_api
import mlm_api as f

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
    ax.set_title('A multilingual vocabulary costs nothing — until the language is left out',
                 pad=14)
    ax.plot([], [], 'o', color=C3, label='XLM-R was trained on this language')
    ax.plot([], [], 'o', color=C2, label='XLM-R was not')
    ax.legend(loc='lower right', fontsize=12)
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
    that length is a lie: $4.30 against $20,000 is 4,650x and the bars looked about 4x apart.
    A dot makes no promise about length; only its position carries the number.
    """
    items = [('Electricity\nfor 83 GPU-hours', 4.30, C3),
             ('Renting the same\ntime in the cloud', 250, C1),
             ('Buying the\nworkstation', 20000, C2)]
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
    ax.set_title('What 105 trained models cost', pad=26, loc='left')
    ax.text(0, 1.03, 'Renting wins below 9,300 GPU-hours of work. This project used 83.',
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
        floor = max((r for r in rows if r.get('task') == task and r.get('steps') == steps
                     and 'yor_random_init' in r['model']), key=lambda r: r['mean'])['mean']
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


if __name__ == '__main__':
    import sys

    ALL = (fig_headline, fig_gradient, fig_matched, fig_bimodal, fig_saturation, fig_cost,
           fig_why_long, fig_scaling, fig_early_signal, fig_metric_validity, fig_floors,
           fig_how_many_seeds, fig_speedup, fig_pipeline, fig_lr_transfer)

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
