"""Does MasakhaNER treat every model alike because of the TASK, or because it has ten times the
labels? Cut its training set to SIB-200's size and find out.

This is the one axis of the original study design that has never been run, and the design document
calls it the decisive experiment: "separate the effect of labelled-data quantity from the effect of
task type by subsampling the larger task's training set to match the smaller one."

WHAT IT MEASURES.

Across the from-scratch models measured on both tasks, MasakhaNER scores span a band of 0.044 while
SIB-200 scores span 0.143 -- three times as much. Entity recognition barely notices which
from-scratch model it is handed; topic classification does. Report 08 explains that by task type:
NER leans on capitalisation and name shape, which transfer without knowing any Yoruba.

But the two tasks differ in label count as well as in kind -- 6,876 against 701 -- and both
explanations predict what we see. "NER is flat because it is a surface task" and "NER is flat
because 6,876 labels are enough for any encoder to reach the same place" are not distinguished by
anything else in the study. Subsampling NER to 701 distinguishes them:

    band at 701 labels widens toward SIB-200's 0.143   -> label QUANTITY
    band at 701 labels stays near its full-data value  -> task TYPE

The instrument is therefore the BAND across from-scratch models, not a single head-to-head. Three
arms of different architectures cannot show whether the from-scratch band widened, because they are
not the thing the band is measured over. mmBERT and the untrained floor run as CONTEXT, off to one
side, and are excluded from the band by name.

THREE CORRECTNESS TRAPS, and this project has paid for the shape of all three already.

1. A RANGE GROWS WITH THE NUMBER OF MODELS IN IT. The published 0.044 is a range over sixteen
   models. Running six at 701 labels and comparing their range against 0.044 would compare a
   range-over-six with a range-over-sixteen and read the difference as a finding. Verified on the
   real records: the same full-split data gives 0.307 over nineteen models, 0.044 over sixteen and
   0.022 over an arbitrary six. So the full-data band is recomputed here over EXACTLY the models
   that ran at 701, and the script refuses to print a comparison across unequal sets. The
   between-model standard deviation is reported beside every range because it does not grow with
   the set size, which makes it the statistic to quote when the sets cannot be matched.

2. THE EXCLUSION IS LOAD-BEARING. The 0.044 is a band over the sixteen models that trained,
   defined as pretraining val_loss < 3.1 -- the same cut figures 11 and 12 use. It leaves out two
   4M-token rungs and the afriberta-preset run, which never left the unigram plateau. Include them
   and the band is 0.307 and the finding evaporates. That is a large consequence for a threshold
   nobody derived, so the report prints the band BOTH ways at every label count. If the reading
   depends on the cut, the reader gets to see that rather than inherit it.

3. THE FIXED LEARNING RATE. Every band cell runs at 3e-5, the rate the 0.044 was measured at. It is
   held fixed on purpose -- the comparison is within-model across label count, so sweeping would
   fold rate selection into the effect -- and stated, because an unstated fixed constant is this
   project's signature failure mode. Two caveats travel with it. 3e-5 is not the best rate for
   these models (ours scores 0.7877 at 3e-5 and 0.8373 at 1e-4), so this is a band at a fixed rate,
   not a band of best-achievable scores. And 3e-5 is not known to suit 701 labels; if the band
   widens, the first follow-up is whether it widens because the rate suits some models better
   there.

   The untrained floor is the exception, and it is the reason to read this comment. Its rate is NOT
   3e-5. Swept over twelve rates its best is 0.626 at 3e-4, against the 0.4140 this project printed
   for a fortnight -- which turns out to be its 3e-5 cell, a rate one tenth of its best. So the
   floor runs at 3e-4 here, and quoting it at the band's rate would reproduce the error that sweep
   just corrected.

WHAT HOLDING THE STEP BUDGET FIXED MEANS HERE. At 701 sentences and 2,150 updates at batch 16 a
model sees each example about 49 times, against 5 times on the full split. That is intended: ft_api
fixes the budget so a subsample does not silently become a compute cut, which would measure a 10x
smaller budget and call it a label-count effect. It does mean these cells train well past fitting
their training set, which is a property of the comparison being about labels rather than compute.

    bash src/a2-nlp/py.sh study_label_quantity.py --report      # what this machine can contribute
    bash src/a2-nlp/py.sh study_label_quantity.py --dry-run
    bash src/a2-nlp/py.sh study_label_quantity.py --gpu 1
    bash src/a2-nlp/py.sh study_label_quantity.py --gpu 1 --levels 701 --context

A from-scratch NER cell is 38 s/seed on a Blackwell card, so one label level over sixteen models at
three seeds is about half an hour. An A100 is 1.6-1.7x slower on these cells, measured from
seconds_per_seed on the records rather than estimated. The part that decides where this runs:
fifteen of the sixteen checkpoints exist only on the workstation, and --report says so plainly
before anything starts.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import fleet_plan
import ft_api
import mlm_api as factory

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'label_quantity.json')

TASK, LANG, STEPS = 'masakhaner', 'yor', 2150
BAND_LR = 3e-5              # the rate the published 0.044 band was measured at -- see trap 3
FLOOR_LR = 3e-4             # the untrained control's SWEPT best, not the band's rate
SEEDS = (0, 1, 2)

# SIB-200's training split size, which is what "match the smaller task" means, plus one rung
# between it and the full 6,876 so a widening band can be seen turning rather than only at its ends.
LEVELS = (701, 2000)

# Trap 2. The cut that defines "the models that trained", and the same one figures 11 and 12 use.
TRAINED_MAX_LOSS = 3.1

# SIB-200's own band at 701 labels: what the label-quantity explanation predicts NER should
# approach. The only external number in the report.
SIB_BAND = 0.143

# Excluded from the band by name, not by a substring test on the path. Matching arms with
# `frag in record['model']` is how fig_headline and claims_audit both went wrong: 'yor_64M' catches
# three different models, and the model field is a filesystem path that differs between machines.
NOT_FROM_SCRATCH = {'mmBERT-base', 'xlm-roberta-base', 'yor-random-init',
                    'xlm-roberta-base-random-init'}

# Context arms: not part of the band, run only with --context. Each carries its own rate, and the
# floor's is deliberately not the band's.
CONTEXT = [('mmBERT base', 'jhu-clsp/mmBERT-base', BAND_LR),
           ('our architecture, untrained (floor)', 'runs/yor_random_init', FLOOR_LR)]


def val_losses() -> dict[str, float]:
    """Pretraining val_loss per run tag, for the trap-2 cut. Through mlm_api, not the files."""
    return {r['tag']: r['val_loss'] for r in factory.results() if r.get('val_loss') is not None}


def band_models() -> list[tuple[str, str, bool]]:
    """The from-scratch models with a full-data NER row at the band's rate.

    Read out of the records rather than listed here, so the set cannot drift from what the 0.044
    was computed over. Keyed on model_slug, which is the basename normalised -- the model field is
    a path and differs between the workstation and a Colab runtime.

    Returns (slug, path-in-this-checkout, trained) where `trained` is the trap-2 cut.
    """
    seen: dict[str, str] = {}
    for r in ft_api.results('*', task=TASK, lang=LANG):
        if r.get('steps') != STEPS or r.get('lr') != BAND_LR or r.get('n_train_requested'):
            continue
        if r['model_slug'] in NOT_FROM_SCRATCH:
            continue
        seen.setdefault(r['model_slug'], r['model'])
    loss = val_losses()
    out = []
    for slug, p in sorted(seen.items()):
        tag = os.path.basename(p.rstrip('/\\'))
        # The record's path is the launching machine's; resolve against this checkout instead.
        out.append((slug, os.path.join('runs', tag),
                    loss.get(tag, float('inf')) < TRAINED_MAX_LOSS))
    return out


def present(path: str) -> bool:
    """A checkpoint is usable if it has a config.json. A bare directory is not a model, and the
    failure it produces otherwise arrives 40 minutes into a sweep rather than before it starts."""
    return os.path.exists(os.path.join(HERE, path, 'config.json'))


def spread(rows: list[dict]) -> dict:
    """The band, the statistic that survives an unequal set size, and the noise to read them
    against.

    `range` is what report 08 quotes and is kept for continuity. `between_sd` is the one to compare
    across sets of different size. `within_sd` pools each cell's own seed spread, which is the
    scale that says whether a band is bigger than the noise inside it -- and it is converted to a
    sample sd, because the sd stored on a record is a population sd over three seeds while every
    "N times the spread" rule in this project was derived for a sample sd. That is a 22% gap at
    three seeds.
    """
    means = [r['mean'] for r in rows]
    n = len(rows)
    k = len(rows[0].get('seeds') or SEEDS) if rows else len(SEEDS)
    scale = (k / (k - 1)) ** 0.5 if k > 1 else 1.0
    within = [r['sd'] * scale for r in rows if r.get('sd') is not None]
    return {'n_models': n,
            'range': (max(means) - min(means)) if n > 1 else 0.0,
            'between_sd': st.stdev(means) if n > 1 else 0.0,
            'within_sd': (st.mean([s ** 2 for s in within]) ** 0.5) if within else float('nan'),
            'lo': min(means) if means else None, 'hi': max(means) if means else None,
            'models': {r['model_slug']: r['mean'] for r in rows}}


def full_data_band(slugs: set[str]) -> dict | None:
    """The full-split band over EXACTLY these models. Trap 1: this is the only thing a 701-label
    band may be compared with."""
    rows = [r for r in ft_api.results('*', task=TASK, lang=LANG)
            if r.get('steps') == STEPS and r.get('lr') == BAND_LR
            and not r.get('n_train_requested') and r['model_slug'] in slugs]
    return spread(rows) if len(rows) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--levels', default=','.join(str(n) for n in LEVELS),
                    help='label counts to run, comma separated. 701 is the experiment; drop 2000 '
                         'first if the window is short.')
    ap.add_argument('--context', action='store_true',
                    help='also run mmBERT and the untrained floor at each level. Not part of the '
                         'band -- context for the panel.')
    ap.add_argument('--all-models', action='store_true',
                    help='include the rungs that never left the unigram plateau. Off by default so '
                         'the band matches the published one; the report prints both cuts anyway.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', action='store_true',
                    help='print what this machine can contribute, and any results already on disk. '
                         'No GPU, no training.')
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(',') if x.strip()]
    band = [(s, p, t) for s, p, t in band_models() if t or a.all_models]
    have = [(s, p) for s, p, _ in band if present(p)]
    missing = [(s, p) for s, p, _ in band if not present(p)]
    skipped = [s for s, _, t in band_models() if not t]

    print(f'from-scratch models with a full-data NER row at lr {BAND_LR:g}: {len(band)}')
    print(f'  on this disk: {len(have)}')
    for s, _ in have:
        print(f'    {s}')
    if missing:
        print(f'  NOT on this disk, so not runnable here: {len(missing)}')
        for s, p in missing:
            print(f'    {s:<32} (expected {p}/)')
    if skipped and not a.all_models:
        print(f'  below the val_loss < {TRAINED_MAX_LOSS} cut, excluded from the band: '
              f'{", ".join(skipped)}')

    # The band is a range over models. Fewer models is not merely a smaller experiment, it is a
    # smaller expected range -- so say plainly when a machine cannot run the instrument at all.
    if len(have) < 6:
        print(f'\n  WARNING: {len(have)} model(s) cannot measure a band. The published 0.044 is a '
              f'range over\n  sixteen; over six of the same models it is 0.022, and that '
              f'difference is arithmetic, not\n  a result. The workstation has all of them, which '
              f'is why this experiment belongs there.\n  What a one-checkpoint machine CAN '
              f'contribute is the --context arms and a single\n  within-model level comparison, '
              f'which is worth having but is not the experiment.')

    if a.report:
        report()
        return

    cells = [(slug, path, n, BAND_LR, 'band') for n in levels for slug, path in have]
    if a.context:
        cells += [(label, path, n, lr, 'context')
                  for n in levels for label, path, lr in CONTEXT]
    est = sum(38 if kind == 'band' else 161 for *_, kind in cells) * len(SEEDS) / 60
    print(f'\n{len(cells)} cells x {len(SEEDS)} seeds = {len(cells)*len(SEEDS)} fine-tuning runs')
    print(f'  ~{est:.0f} min on a Blackwell card, ~{est*1.7:.0f} min on an A100, from '
          f'seconds_per_seed on the records')

    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return
    if not cells:
        print('\nNothing to run on this machine.')
        return

    fleet_plan.announce(
        'MasakhaNER at SIB-200s label count: quantity or task type',
        [fleet_plan.cell(ft_api.record_tag(path, TASK, LANG, n, lr, STEPS),
                         f'{label}  n={n}', kind='finetune', steps=STEPS, eta_s=135)
         for label, path, n, lr, _ in cells],
        owner='Patrick')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    # Resume rather than restart. reuse=True makes a finished cell free to re-request, but the
    # results file should not lose the cells an interrupted run already wrote.
    rows: list[dict] = []
    if os.path.exists(OUT):
        rows = [r for r in json.load(open(OUT, encoding='utf-8')) if 'mean' in r]
    t0 = time.time()
    for i, (label, path, n, lr, kind) in enumerate(cells, 1):
        try:
            rec = ft_api.evaluate(path, task=TASK, lang=LANG, steps=STEPS, lr=lr, n_train=n,
                                  seeds=SEEDS, reuse=True, label=f'{label} n={n}')
            rows = [r for r in rows
                    if not (r.get('model_slug') == rec['model_slug']
                            and r.get('n_train_requested') == n)]
            rows.append({'kind': kind, 'label': label, 'model_slug': rec['model_slug'],
                         'n_train_requested': n, 'n_train': rec['n_train'], 'lr': lr,
                         'mean': rec['mean'], 'sd': rec.get('sd'), 'ci': rec.get('ci'),
                         'scores': rec.get('scores'), 'seeds': rec.get('seeds')})
        except Exception as e:                   # noqa: BLE001 -- one cell must not stop the rest
            print(f'  FAILED {label} n={n}: {repr(e)[:120]}', flush=True)
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(cells)}] {label} n={n}  ({(time.time()-t0)/60:.0f} min elapsed)',
              flush=True)

    report(rows)


def reading(at: dict, base: dict) -> list[str]:
    """The decision rule, written before the runs and applied to whatever came back.

    Order matters and it is not the order it was first written in. "Did the band widen at all"
    is tested BEFORE "is it as wide as SIB-200", because a set whose baseline range is already
    large would otherwise satisfy the second test while showing no change -- which is exactly what
    happened on the first synthetic pass, where an unchanged band read as LABEL QUANTITY.
    """
    n = at['n_models']
    if at['range'] <= base['range'] + at['within_sd']:
        return [f'-> TASK TYPE. Cutting the labels to SIB-200s count did not make NER '
                f'discriminate:',
                f'   {at["range"]:.3f} against {base["range"]:.3f} on the full split, inside the '
                f'within-cell spread',
                f'   of {at["within_sd"]:.3f}. The divergence is about what the task needs, not '
                f'about how many',
                f'   labels it has. Report 08 section 3.6c stands, and the decisive experiment '
                f'says so.']
    if at['range'] >= 0.75 * SIB_BAND:
        return [f'-> LABEL QUANTITY. At this label count NER spreads models nearly as widely as '
                f'topic',
                f'   classification does ({at["range"]:.3f} against {SIB_BAND:.3f}). NERs flatness '
                f'on the full split is',
                f'   about having 6,876 labels, not about being a surface task. Report 08 section '
                f'3.6c needs',
                f'   rewriting and so does the poster panel.']
    return [f'-> BETWEEN THE TWO. {at["range"]:.3f} is above the full-split {base["range"]:.3f} '
            f'but short of topic',
            f'   classifications {SIB_BAND:.3f}. Three seeds and {n} models cannot split these; '
            f'say so rather than',
            f'   rounding it to whichever explanation the panel prefers.']


def report(rows=None):
    if rows is None:
        if not os.path.exists(OUT):
            print('\nNothing on disk yet.')
            return
        rows = json.load(open(OUT, encoding='utf-8'))
    rows = [r for r in rows if 'mean' in r]
    if not rows:
        print('\nNothing on disk yet.')
        return

    print('\n' + '=' * 78)
    print('IS NERs FLATNESS ABOUT THE TASK, OR ABOUT HAVING TEN TIMES THE LABELS')
    print('=' * 78)

    trained = {s for s, _, t in band_models() if t}
    band_rows = [r for r in rows if r.get('kind') == 'band']

    # Trap 2: print it both ways rather than inheriting the cut.
    for cut_label, keep in (('the models that trained (val_loss < %g)' % TRAINED_MAX_LOSS, trained),
                            ('every from-scratch model', None)):
        sel = [r for r in band_rows if keep is None or r['model_slug'] in keep]
        if len(sel) < 2:
            continue
        slugs = {r['model_slug'] for r in sel}
        base = full_data_band(slugs)
        print(f'\n\n--- {cut_label} ---')
        if base is None:
            print('  No full-split rows for these models at the band rate; nothing to compare.')
            continue
        print(f'\n  Full split (6,876 labels), {base["n_models"]} models, lr {BAND_LR:g}:')
        print(f'    range {base["range"]:.4f}   between-model sd {base["between_sd"]:.4f}   '
              f'within-cell sd {base["within_sd"]:.4f}')
        print(f'    {base["lo"]:.4f} - {base["hi"]:.4f}')
        if base['n_models'] != 16:
            print(f'    NB: a range over {base["n_models"]}, not the published 0.044 over 16. '
                  f'Compare only with the\n        lines below, which are over the same '
                  f'{base["n_models"]}.')

        for n in sorted({r['n_train_requested'] for r in sel}):
            at = spread([r for r in sel if r['n_train_requested'] == n])
            if at['n_models'] < 2:
                print(f'\n  {n} labels: {at["n_models"]} model(s) -- a band needs more than one.')
                continue
            print(f'\n  {n} labels, {at["n_models"]} models, lr {BAND_LR:g}:')
            print(f'    range {at["range"]:.4f}   between-model sd {at["between_sd"]:.4f}   '
                  f'within-cell sd {at["within_sd"]:.4f}')
            print(f'    {at["lo"]:.4f} - {at["hi"]:.4f}')
            for slug, m in sorted(at['models'].items(), key=lambda kv: -kv[1]):
                was = base['models'].get(slug)
                d = f'  ({m - was:+.4f} vs full split)' if was is not None else ''
                print(f'      {slug:<32} {m:.4f}{d}')
            if at['n_models'] != base['n_models']:
                print('    Not comparable with the full-split line: different model sets. Trap 1.')
                continue
            print(f'\n    band x{at["between_sd"]/base["between_sd"]:.2f} on between-model sd, '
                  f'x{at["range"]/base["range"]:.2f} on range'
                  if base['between_sd'] and base['range'] else '')
            for line in reading(at, base):
                print('    ' + line)

    ctx = [r for r in rows if r.get('kind') == 'context']
    if ctx:
        print('\n\n  Context (not part of the band):')
        for r in sorted(ctx, key=lambda r: (r['n_train_requested'], -r['mean'])):
            print(f'    {r["label"]:<38} n={r["n_train_requested"]:<5} lr {r["lr"]:<8g} '
                  f'{r["mean"]:.4f}')
        print(f'    The floor runs at {FLOOR_LR:g}, its swept best. At 3e-5 it scores 0.414, and '
              f'that cell is a\n    rate one tenth of its best -- see trap 3 in the docstring.')


if __name__ == '__main__':
    main()
