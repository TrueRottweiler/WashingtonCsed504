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

FOUR CORRECTNESS TRAPS, and this project has paid for the shape of all four already.

1. A RANGE GROWS WITH THE NUMBER OF MODELS IN IT. The published 0.044 is a range over sixteen
   models. Running six at 701 labels and comparing their range against 0.044 would compare a
   range-over-six with a range-over-sixteen and read the difference as a finding. Verified on the
   real records: the same full-split data gives 0.307 over nineteen models, 0.044 over sixteen and
   0.022 over an arbitrary six. So the full-data band is recomputed here over EXACTLY the models
   that ran at each label count -- per level, not once over the union of the levels, which is a
   distinction that cost a printed verdict; see trap 4 -- and the script refuses to print a
   comparison across unequal sets. The between-model standard deviation is reported beside every
   range because it does not grow with the set size, which makes it the statistic to quote when
   the sets cannot be matched, and it is what `reading()` decides on.

2. THE EXCLUSION IS LOAD-BEARING. The 0.044 is a band over the sixteen models that trained,
   defined as pretraining val_loss < 3.1 -- the same cut figures 11 and 12 use. It leaves out two
   4M-token rungs and the afriberta-preset run, which never left the unigram plateau. Include them
   and the band is 0.307 and the finding evaporates. That is a large consequence for a threshold
   nobody derived, so the report prints the band BOTH ways at every label count. If the reading
   depends on the cut, the reader gets to see that rather than inherit it.

2b. AND THE CUT ONLY MEANS ANYTHING WITHIN ONE VOCABULARY. `val_loss` is nats per token, so it
   compares two models only when they score over the same vocabulary -- the project's oldest
   gotcha, and the reason `vocab_fingerprint` is on every pretraining record. The four
   swap_yor_xlmr_* models are the deliberately-crippled arm of the tokenizer-swap experiment,
   trained on XLM-R's 250k vocabulary, and they sail through `val_loss < 3.1` at 1.24-1.70 for a
   reason that has nothing to do with training quality: fewer nats per token because there are
   more tokens. They are also 153.8M parameters against 33.8M. So the band is restricted to one
   vocabulary fingerprint, and the models dropped are printed by name. See trap 4 for what this
   cost before it was guarded.

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

4. THE MODEL SET IS NOT A CONSTANT, AND THIS ONE ACTUALLY FIRED. `band_models()` reads the set out
   of the records rather than listing it, so that it cannot drift from what the 0.044 was computed
   over. That is the right instinct and it was not sufficient: the set is "every from-scratch model
   with a full-split NER row at 3e-5", which is a property of what ANYONE has run. Between the run
   that produced these results and the merge of the records that justify them, a downstream sweep
   elsewhere in the project gave the four XLM-R-vocabulary swap models their missing full-split
   rows. The set went 16 -> 21 with no change to this file, the band went 0.069 -> 0.182, and the
   verdict printed LABEL QUANTITY where the run had said BETWEEN THE TWO -- a reversed conclusion,
   from records landing.

   Two things were wrong and both are fixed above. The four models did not belong in the band at
   all (trap 2b). And `reading()` compared an absolute range against SIB-200's 0.143, a range over
   sixteen -- so the moment the set was not sixteen, the decision rule was itself committing trap 1.
   It decides on between-model sd now, which does not grow with the set size.

   The lesson is the project's own, one noun further out: it is not enough for a derived set to be
   computed rather than hardcoded. A set computed from a shared directory is a query against other
   people's work, and it has to be pinned by the property that defines it -- here one vocabulary
   and one architecture -- or it will answer a different question next week.

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

# SIB-200's own spread across the SAME sixteen models: what the label-quantity explanation predicts
# NER should approach. The only external numbers in the report.
#
# Both are quoted because both get printed, but the RULE decides on the sd. A range is only
# comparable against another range over the same number of models (trap 1), and this one is over
# sixteen while the band being judged is over whatever ran -- which is exactly how the rule came to
# contradict its own docstring. The sd does not grow with the set size, so it can be a constant.
SIB_BAND = 0.1426           # range over the sixteen, for printing
SIB_BETWEEN_SD = 0.0457     # between-model sd over the same sixteen -- what reading() tests

# Excluded from the band by name, not by a substring test on the path. Matching arms with
# `frag in record['model']` is how fig_headline and claims_audit both went wrong: 'yor_64M' catches
# three different models, and the model field is a filesystem path that differs between machines.
NOT_FROM_SCRATCH = {'mmBERT-base', 'xlm-roberta-base', 'yor-random-init',
                    'xlm-roberta-base-random-init'}

# Context arms: not part of the band, run only with --context. Each carries its own rate, and the
# floor's is deliberately not the band's.
CONTEXT = [('mmBERT base', 'jhu-clsp/mmBERT-base', BAND_LR),
           ('our architecture, untrained (floor)', 'runs/yor_random_init', FLOOR_LR)]


def pretrain_index() -> dict[str, dict]:
    """Pretraining record per run tag, for the trap-2 cut and the trap-2b vocabulary guard.
    Through mlm_api, not the files."""
    return {r['tag']: r for r in factory.results() if r.get('val_loss') is not None}


def band_models() -> tuple[list[tuple[str, str, bool]], list[tuple[str, str]]]:
    """The from-scratch models with a full-data NER row at the band's rate.

    Read out of the records rather than listed here, so the set cannot drift from what the 0.044
    was computed over. Keyed on model_slug, which is the basename normalised -- the model field is
    a path and differs between the workstation and a Colab runtime.

    Reading it out of the records is necessary and not sufficient (trap 4): the query is against a
    shared directory, so it grows whenever anyone adds a full-split row. It is therefore pinned to
    ONE VOCABULARY -- the band is a spread over models that differ in pretraining, and a model
    scored over a different vocabulary is not another point on that axis, it is a different
    experiment. The majority fingerprint defines the family rather than a hardcoded hash, so this
    keeps working if the shared BPE is ever retrained.

    Returns (band, dropped): band is (slug, path-in-this-checkout, trained) where `trained` is the
    trap-2 cut; dropped is (slug, reason) for everything the vocabulary guard removed, so it can be
    printed by name instead of vanishing.
    """
    seen: dict[str, str] = {}
    for r in ft_api.results('*', task=TASK, lang=LANG):
        if r.get('steps') != STEPS or r.get('lr') != BAND_LR or r.get('n_train_requested'):
            continue
        if r['model_slug'] in NOT_FROM_SCRATCH:
            continue
        seen.setdefault(r['model_slug'], r['model'])

    index = pretrain_index()
    tags = {slug: os.path.basename(p.rstrip('/\\')) for slug, p in seen.items()}

    # The vocabulary the band is measured over: the one most of these models share. Taking the
    # majority rather than naming a hash means a retrained shared BPE does not empty the band.
    counts: dict[str, int] = {}
    for tag in tags.values():
        fp = (index.get(tag) or {}).get('vocab_fingerprint')
        if fp:
            counts[fp] = counts.get(fp, 0) + 1
    home = max(counts, key=counts.get) if counts else None

    band, dropped = [], []
    for slug, tag in sorted(tags.items()):
        rec = index.get(tag) or {}
        fp = rec.get('vocab_fingerprint')
        if home and fp and fp != home:
            dropped.append((slug, f'vocabulary {fp}, not the band\'s {home}'))
            continue
        # The record's path is the launching machine's; resolve against this checkout instead.
        band.append((slug, os.path.join('runs', tag),
                     rec.get('val_loss', float('inf')) < TRAINED_MAX_LOSS))
    return band, dropped


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
    ap.add_argument('--shard', default=None, metavar='i/k',
                    help='run only every k-th cell, starting at the i-th. Two cards on the same '
                         'levels: --shard 1/2 on one, --shard 2/2 on the other. Safe to combine '
                         'because the results file is merged on every write, not held in memory.')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--report', action='store_true',
                    help='print what this machine can contribute, and any results already on disk. '
                         'No GPU, no training.')
    a = ap.parse_args()

    levels = [int(x) for x in a.levels.split(',') if x.strip()]
    all_band, dropped = band_models()
    band = [(s, p, t) for s, p, t in all_band if t or a.all_models]
    have = [(s, p) for s, p, _ in band if present(p)]
    missing = [(s, p) for s, p, _ in band if not present(p)]
    skipped = [s for s, _, t in all_band if not t]

    if dropped:
        # Trap 2b/4: printed by name, because these are the models whose silent inclusion reversed
        # the verdict once already.
        print(f'  not on the band\'s vocabulary, excluded: {len(dropped)}')
        for s, why in dropped:
            print(f'    {s:<32} ({why})')

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

    # Sharding, so two cards can share one level rather than taking a level each.
    #
    # Splitting by --levels looks like the obvious way to use both GPUs and is not: the levels are
    # not equal work, and once a level's cells are mostly on disk its card finishes in a minute
    # and then sits idle while the other grinds through a full pass. Dealing every k-th cell puts
    # the same number on each card whatever is already cached, and because reuse=True makes a
    # finished cell free, a shard that happens to draw cached cells simply finishes early rather
    # than doing someone else's work twice.
    #
    # Interleaved rather than split down the middle on purpose: the band models are ordered by
    # size, so contiguous halves would hand one card all the large-vocabulary checkpoints.
    if a.shard:
        i, k = (int(x) for x in a.shard.split('/'))
        cells = cells[i - 1::k]
        print(f'shard {i} of {k}: {len(cells)} of the cells')

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
    def merge(rec=None, n=None, kind=None, label=None):
        """Re-read, merge, write. Never hold the results file in memory across a cell.

        The original held `rows` in memory and rewrote the whole file after each cell, which is
        correct for one process and destructive for two: each would write back its own view and
        drop the other's cells. That is not hypothetical -- the same shape made ten finished cells
        appear to vanish from the NER floor sweep on the 9th, and the per-cell ft_*.json records
        were the only reason nothing was actually lost.

        Re-reading immediately before each write costs a few milliseconds and makes it safe to
        run one level per card, which is what turns this from a ninety-minute job into a
        forty-five-minute one.
        """
        try:
            with open(OUT, encoding='utf-8') as fh:
                cur = [r for r in json.load(fh) if 'mean' in r]
        except (OSError, ValueError):
            cur = []
        if rec is not None:
            cur = [r for r in cur
                   if not (r.get('model_slug') == rec['model_slug']
                           and r.get('n_train_requested') == n)]
            cur.append({'kind': kind, 'label': label, 'model_slug': rec['model_slug'],
                        'n_train_requested': n, 'n_train': rec['n_train'], 'lr': rec['lr'],
                        'mean': rec['mean'], 'sd': rec.get('sd'), 'ci': rec.get('ci'),
                        'scores': rec.get('scores'), 'seeds': rec.get('seeds')})
        tmp = f'{OUT}.{os.getpid()}.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(cur, fh, indent=2)
        os.replace(tmp, OUT)
        return cur

    rows = merge()
    t0 = time.time()
    for i, (label, path, n, lr, kind) in enumerate(cells, 1):
        try:
            rec = ft_api.evaluate(path, task=TASK, lang=LANG, steps=STEPS, lr=lr, n_train=n,
                                  seeds=SEEDS, reuse=True, label=f'{label} n={n}')
            rows = merge(rec, n=n, kind=kind, label=label)
        except Exception as e:                   # noqa: BLE001 -- one cell must not stop the rest
            print(f'  FAILED {label} n={n}: {repr(e)[:120]}', flush=True)
            rows = merge()
        print(f'  [{i}/{len(cells)}] {label} n={n}  ({(time.time()-t0)/60:.0f} min elapsed)',
              flush=True)

    report(rows)


def reading(at: dict, base: dict) -> list[str]:
    """The decision rule, written before the runs and applied to whatever came back.

    Order matters and it is not the order it was first written in. "Did the band widen at all"
    is tested BEFORE "is it as wide as SIB-200", because a set whose baseline spread is already
    large would otherwise satisfy the second test while showing no change -- which is exactly what
    happened on the first synthetic pass, where an unchanged band read as LABEL QUANTITY.

    It decides on BETWEEN-MODEL SD, not on the range, and that is a correction rather than a
    preference. `at` and `base` are always over the same models, so comparing their ranges is fair;
    but the second test compares against SIB-200, whose figure is over sixteen. The moment this
    experiment ran over some other number, testing `range >= 0.75 * SIB_BAND` was committing the
    very trap the file's docstring opens with. The sd does not grow with the set size, so it is the
    statistic that can be held against an outside constant. Trap 4.
    """
    n = at['n_models']
    if at['between_sd'] <= base['between_sd'] + at['within_sd']:
        return [f'-> TASK TYPE. Cutting the labels to SIB-200s count did not make NER '
                f'discriminate:',
                f'   between-model sd {at["between_sd"]:.4f} against {base["between_sd"]:.4f} on '
                f'the full split, inside',
                f'   the within-cell spread of {at["within_sd"]:.4f}. The divergence is about what '
                f'the task needs,',
                f'   not about how many labels it has. Report 08 section 3.6c stands, and the '
                f'decisive',
                f'   experiment says so.']
    if at['between_sd'] >= 0.75 * SIB_BETWEEN_SD:
        return [f'-> LABEL QUANTITY. At this label count NER spreads models nearly as widely as '
                f'topic',
                f'   classification does (between-model sd {at["between_sd"]:.4f} against '
                f'{SIB_BETWEEN_SD:.4f}). NERs',
                f'   flatness on the full split is about having 6,876 labels, not about being a '
                f'surface',
                f'   task. Report 08 section 3.6c needs rewriting and so does the poster panel.']
    return [f'-> BETWEEN THE TWO. Between-model sd {at["between_sd"]:.4f} is above the full-split '
            f'{base["between_sd"]:.4f}',
            f'   but short of topic classifications {SIB_BETWEEN_SD:.4f} -- '
            f'{at["between_sd"]/SIB_BETWEEN_SD:.0%} of the way, on a',
            f'   statistic that does not grow with the set size. Three seeds and {n} models cannot '
            f'split',
            f'   these; say so rather than rounding it to whichever explanation the panel prefers.',
            f'   (Ranges, over these same {n}: {at["range"]:.4f} at this level against '
            f'{base["range"]:.4f} on the full',
            f'   split, and {SIB_BAND:.4f} for topic.)']


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

    all_band, dropped = band_models()
    trained = {s for s, _, t in all_band if t}
    on_band = {s for s, _, _ in all_band}
    band_rows = [r for r in rows if r.get('kind') == 'band']

    # Trap 2b/4. Results written before the vocabulary guard existed can contain cells for models
    # the band no longer includes, so say what is being left out rather than quietly dropping it.
    extra = {r['model_slug'] for r in band_rows} - on_band
    if extra:
        print(f'\n  Excluded from the band, not on its vocabulary: {", ".join(sorted(extra))}')
        print('  They are a different experiment (the tokenizer swap), not another point on this')
        print('  axis. Including them takes the 701-label band from 0.069 to 0.182 and reverses')
        print('  the verdict -- see trap 4.')

    # Trap 2: print it both ways rather than inheriting the cut.
    for cut_label, keep in (('the models that trained (val_loss < %g)' % TRAINED_MAX_LOSS, trained),
                            ('every from-scratch model', on_band)):
        sel = [r for r in band_rows if r['model_slug'] in keep]
        if len(sel) < 2:
            continue
        print(f'\n\n--- {cut_label} ---')

        for n in sorted({r['n_train_requested'] for r in sel}):
            at = spread([r for r in sel if r['n_train_requested'] == n])
            if at['n_models'] < 2:
                print(f'\n  {n} labels: {at["n_models"]} model(s) -- a band needs more than one.')
                continue

            # Trap 1, and the part of it that was got wrong: the baseline is recomputed over the
            # models at THIS level, not once over the union of every level. Computing it over the
            # union and then comparing a 16-model level against it is the very comparison across
            # unequal sets the trap forbids -- and it silently suppressed the 2,000-label reading,
            # which is where the sharpest result turned out to be.
            base = full_data_band(set(at['models']))
            print(f'\n  {n} labels, {at["n_models"]} models, lr {BAND_LR:g}:')
            print(f'    range {at["range"]:.4f}   between-model sd {at["between_sd"]:.4f}   '
                  f'within-cell sd {at["within_sd"]:.4f}')
            print(f'    {at["lo"]:.4f} - {at["hi"]:.4f}')
            if base is None:
                print('    No full-split rows for these models at the band rate; '
                      'nothing to compare.')
                continue
            print(f'    full split over the same {base["n_models"]}: range {base["range"]:.4f}   '
                  f'between-model sd {base["between_sd"]:.4f}   '
                  f'[{base["lo"]:.4f} - {base["hi"]:.4f}]')
            for slug, m in sorted(at['models'].items(), key=lambda kv: -kv[1]):
                was = base['models'].get(slug)
                d = f'  ({m - was:+.4f} vs full split)' if was is not None else ''
                print(f'      {slug:<32} {m:.4f}{d}')
            if at['n_models'] != base['n_models']:
                print('    Not comparable with the full-split line: different model sets. Trap 1.')
                continue
            if base['between_sd'] and base['range']:
                print(f'\n    band x{at["between_sd"]/base["between_sd"]:.2f} on between-model sd, '
                      f'x{at["range"]/base["range"]:.2f} on range')
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
