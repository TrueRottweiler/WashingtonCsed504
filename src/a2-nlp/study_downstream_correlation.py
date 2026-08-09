"""Does the number we spent a term minimizing predict the number we actually care about?

Every pretraining decision in this project was made on validation loss. Not one of them was
checked against downstream usefulness, because of an accident of history: we have 107 trained
checkpoints on disk and exactly ONE has ever been fine-tuned. The correlation everyone assumes is
tight has never been looked at here, and it is cheap to look at -- fine-tuning costs 1.2 minutes
per seed against the ninety minutes the pretraining cost, all of which is already paid for.

The design is deliberately dull, because the question is not:

  * every Yoruba checkpoint whose vocabulary fingerprint matches the corpus it was trained on --
    a mismatched vocabulary would compare two different things and the fingerprint is what stops
    that happening silently;
  * both downstream tasks, because report 06 established they diverge and a correlation that
    holds on one and not the other is the interesting outcome;
  * three seeds per cell, because that is the rule this project learned the hard way;
  * default learning rate for every arm -- this measures the RANK ORDER induced by pretraining
    loss, not the best score any model can reach, so a per-model sweep would answer a different
    question and cost twelve times more.

What we expect: a tight negative correlation, because that is what the field assumes. What would
be more interesting: the correlation breaking down somewhere, most plausibly among the runs that
never learned, where every model is bad in the same way but their losses still differ by nats.

    bash src/a2-nlp/py.sh study_downstream_correlation.py --dry-run
    bash src/a2-nlp/py.sh study_downstream_correlation.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import ft_api
import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'downstream_correlation.json')

TASKS = [('sib200', 'yor_Latn', 1056), ('masakhaner', 'yor', 2150)]
SEEDS = (0, 1, 2)


def candidates():
    """Yoruba checkpoints that exist on disk and share the corpus's vocabulary.

    The fingerprint check is the whole reason this is trustworthy. Seven of our older runs
    predate fingerprint recording; they are skipped rather than assumed, because a checkpoint
    whose tokenizer we cannot verify is a checkpoint that could be scoring on a different
    vocabulary than it was trained on.
    """
    want = f.corpus_info('yor').get('tokenizer_fingerprint')
    out = []
    for r in f.results('*'):
        if r.get('corpus') != 'yor':
            continue
        d = os.path.join(HERE, 'runs', r['tag'])
        if not os.path.exists(os.path.join(d, 'model.safetensors')):
            continue
        fp = r.get('vocab_fingerprint')
        if fp != want:
            continue
        out.append(r)

    # Three tags carry TWO records with different losses pointing at the SAME directory: the run
    # was repeated and the checkpoint overwritten while both result files survived. The weights
    # on disk belong to one of those losses and nothing records which. Pairing a checkpoint with
    # the wrong loss would put a wrong point on the very curve this study is drawing, so an
    # ambiguous tag is dropped rather than guessed at.
    seen = {}
    for r in out:
        seen.setdefault(r['tag'], []).append(r)
    ambiguous = sorted(k for k, v in seen.items() if len(v) > 1)
    out = [v[0] for k, v in seen.items() if len(v) == 1]
    out.sort(key=lambda r: r['val_loss'])
    return out, want, ambiguous


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None, help='use only the first N checkpoints')
    ap.add_argument('--dry-run', action='store_true', help='print the plan and the cost, run nothing')
    a = ap.parse_args()

    models, fp, ambiguous = candidates()
    if a.limit:
        models = models[:a.limit]

    print(f'{len(models)} Yoruba checkpoints at vocabulary fingerprint {fp}')
    if ambiguous:
        print(f'{len(ambiguous)} excluded as ambiguous (two losses, one overwritten directory):')
        for tag in ambiguous:
            print(f'    {tag}')
    print(f'validation loss spans {models[0]["val_loss"]:.3f} to {models[-1]["val_loss"]:.3f}\n')
    cells = len(models) * len(TASKS)
    print(f'{cells} cells x {len(SEEDS)} seeds = {cells*len(SEEDS)} fine-tuning runs')
    print(f'at ~1.2 min/seed that is about {cells*len(SEEDS)*1.2/60:.1f} GPU-hours\n')
    for r in models:
        print(f'  {r["tag"]:<42} val={r["val_loss"]:.3f}')
    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    rows, t0 = [], time.time()
    for i, r in enumerate(models, 1):
        for task, lang, steps in TASKS:
            # reuse=True so an interrupted night resumes instead of re-spending what it already
            # spent -- the same property that makes the notebooks cheap to re-run.
            rec = ft_api.evaluate(f'runs/{r["tag"]}', task=task, lang=lang, steps=steps,
                                  seeds=SEEDS, reuse=True,
                                  label=f'corr_{r["tag"]}')
            rows.append({'tag': r['tag'], 'val_loss': r['val_loss'], 'task': task,
                         'mean': rec['mean'], 'sd': rec['sd'], 'ci': rec.get('ci'),
                         'n_tokens': r['n_tokens'], 'steps': r['steps'],
                         'preset': r.get('preset') or 'poc'})
            json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(models)}] {r["tag"]}  ({(time.time()-t0)/60:.0f} min elapsed)',
              flush=True)

    report(rows)


def report(rows=None):
    """Correlation between pretraining loss and downstream score, per task."""
    rows = rows or json.load(open(OUT, encoding='utf-8'))
    print('\n' + '=' * 74)
    print('DOES PRETRAINING LOSS PREDICT DOWNSTREAM SCORE?')
    print('=' * 74)
    for task, _, _ in TASKS:
        g = [r for r in rows if r['task'] == task]
        if len(g) < 4:
            continue
        x = [r['val_loss'] for r in g]
        y = [r['mean'] for r in g]
        r_p = st.correlation(x, y)
        # Spearman as well, because the question is whether the RANK ORDER is preserved -- a
        # curved but monotone relationship would still make validation loss a usable proxy.
        rx = _rank(x); ry = _rank(y)
        r_s = st.correlation(rx, ry)
        print(f'\n  {task}  (n={len(g)})')
        print(f'    Pearson  {r_p:+.3f}   Spearman {r_s:+.3f}')
        print(f"    {'val_loss':>10}{'score':>9}   {'tag'}")
        for r in sorted(g, key=lambda r: r['val_loss']):
            print(f'    {r["val_loss"]:>10.3f}{r["mean"]:>9.3f}   {r["tag"]}')


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    for pos, i in enumerate(order):
        out[i] = float(pos)
    return out


if __name__ == '__main__':
    main()
