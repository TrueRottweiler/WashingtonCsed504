"""Sweep the MasakhaNER untrained controls, so the floor is chosen the way everything above it is.

Patrick's dev-split sweep closed this hole on SIB-200 and left it wide open on MasakhaNER. The
asymmetry is exact and it is ours:

    mmBERT            5 rates, best 0.8628
    XLM-R             5 rates, best 0.8513
    from-scratch      8 rates, best 0.8373
    untrained control 1 rate,        0.4140   <- everything above it was swept; this was not

Every "clears the floor by" figure on NER inherits that. Worse, once the floors go on a wall side
by side, SIB's control will have been swept nine ways and NER's once, so a cross-task floor
comparison would be setting a tuned control against an untuned one -- which is the same
best-of-sweep-against-a-default asymmetry report 08 section 2b spent three passes removing, just
pointed at the bottom of the chart instead of the top.

Both untrained arms, the full rate range every other arm has seen, three seeds. Selection is on
test here rather than on a dev split, because MasakhaNER as we load it has no dev split and
inventing one for the control alone would be a different asymmetry. That limitation is stated
rather than hidden: this makes the floor an UPPER bound on what an untrained model scores, which
is the conservative direction for every gap measured against it.

    bash src/a2-nlp/py.sh study_ner_control_sweep.py --dry-run
    bash src/a2-nlp/py.sh study_ner_control_sweep.py --gpu 1
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
OUT = os.path.join(HERE, 'runs', 'ner_control_sweep.json')

TASK, LANG, STEPS = 'masakhaner', 'yor', 2150
# Started as the same eight rates the from-scratch arm was swept over, so the control could not be
# accused of a narrower search than the thing it is the floor for. That turned out not to be
# enough: the first pass rose monotonically across all eight and peaked at 0.594 on the top rate,
# which is a sweep reporting its own boundary rather than a maximum. Extended upward until the
# curve turns over. The floor matters more than most cells here because every "clears the floor
# by" number on NER is measured against it, and a floor quoted from an unfinished sweep is
# an overstatement of every one of those gaps.
RATES = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3]
SEEDS = (0, 1, 2)
ARMS = [('our architecture, untrained', 'runs/yor_random_init'),
        ("XLM-R's architecture, untrained", 'runs/xlm-roberta-base_random_init')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    # Build the controls if they are not on disk. This is the one-line call the API exists for,
    # and it is why sweeping a floor is affordable at all.
    for label, path in ARMS:
        if not os.path.exists(os.path.join(HERE, path, 'config.json')):
            print(f'building {label} -> {path}')
            if not a.dry_run:
                if 'xlm' in path:
                    factory.random_init_like('FacebookAI/xlm-roberta-base')
                else:
                    factory.random_init('yor')

    cells = [(label, path, lr) for label, path in ARMS for lr in RATES]
    print(f'\n{len(ARMS)} arms x {len(RATES)} rates x {len(SEEDS)} seeds = '
          f'{len(cells)*len(SEEDS)} fine-tuning runs')
    have = [r for r in ft_api.results('*')
            if r.get('task') == TASK and r.get('steps') == STEPS and 'random_init' in r['model']]
    print(f'already on disk: {len(have)} control cell(s) at rates '
          f'{sorted({r["lr"] for r in have})}')
    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return

    fleet_plan.announce('MasakhaNER: sweeping the untrained floor',
                        [fleet_plan.cell(
                            ft_api.record_tag(path, TASK, LANG, None, lr, STEPS),
                            f'{label}  lr={lr:g}', kind='finetune', steps=STEPS, eta_s=135)
                         for label, path, lr in cells],
                        owner='Patrick',
                        replace_prefix='ft_masakhaner_yor_yor-random-init')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    rows, t0 = [], time.time()
    for i, (label, path, lr) in enumerate(cells, 1):
        try:
            rec = ft_api.evaluate(path, task=TASK, lang=LANG, steps=STEPS, lr=lr,
                                  seeds=SEEDS, reuse=True, label=f'nerfloor {label} lr{lr:g}')
            rows.append({'arm': label, 'model': path, 'lr': lr, 'mean': rec['mean'],
                         'sd': rec.get('sd'), 'ci': rec.get('ci')})
        except Exception as e:                     # noqa: BLE001 -- one cell must not stop the rest
            print(f'  FAILED {label} lr={lr:g}: {repr(e)[:120]}', flush=True)
            rows.append({'arm': label, 'lr': lr, 'error': repr(e)[:200]})
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(cells)}] {label} lr={lr:g}  '
              f'({(time.time()-t0)/60:.0f} min elapsed)', flush=True)

    report(rows)


def report(rows=None):
    rows = [r for r in (rows or json.load(open(OUT, encoding='utf-8'))) if 'mean' in r]
    print('\n' + '=' * 74)
    print('WHAT AN UNTRAINED MODEL SCORES ON NER, ONCE IT GETS A SWEPT RATE')
    print('=' * 74)
    for label, _ in ARMS:
        g = sorted((r for r in rows if r['arm'] == label), key=lambda r: -r['mean'])
        if not g:
            continue
        best = g[0]
        print(f"\n  {label}")
        print('    ' + '  '.join(f"{r['lr']:g}:{r['mean']:.3f}" for r in
                                 sorted(g, key=lambda r: r['lr'])))
        print(f"    best {best['mean']:.4f} at lr {best['lr']:g}"
              + (f", CI [{best['ci'][0]:.3f}, {best['ci'][1]:.3f}]" if best.get('ci') else ''))
        if best['lr'] in (min(RATES), max(RATES)):
            print('    CAUTION: peaked at the edge of the range. Extend before quoting -- this is'
                  '\n    the fifth sweep in this project to do it.')
        old = 0.4140
        print(f"    against the single unswept cell we had been printing: {old:.4f} "
              f"({best['mean']-old:+.4f})")


if __name__ == '__main__':
    main()
