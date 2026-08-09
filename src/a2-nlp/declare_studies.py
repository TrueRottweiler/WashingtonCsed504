"""Publish tonight's two studies so the dashboard shows them while they run.

The correlation study is thirty-eight fine-tuning cells driven from one Python process rather than
from mlm_fleet, so nothing announces it: the live panel watches for per-run pretraining logs and
finds none, and the page looks empty while a card sits at 84%. A dashboard you cannot trust to be
empty when it says empty is worse than no dashboard, which is the whole argument for declaring the
queue up front.

Cells already recorded keep whatever the previous plan knew about them, so re-running this is
safe and idempotent, and re-running it mid-study fills in the finished rows rather than resetting
them.

    bash src/a2-nlp/py.sh declare_studies.py
"""
from __future__ import annotations

import json
import os
import time

import ft_api
import mlm_api
import mlm_train

f_compact = mlm_api.compact
import study_clip_prevention as clipprev
import study_downstream_correlation as study
import study_lr_transfer as lrx

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
QUEUE = 'two days: correlation + LR transfer, then clipping prevention on both cards'

# Measured from the 46 existing downstream records: 1.2 min per seed, three seeds per cell.
ETA_PER_CELL_S = int(1.2 * 60 * 3)


def main():
    models, fp, ambiguous = study.candidates()
    cells = []
    for r in models:
        for task, lang, steps in study.TASKS:
            lr = ft_api.NER_LR if task == 'masakhaner' else ft_api.FT_LR
            tag = ft_api.record_tag(f'runs/{r["tag"]}', task, lang, None, lr, steps)
            short = 'NER' if task == 'masakhaner' else 'SIB'
            cells.append({'kind': 'finetune', 'tag': tag,
                          'label': f'{short}  {r["tag"]}  (pretrain val {r["val_loss"]:.3f})',
                          'eta_s': ETA_PER_CELL_S, 'update_tokens': 0,
                          'preset': 'poc', 'steps': steps})

    # Study 2 runs pretraining from a plain loop rather than through mlm_fleet, so nothing
    # else would announce its forty cells either.
    for lang in lrx.LANGS:
        for lr in lrx.LRS:
            for seed in lrx.SEEDS:
                cells.append({'kind': 'pretrain', 'tag': f'lrx_{lang}_{lr:g}_s{seed}',
                              'label': f'{lang}  lr={lr:g}  seed {seed}',
                              'corpus': lang, 'eta_s': 520,
                              'update_tokens': lrx.STEPS * 128 * 128,
                              'preset': 'poc', 'steps': lrx.STEPS})

    # Study 4 is chained behind the first two by chain_card0.sh / chain_card1.sh, split by SEED
    # across the cards so a hardware difference cannot masquerade as a clipping effect.
    for n in clipprev.TOKENS:
        for c in clipprev.CLIPS:
            for seed in clipprev.SEEDS:
                card = 0 if seed < 6 else 1
                cells.append({'kind': 'pretrain',
                              'tag': f'clipprev_{f_compact(n)}_c{c:g}_s{seed}',
                              'label': f'clip {c:g}  {f_compact(n)} tokens  seed {seed} '
                                       f'(card {card})',
                              'corpus': clipprev.CORPUS, 'eta_s': 5100,
                              'update_tokens': clipprev.STEPS * 128 * 128,
                              'preset': clipprev.PRESET, 'steps': clipprev.STEPS})

    prev = {}
    try:
        with open(os.path.join(RUNS, '_fleet_plan.json'), encoding='utf-8') as f:
            prev = {c['tag']: c for c in json.load(f).get('cells', [])}
    except (OSError, ValueError):
        pass

    plan = {'corpus': None, 'queue': QUEUE, 'batch': 128, 'seq_len': 128,
            'n_gpu': 2, 'started': time.time(),
            'cells': [{**c, **{k: v for k, v in prev.get(c['tag'], {}).items()
                               if v is not None}} for c in cells]}
    tmp = os.path.join(RUNS, f'_fleet_plan.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, os.path.join(RUNS, '_fleet_plan.json'))

    n_ft = sum(1 for c in cells if c['kind'] == 'finetune')
    print(f'declared {len(cells)} cells: {n_ft} fine-tuning, {len(cells)-n_ft} pretraining')
    print(f'study 1: {len(models)} checkpoints at fingerprint {fp}, '
          f'{len(ambiguous)} excluded as ambiguous')
    print(f'estimated {sum(c["eta_s"] for c in cells)/3600:.1f} GPU-hours total across 2 cards')


if __name__ == '__main__':
    main()
