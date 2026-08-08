"""Publish everything queued for tonight, so the panel shows what is coming rather than what began.

declare_plan.py does this for one specific study. This does it for the actual queue in flight:
the swap already running, the seeded cells behind it, four downstream fine-tunes, and the swap
again at the ladder budget. Written as one file because the queue is a fact about tonight, not a
reusable shape.

Each fleet updates its own cells in place when it launches, so this declaration fills in rather
than being overwritten.

    bash src/a2-nlp/py.sh declare_tonight.py
"""
from __future__ import annotations

import json
import os
import time

import ft_api
import mlm_train

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
BATCH, SEQ = 128, 128
QUEUE = 'tonight'


def cell(corpus, tokens, upd, seed, preset='poc', prefix=''):
    steps = max(1, round(upd / (BATCH * SEQ)))
    tag = mlm_train.cell_tag(corpus, tokens, steps, seed, preset)
    return {'kind': 'pretrain', 'corpus': corpus, 'tag': f'{prefix}_{tag}' if prefix else tag,
            'tokens': tokens, 'update_tokens': upd, 'steps': steps,
            'preset': preset, 'seed': seed}


def ft_cell(model_path, task, lang, steps, label, lr=None, eta_s=300):
    lr = lr if lr is not None else (ft_api.NER_LR if task == 'masakhaner' else ft_api.FT_LR)
    tag = ft_api.record_tag(model_path, task, lang, None, lr, steps)
    return {'kind': 'finetune', 'tag': tag, 'label': label, 'eta_s': eta_s,
            'update_tokens': 0, 'preset': 'poc', 'steps': steps}


CKPT = 'runs/yor_64M_62.5k_s0'
XLMR_UNTRAINED = 'runs/xlm-roberta-base_random_init'

cells = (
    # in flight: the tokenizer swap at the multi_yor budget
    [cell('yor', 69_096_452, 196_608_000, s, prefix='swap') for s in (0, 1, 2)]
    + [cell('yor_xlmr', 121_339_416, 196_608_000, s, prefix='swap') for s in (0, 1, 2)]
    # the two single-seed cells Patrick flagged on report 07's table
    + [cell('eng_1b', n, 1_024_000_000, s)
       for n in (4_000_000, 1_024_000_000) for s in (1, 2)]
    # the downstream rows report 06 is missing, plus the vocabulary control
    + [ft_cell(CKPT, 'masakhaner', 'yor', 2150, 'from-scratch NER, NFC', eta_s=150),
       ft_cell(CKPT, 'sib200', 'yor_Latn', 352, 'from-scratch SIB @352', eta_s=60),
       ft_cell(CKPT, 'sib200', 'yor_Latn', 1056, 'from-scratch SIB @1056', eta_s=180),
       ft_cell(XLMR_UNTRAINED, 'sib200', 'yor_Latn', 1056, 'XLM-R arch untrained @1056',
               eta_s=600)]
    # the swap again, at the budget the ladders use
    # Only the 16k arm. Matched COMPUTE is 250k at 12k steps against 16k at ~61k steps, because
    # the 250k output head makes a step 5.1x more expensive; the 250k arm at 62.5k steps would
    # cost ten hours to re-answer the matched-STEPS question that turned out to be confounded.
    + [cell('yor', 69_096_452, 1_024_000_000, s, prefix='swap62k') for s in (0, 1, 2)]
)


def main():
    try:
        import torch
        cards = max(1, torch.cuda.device_count())
    except Exception:                       # noqa: BLE001
        cards = 2

    prev = {}
    try:
        with open(os.path.join(RUNS, '_fleet_plan.json'), encoding='utf-8') as f:
            old = json.load(f)
        prev = {c['tag']: c for c in old.get('cells', [])}
    except (OSError, ValueError):
        pass

    plan = {'corpus': None, 'queue': QUEUE, 'batch': BATCH, 'seq_len': SEQ,
            'n_gpu': cards, 'started': time.time(),
            'cells': [{**c, **{k: v for k, v in prev.get(c['tag'], {}).items()
                               if v is not None}} for c in cells]}
    tmp = os.path.join(RUNS, f'_fleet_plan.{os.getpid()}.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, os.path.join(RUNS, '_fleet_plan.json'))

    n_ft = sum(1 for c in cells if c['kind'] == 'finetune')
    print(f'declared {len(cells)} cells for {QUEUE!r} across {cards} cards '
          f'({len(cells) - n_ft} pretraining, {n_ft} fine-tuning)')
    for c in plan['cells']:
        print(f"  {c['kind']:9s} {c.get('label') or c['tag']}")


if __name__ == '__main__':
    main()
