"""Publish a whole night's queue before it starts, so the dashboard shows what is coming.

mlm_fleet writes its cells when it launches, which is right for a single fleet and useless for a
driver that runs a dozen of them: the panel can only ever show what has already begun, so a queue
of twelve languages reads as three. Watching a run and being unable to tell whether it is the
whole study or a twelfth of it is the thing the panel exists to prevent.

A driver calls this once, up front, with everything it intends to run. Each fleet then updates
its own cells in place and leaves the rest alone -- write_plan keeps any cell whose tag it does
not own -- so the declaration survives and fills in as the night proceeds.

    bash py.sh declare_plan.py overnight3
"""
from __future__ import annotations

import json
import os
import sys

import mlm_train

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
BATCH, SEQ = 128, 128


def cell(corpus, tokens, update_tokens, seed, preset, prefix):
    steps = max(1, round(update_tokens / (BATCH * SEQ)))
    tag = mlm_train.cell_tag(corpus, tokens, steps, seed, preset)
    return {'corpus': corpus, 'tag': f'{prefix}_{tag}' if prefix else tag,
            'tokens': tokens, 'update_tokens': update_tokens, 'steps': steps,
            'preset': preset, 'seed': seed}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'queue'

    try:
        with open(os.path.join(RUNS, 'gradient_languages.json'), encoding='utf-8') as f:
            langs = [r['corpus'] for r in json.load(f) if r['status'] in ('ok', 'existing')]
    except (OSError, ValueError):
        langs = []

    cells = [cell(c, 50_000_000, 196_608_000, 0, 'poc', 'grad') for c in langs]
    for prefix in ('lr15', 'clip05'):
        cells += [cell('eng_1b', 64_000_000, 1_024_000_000, s, 'afriberta', prefix)
                  for s in (0, 1, 2)]

    try:
        import torch
        cards = max(1, torch.cuda.device_count())
    except Exception:                       # noqa: BLE001
        cards = 2

    # Preserve anything already recorded for this queue -- a fleet that has started knows more
    # about its own cells than this declaration does.
    prev = {}
    try:
        with open(os.path.join(RUNS, '_fleet_plan.json'), encoding='utf-8') as f:
            old = json.load(f)
        if old.get('queue') == name:
            prev = {c['tag']: c for c in old.get('cells', [])}
    except (OSError, ValueError):
        pass

    import time
    plan = {'corpus': None, 'queue': name, 'batch': BATCH, 'seq_len': SEQ,
            'n_gpu': cards, 'started': time.time(),
            'cells': [{**c, **{k: v for k, v in prev.get(c['tag'], {}).items()
                               if v is not None}} for c in cells]}
    os.makedirs(RUNS, exist_ok=True)
    tmp = os.path.join(RUNS, '_fleet_plan.json.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, os.path.join(RUNS, '_fleet_plan.json'))

    print(f'declared {len(plan["cells"])} cells for queue {name!r} across {cards} cards')
    for c in plan['cells']:
        print(f'  {c["tag"]}')


if __name__ == '__main__':
    main()
