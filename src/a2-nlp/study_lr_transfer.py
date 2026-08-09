"""Does a learning rate tuned on one language transfer to another?

The bottom poster claims that adding a language to this factory is one function call. That claim
is only true if the SETTINGS transfer too -- if every new language needs its own sweep, then
adding one costs a night rather than a call, and we should say so on the board instead.

We have never checked. Yoruba and English were tuned separately and the two argmins were never
compared, so the claim we are printing rests on nothing.

The design holds everything fixed except the two things in question:

  * five languages at a fixed 16M-token budget, so no language is advantaged by having more text
    (Wolof is excluded -- 4.8M tokens total, it cannot reach the budget);
  * four learning rates bracketing the useful range. 1e-3 is deliberately left out: the single
    run we have at that rate never learned anything, and spending 10 runs to confirm it would buy
    nothing;
  * two seeds per cell, because a single run is a coin flip and this project has paid for that
    lesson more than once. Two is not three; it is what fits the night, and the write-up should
    say so rather than pretend otherwise.

A second thing falls out for free. Panel 13 found the plateau ends at 7,200 steps for the 33.8M
model and 30,000 for the 98M -- but learning rate and model size are perfectly confounded across
all 105 of our existing runs, so we cannot say which one moves it. This grid varies the rate at a
FIXED model size, which breaks the confound at no extra cost.

    bash src/a2-nlp/py.sh study_lr_transfer.py --dry-run
    bash src/a2-nlp/py.sh study_lr_transfer.py --gpu 1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import mlm_api as f

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'lr_transfer.json')

# swh is covered by XLM-R; the rest are not. That is not what this study measures, but keeping
# one covered language in means the result can be checked against the coverage gradient later.
LANGS = ['swh', 'hau', 'yor', 'ibo', 'nya']
LRS = [1.5e-4, 3e-4, 5e-4, 7e-4]
SEEDS = [0, 1]
TOKENS = 16_000_000
STEPS = 12_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    cells = [(lang, lr, s) for lang in LANGS for lr in LRS for s in SEEDS]
    per_run_min = STEPS * 128 * 128 / 470_000 / 60
    print(f'{len(LANGS)} languages x {len(LRS)} rates x {len(SEEDS)} seeds = {len(cells)} runs')
    print(f'{TOKENS:,} tokens, {STEPS:,} steps each')
    print(f'~{per_run_min:.0f} min per run => about {len(cells)*per_run_min/60:.1f} GPU-hours\n')
    for lang in LANGS:
        info = f.corpus_info(lang)
        print(f"  {lang:<5} corpus {info['n_tokens']['train']:>11,} tokens  "
              f"fingerprint {info.get('tokenizer_fingerprint')}")
    if a.dry_run:
        print('\n--dry-run: nothing was executed.')
        return

    rows, t0 = [], time.time()
    for i, (lang, lr, seed) in enumerate(cells, 1):
        # The tag carries the rate, or four cells that differ only in learning rate would
        # overwrite each other -- the collision this factory has been bitten by twice.
        tag = f'lrx_{lang}_{lr:g}_s{seed}'
        try:
            rec = f.pretrain(lang, tokens=TOKENS, steps=STEPS, seed=seed, lr=lr,
                             gpu=a.gpu, tag=tag, reuse=True)
            rows.append({'lang': lang, 'lr': lr, 'seed': seed, 'tag': tag,
                         'val_loss': rec['val_loss'], 'seconds': rec['seconds']})
        except Exception as e:                       # noqa: BLE001 -- one cell must not stop the night
            print(f'  FAILED {tag}: {repr(e)[:120]}', flush=True)
            rows.append({'lang': lang, 'lr': lr, 'seed': seed, 'tag': tag, 'error': repr(e)[:200]})
        json.dump(rows, open(OUT, 'w', encoding='utf-8'), indent=2)
        print(f'  [{i}/{len(cells)}] {tag}  ({(time.time()-t0)/60:.0f} min elapsed)', flush=True)

    report(rows)


def report(rows=None):
    """Best rate per language, and whether it is the same rate everywhere."""
    rows = [r for r in (rows or json.load(open(OUT, encoding='utf-8'))) if 'val_loss' in r]
    print('\n' + '=' * 72)
    print('DOES THE BEST LEARNING RATE TRANSFER?')
    print('=' * 72)
    print(f"\n{'':>6}" + ''.join(f'{lr:>11.1e}' for lr in LRS) + f"{'best':>10}")
    argmins = {}
    for lang in LANGS:
        cells = {}
        for lr in LRS:
            v = [r['val_loss'] for r in rows if r['lang'] == lang and r['lr'] == lr]
            if v:
                cells[lr] = st.mean(v)
        if not cells:
            continue
        best = min(cells, key=cells.get)
        argmins[lang] = best
        line = ''.join(f'{cells[lr]:>11.3f}' if lr in cells else f'{"--":>11}' for lr in LRS)
        print(f'{lang:>6}{line}{best:>10.1e}')

    if argmins:
        uniq = set(argmins.values())
        print(f'\n  best rate is the same for all {len(argmins)} languages: '
              f'{"YES" if len(uniq) == 1 else "NO"}')
        if len(uniq) == 1:
            print('  => "adding a language is one function call" holds, with evidence.')
        else:
            print(f'  => it varies over {sorted(uniq)}. Every new language costs a sweep, and')
            print('     the poster should say that instead.')
        edge = [l for l, b in argmins.items() if b in (LRS[0], LRS[-1])]
        if edge:
            print(f'\n  CAUTION: {", ".join(edge)} peaked at the edge of the swept range. Three of'
                  '\n  five sweeps in this project did that, and a best-of-sweep number means'
                  '\n  nothing if the sweep does not contain the best. Extend before quoting.')


if __name__ == '__main__':
    main()
