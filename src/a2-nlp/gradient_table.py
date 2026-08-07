"""The tokenizer-fit gradient across every prepared language. No GPU, no training.

Report 04 4 measured what XLM-R's 250k multilingual vocabulary costs against a language's own
16k BPE: 1.04x for English and French, 1.00x for Indonesian, 0.95x for Mandarin -- better than a
dedicated vocabulary -- and 1.65x for Yoruba. The penalty appeared only where the language was
under-represented, which is the group's thesis with a control arm.

That was four languages. The study's headline contrast is computed from two per group and the
POC's own output says it has no degrees of freedom behind it. This recomputes the same
measurement over every corpus on disk, labelled by whether XLM-R pretrained on the language, so
the claim rests on a gradient rather than on two points.

Uses each corpus's committed sample_docs, so it measures the same text the corpus was built from
and needs nothing downloaded except XLM-R's tokenizer.

    bash src/a2-nlp/py.sh gradient_table.py
"""
from __future__ import annotations

import glob
import json
import os

import audit_corpus as A
import mlm_data as D

XLMR = 'FacebookAI/xlm-roberta-base'
HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, 'runs', 'gradient_languages.json')
OUT = os.path.join(HERE, 'runs', 'gradient_table.json')

# Which languages XLM-R pretrained on. The corpora prepared before this sweep are labelled here;
# anything added by prepare_gradient_languages.py carries its own flag in that file.
KNOWN = {'eng': True, 'fra': True, 'ind': True, 'cmn': True, 'yor': False, 'eng_1b': True}


def coverage_map() -> dict:
    m = dict(KNOWN)
    try:
        with open(PLAN, encoding='utf-8') as f:
            for row in json.load(f):
                m[row['corpus']] = row['in_xlmr']
    except (OSError, ValueError):
        pass
    return m


def main():
    cover = coverage_map()
    xlmr = D.load_shared_tokenizer(XLMR, 512)

    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, 'data', '*', 'stats.json'))):
        name = os.path.basename(os.path.dirname(path))
        if name in ('bpe16k', 'shakespeare', 'wikitext2', 'wikitext103'):
            continue                       # not language corpora in this sense
        try:
            docs = D.sample_docs(name)
            own = D.load_tokenizer(name)
        except Exception as e:             # noqa: BLE001
            print(f'{name}: skipped ({repr(e)[:70]})')
            continue
        if not docs:
            continue

        # Every committed sample document, not a slice. At 400 the xho/wol boundary
        # separated the two groups by 0.006, and at 800 it reversed -- so a fixed
        # small sample produced an ordering that looked perfect and was not stable.
        text = '\n'.join(docs)
        # Per word where words exist, per character where they do not. Mandarin has no whitespace
        # boundaries, and measuring it per "word" once reported 15.57 tokens/word -- meaningless,
        # and plausible enough to reach a table.
        by_word = A.uses_word_boundaries(text)
        unit = 'word' if by_word else 'char'
        f_own = A.fertility(own, docs) if by_word else A.fertility_per_char(own, docs)
        f_xlm = A.fertility(xlmr, docs) if by_word else A.fertility_per_char(xlmr, docs)

        rows.append({'corpus': name, 'in_xlmr': cover.get(name), 'n_docs': len(docs),
                     'unit': unit, 'own': round(f_own, 4), 'xlmr': round(f_xlm, 4),
                     'penalty': round(f_xlm / f_own, 4) if f_own else None})
        print(f'  {name:8s} {unit:5s} own {f_own:6.3f}  xlmr {f_xlm:6.3f}  '
              f'penalty {f_xlm / f_own:5.2f}x', flush=True)

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    seen = [r for r in rows if r['in_xlmr'] is True and r['penalty']]
    unseen = [r for r in rows if r['in_xlmr'] is False and r['penalty']]
    print(f'\n{"in XLM-R":>12s}  n={len(seen):2d}  mean penalty '
          f'{sum(r["penalty"] for r in seen) / max(len(seen), 1):.3f}')
    print(f'{"not in XLM-R":>12s}  n={len(unseen):2d}  mean penalty '
          f'{sum(r["penalty"] for r in unseen) / max(len(unseen), 1):.3f}')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
