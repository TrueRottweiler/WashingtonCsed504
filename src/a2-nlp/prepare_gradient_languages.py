"""Widen the language gradient, so the study's contrast stops resting on two languages a side.

The headline contrast -- unseen-by-XLM-R against seen-by-XLM-R -- is computed from two languages
in each group, and the POC's own output says so: "the contrast has no degrees of freedom behind
it. Add languages before quoting it." This adds them.

The measurement that needs no training at all is the tokenizer one: fertility under a language's
own 16k BPE against XLM-R's 250k vocabulary. Report 04 4 found that penalty is 1.00-1.04x for
well-covered languages, 0.95x for Mandarin, and 1.65x for Yoruba. If that gradient holds across
ten languages split by XLM-R coverage rather than four, it stops being an anecdote.

Runs unattended, so every language is independent: one that is absent from FineWeb-2, too small,
or slow is recorded and skipped rather than taking the queue down with it.

    bash src/a2-nlp/py.sh prepare_gradient_languages.py
"""
from __future__ import annotations

import json
import os
import time
import traceback

import mlm_data as D

# Split by whether XLM-R pretrained on the language. Its 100 languages include Swahili, Hausa,
# Amharic, Afrikaans, Somali and Xhosa from this region; they exclude Yoruba, Igbo, Kinyarwanda,
# Shona, Luganda and Wolof. Both groups are the same script family, which keeps the comparison
# about coverage rather than about writing system -- the Mandarin result in report 04 4 showed
# script confounds this badly.
CANDIDATES = [
    # (corpus name, FineWeb-2 config, in XLM-R?)
    ('swh', 'swh_Latn', True),
    ('hau', 'hau_Latn', True),
    ('amh', 'amh_Ethi', True),
    ('afr', 'afr_Latn', True),
    ('som', 'som_Latn', True),
    ('xho', 'xho_Latn', True),
    ('ibo', 'ibo_Latn', False),
    ('kin', 'kin_Latn', False),
    ('sna', 'sna_Latn', False),
    ('lug', 'lug_Latn', False),
    ('wol', 'wol_Latn', False),
    ('nya', 'nya_Latn', False),
]

# The same character budget the five-language set used, so the new rows are comparable with the
# old ones rather than a separate experiment.
MAX_CHARS = 260_000_000
MAX_SECONDS = 900          # a language that cannot stream this in 15 minutes is too thin to use

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs',
                   'gradient_languages.json')


def main():
    results = []
    for name, cfg, in_xlmr in CANDIDATES:
        if os.path.exists(os.path.join(D.P.out_dir(name), 'stats.json')):
            print(f'{name}: already prepared, skipping')
            results.append({'corpus': name, 'lang': cfg, 'in_xlmr': in_xlmr, 'status': 'existing'})
            continue

        print(f'\n=== {name} ({cfg}, {"in" if in_xlmr else "not in"} XLM-R) ===', flush=True)
        t0 = time.time()
        try:
            stats = D.prepare_corpus(name=name, lang=cfg, source='fineweb2',
                                     max_chars=MAX_CHARS, max_seconds=MAX_SECONDS,
                                     val_tokens=500_000)
            results.append({'corpus': name, 'lang': cfg, 'in_xlmr': in_xlmr,
                            'status': 'ok', 'minutes': round((time.time() - t0) / 60, 1),
                            'train_tokens': stats['n_tokens']['train'],
                            'chars': stats.get('chars'),
                            'chars_per_token': stats.get('chars_per_token'),
                            'fingerprint': stats.get('tokenizer_fingerprint')})
            print(f'  {name}: {stats["n_tokens"]["train"]:,} train tokens '
                  f'in {(time.time() - t0) / 60:.1f} min', flush=True)
        except Exception as e:                      # noqa: BLE001 -- one language must not stop the rest
            print(f'  {name}: FAILED -- {repr(e)[:160]}', flush=True)
            traceback.print_exc()
            results.append({'corpus': name, 'lang': cfg, 'in_xlmr': in_xlmr,
                            'status': 'failed', 'error': repr(e)[:300]})

        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

    ok = [r for r in results if r['status'] in ('ok', 'existing')]
    print(f'\n{len(ok)}/{len(CANDIDATES)} prepared')
    print(f'  in XLM-R    : {sum(1 for r in ok if r["in_xlmr"])}')
    print(f'  not in XLM-R: {sum(1 for r in ok if not r["in_xlmr"])}')
    print(f'\nwritten to {OUT}')


if __name__ == '__main__':
    main()
