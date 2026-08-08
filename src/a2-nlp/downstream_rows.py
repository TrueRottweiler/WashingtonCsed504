"""The downstream rows report 06 is missing, plus the control that isolates the vocabulary.

Patrick re-measured both published baselines under NFC and neither moved -- 0.841 and 0.851,
within 0.003 of their pre-fix values. The from-scratch model is the one whose tokenizer fertility
moved 47% under the encoding fix, so it is the only row the fix can plausibly have changed, and
it is the one still unmeasured because the checkpoint exists nowhere but this workstation.

SIB-200 is run at BOTH budgets deliberately. 352 steps was the constant that produced the
project's central downstream conclusion; 1056 is where XLM-R turned out to train at all. A
from-scratch row measured at only one of them inherits exactly the bug that cost us that
conclusion.

The last row is the control Patrick asked for: XLM-R's architecture and 250k vocabulary with no
pretrained weights. Our existing control differs from XLM-R in size, tokenizer and pretraining at
once, so it cannot separate "XLM-R's Yoruba is weak" from "XLM-R's vocabulary caps it" -- and the
second is the study's thesis.

    bash src/a2-nlp/py.sh downstream_rows.py
"""
from __future__ import annotations

import ft_api

# The strongest from-scratch Yoruba model on disk, val 2.315. Every from-scratch row uses this
# one checkpoint, so the rows differ by task and budget rather than by which model they got.
CKPT = 'runs/yor_64M_62.5k_s0'
XLMR_UNTRAINED = 'runs/xlm-roberta-base_random_init'

ROWS = [
    (CKPT,           'masakhaner', 'yor',      2150, 'from-scratch NER, NFC'),
    (CKPT,           'sib200',     'yor_Latn',  352, 'from-scratch SIB @352'),
    (CKPT,           'sib200',     'yor_Latn', 1056, 'from-scratch SIB @1056'),
    (XLMR_UNTRAINED, 'sib200',     'yor_Latn', 1056, 'XLM-R arch, untrained'),
]


def main():
    for path, task, lang, steps, label in ROWS:
        try:
            r = ft_api.evaluate(path, task=task, lang=lang, seeds=(0, 1, 2),
                                steps=steps, label=label)
            print(f'{label:24s} {r["mean"]:.4f}  CI [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]  '
                  f'degenerate={r["degenerate"]}', flush=True)
        except Exception as e:                  # noqa: BLE001 -- one row must not stop the rest
            print(f'{label:24s} FAILED: {repr(e)[:140]}', flush=True)


if __name__ == '__main__':
    main()
