"""Sweep the baselines' NER learning rate too, so the comparison is best-against-best.

Patrick swept SIB-200 for both baselines. Nobody swept MasakhaNER for anything -- every NER row
in the project uses NER_LR = 3e-5, the default. That was symmetric and therefore fair.

Sweeping only OUR model broke that symmetry in our favor, and it matters: our number moved from
0.788 at the default to 0.837 at 1e-4. Publishing that against baselines pinned at a default
would be precisely the asymmetry this sweep was run to remove.

    bash src/a2-nlp/py.sh sweep_ner_baselines.py
"""
from __future__ import annotations

import ft_api

MODELS = [('FacebookAI/xlm-roberta-base', 'XLM-R'), ('jhu-clsp/mmBERT-base', 'mmBERT')]
LRS = (2e-5, 3e-5, 5e-5, 7e-5, 1e-4)      # the range where our model peaked


def main():
    for path, label in MODELS:
        print(f'\n=== {label} on MasakhaNER, 2150 steps ===', flush=True)
        best = None
        for lr in LRS:
            try:
                r = ft_api.evaluate(path, task='masakhaner', lang='yor', lr=lr, seeds=(0, 1, 2),
                                    steps=2150, label=f'{label} NER lr{lr:g}')
                print(f'  lr {lr:7.0e}  {r["mean"]:.4f}  [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]',
                      flush=True)
                if best is None or r['mean'] > best[1]:
                    best = (lr, r['mean'])
            except Exception as e:          # noqa: BLE001
                print(f'  lr {lr:7.0e}  FAILED: {repr(e)[:90]}', flush=True)
        if best:
            print(f'  best: lr {best[0]:g} at {best[1]:.4f}', flush=True)


if __name__ == '__main__':
    main()
