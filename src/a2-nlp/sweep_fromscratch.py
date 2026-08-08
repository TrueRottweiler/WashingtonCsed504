"""A learning-rate sweep for our own model downstream -- the one every baseline got and ours did not.

Report 08 reports the from-scratch model beating mmBERT on SIB-200, 0.632 against 0.574. But
mmBERT's number is the best of a five-rate sweep and ours is a single default. That asymmetry
favors us, and after report 06 -- where a step budget nobody questioned produced the project's
central downstream conclusion -- it is exactly the kind of thing to measure rather than caveat.

Two outcomes and both are worth having. If a better rate exists the win grows and is on firmer
footing. If 2e-5 was already best, the comparison was fair and can be quoted without the asterisk.

    bash src/a2-nlp/py.sh sweep_fromscratch.py
"""
from __future__ import annotations

import ft_api

CKPT = 'runs/yor_64M_62.5k_s0'
LRS = (5e-6, 1e-5, 2e-5, 3e-5, 5e-5)     # the same five Patrick swept for XLM-R and mmBERT


def main():
    for task, lang, steps in (('sib200', 'yor_Latn', 1056), ('masakhaner', 'yor', 2150)):
        print(f'\n=== {task} @ {steps} steps ===', flush=True)
        best = None
        for lr in LRS:
            try:
                r = ft_api.evaluate(CKPT, task=task, lang=lang, lr=lr, seeds=(0, 1, 2),
                                    steps=steps, label=f'from-scratch sweep lr{lr:g}')
                print(f'  lr {lr:7.0e}  {r["mean"]:.4f}  [{r["ci"][0]:.3f}, {r["ci"][1]:.3f}]  '
                      f'degenerate={r["degenerate"]}', flush=True)
                if best is None or r['mean'] > best[1]:
                    best = (lr, r['mean'])
            except Exception as e:          # noqa: BLE001 -- one rate must not stop the sweep
                print(f'  lr {lr:7.0e}  FAILED: {repr(e)[:100]}', flush=True)
        if best:
            print(f'  best: lr {best[0]:g} at {best[1]:.4f}', flush=True)


if __name__ == '__main__':
    main()
