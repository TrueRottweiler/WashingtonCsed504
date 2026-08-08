"""Read the tokenizer swap: does a badly-fitting vocabulary cost anything, and how much?

Everything else in this project measures that the penalty EXISTS -- 1.65x in report 04, 1.76x
across seventeen corpora in report 07. Nothing measures what it COSTS. This does, by holding the
text, the architecture and the compute fixed and changing only which vocabulary turned characters
into tokens.

The unit is bits per character, and that choice is the whole experiment. Nats per token cannot
compare two vocabularies, and the two arms differ in exactly the way that makes per-token loss
misleading: the 250k vocabulary has an EASIER per-token job (more of the probability mass is
already spent choosing between finer-grained pieces) and has to do MORE of them for the same
text. Per-token loss flatters it; per-character loss does not.

    bash src/a2-nlp/py.sh swap_table.py
"""
from __future__ import annotations

import math
import statistics as st

import mlm_api as factory

ARMS = [
    ('our 16k BPE', 'swap_yor_*', 'yor'),
    ("XLM-R's 250k", 'swap_yor_xlmr_*', 'yor_xlmr'),
]


def arm(pattern: str, corpus: str) -> dict | None:
    rows = [r for r in factory.results(pattern) if r.get('corpus') == corpus]
    if not rows:
        return None
    info = factory.corpus_info(corpus)
    cpt = info['chars_per_token']
    losses = sorted(r['val_loss'] for r in rows)
    bpc = [v / math.log(2) / cpt for v in losses]
    return {'n': len(losses), 'vocab': info['vocab_size'], 'cpt': cpt,
            'tokens': info['n_tokens']['train'],
            'loss': st.mean(losses), 'loss_sd': st.stdev(losses) if len(losses) > 1 else 0.0,
            'bpc': st.mean(bpc), 'bpc_sd': st.stdev(bpc) if len(bpc) > 1 else 0.0,
            'passes': rows[0]['steps'] * rows[0]['batch'] * rows[0]['seq_len'] / info['n_tokens']['train']}


def main():
    got = [(label, arm(pat, corpus)) for label, pat, corpus in ARMS]
    got = [(l, a) for l, a in got if a]
    if len(got) < 2:
        have = ', '.join(l for l, _ in got) or 'none'
        print(f'not enough arms finished yet (have: {have})')
        return

    print('Same Yoruba text, same architecture, same tokens of updates.')
    print('The only difference is which vocabulary turned characters into tokens.\n')
    print(f"{'arm':14s} {'vocab':>8s} {'chars/tok':>10s} {'tokens':>12s} {'epochs':>7s} "
          f"{'nats/tok':>16s} {'bits/char':>16s}")
    for label, a in got:
        print(f"{label:14s} {a['vocab']:8,d} {a['cpt']:10.3f} {a['tokens']:12,d} "
              f"{a['passes']:7.2f} "
              f"{a['loss']:8.3f} ±{a['loss_sd']:5.3f} {a['bpc']:8.3f} ±{a['bpc_sd']:5.3f}")

    (l0, a0), (l1, a1) = got[0], got[1]
    d_loss = a1['loss'] - a0['loss']
    d_bpc = a1['bpc'] - a0['bpc']
    pooled = (a0['bpc_sd'] + a1['bpc_sd']) / 2

    print(f"\nper-token loss says {l1} is "
          f"{'WORSE' if d_loss > 0 else 'BETTER'} by {abs(d_loss):.3f} nats/token")
    print(f"per-character   says {l1} is "
          f"{'WORSE' if d_bpc > 0 else 'BETTER'} by {abs(d_bpc):.3f} bits/char"
          + (f'  ({abs(d_bpc)/pooled:.1f}x the seed spread)' if pooled else ''))
    if (d_loss > 0) != (d_bpc > 0):
        print('\n  The two units DISAGREE ON THE SIGN. This is the trap the experiment was built')
        print('  to avoid: the bigger vocabulary has an easier per-token job and does more of')
        print('  them, so per-token loss flatters it. Only the per-character number compares.')

    pct = 100 * d_bpc / a0['bpc']
    print(f"\nthe penalty costs {pct:+.1f}% in bits per character")
    print(f"for reference, it costs {a1['cpt'] and a0['cpt']/a1['cpt']:.2f}x the tokens for the "
          f"same text -- the fertility number reports 04 and 07 measure")


if __name__ == '__main__':
    main()
