"""Fine-tune both swap arms on the card that already holds them, with symmetric rate selection.

This is Patrick's blocked half of the tokenizer-swap experiment, run here because the checkpoints
are here. Moving them is the expensive option and not by a little: each XLM-R-vocabulary
checkpoint directory is 2.36 GB on disk, of which 1.85 GB is `train_state.pt` -- optimizer moments
that fine-tuning never reads. The weights really are the 0.62 GB Patrick quoted; it is the
directory that is not. Six checkpoints is 8.6 GB moved, or 3.4 GB if stripped, against minutes of
compute on the card they are already sitting on.

The arms:

    swap_yor_xlmr_121.3M_12k_s{0,1,2}   our corpus, XLM-R's vocabulary   -- no ft rows at all
    swap_yor_69.1M_12k_s{0,1,2}         our corpus, our vocabulary       -- single-rate rows only

The second arm is the reason this is a sweep and not three fine-tunes. Its existing rows were
scored at one learning rate that nobody chose on a dev split, so sweeping only the new arm would
hand the new arm a best-of-nine and leave the comparison arm with a default -- the asymmetry
report 11 spent a day removing from the main table, reintroduced at the bottom of the same board.
Both arms get the same nine rates, the same three seeds, on the same 99 dev items.

MasakhaNER is handled differently, and the difference is worth stating rather than hiding.
MasakhaNER as we load it has no dev split, so there is nothing to select on. Rather than invent
one for these two arms alone, both stay at the single rate they already share (3e-05). That is
symmetric by being equally unswept, which is a defensible comparison; a swept new arm against an
unswept old one would not be. The new arm's NER cells are the only thing missing there.

    bash src/a2-nlp/py.sh study_swap_downstream.py --dry-run
    bash src/a2-nlp/py.sh study_swap_downstream.py --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import fleet_plan
import ft_api

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'runs', 'swap_downstream.json')

# Patrick's dev protocol, read off his own records rather than retyped from the report.
RATES = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7e-5, 1e-4, 2e-4, 3e-4]
DEV_SEEDS = (0, 1, 2)
TEST_SEEDS = (0, 1, 2, 3, 4)
SIB, SIB_STEPS = 'sib200', 1056
NER, NER_STEPS, NER_LR = 'masakhaner', 2150, 3e-5

ARMS = [("XLM-R's vocabulary", 'swap_yor_xlmr_121.3M_12k_s%d'),
        ('our vocabulary', 'swap_yor_69.1M_12k_s%d')]
SEEDS_OF_PRETRAIN = (0, 1, 2)


def checkpoints():
    return [(label, s, f'runs/{tmpl % s}') for label, tmpl in ARMS for s in SEEDS_OF_PRETRAIN]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--skip-ner', action='store_true')
    a = ap.parse_args()

    ck = checkpoints()
    missing = [p for _, _, p in ck if not os.path.exists(os.path.join(HERE, p, 'config.json'))]
    if missing:
        raise SystemExit('missing checkpoints, nothing to run:\n  ' + '\n  '.join(missing))

    n_dev = len(ck) * len(RATES)
    print(f'{len(ck)} checkpoints x {len(RATES)} rates x {len(DEV_SEEDS)} seeds = '
          f'{n_dev * len(DEV_SEEDS)} dev fine-tunes')
    print(f'then {len(ck)} x {len(TEST_SEEDS)} = {len(ck)*len(TEST_SEEDS)} test fine-tunes '
          f'at each checkpoint\'s dev-chosen rate')
    if not a.skip_ner:
        print(f'then MasakhaNER at a shared {NER_LR:g} on whichever checkpoints lack it')
    if a.dry_run:
        for label, s, p in ck:
            have = [r for r in ft_api.results('*', eval_split=None)
                    if r['model'].endswith(os.path.basename(p))]
            print(f'  {label:20s} s{s}  {len(have)} existing row(s)')
        print('\n--dry-run: nothing was executed.')
        return

    fleet_plan.announce(
        'tokenizer swap: both arms, rate chosen on dev',
        [fleet_plan.cell(
            ft_api.record_tag(p, SIB, 'yor_Latn', None, lr, SIB_STEPS, eval_split='validation'),
            f'{label} s{s}  dev lr={lr:g}', kind='finetune', steps=SIB_STEPS, eta_s=90)
         for label, s, p in ck for lr in RATES],
        owner='Patrick', replace_prefix='ft_sib200_yor_Latn_swap')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(a.gpu)
    rows, t0 = [], time.time()

    def note(**kw):
        rows.append(kw)
        with open(OUT, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2)

    for label, s, path in ck:
        # --- choose the rate on the 99 dev items ------------------------------------------
        dev = []
        for lr in RATES:
            try:
                r = ft_api.evaluate(path, task=SIB, steps=SIB_STEPS, lr=lr, seeds=DEV_SEEDS,
                                    eval_split='validation', reuse=True,
                                    label=f'swap dev {label} s{s} lr{lr:g}')
                dev.append(r)
                note(stage='dev', arm=label, pretrain_seed=s, lr=lr, mean=r['mean'],
                     sd=r.get('sd'))
            except Exception as e:                 # noqa: BLE001 -- one rate must not stop the arm
                print(f'  FAILED dev {label} s{s} lr={lr:g}: {repr(e)[:120]}', flush=True)
                note(stage='dev', arm=label, pretrain_seed=s, lr=lr, error=repr(e)[:200])
            print(f'  [{(time.time()-t0)/60:5.1f} min] dev {label} s{s} lr={lr:g}', flush=True)
        if not dev:
            continue
        best = max(dev, key=lambda r: r['mean'])

        # --- and only then score it on the 204 test items ---------------------------------
        try:
            t = ft_api.evaluate(path, task=SIB, steps=SIB_STEPS, lr=best['lr'], seeds=TEST_SEEDS,
                                reuse=True, label=f'swap test {label} s{s}')
            note(stage='test', arm=label, pretrain_seed=s, lr=best['lr'], task=SIB,
                 dev_mean=best['mean'], mean=t['mean'], sd=t.get('sd'), ci=t.get('ci'),
                 scores=t.get('scores'))
            print(f"  -> {label} s{s}: dev picked {best['lr']:g}, test {t['mean']:.4f}",
                  flush=True)
        except Exception as e:                     # noqa: BLE001
            print(f'  FAILED test {label} s{s}: {repr(e)[:120]}', flush=True)
            note(stage='test', arm=label, pretrain_seed=s, error=repr(e)[:200])

        # --- MasakhaNER, at the rate both arms already share ------------------------------
        if not a.skip_ner:
            try:
                n = ft_api.evaluate(path, task=NER, steps=NER_STEPS, lr=NER_LR, seeds=DEV_SEEDS,
                                    reuse=True, label=f'swap ner {label} s{s}')
                note(stage='ner', arm=label, pretrain_seed=s, lr=NER_LR, task=NER,
                     mean=n['mean'], sd=n.get('sd'), ci=n.get('ci'), scores=n.get('scores'))
                print(f"  -> {label} s{s}: NER {n['mean']:.4f}", flush=True)
            except Exception as e:                 # noqa: BLE001
                print(f'  FAILED ner {label} s{s}: {repr(e)[:120]}', flush=True)
                note(stage='ner', arm=label, pretrain_seed=s, error=repr(e)[:200])

    report(rows)


def report(rows=None):
    rows = rows or json.load(open(OUT, encoding='utf-8'))
    print('\n' + '=' * 78)
    print('THE TOKENIZER SWAP, DOWNSTREAM, WITH BOTH ARMS SELECTED THE SAME WAY')
    print('=' * 78)
    for task, stage in ((SIB, 'test'), (NER, 'ner')):
        got = {}
        for label, _ in ARMS:
            g = [r for r in rows if r.get('stage') == stage and r.get('arm') == label
                 and 'mean' in r]
            if not g:
                continue
            got[label] = g
            per = ', '.join(f"s{r['pretrain_seed']}:{r['mean']:.4f}" for r in
                            sorted(g, key=lambda r: r['pretrain_seed']))
            rate = ' '.join(sorted({f"lr{r['lr']:g}" for r in g}))
            print(f'\n  {task:11s} {label:22s} {per}   [{rate}]')
            print(f"    mean over pretraining seeds {st.mean(r['mean'] for r in g):.4f}")
        if len(got) == 2:
            (_, a), (_, b) = got.items()
            ma, mb = st.mean(r['mean'] for r in a), st.mean(r['mean'] for r in b)
            print(f'\n    difference {ma-mb:+.4f} across {len(a)} vs {len(b)} pretraining seeds')
            print('    n=3 a side: a difference must be about 2.27x the seed spread to clear'
                  '\n    p<0.05, and the exact test cannot go below p=0.10 at all at that size.'
                  '\n    Both bars come from claims_audit.py rather than from memory.')


if __name__ == '__main__':
    main()
