"""Every record a report cites, resolved against `runs/` — and every record nothing cites.

Jeffrey asked the question this exists to answer: *where did the 20.7 seconds on the 12.7M-character
sample come from?* The answer is `runs/pipeline_bench.json`, stage "2. train 16k BPE tokenizer",
written by `pipeline_bench.py`. Before this file there was no way to ask that of an arbitrary
number except by reading the code that produced the figure beside it.

So the reports cite their sources the way they cite their references, and this resolves the
citations. Two directions, both of which matter:

    a cited record that is not on disk    a number nobody can check
    a record nothing cites                an experiment that ran and never reached a reader

The second is the one that is easy to miss and expensive: 143 GPU-hours of runs are worth nothing
if no sentence anywhere points at them.

    bash src/a2-nlp/py.sh check_provenance.py
    bash src/a2-nlp/py.sh check_provenance.py --table     # the appendix table, as markdown

Exits non-zero if a citation dangles.
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'runs')
REPORTS = os.path.join(HERE, 'reports')

# The study-level records: one per experiment, as opposed to the ~700 per-run files those
# experiments are computed FROM. A report cites the study record; the study record is derived
# from the run records; the run records are what the GPU actually wrote. Citing the middle layer
# is the useful altitude -- a reader can open it, and it is small enough to read.
STUDIES = {
    'pipeline_bench.json': ('pipeline_bench.py', 'what each stage of a run costs in wall-clock'),
    'budget.json': ('study_budget.py', 'luck vs skill vs search budget on the 60-run grid'),
    'lr_transfer.json': ('study_lr_transfer.py', 'five languages x six rates x two seeds'),
    'early_signal.json': ('early_signal.py', 'can a doomed run be detected early'),
    'clip_prevention.json': ('study_clip_prevention.py', 'does tighter clipping prevent divergence'),
    'swap_downstream.json': ('study_swap_downstream.py', 'the vocabulary swap, carried downstream'),
    'tokenizer_seeds.json': ('study_tokenizer_seeds.py', 'six pre-registered seeds a side'),
    'label_quantity.json': ('study_label_quantity.py', 'the decisive labelled-data experiment'),
    'ner_control_sweep.json': ('study_ner_control_sweep.py', 'the untrained NER floor, twelve rates'),
    'downstream_correlation.json': ('study_downstream_correlation.py',
                                    'does validation loss predict downstream score'),
    'gradient_table.json': ('gradient_table.py', 'the vocabulary penalty across seventeen languages'),
    'gradient_languages.json': ('prepare_gradient_languages.py', 'which languages were prepared'),
    'scaling_law.json': ('scaling_law.py', 'the fitted data/compute surface'),
    'claims_audit.json': ('claims_audit.py', 'every comparative claim against its null'),
    'hardware.json': ('bench_portable.py', 'one run costed on each machine — NOT YET COLLECTED'),
}

# `runs/pipeline_bench.json` in prose or code, with or without the runs/ prefix.
CITE = re.compile(r'`runs/([A-Za-z0-9_.-]+\.json)`')


def cited():
    """Which records each report names, keyed by record."""
    out = {}
    for path in sorted(glob.glob(os.path.join(REPORTS, '*.md'))):
        with io.open(path, encoding='utf-8') as fh:
            for n, line in enumerate(fh, 1):
                for m in CITE.finditer(line):
                    out.setdefault(m.group(1), []).append(
                        f'{os.path.basename(path)}:{n}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--table', action='store_true',
                    help='print the appendix table as markdown and exit')
    args = ap.parse_args()

    refs = cited()

    if args.table:
        print('| record | written by | what it holds |')
        print('|---|---|---|')
        for rec, (script, what) in STUDIES.items():
            here = '' if os.path.exists(os.path.join(RUNS, rec)) else ' *(not yet on disk)*'
            print(f'| `runs/{rec}`{here} | `{script}` | {what} |')
        return 0

    dangling = sorted(r for r in refs if not os.path.exists(os.path.join(RUNS, r)))
    uncited = sorted(r for r in STUDIES
                     if r not in refs and os.path.exists(os.path.join(RUNS, r)))
    unlisted = sorted(r for r in refs if r not in STUDIES)

    print(f'{len(refs)} records cited across the reports, '
          f'{len(STUDIES)} study records known')
    for r in sorted(refs):
        mark = 'ok  ' if os.path.exists(os.path.join(RUNS, r)) else 'GONE'
        print(f'  {mark} runs/{r:32} cited at {", ".join(refs[r][:3])}'
              f'{" …" if len(refs[r]) > 3 else ""}')

    for r in dangling:
        print(f'  DANGLING  runs/{r} is cited and is not on disk')
    for r in uncited:
        print(f'  uncited   runs/{r} exists and no report points at it')
    for r in unlisted:
        print(f'  unlisted  runs/{r} is cited but is not in STUDIES — add it or fix the citation')

    if dangling:
        print(f'\n{len(dangling)} dangling citation(s)')
        return 1
    print('\nevery citation resolves')
    return 0


if __name__ == '__main__':
    sys.exit(main())
