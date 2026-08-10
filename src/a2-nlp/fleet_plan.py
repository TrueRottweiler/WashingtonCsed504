"""Let a study announce its own queue, so nothing can run undeclared.

The dashboard has now gone blind three times, and each time the fix was to remember to re-run a
declaration script. That is not a fix, it is a habit, and the habit failed at the extended
learning-rate sweep, the tokenizer seeds and the NER control sweep -- three for three.

The cause is structural. `declare_studies.py` is a hand-written snapshot of what somebody
believed was queued at the moment they wrote it. Every study driven from a plain Python loop --
which is all of them, because mlm_fleet only covers pretraining grids -- has to be added to that
file by hand. A dashboard that is only correct when a human remembers to update it will be wrong
exactly when it matters, which is at 3 a.m. while nobody is watching.

So the announcement becomes a side effect of running. Each study calls `announce()` at the top of
its own main(), with the cell list it is about to iterate over -- which it already has, because it
built it to loop on. Forgetting is no longer possible without also forgetting to run the study.

Cells from other studies are preserved, so two studies on two cards do not erase each other, and
re-announcing an in-flight study fills in its finished rows rather than resetting them.
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN = os.path.join(HERE, 'runs', '_fleet_plan.json')


def cell(tag, label, *, kind='pretrain', eta_s=600, corpus=None, steps=0, update_tokens=0,
         preset='poc'):
    """One row for the dashboard's queue panel."""
    return {'kind': kind, 'tag': tag, 'label': label, 'eta_s': eta_s, 'corpus': corpus,
            'steps': steps, 'update_tokens': update_tokens, 'preset': preset}


def announce(queue, cells, *, n_gpu=2, replace_prefix=None, owner=None):
    """Publish `cells` under `queue`, keeping everything another study already declared.

    `replace_prefix` drops this study's own previous rows before adding the new ones, which is
    what a re-announce after extending a grid wants -- otherwise the old and new cell lists both
    sit in the panel and the totals double.

    `queue` and `owner` are stamped onto every CELL, not just onto the file. The top-level
    'queue' key is whoever announced last, so with three studies running the panel used to put
    two hundred cells under one heading -- and the heading belonged to whichever script started
    most recently. You could see that the machine was busy and not whose work it was busy with,
    which is the question anyone actually walks over to the screen to ask.
    """
    try:
        with open(PLAN, encoding='utf-8') as f:
            old = json.load(f)
        keep = {c['tag']: c for c in old.get('cells', [])}
    except (OSError, ValueError):
        old, keep = {}, {}

    if replace_prefix:
        keep = {t: c for t, c in keep.items() if not t.startswith(replace_prefix)}

    # Anything already recorded for a tag wins over the freshly declared placeholder, so state
    # that a run has finished survives a re-announce. Attribution is the exception: it comes from
    # the caller announcing now, so a study that gets renamed or reassigned is not stuck with
    # whatever the first announce happened to say.
    # Which script published this queue. The dashboard uses it to answer "what is running right
    # now" from the process list rather than from run files: a study doing fine-tuning writes
    # nothing until each cell lands, so file-based liveness cannot see it, but the study's own
    # process is sitting there in plain view the whole time. Without this the panel could only
    # say it did not know, which is a poor answer when the answer is one process scan away.
    script = os.path.basename(sys.argv[0]) or None

    for c in cells:
        prev = keep.get(c['tag'], {})
        merged = {**c, **{k: v for k, v in prev.items() if v is not None}}
        keep[c['tag']] = {**merged, 'study': queue, 'owner': owner, 'script': script}

    plan = {**old, 'queue': queue, 'n_gpu': n_gpu, 'started': old.get('started') or time.time(),
            'announced': time.time(), 'batch': 128, 'seq_len': 128,
            'cells': list(keep.values())}
    tmp = f'{PLAN}.{os.getpid()}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    os.replace(tmp, PLAN)
    print(f'[fleet_plan] announced {len(cells)} cells for {queue!r} '
          f'({len(plan["cells"])} in the panel total)')
    return plan
