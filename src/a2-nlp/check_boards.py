"""Which figure each poster claims, and whether the two boards claim the same one.

The companion to check_links.py, for the same class of problem: something that is wrong only in
the space *between* two documents, so neither can see it from the inside and no review of either
one catches it.

Here the failure is two posters hanging side by side carrying the identical chart. Both documents
are individually correct. Report 12 v2's week 7 names `03-matched-steps-vs-compute.svg`, report
13's matched-compute panel names Figure 03, and each is the right figure for the argument it sits
under. Printed and mounted together they read as a mistake, and the first person to notice is
somebody standing in front of them at the poster session.

The two boards name figures differently -- report 12 gives filenames, report 13 gives "Figure NN"
-- so both spellings are read and reduced to a number. That difference is itself worth knowing
about: it is why a plain text search for the collision finds nothing.

Also reports figures that exist and no board carries, which is the cheaper question and the one
worth asking before drawing a new one.

    bash src/a2-nlp/py.sh check_boards.py

Exits non-zero if any figure is claimed twice, so it can go in front of a print run.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

BOARDS = (
    ('bottom (report 12 v2)', 'reports/12-poster-build-sheet-v2.md'),
    ('top (report 13)', 'reports/13-the-top-board.md'),
)

# `03-matched-steps-vs-compute.svg` and `Figure 03` are the same claim written two ways.
BY_FILENAME = re.compile(r'\b(\d\d)-[a-z0-9-]+\.(?:svg|png)')
BY_NUMBER = re.compile(r'\bFigures? (\d\d)\b')

# Both boards discuss figures outside their panels -- which ones are free, which went upstairs,
# what still needs drawing -- and that is commentary rather than a panel claiming wall space. The
# rule is structural rather than a pattern on the citation line: a figure named under a content
# heading is claimed, a figure named under a housekeeping heading is not. Keyed on the heading
# because an earlier version keyed on the `**Figure:**` line and missed the bottom strip, which
# carries figure 06 in running prose and is as much a panel as any cell.
HOUSEKEEPING = ('gaps', 'order of work', 'what the board must not do', 'references',
                'appendix', 'format', 'the arc, in one paragraph')


def claims(path):
    """Figure numbers this document puts on the wall, not ones it merely talks about."""
    out = {}
    with io.open(path, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    counting = True
    for i, line in enumerate(lines):
        if line.startswith('#'):
            title = line.lstrip('#').strip().lower()
            counting = not any(title.startswith(h) for h in HOUSEKEEPING)
        if not counting:
            continue
        for pat in (BY_FILENAME, BY_NUMBER):
            for m in pat.finditer(line):
                out.setdefault(m.group(1), i + 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figures', default=os.path.join(HERE, 'reports', 'figures'))
    args = ap.parse_args()

    claimed, missing = {}, []
    for name, rel in BOARDS:
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            missing.append(rel)
            continue
        got = claims(path)
        claimed[name] = got
        print(f'{name}: {len(got)} figures — {", ".join(sorted(got))}')
    for rel in missing:
        print(f'  (not on disk: {rel})')

    on_disk = {m.group(1) for f in os.listdir(args.figures) if (m := BY_FILENAME.match(f))}

    both = set.intersection(*(set(c) for c in claimed.values())) if len(claimed) > 1 else set()
    for n in sorted(both):
        where = '; '.join(f'{b} line {claimed[b][n]}' for b in claimed)
        print(f'  CLAIMED TWICE  figure {n} — {where}')

    unused = sorted(on_disk - set().union(*(set(c) for c in claimed.values()), set()))
    if unused:
        print(f'  unclaimed by either board: {", ".join(unused)}')

    ghosts = sorted(set().union(*(set(c) for c in claimed.values()), set()) - on_disk)
    if ghosts:
        print(f'  named but not on disk: {", ".join(ghosts)}')

    if both:
        print(f'\n{len(both)} figure(s) claimed by both boards')
        return 1
    print('\nno figure is claimed by both boards')
    return 0


if __name__ == '__main__':
    sys.exit(main())
