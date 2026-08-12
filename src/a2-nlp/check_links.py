"""Every relative link in the markdown, resolved against the filesystem.

Patrick found the case this exists for. PR #69 renamed `12-poster-build-sheet.md` and did not
touch `reports/README.md`; PR #70 edited `reports/README.md` and did not touch report 12. Neither
diff contains the broken line, so no review of either PR can see it, and git has nothing to flag.
The index row simply starts pointing at a file that is no longer there, and the first person to
notice is a reader who clicks it.

That is a whole class rather than one incident: a rename in one PR and an edit in another, with
the damage in the intersection that neither shows. The check is four seconds and needs no GPU, no
network and no records.

    bash src/a2-nlp/py.sh check_links.py
    bash src/a2-nlp/py.sh check_links.py --root ..

Exits non-zero if anything is dangling, so it can go in front of a merge.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from urllib.parse import unquote

# Inline links, `[text](target)`. Reference-style definitions and bare URLs are left alone: the
# first are rare here and the second are not ours to verify.
LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')

SKIP_PREFIX = ('http://', 'https://', 'mailto:', '#')


def targets(path):
    with io.open(path, encoding='utf-8') as fh:
        for n, line in enumerate(fh, 1):
            for m in LINK.finditer(line):
                t = m.group(1).strip()
                if t.startswith(SKIP_PREFIX):
                    continue
                # Strip a fragment and any title in quotes: [x](y.md#section "title").
                t = t.split(' ')[0].split('#')[0]
                if t:
                    yield n, unquote(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)),
                    help='directory to walk; defaults to this file\'s directory')
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    bad, checked, files = [], 0, 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in ('.git', 'node_modules', '__pycache__', 'runs', 'data')]
        for name in sorted(filenames):
            if not name.endswith('.md'):
                continue
            files += 1
            src = os.path.join(dirpath, name)
            for line_no, target in targets(src):
                checked += 1
                if not os.path.exists(os.path.normpath(os.path.join(dirpath, target))):
                    bad.append((os.path.relpath(src, root), line_no, target))

    print(f'{checked} relative links in {files} markdown files under {root}')
    for src, line_no, target in bad:
        print(f'  BROKEN  {src}:{line_no} -> {target}')
    if bad:
        print(f'\n{len(bad)} broken')
        return 1
    print('all resolve')
    return 0


if __name__ == '__main__':
    sys.exit(main())
