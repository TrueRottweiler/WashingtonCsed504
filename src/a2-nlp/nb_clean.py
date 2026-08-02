"""
nb_clean.py -- make an executed notebook render everywhere, and shrink it.

Run this after `jupyter nbconvert --execute`. Executing a notebook that touches Hugging Face
leaves three kinds of debris behind, and the first one stops cells from rendering at all:

  ipywidgets views      tqdm and the datasets/transformers download bars emit outputs of type
                        application/vnd.jupyter.widget-view+json. A widget view is only a
                        POINTER; the actual widget lives in the notebook's metadata.widgets
                        block. If that block is missing or stale -- and it is missing whenever
                        the notebook was executed headlessly, or whenever someone stripped it to
                        save space -- the viewer has a pointer to nothing and shows
                        "Could not render content" where the output should be. Every one of
                        these outputs also carries a text/plain fallback, so dropping the widget
                        mime type loses nothing readable and fixes the render.

  ANSI escape codes     transformers prints its load reports with \\x1b[1m bold sequences. Some
                        viewers interpret them, some print them literally as [1m.

  missing cell ids      nbformat 4.5 gave every cell an id, and tooling increasingly assumes it.
                        A notebook rebuilt by hand (or by a transform script that copies cells)
                        can easily lose them.

Usage:
    python nb_clean.py POC_v4_factory.ipynb
    python nb_clean.py *.ipynb --quiet
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

# The mime type that renders as "Could not render content" without its backing state.
WIDGET_MIME = 'application/vnd.jupyter.widget-view+json'

# CSI escape sequences: ESC [ ... final-byte.
ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


def clean_notebook(nb: dict) -> dict:
    """Strip widget views, ANSI codes, and stale widget state; restore cell ids."""
    stats = {'widget_outputs': 0, 'ansi': 0, 'ids': 0, 'empty_outputs': 0}

    for n, cell in enumerate(nb.get('cells', [])):
        # nbformat 4.5 wants an id on every cell. Derive it from the index so re-running this
        # is stable -- a random id would make every clean produce a spurious diff.
        if not cell.get('id'):
            cell['id'] = f'cell-{n:03d}'
            stats['ids'] += 1

        kept = []
        for out in cell.get('outputs', []):
            data = out.get('data')
            if data and WIDGET_MIME in data:
                del data[WIDGET_MIME]
                stats['widget_outputs'] += 1
                # A widget view with nothing but the pointer has no fallback to keep.
                if not data:
                    stats['empty_outputs'] += 1
                    continue

            if out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
                if '\x1b[' in text:
                    out['text'] = ANSI.sub('', text).splitlines(keepends=True)
                    stats['ansi'] += 1
            elif data and 'text/plain' in data:
                plain = ''.join(data['text/plain'])
                if '\x1b[' in plain:
                    data['text/plain'] = ANSI.sub('', plain).splitlines(keepends=True)
                    stats['ansi'] += 1

            kept.append(out)

        if 'outputs' in cell:
            cell['outputs'] = kept

    # The widget state block is what the views pointed at. With the views gone it is dead weight
    # -- and it is the single largest thing in a Colab-exported notebook, routinely 800 KB.
    if nb.get('metadata', {}).pop('widgets', None) is not None:
        stats['widget_state'] = True

    # Declare the version whose rules we just satisfied.
    nb['nbformat'], nb['nbformat_minor'] = 4, 5
    return stats


def main():
    p = argparse.ArgumentParser(description='Fix rendering of executed notebooks.')
    p.add_argument('paths', nargs='+', help='notebook paths or globs')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()

    targets = []
    for pattern in args.paths:
        targets.extend(glob.glob(pattern) if any(c in pattern for c in '*?[') else [pattern])

    for path in targets:
        before = os.path.getsize(path)
        with open(path, encoding='utf-8') as f:
            nb = json.load(f)
        stats = clean_notebook(nb)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write('\n')
        after = os.path.getsize(path)
        if not args.quiet:
            print(f'{os.path.basename(path)}: {before/1024:.0f} -> {after/1024:.0f} KB | '
                  f'{stats["widget_outputs"]} widget views removed '
                  f'({stats["empty_outputs"]} outputs dropped entirely), '
                  f'{stats["ansi"]} ANSI-stripped, {stats["ids"]} ids added'
                  + (', widget state dropped' if stats.get('widget_state') else ''))


if __name__ == '__main__':
    main()
