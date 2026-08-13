"""The bottom board's panel text, read out of the build sheet rather than retyped here.

`build_posters.py` arrived from PR #77 with its copy of both posters' words in two hand-written
dicts at the top of the file. That is the one rule this project keeps -- the repository carries
numbers, prose carries reasoning -- and the dicts had already drifted in the two days between
being written and being reviewed: they cite `reports/09-the-poster.md` and
`12-poster-build-sheet-v2.md`, both of which were renamed, and they quote counts that have since
moved.

So the layout tool reads the board file. `12-the-bottom-board.md` already holds the nine cells and
three strip blocks written to the 55-word measure, each as a blockquote under its own heading, and
those blockquotes are the words that go on the wall. Parsing them means a poster cannot say
something the build sheet does not.

    from board_content import bottom_board
    cells = bottom_board()          # {'1': Panel(...), ..., 'strip1': Panel(...)}

The top board is deliberately NOT parsed here. Report 13 is Patrick's and is written as prose
rather than to a measure, so there is nothing to lift; his half of `build_posters.py` still
carries its hand-written copy and is flagged in that file as the remaining hardcoded content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

BOARD = Path(__file__).resolve().parent.parent / 'reports' / '12-the-bottom-board.md'

# "### 3 · What belongs in a notebook, and what belongs in a queue?"
CELL = re.compile(r'^### (\d+) · (.+)$', re.M)
# "**Big number:** `53 s vs 85 min` / `96×` · **Figure:** `15-what-a-run-is-made-of.svg`"
BIG = re.compile(r'\*\*Big number:\*\*\s*(.+?)(?:\s*·\s*\*\*Figure:\*\*\s*(.+))?$', re.M)
WORDCOUNT = re.compile(r'^\*(\d+) words[^*]*\*$', re.M)


@dataclass
class Panel:
    number: str
    title: str
    big: str
    figure: str | None
    body: str
    words: int

    @property
    def big_lines(self) -> list[str]:
        """The big number as the lines it is set on -- `2.07×` / `1.32× real` is two lines.

        The board file writes the break as ` / ` because a 6.35 in column at that point size holds
        about eighteen characters, and the constraint is invisible until something overflows.
        """
        return [p.strip(' `') for p in self.big.split(' / ')]


def _blockquote(chunk: str) -> str:
    """The panel text: the first run of '> ' lines, unwrapped."""
    lines, out = chunk.split('\n'), []
    started = False
    for line in lines:
        if line.startswith('> '):
            started, _ = True, out.append(line[2:].rstrip())
        elif started and not line.startswith('>'):
            break
    return ' '.join(out).strip()


def bottom_board(path: Path | None = None) -> dict[str, Panel]:
    """Every cell and strip block of the bottom board, keyed by number."""
    text = (path or BOARD).read_text(encoding='utf-8')
    heads = list(CELL.finditer(text))
    out: dict[str, Panel] = {}
    for i, m in enumerate(heads):
        chunk = text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        big = BIG.search(chunk)
        counted = WORDCOUNT.search(chunk)
        body = _blockquote(chunk)
        out[m.group(1)] = Panel(
            number=m.group(1),
            title=m.group(2).strip(),
            big=(big.group(1).strip() if big else ''),
            figure=(big.group(2).strip(' `*') if big and big.group(2) else None),
            body=body,
            words=int(counted.group(1)) if counted else len(body.split()),
        )
    return out


def check(path: Path | None = None) -> list[str]:
    """What would make a panel unprintable. Empty list means the board is settable.

    The two failures worth catching before a print run, both of which have happened: a cell whose
    prose has grown past what the column holds, and a big number too wide for the measure.
    """
    problems = []
    for n, p in sorted(bottom_board(path).items()):
        actual = len(p.body.split())
        if not p.body:
            problems.append(f'cell {n}: no panel text found')
            continue
        if actual > 75:
            problems.append(f'cell {n}: {actual} words of body, over the ~55 the column holds')
        for line in p.big_lines:
            if len(line) > 22:
                problems.append(f'cell {n}: big-number line {line!r} is {len(line)} chars, '
                                f'over the ~18 that fit at 6.35 in')
    return problems


if __name__ == '__main__':
    import sys

    board = bottom_board()
    print(f'{len(board)} panels parsed from {BOARD.name}\n')
    for n, p in sorted(board.items(), key=lambda kv: int(kv[0])):
        fig = p.figure or '— no figure'
        print(f'  {n}. {p.title[:44]:46} {len(p.body.split()):3}w  {fig}')
        print(f'      big: {" / ".join(p.big_lines)}')
    problems = check()
    print()
    for msg in problems:
        print('  ' + msg)
    print(f'{len(problems)} problem(s)' if problems else 'every panel is settable as written')
    sys.exit(1 if problems else 0)
