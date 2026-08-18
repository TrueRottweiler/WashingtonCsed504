"""Both boards' panel text, read out of the build sheets rather than retyped here.

`build_posters.py` arrived from PR #77 with its copy of both posters' words in two hand-written
dicts at the top of the file. That is the one rule this project keeps -- the repository carries
numbers, prose carries reasoning -- and the dicts had already drifted in the two days between
being written and being reviewed: they cite `reports/09-the-poster.md` and
`12-poster-build-sheet-v2.md`, both of which were renamed, and they quote counts that have since
moved.

So the layout tool reads the board files. Each holds its cells and strip blocks written to the
55-word measure, each as a blockquote under its own heading, and those blockquotes are the words
that go on the wall. Parsing them means a poster cannot say something the build sheet does not.

    from board_content import bottom_board, top_board
    cells = bottom_board()          # {'1': Panel(...), ..., 'strip1': Panel(...)}

WHAT CHANGED ON 17 AUGUST, and the first item is the one that matters.

**The parser was imported and never called.** `build_posters.py` has
`from board_content import bottom_board, check as check_board` at the top and no call to either,
so both content dicts were still hand-written and the claim in three docstrings -- that a poster
cannot say something the build sheet does not -- was not true of the code. The proof was sitting
in the file it described: `FACTORY['sources']` still cited the two renamed filenames that
parsing was introduced to stop citing. `check_links.py` could not see it, because a filename
inside a Python string is not a markdown link. That is this project's own pattern one level up
from where it usually lands -- not a constant deciding a result, but a *claim about the code that
the code does not implement*.

**The strip blocks were never parsed.** The docstring above promised `'strip1'` and the heading
regex required `### <digit> · `, so the three blocks across the foot of each board were silently
absent -- and with them `06-what-it-cost.svg`, which is the bottom board's only figure that lives
in the strip. Nine panels came back where the docstring said twelve.

**A `**Figure:**` line without a `**Big number:**` before it was dropped.** That is exactly how
the strip's one figure is written, so the two defects hid each other.

**The word cap is figure-aware now.** Report 12 measures the column at ~55 words with a figure
and ~110 without; the old flat cap of 75 would have failed any cell that legitimately runs long
as type. Both of the top board's figureless cells do.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / 'reports'
BOTTOM = REPORTS / '12-the-bottom-board.md'
TOP = REPORTS / '13-the-top-board.md'

# The two sections of a board file that hold settable text. Everything else in these documents is
# commentary, and the appendices carry '### ' headings of their own -- so the parse is bounded to
# these regions rather than run over the whole file.
# "The cells" as well as "The nine cells": the bottom board's 17 Aug redesign dropped the uniform
# 3x3 (four chart panels, a stat rail, three statement cards), so its section stopped counting
# itself. The top board still says nine and still has nine.
CELLS_HEAD = re.compile(r'^## The (?:nine )?cells\b.*$', re.M)
STRIP_HEAD = re.compile(r'^## The strip\b.*$', re.M)
TITLE_HEAD = re.compile(r'^## Title block\s*$', re.M)
NEXT_SECTION = re.compile(r'^## ', re.M)
# "Author line, in the header band: *Jeffrey Stall · A2-NLP · CSED 504 · ...*", which wraps.
AUTHOR = re.compile(r'Author line, in the header band: \*(.+?)\*', re.S)

# "### 3 · What belongs in a notebook, and what belongs in a queue?" -- and, in the strip,
# "### What it cost", which carries no number.
PANEL = re.compile(r'^### (?:(\d+) · )?(.+)$', re.M)
# "**Big number:** `53 s vs 85 min` / `96×` · **Figure:** `15-what-a-run-is-made-of.svg`"
# Either half may appear without the other: the strip's cost block is Figure-only.
BIG = re.compile(r'\*\*Big number:\*\*\s*(.*?)(?=\s*·\s*\*\*Figure:\*\*|\s*$)', re.M)
FIG = re.compile(r'\*\*Figure:\*\*\s*(.+?)\s*$', re.M)
# Anchored on the opening "*NN words" only. Requiring the line to close with an unbroken run of
# non-asterisks -- '^\*(\d+) words[^*]*\*$', which is what this was -- silently failed on every
# annotation citing a glob, because `runs/ft_sib200_*.json` puts an asterisk mid-line. Four cells
# skipped the count check that way, and a check that quietly matches nothing is worse than none.
WORDCOUNT = re.compile(r'^\*(\d+) words\b', re.M)

# Report 12's writing guides: ~55 words in a cell with a figure, ~110 without, ~100 in a strip
# block. Keep them -- they are what the prose is written to.
GUIDE = {'figure': 55, 'type': 110, 'strip': 100}

# The caps are MEASURED off the rendered boards, not the guide times a slack factor. The body
# boxes are 194.4 pt with a figure, 329.8 pt without and 167.8 pt in a strip.
#
# A WORD COUNT IS A PROXY AND THESE ARE THE CONSERVATIVE END OF IT. The observed rate runs
# 2.82-3.23 pt of column per word at 18 pt, because it depends on how long the words are: plain
# prose sets at 2.82, and a cell carrying `121,339,416`, `yor-bpe16k` and `macro-F1` sets at 3.23.
# No single word cap is both safe for the second and permissive for the first, so these divide by
# the worst rate. A panel over the cap is not necessarily unprintable -- it needs looking at.
#
# THE AUTHORITY IS THE RENDERER. `build_posters.main()` fails on any box whose laid-out text is
# taller than the box it was given, which is measured rather than estimated. This gate is the
# cheap pre-check that runs without PowerPoint; that one is the gate.
#
# (The first version of these derived the caps as guide x (75/55), giving 'type' a cap of 150. A
# 133-word cell passed and then overflowed by 100 pt. A cap inferred from another cap.)
CAP = {'figure': 65, 'type': 102, 'strip': 80}

# ...and those three assume the panel carries a big number, because eleven of the twelve on each
# board do. A panel without one gets that block's 1.03 in back, which is 26 more words at 18 pt
# and 38 at 15 pt. Without this the cap said 86 where report 12's own guide for a strip block is
# ~100, and it rejected the bottom board's Next/sources/AI block at 87 words -- a block that fits
# with 37 to spare. Same shape as the bug above, one variable further in: a cap that quietly
# assumes a layout the panel does not have.
BIG_BLOCK_WORDS = {'figure': 26, 'type': 26, 'strip': 38}

# The measure holds about 18 characters on a big-number line; past ~22 it certainly overflows.
BIG_LINE_CHARS = 22


@dataclass
class Panel:
    number: str
    title: str
    big: str
    figure: str | None
    body: str
    words: int
    strip: bool = False

    @property
    def big_lines(self) -> list[str]:
        """The big number as the lines it is set on -- `2.07×` / `1.32× real` is two lines.

        The board file writes the break as ` / ` because a 6.35 in column at that point size holds
        about eighteen characters, and the constraint is invisible until something overflows.
        """
        return [p.strip(' `') for p in self.big.split(' / ')] if self.big else []

    @property
    def body_plain(self) -> str:
        """The panel text with markdown emphasis removed -- a PowerPoint run carries no markup."""
        return _plain(self.body)

    @property
    def budget(self) -> str:
        """Which word budget this panel is held to -- the column is what decides it."""
        if self.strip:
            return 'strip'
        return 'figure' if self.figure else 'type'

    @property
    def limit(self) -> int:
        b = self.budget
        return CAP[b] + (0 if self.big_lines else BIG_BLOCK_WORDS[b])


def _blockquote(chunk: str) -> str:
    """The panel text: the first run of '> ' lines, unwrapped.

    The first run, not every run -- a cell may carry a second blockquote of commentary below its
    word count, and that is a note to whoever sets the board rather than words for the wall.
    """
    lines, out = chunk.split('\n'), []
    started = False
    for line in lines:
        if line.startswith('> '):
            started, _ = True, out.append(line[2:].rstrip())
        elif started and not line.startswith('>'):
            break
    return ' '.join(out).strip()


def _figure(chunk: str) -> str | None:
    """The figure filename, or None where the panel says it is set as type instead."""
    m = FIG.search(chunk)
    if not m:
        return None
    name = m.group(1).strip(' `*')
    return None if name.lower().startswith('none') else name


def _region(text: str, head: re.Pattern) -> str:
    """One bounded section of a board file, from its heading to the next '## '."""
    m = head.search(text)
    if not m:
        return ''
    rest = text[m.end():]
    nxt = NEXT_SECTION.search(rest)
    return rest[: nxt.start()] if nxt else rest


def _panels(region: str, *, strip: bool) -> list[Panel]:
    heads = list(PANEL.finditer(region))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(region)
        chunk = region[m.end():end]
        big = BIG.search(chunk)
        counted = WORDCOUNT.search(chunk)
        body = _blockquote(chunk)
        out.append(Panel(
            number=m.group(1) or f'strip{len(out) + 1}',
            # Cleaned here, not at render time: a heading like "Does the vocabulary *cause* it?"
            # printed its asterisks on the first proof. Bodies went through _plain and titles did
            # not, which is the same omission one field over.
            title=_plain(m.group(2).strip()),
            big=(big.group(1).strip() if big else ''),
            figure=_figure(chunk),
            body=body,
            words=int(counted.group(1)) if counted else len(body.split()),
            strip=strip,
        ))
    return out


@dataclass
class TitleBlock:
    title: str
    subtitle: str
    takeaway: str
    author: str
    source_file: str
    goals: str = ''
    citations: str = ''

    @property
    def sources_line(self) -> str:
        """The footer citation, built from the filename rather than typed beside it.

        This is the line that carried the defect: `FACTORY['sources']` named
        `reports/09-the-poster.md` and `12-poster-build-sheet-v2.md` months after both were
        renamed, and no link checker could see it because a filename inside a Python string is
        not a link. Deriving it from the Path that was actually read makes the class of error
        impossible rather than merely fixed.
        """
        provenance = (f'Source: reports/{self.source_file} · every number on this board '
                      f'regenerates from committed runs/ records')
        return f'{self.citations} · {provenance}' if self.citations else provenance


def _quote_runs(region: str) -> list[list[str]]:
    """Every separate blockquote in a region, as a list of unwrapped line lists.

    Separate, not merged: the top board's Title block holds two -- the title block proper and the
    goals line that sits under the author line. Joining them put the whole of the goals text
    inside the takeaway box on the first render.
    """
    runs, current = [], []
    for line in region.split('\n'):
        if line.startswith('>'):
            text = line[2:].rstrip() if line.startswith('> ') else ''
            current.append(text)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def title_block(path: Path) -> TitleBlock:
    """The header band: title, subtitle, the takeaway sentence, the author line, and goals."""
    region = _region(path.read_text(encoding='utf-8'), TITLE_HEAD)
    runs = _quote_runs(region)
    head = runs[0] if runs else []
    title = next((ln[3:].strip() for ln in head if ln.startswith('## ')), '')
    subtitle = next((ln.strip('*') for ln in head
                     if ln.startswith('*') and not ln.startswith('**')), '')
    takeaway = ' '.join(ln for ln in head
                        if ln and not ln.startswith('## ') and ln.strip('*') != subtitle)
    author = AUTHOR.search(region)
    return TitleBlock(
        title=title,
        subtitle=subtitle,
        takeaway=_plain(takeaway),
        author=_plain(' '.join(author.group(1).split())) if author else '',
        goals=_plain(_labelled_quote(region, 'Goals')),
        citations=_plain(_labelled_quote(region, 'Citations')),
        source_file=path.name,
    )


def _labelled_quote(region: str, label: str) -> str:
    """The blockquote introduced by a bold **Label** paragraph, or ''.

    By label rather than by position. Positionally -- runs[1] is goals, runs[2] is citations --
    the bottom board, which needs citations and has no goals, would have printed its reference
    list across the header band as though it were the project's objectives. Two boards do not
    have to carry the same optional blocks, so the block has to say what it is.
    """
    m = re.search(rf'^\*\*{label}\*\*', region, re.M)
    if not m:
        return ''
    runs = _quote_runs(region[m.end():])
    return ' '.join(ln for ln in (runs[0] if runs else []) if ln)


def _plain(text: str) -> str:
    """Markdown emphasis stripped -- PowerPoint runs carry no inline markup."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    return re.sub(r'\*(.+?)\*', r'\1', text).strip()


def board(path: Path) -> dict[str, Panel]:
    """Every cell and strip block of one board, keyed by number then 'strip1'..'strip3'."""
    text = path.read_text(encoding='utf-8')
    panels = (_panels(_region(text, CELLS_HEAD), strip=False)
              + _panels(_region(text, STRIP_HEAD), strip=True))
    return {p.number: p for p in panels}


def bottom_board(path: Path | None = None) -> dict[str, Panel]:
    return board(path or BOTTOM)


def top_board(path: Path | None = None) -> dict[str, Panel]:
    return board(path or TOP)


def check(path: Path | None = None) -> list[str]:
    """What would make a panel unprintable. Empty list means the board is settable.

    The two failures worth catching before a print run, both of which have happened: a cell whose
    prose has grown past what the column holds, and a big number too wide for the measure.
    """
    problems = []
    for name, panels in _boards(path):
        for n, p in panels.items():
            if not p.body:
                problems.append(f'{name} {n}: no panel text found')
                continue
            for line in p.big_lines:
                if len(line) > BIG_LINE_CHARS:
                    problems.append(f'{name} {n}: big-number line {line!r} is {len(line)} chars, '
                                    f'over the ~18 that fit at 6.35 in')
    return problems


def long_panels(path: Path | None = None) -> list[str]:
    """Panels past the word cap -- advisory, because a word count is a proxy and the caps are the
    conservative end of a range.

    This blocked, briefly, and blocking was wrong in both directions. Set to the safe end it
    rejected four of the bottom board's cells at 66-69 words, every one of which renders with room
    to spare because Jeffrey writes plainer prose than the rate assumed. Set to the permissive end
    it passed a 133-word cell that then overflowed by 100 pt. The quantity it is estimating --
    laid-out height -- depends on how long the words are, so no word count can decide it.

    `build_posters.main()` measures the real thing and fails on it. This one says "look at that".
    """
    notes = []
    for name, panels in _boards(path):
        for n, p in panels.items():
            actual = len(p.body.split())
            if p.body and actual > p.limit:
                notes.append(f'{name} {n}: {actual} words, over the ~{p.limit} a {p.budget} '
                             f'panel usually holds -- check it in the render')
    return notes


def stale_counts(path: Path | None = None) -> list[str]:
    """Panels whose "*NN words*" annotation no longer matches the prose above it.

    Advisory rather than blocking: the annotation is a note to whoever sets the board, and being
    eight words out has never stopped anything printing. It is still worth printing, because it is
    this project's most-repeated lesson in miniature -- a count written in prose is a cache with
    no invalidation, and both boards were carrying drifted ones the first time this ran.
    """
    notes = []
    for name, panels in _boards(path):
        for n, p in panels.items():
            actual = len(p.body.split())
            if p.body and p.words != actual:
                notes.append(f'{name} {n}: annotated "{p.words} words", counts {actual}')
    return notes


def _boards(path: Path | None):
    if path is not None:
        return [(path.stem, board(path))]
    return [('bottom', bottom_board()), ('top', top_board())]


if __name__ == '__main__':
    import sys

    # A Windows console defaults to cp1252, which has no U+2212 MINUS SIGN -- and one board's big
    # number is "−0.080". Found by cloning fresh and running this the way the README says to,
    # rather than through py.sh, which sets UTF-8 for exactly this reason. A gate that only runs
    # under its wrapper is a gate that fails for the first person who invokes it directly.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    problems, notes = check(), long_panels() + stale_counts()
    for name, panels in _boards(None):
        src = BOTTOM if name == 'bottom' else TOP
        print(f'\n{name} board -- {len(panels)} panels from {src.name}')
        for n, p in panels.items():
            fig = p.figure or f'— no figure, {p.budget}'
            print(f'  {n:>6}. {p.title[:42]:44} {len(p.body.split()):3}w/{p.limit:<3} {fig}')
            if p.big_lines:
                print(f'          big: {" / ".join(p.big_lines)}')
    if notes:
        print(f'\n{len(notes)} drifted word count(s) -- advisory, nothing here blocks a print run')
        for msg in notes:
            print('  ' + msg)
    print()
    for msg in problems:
        print('  ' + msg)
    print(f'{len(problems)} problem(s)' if problems else 'every panel is settable as written')
    sys.exit(1 if problems else 0)
