# poster/ — turning a board into something printable

`build_posters.py` renders both posters into editable PowerPoint, print PDF and a PNG preview,
from the UW template. It came from Leon's PR #77 via #78 and is the only tool in the project that
produces something you can hand to a print shop.

```bash
bash src/a2-nlp/py.sh poster/board_content.py     # are the boards settable? (no deps)
bash src/a2-nlp/py.sh poster/build_posters.py     # render (needs Windows + PowerPoint)
```

Rendering needs `python-pptx PyMuPDF Pillow pywin32` and a real PowerPoint install — it drives it
over COM. Outputs land in `poster/outputs/`, which is **not tracked**: they are generated, they
are ~11 MB, and #78 already rejected carrying built posters in the repo. Regenerate rather than
commit.

## What prints

**`board-yoruba-findings`** and **`board-model-factory`** — the two boards, set to report 12's
measured grid: nine cells in a 3 × 3, three strip blocks across the foot, 24 × 36 in portrait.
**These are the posters.**

The three `design-*` builders are alternates inherited from #77. They arrange semantic slots
("headline", "causal") that predate the grid, so each shows a chosen subset of the board rather
than the board. Useful to look at; not the deliverable.

## Both boards' words come from the board files

`board_content.py` parses `12-the-bottom-board.md` and `13-the-top-board.md` — nine cells, three
strip blocks and the title block each — and `build_posters.py` builds its content dict from that.
**A poster cannot say something the build sheet does not.**

That sentence was in three docstrings before it was true of the code. #78 wrote the parser and
left the hand-written dicts in place: `bottom_board` was imported and never called. The evidence
was in the file it described — the factory dict's sources line still cited
`reports/09-the-poster.md` and `12-poster-build-sheet-v2.md` weeks after both were renamed, and
no link checker could see it, because a filename inside a Python string is not a link. The footer
citation is now derived from the `Path` that was actually read.

## Three things found by rendering, in the order they hid each other

**The overflow validator could never fire.** `layout-validation.json` flags a text box whose text
is taller than the box. python-pptx writes `<a:spAutoFit/>` into every textbox it creates, so
PowerPoint had already grown each box to fit — `BoundHeight` was always a few points *under*
`Height`. It reported 55 shapes examined, 0 skipped, **0 overflows**, on a proof whose cells
visibly ran over each other. Turning autofit off makes the declared geometry authoritative and
the check real.

**Every poster was 36 × 48.** `assert_template_size()` was added in #78 to stop exactly that, and
it passes: it measures the *template* as it is opened, and `reflow_for_print` rescales the sheet
to 36 × 48 afterwards, multiplying every font by 1.6. A guard on the input to a transform says
nothing about its output. Reflow is off by default now, and `main()` checks the exported page
size — the half the original assertion could not see.

**Report 12's word budget overshoots its own geometry by ~13%.** Measured off the rendered
boards, the rate is **~2.82 pt of column per word** at 18 pt in a 5.77 in measure, so the 1.9 in a
cell keeps for prose holds about **48 words, not 55**. Both boards were written to the guide. The
body now takes 2.72 in out of the figure and the big number rather than re-cutting two boards.
The strip is worse: at 18 pt a strip block carrying a figure has 1.50 in left, which is 38 words
against the ~100 budgeted, so the strip is set at 15 pt.

## board_content.py as a gate

Run on its own it parses every panel of both boards and reports what would make one unprintable —
body past what the column holds, or a big number too wide for the measure. Word caps are
figure-aware, because a cell with no figure legitimately runs long as type.

It also prints, as **advisory** rather than blocking, any panel whose `*NN words*` annotation no
longer matches the prose above it. Eleven of the bottom board's twelve were out, by up to eleven
words. Nothing about that stops a poster printing — it is just this project's most-repeated
lesson in miniature, a count written in prose with no invalidation.

## What was not taken from #77

| dropped | reason |
|---|---|
| 8 generated PPTX/PDF/PNG outputs | built at **36 × 48 in**; the template is 24 × 36 |
| 12 font files, ~33 MB | Fontfabric Uni Sans is a commercial licence, and the figures do not use them — PowerPoint substitutes |
| `__MACOSX/`, 30-odd `._*` files | macOS resource forks |
| the horizontal template | neither board is landscape |
| the `FACTORY` and `YORUBA` content dicts | hand-typed prose and numbers, already citing two renamed files |

`build_evidence_grid` is the landscape concept and is **unreachable** — it opens the template that
was not taken, so it raises before drawing. It is kept, out of `BUILDERS` and documented, because
restoring it needs one file rather than a rewrite. It is also the source of every "content runs
out to x = 34.75 in" warning in the module: those blocks are correct for a 48 in board and were
never a defect in the portrait path, which has always fitted 24 in. That warning was inherited
through two hands without being checked.
