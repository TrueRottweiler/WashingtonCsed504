# poster/ — turning a board into something printable

`build_posters.py` renders the two posters into editable PowerPoint, print PDF and a PNG preview,
from the UW templates. It came from Leon's PR #77 and is the only tool in the project that
produces something you can hand to a print shop.

```bash
bash src/a2-nlp/py.sh poster/board_content.py     # is the board settable? (no deps)
bash src/a2-nlp/py.sh poster/build_posters.py     # render (needs Windows + PowerPoint)
```

## What was taken from #77, and what was not

**Taken:** the generator, the vertical template, the four layout concepts, and the
overflow-validation idea — `layout-validation.json` records any text box whose content does not
fit, which is the right shape for this project.

**Not taken, and why.**

| dropped | reason |
|---|---|
| 8 generated PPTX/PDF/PNG outputs | built at **36 × 48 in**; the template is 24 × 36 |
| 12 font files, ~33 MB | Fontfabric Uni Sans is a commercial licence, and the figures do not use them — PowerPoint substitutes |
| `__MACOSX/`, 30-odd `._*` files | macOS resource forks |
| the horizontal template | neither board is landscape |
| the `FACTORY` content dict | hand-typed prose and numbers, already citing two renamed files |

Total: 33.6 MB of binaries and eight posters at the wrong size, against one 58 KB source file
that does the actual work.

## Three things to know before rendering

**The size guard.** `assert_template_size()` refuses to build at anything but 24 × 36. The old
outputs were 36 × 48 and there is no way to tell on screen — a wrong-size poster looks exactly
like a right-size one until the printer refuses it. This was my error first: the build sheet
asserted 3 ft × 4 ft for a fortnight and this file inherited it.

**The layouts still need re-fitting.** Seven geometry blocks place content out to **x = 34.75 in**,
ten inches past the right edge of a 24 in board. That is design work, not a bug fix, so it has been
left rather than guessed at. Until it is done the generator will run and the output will be wrong
in the horizontal direction.

**The bottom board's words come from the board file.** `board_content.py` parses the nine cells and
three strip blocks out of `12-the-bottom-board.md`, where they are already written to the 55-word
measure. A poster therefore cannot say something the build sheet does not. The `YORUBA` dict in
`build_posters.py` is still hand-written, because report 13 is prose rather than panel text — that
is the remaining piece of hardcoded content in this folder.

## board_content.py as a gate

Run on its own it parses every panel and reports what would make one unprintable — body text past
what the column holds, or a big number too wide for the measure. It found three problems in the
board on its first run, including two word counts that had been estimated rather than counted, and
one big number 28 characters long in a column that holds about 18. No dependencies, so it runs
anywhere.
