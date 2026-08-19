# poster/final — what actually prints

This directory is tracked; [`poster/outputs/`](../outputs) is not. The split is deliberate and
it is the #78 precedent applied, not reversed: a **generated** board rebuilds from the repository
in one command, so committing every build would be noise — but the **printed** artifact is a
deliverable the grader should not need PowerPoint and a Windows build machine to see, and a
**hand-built** board is not reproducible from anything in the repository at all, which is the
whole reason it must be in git. Jeffrey ruled on the path on 18 August, on Patrick's proposal.

Each file carries its provenance here, because the pair look alike and are made in opposite ways.

| file | provenance |
|---|---|
| `board-model-factory_v2.pptx` / `.pdf` | **Generated** by [`poster/build_posters.py`](../build_posters.py) from the build sheet [`reports/12-the-bottom-board.md`](../../reports/12-the-bottom-board.md) — every number regenerates from committed `runs/` records. Rebuild with `bash src/a2-nlp/py.sh poster/build_posters.py` (needs PowerPoint + pywin32); the committed copy is the printed version, from the commit that carries it. |
| *top board — Leon's, to be added* | **Hand-built in PowerPoint** by Leon Wan from the numbers in [`reports/13-the-top-board.md`](../../reports/13-the-top-board.md) — **not** produced by `build_posters.py`, and not reproducible from the repository. Patrick commits it here (PPTX and PDF, so the editable source travels with the printed artifact) when Leon sends the final version with the five rubric blocks. |

Both boards are 24 × 36 in on the UW research-poster template and hang as a pair — the top board
is the experiment, the bottom board the machinery that ran it.
