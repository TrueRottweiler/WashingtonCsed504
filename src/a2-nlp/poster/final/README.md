# poster/final — the two boards as submitted

The project's poster is a pair of 24 × 36 in boards on the UW research-poster template, hung
together: the **top board is the experiment** — from-scratch Yorùbá pretraining against
multilingual transfer — and the **bottom board is the machinery** that ran it, framed as the
course it would take to teach. This directory holds both, each as the editable PPTX and the
printed PDF.

The two are made in opposite ways, so each file's provenance is stated here:

| file | provenance |
|---|---|
| `NLP_Project_Poster.pptx` / `.pdf` — **the top board** | **Hand-built in PowerPoint** by Leon Wan from the numbers in [`reports/13-the-top-board.md`](../../reports/13-the-top-board.md). It was **not** produced by `poster/build_posters.py` and is not reproducible from the repository — which is exactly why the editable PPTX is committed alongside the PDF. Its numbers were independently checked against the committed `runs/` records (about twenty of them, none wrong), and the generated `board-yoruba-findings` from the same build sheet serves as the rebuildable cross-check. |
| `board-model-factory_v2.pptx` / `.pdf` — **the bottom board** | **Generated** by [`poster/build_posters.py`](../build_posters.py) from the build sheet [`reports/12-the-bottom-board.md`](../../reports/12-the-bottom-board.md). Every number on it regenerates from committed `runs/` records, and the build fails if any text overflows its box or a claim disagrees with its record. Rebuild with `bash src/a2-nlp/py.sh poster/build_posters.py` (needs PowerPoint + pywin32); the committed copy is the printed version, byte-reproducible from the commit that carries it. |

Why this directory is tracked while [`poster/outputs/`](../outputs) is ignored: a generated
board rebuilds from the repository in one command, so committing every build would be noise —
but the **printed artifact is a deliverable** a reader should not need PowerPoint and a Windows
build machine to see, and a hand-built board cannot be rebuilt from anything in the repository
at all. What prints lives in git; what merely built does not.
