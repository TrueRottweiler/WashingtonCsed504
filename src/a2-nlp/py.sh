#!/usr/bin/env bash
# Run this folder's Python in the uw-csed504 conda env.
#
# Every a2-nlp script wants the same three things: the conda env's interpreter (not the system
# Python 3.13, which has no torch), this folder as the working directory (the modules import
# each other by bare name, and data/ runs/ logs/ are all relative), and UTF-8 stdout (the
# default cp1252 console encoding raises UnicodeEncodeError on the arrows and box characters
# the reports print). Wrapping them here means one stable command prefix instead of a different
# ad-hoc incantation each time.
#
# Usage, from anywhere in the repo:
#   bash src/a2-nlp/py.sh mlm_run.py --corpus yor --smoke
#   bash src/a2-nlp/py.sh -m unittest test_store_dtype -v
#   bash src/a2-nlp/py.sh store_bench.py --dataset wikitext103 --model gpt
#
# WHICH INTERPRETER, in order:
#   1. $UW_CSED504_PY, if set. Honoured strictly -- naming an interpreter that is not there is an
#      error, not something to paper over.
#   2. the workstation's conda env, which is where the long jobs run
#   3. a repo-local .venv/, if someone has made one
#   4. python3 or python on PATH
#
# Steps 3 and 4 were added on 12 August. The conda path below is one person's home directory, and
# it was the ONLY candidate -- so anyone else's first command out of the README died on
# "python not found at: C:/Users/truer/...", with the override documented nowhere but this file.
# That is exactly what the standing "clone fresh and run QUICKSTART cold" request exists to find,
# and it was found by accident instead. A fallback interpreter may well lack torch; that failure
# names the missing package, which is a better error than a path nobody else can have.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

if [ -n "${UW_CSED504_PY:-}" ]; then
    CANDIDATES=("$UW_CSED504_PY")
else
    CANDIDATES=("C:/Users/truer/.conda/envs/uw-csed504/python.exe"
                "$REPO/.venv/Scripts/python.exe"
                "$REPO/.venv/bin/python")
fi

PYEXE=""
for c in "${CANDIDATES[@]}"; do
    if [ -x "$c" ]; then PYEXE="$c"; break; fi
done

if [ -z "$PYEXE" ] && [ -z "${UW_CSED504_PY:-}" ]; then
    for c in python3 python; do
        if command -v "$c" >/dev/null 2>&1; then PYEXE="$(command -v "$c")"; break; fi
    done
fi

if [ -z "$PYEXE" ]; then
    echo "no usable python found. Tried:" >&2
    for c in "${CANDIDATES[@]}"; do echo "  $c" >&2; done
    [ -z "${UW_CSED504_PY:-}" ] && echo "  python3, python (on PATH)" >&2
    echo "Set UW_CSED504_PY=/path/to/python to name one explicitly." >&2
    exit 1
fi

# Say so when this is NOT the first candidate, so a run on a fallback interpreter is never a
# surprise -- and stay silent on the workstation, where the first candidate always wins.
if [ "$PYEXE" != "${CANDIDATES[0]}" ]; then
    echo "py.sh: using $PYEXE" >&2
fi

export PYTHONIOENCODING=utf-8
cd "$HERE"
exec "$PYEXE" -W ignore "$@"
