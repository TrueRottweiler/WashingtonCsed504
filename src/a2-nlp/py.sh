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
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYEXE="${UW_CSED504_PY:-C:/Users/truer/.conda/envs/uw-csed504/python.exe}"

if [ ! -x "$PYEXE" ]; then
    echo "python not found at: $PYEXE" >&2
    echo "set UW_CSED504_PY to override, or check the env still exists." >&2
    exit 1
fi

export PYTHONIOENCODING=utf-8
cd "$HERE"
exec "$PYEXE" -W ignore "$@"
