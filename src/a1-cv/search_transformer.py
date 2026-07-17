"""Thin Transformer launcher over the shared cv_search engine."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_search.cli.common import run


if __name__ == "__main__":
    raise SystemExit(run("transformer"))
