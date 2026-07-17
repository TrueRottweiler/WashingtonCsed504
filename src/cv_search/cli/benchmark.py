"""Pilot benchmark command; equivalent to search --estimate-only."""

from __future__ import annotations

import sys

from .common import run


def main() -> int:
    arguments = sys.argv[1:]
    if "--estimate-only" not in arguments:
        arguments.append("--estimate-only")
    return run(argv=arguments)


if __name__ == "__main__":
    raise SystemExit(main())
