"""Generic installed search command."""

from __future__ import annotations

from .common import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
