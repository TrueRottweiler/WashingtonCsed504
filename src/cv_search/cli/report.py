"""Regenerate CSV, Pareto, plots, and Markdown reports from results.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config
from ..plotting import generate_plots
from ..reporting import write_reports
from ..storage import StudyStorage


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate reports for a saved study")
    parser.add_argument("--study", required=True, help="Study result directory")
    parser.add_argument("--config", required=True, help="Original/resolved TOML configuration")
    args = parser.parse_args()
    root = Path(args.study)
    storage = StudyStorage(root)
    config = load_config(args.config)
    results = storage.load()
    storage.write_csv(storage.paths.leaderboard_csv, results)
    generate_plots(
        results,
        png_dir=storage.paths.plots_png,
        svg_dir=storage.paths.plots_svg,
        html_dir=storage.paths.plots_html,
    )
    hardware_path = root / "hardware.json"
    hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else {}
    estimate_path = root / "runtime_estimate.json"
    estimate = json.loads(estimate_path.read_text()) if estimate_path.exists() else None
    write_reports(
        root,
        results,
        config.objectives,
        str(config.selection.get("policy", "weighted_pareto")),
        hardware,
        estimate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
