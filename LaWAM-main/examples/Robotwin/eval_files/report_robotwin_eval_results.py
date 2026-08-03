#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract task success rates from a Robotwin eval run directory."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Robotwin eval run dir, e.g. results/eval_runs/robotwin/<group>/<run_tag>",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <run_dir>/task_success_rates.json.",
    )
    return parser.parse_args()


def collect_success_rates(run_dir: Path) -> dict[str, float]:
    tasks_dir = run_dir / "tasks"
    if not tasks_dir.is_dir():
        raise FileNotFoundError(f"tasks directory not found: {tasks_dir}")

    success_rates: dict[str, float] = {}
    for task_dir in sorted(path for path in tasks_dir.iterdir() if path.is_dir()):
        result_path = task_dir / "_result.txt"
        if not result_path.is_file():
            continue

        text = result_path.read_text(encoding="utf-8").strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"empty result file: {result_path}")

        success_rates[task_dir.name] = float(lines[-1])

    return success_rates


def compute_average_success_rate(success_rates: dict[str, float]) -> float:
    if not success_rates:
        return 0.0
    return sum(success_rates.values()) / len(success_rates)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_path = args.output if args.output is not None else run_dir / "task_success_rates.json"

    success_rates = collect_success_rates(run_dir)
    report = {
        "task_success_rates": success_rates,
        "average_success_rate": compute_average_success_rate(success_rates),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
