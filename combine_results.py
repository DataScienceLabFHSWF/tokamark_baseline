"""Combine TokaMark task metric files into comparison tables."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def collect_task_metrics(results_dir: Path, seed: int, task: str) -> list[dict[str, str]]:
    """Read task metrics for every model using the requested seed and task."""
    rows: list[dict[str, str]] = []
    pattern = f"*/seed_{seed}/{task}/task_metrics.csv"

    for metrics_path in sorted(results_dir.glob(pattern)):
        model = metrics_path.parents[2].name
        with metrics_path.open(newline="") as metrics_file:
            for row in csv.DictReader(metrics_file):
                rows.append(
                    {
                        "model": model,
                        "seed": str(seed),
                        "task": task,
                        **row,
                    }
                )

    if not rows:
        raise FileNotFoundError(
            f"No task_metrics.csv files found under {results_dir / pattern}"
        )

    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Write rows using the union of their field names."""
    fieldnames = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_comparison(rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    """Select and rank the aggregate task row for each model."""
    comparison = [row for row in rows if row.get("feature_name") == task]
    if not comparison:
        raise ValueError(f"No aggregate row with feature_name={task!r} was found")

    comparison.sort(
        key=lambda row: (float(row["NRMSE_mean"]), float(row["NMAE_mean"]))
    )
    for rank, row in enumerate(comparison, start=1):
        row["NRMSE_rank"] = str(rank)

    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine task_metrics.csv files into comparison CSVs."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results/random"))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--task", default="task_3-1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <results-dir>/comparison/seed_<seed>/<task>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.results_dir / "comparison" / f"seed_{args.seed}" / args.task
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_task_metrics(args.results_dir, args.seed, args.task)
    write_csv(output_dir / "combined_task_metrics.csv", rows)
    write_csv(output_dir / "model_comparison.csv", build_comparison(rows, args.task))

    model_count = len({row["model"] for row in rows})
    print(f"Combined {len(rows)} metric rows from {model_count} models")
    print(f"Wrote {output_dir / 'combined_task_metrics.csv'}")
    print(f"Wrote {output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()