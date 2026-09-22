"""Combine official Task 3-1 outputs into one comparison CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODEL_METADATA = {
    "cnn": ("CNN", "direct window", "output MSE"),
    "lstm": ("CNN-LSTM", "direct window", "output MSE"),
    "jepa_rollout": ("JEPA rollout", "recursive latent rollout", "output MSE"),
    "jepa_direct_mse": ("Direct JEPA MSE", "direct window", "output MSE"),
    "jepa_direct": ("Direct JEPA", "direct window", "output + latent MSE"),
}


def read_metrics(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            row["feature_name"]: row
            for row in csv.DictReader(handle)
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="random")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    rows = []
    for model_key, (label, protocol, objective) in MODEL_METADATA.items():
        metrics_path = (
            args.results_root
            / args.split
            / model_key
            / f"seed_{args.seed}"
            / "task_3-1"
            / "task_metrics.csv"
        )
        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Missing {metrics_path}. Run training and evaluation for {model_key}."
            )

        metrics = read_metrics(metrics_path)
        task = metrics["task_3-1"]
        te = metrics["thomson_scattering-t_e"]
        ne = metrics["thomson_scattering-n_e"]

        rows.append(
            {
                "model": label,
                "model_key": model_key,
                "forecast_protocol": protocol,
                "objective": objective,
                "n_shots": task["n_shots"],
                "task_NRMSE_mean": task["NRMSE_mean"],
                "task_NRMSE_std_pop": task["NRMSE_std_pop"],
                "t_e_NRMSE_mean": te["NRMSE_mean"],
                "t_e_NRMSE_std_pop": te["NRMSE_std_pop"],
                "n_e_NRMSE_mean": ne["NRMSE_mean"],
                "n_e_NRMSE_std_pop": ne["NRMSE_std_pop"],
            }
        )

    output_path = (
        args.results_root
        / args.split
        / f"task_3-1_comparison_seed_{args.seed}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(output_path)


if __name__ == "__main__":
    main()

