"""Run the controlled EDMD Task 3-1 ridge/backend sweep.

Run from this directory, for example:

    uv run --project ../.. python sweep_controlled_edmd.py \
        --split random --structures affine bilinear --backends torch qant \
        --data-root /mnt/data2/datasets/tokamark/data

Each condition receives its own checkpoint and metrics directory through
``--run-id``. EDMD is deterministic for a fixed split and ridge value; seeds
are retained for benchmark split parity and resampling experiments.
"""

from __future__ import annotations

import argparse
import subprocess
import sys


MODEL = "plume_controlled_edmd"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="task_3-1")
    parser.add_argument("--config", default="/src/config/config_model_test.yaml")
    parser.add_argument("--split", choices=["random", "temporal"], default="random")
    parser.add_argument(
        "--ridges",
        nargs="+",
        type=float,
        default=[1e-8, 1e-6, 1e-4, 1e-2],
    )
    parser.add_argument("--backends", nargs="+", choices=["torch", "qant"], default=["torch"])
    parser.add_argument("--structures", nargs="+", choices=["affine", "bilinear"], default=["affine"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[23])
    parser.add_argument("--data-root", type=str, required=True)
    args = parser.parse_args()

    for seed in args.seeds:
        for ridge in args.ridges:
            ridge_label = f"{ridge:.0e}".replace("+", "")
            for structure in args.structures:
                for backend in args.backends:
                    run_id = f"seed_{seed}__{structure}__ridge_{ridge_label}__{backend}"
                    common = [
                        sys.executable,
                        "run_training.py",
                        "--task",
                        args.task,
                        "--config",
                        args.config,
                        "--split",
                        args.split,
                        "--seed",
                        str(seed),
                        "--model",
                        MODEL,
                        "--edmd-ridge",
                        str(ridge),
                        "--edmd-backend",
                        backend,
                        "--edmd-structure",
                        structure,
                        "--run-id",
                        run_id,
                        "--data-root",
                        args.data_root,
                    ]
                    print(f"[fit] {run_id}", flush=True)
                    subprocess.run(common, check=True)

                    evaluate = common.copy()
                    evaluate[1] = "run_evaluation.py"
                    print(f"[eval] {run_id}", flush=True)
                    subprocess.run(evaluate, check=True)


if __name__ == "__main__":
    main()
