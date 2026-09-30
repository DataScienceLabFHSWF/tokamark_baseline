#!/usr/bin/env python3
"""
Extract and summarize evaluation metrics from the sweep results.
Generates tables suitable for the paper.
"""

import json
import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_BASE = Path("results")
SWEEP_DIR = RESULTS_BASE / "sweep_multiseed_20260928_120944"
RANDOM_DIR = RESULTS_BASE / "random"

MODELS = [
    "plume_direct_mse",
    "plume_vanilla_mse", 
    "plume_direct_jepa",
    "plume_koopman_mse",
    "plume_koopman_jepa",
    "plume_vanilla_jepa",
    "cnn",
    "lstm",
]

SEEDS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]


def _condition_metadata(path: Path) -> Dict:
    """Parse model condition metadata from a task-metrics path."""
    parts = path.parts
    seed = None
    for part in parts:
        match = re.fullmatch(r"seed_(\d+)", part)
        if match:
            seed = int(match.group(1))
            break

    condition = next(
        (part for part in parts if part.startswith("plume_controlled_edmd__")),
        None,
    )
    if condition is not None:
        metadata = {
            "Model": "controlled_edmd",
            "structure": "unknown",
            "observable": "physical",
            "delay_steps": 1,
            "ridge": "unknown",
            "backend": "unknown",
            "seed": seed,
            "status": "preliminary",
        }
        has_observable_label = False
        has_delay_label = False
        for field in condition.split("__")[1:]:
            if field in {"affine", "bilinear"}:
                metadata["structure"] = field
            elif field in {"physical", "pod_quadratic"}:
                metadata["observable"] = field
                has_observable_label = True
            elif field.startswith("delay_"):
                metadata["delay_steps"] = int(field.removeprefix("delay_"))
                has_delay_label = True
            elif field.startswith("ridge_"):
                metadata["ridge"] = field.removeprefix("ridge_")
            elif field in {"torch", "qant", "auto"}:
                metadata["backend"] = field
        if has_observable_label and has_delay_label:
            metadata["status"] = "corrected"
        return metadata

    if seed is not None:
        seed_index = parts.index(f"seed_{seed}")
        return {"Model": parts[seed_index - 1], "seed": seed}
    return {"Model": path.parent.name, "seed": seed}


def collect_task_metrics(results_base: Path) -> pd.DataFrame:
    """Collect task-level metrics from legacy and EDMD result layouts."""
    records = []
    for path in results_base.rglob("task_metrics.csv"):
        metrics = pd.read_csv(path)
        rows = metrics[metrics["feature_name"].isin(["task-3-1", "task_3-1"])]
        if rows.empty:
            continue
        row = rows.iloc[0]
        record = _condition_metadata(path)
        record.update({
            "NRMSE_mean": row.get("NRMSE_mean", np.nan),
            "NRMSE_std": row.get("NRMSE_std_pop", np.nan),
            "NMAE_mean": row.get("NMAE_mean", np.nan),
            "NMAE_std": row.get("NMAE_std_pop", np.nan),
            "n_shots": row.get("n_shots", 0),
            "source_path": str(path),
        })
        records.append(record)
    return pd.DataFrame(records)


def load_task_metrics(model: str, seed: int, source: str = "sweep") -> Dict:
    """Load task_metrics.csv for a model and seed."""
    if source == "sweep":
        path = SWEEP_DIR / f"{model}_seed{seed}_eval.log"
    else:
        path = RANDOM_DIR / model / f"seed_{seed}" / "task_3-1" / "task_metrics.csv"
    
    # Try to find CSV file first (actual metrics)
    csv_candidates = list(RANDOM_DIR.glob(f"{model}/seed_{seed}/**/task_metrics.csv"))
    if csv_candidates:
        df = pd.read_csv(csv_candidates[0])
        # Get task_3-1 row
        task_row = df[df["feature_name"] == "task_3-1"]
        if not task_row.empty:
            return task_row.iloc[0].to_dict()
    
    return None


def extract_metrics_from_sweep() -> Dict[str, Dict[str, Dict]]:
    """
    Extract NRMSE and NMAE metrics from all sweep results.
    Returns: {model: {seed: {metric: value}}}
    """
    results = {}
    
    for model in MODELS:
        results[model] = {}
        for seed in SEEDS:
            metrics = load_task_metrics(model, seed, source="sweep")
            if metrics:
                results[model][seed] = {
                    "NRMSE_mean": metrics.get("NRMSE_mean", np.nan),
                    "NRMSE_std": metrics.get("NRMSE_std_pop", np.nan),
                    "NMAE_mean": metrics.get("NMAE_mean", np.nan),
                    "NMAE_std": metrics.get("NMAE_std_pop", np.nan),
                    "n_shots": metrics.get("n_shots", 0),
                }
            else:
                results[model][seed] = None
    
    return results


def compute_summary_stats(results: Dict) -> pd.DataFrame:
    """Compute mean and std across all seeds for each model."""
    summary = []
    
    for model in MODELS:
        nrmse_vals = []
        nmae_vals = []
        
        for seed in SEEDS:
            if results[model][seed] is not None:
                nrmse_vals.append(results[model][seed]["NRMSE_mean"])
                nmae_vals.append(results[model][seed]["NMAE_mean"])
        
        if nrmse_vals:
            summary.append({
                "Model": model,
                "NRMSE (mean ± std)": f"{np.mean(nrmse_vals):.4f} ± {np.std(nrmse_vals):.4f}",
                "NMAE (mean ± std)": f"{np.mean(nmae_vals):.4f} ± {np.std(nmae_vals):.4f}",
                "n_seeds": len(nrmse_vals),
            })
    
    return pd.DataFrame(summary)


def load_bfloat16_errors() -> Dict:
    """Load bfloat16 quantization error measurements."""
    bfloat_file = RESULTS_BASE / "bfloat16_error_smoke.json"
    if bfloat_file.exists():
        with open(bfloat_file) as f:
            data = json.load(f)
        return {entry["model"]: entry for entry in data}
    return {}


def generate_latex_table(summary_df: pd.DataFrame) -> str:
    """Generate LaTeX table from summary statistics."""
    latex = "\\begin{table}[]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lcc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{NRMSE (mean ± std)} & \\textbf{NMAE (mean ± std)} \\\\\n"
    latex += "\\midrule\n"
    
    for _, row in summary_df.iterrows():
        model_name = row["Model"].replace("_", "\\_")
        latex += f"{model_name} & {row['NRMSE (mean ± std)']} & {row['NMAE (mean ± std)']} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Task 3-1 evaluation metrics across 10 seeds (23--32).}\n"
    latex += "\\label{tab:results}\n"
    latex += "\\end{table}\n"
    
    return latex


def generate_bfloat16_table(bfloat_errors: Dict) -> str:
    """Generate LaTeX table for bfloat16 quantization errors."""
    latex = "\\begin{table}[]\n"
    latex += "\\centering\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "\\textbf{Model} & \\textbf{NRMSE (PyTorch)} & \\textbf{NRMSE (bfloat16)} & \\textbf{Rel. $\\Delta$ (\\\%)} \\\\\n"
    latex += "\\midrule\n"
    
    for model, data in bfloat_errors.items():
        torch_nrmse = data["combined_nrmse_torch"]
        qant_nrmse = data["combined_nrmse_qant"]
        rel_delta = data["combined_nrmse_rel_delta"] * 100
        
        model_name = model.replace("_", "\\_")
        latex += f"{model_name} & {torch_nrmse:.4f} & {qant_nrmse:.4f} & {rel_delta:+.2f}\\% \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\caption{Q.ANT SDK bfloat16 quantization error on CPU-emulated hardware. Relative difference is (Q.ANT - PyTorch) / PyTorch.}\n"
    latex += "\\label{tab:bfloat16}\n"
    latex += "\\end{table}\n"
    
    return latex


def generate_edmd_table(records: pd.DataFrame) -> str:
    """Generate a LaTeX table aggregated by EDMD structure and backend."""
    edmd = records[
        (records["Model"] == "controlled_edmd")
        & (records.get("status", "preliminary") == "corrected")
    ].copy()
    if edmd.empty:
        return "% No controlled EDMD task metrics found.\n"
    summary = (
        edmd.groupby(
            ["observable", "delay_steps", "structure", "ridge", "backend"],
            dropna=False,
        )
        .agg(
            nrmse=("NRMSE_mean", "mean"),
            nrmse_seed_std=("NRMSE_mean", "std"),
            nmae=("NMAE_mean", "mean"),
            nmae_seed_std=("NMAE_mean", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
        .fillna(0.0)
    )
    lines = [
        "\\begin{table}[]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lllllccc}",
        "\\toprule",
        "Observable & Delay & Structure & Ridge & Backend & NRMSE & NMAE & Seeds \\\\",
        "\\midrule",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"{row['observable']} & {int(row['delay_steps'])} & {row['structure']} & "
            f"{row['ridge']} & {row['backend']} & "
            f"{row['nrmse']:.4f} $\\pm$ {row['nrmse_seed_std']:.4f} & "
            f"{row['nmae']:.4f} $\\pm$ {row['nmae_seed_std']:.4f} & "
            f"{int(row['seeds'])} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Controlled EDMD Task 3-1 metrics aggregated across available seeds.}",
        "\\label{tab:controlled_edmd}",
        "\\end{table}",
    ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-results-base", type=Path, default=Path(
        "/mnt/data2/plume_shared/results/paper1/tokamark/results"
    ))
    args = parser.parse_args()
    print("Extracting metrics from sweep results...\n")
    
    # Load results
    results = extract_metrics_from_sweep()
    summary = compute_summary_stats(results)
    bfloat_errors = load_bfloat16_errors()
    
    print("=" * 80)
    print("SUMMARY STATISTICS (across all seeds)")
    print("=" * 80)
    print(summary.to_string(index=False))
    print()
    
    print("=" * 80)
    print("BFLOAT16 QUANTIZATION ERRORS")
    print("=" * 80)
    if bfloat_errors:
        for model, data in bfloat_errors.items():
            print(f"\n{model}:")
            print(f"  PyTorch NRMSE: {data['combined_nrmse_torch']:.4f}")
            print(f"  bfloat16 NRMSE: {data['combined_nrmse_qant']:.4f}")
            print(f"  Rel. Δ: {data['combined_nrmse_rel_delta']*100:+.2f}%")
    else:
        print("  No bfloat16 error data found")
    
    print()
    print("=" * 80)
    print("LATEX TABLES FOR PAPER")
    print("=" * 80)
    print(generate_latex_table(summary))
    print()
    print(generate_bfloat16_table(bfloat_errors))

    central_records = collect_task_metrics(args.shared_results_base)
    print()
    print("=" * 80)
    print("CONTROLLED EDMD RESULTS FROM SHARED DIRECTORY")
    print("=" * 80)
    if central_records.empty:
        print("No controlled EDMD task metrics found.")
    else:
        print(central_records.to_string(index=False))
        preliminary_count = int(
            ((central_records["Model"] == "controlled_edmd")
             & (central_records["status"] != "corrected")).sum()
        )
        print(f"\nExcluded preliminary/legacy EDMD records from LaTeX table: {preliminary_count}")
        print()
        print(generate_edmd_table(central_records))
