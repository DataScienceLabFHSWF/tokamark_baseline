"""Quantify the bfloat16 (Q.ANT native backend) prediction-accuracy error
against the float32 PyTorch backend for every Q.ANT-capable model.

Unlike the ad-hoc single-model, single-subset comparison in
``experiments/paper1/eval.py`` (PLUME's original pipeline), this script:

- runs entirely inside the ``tokamark_baseline`` extension, so it covers the
  *actual* models used for the multi-seed sweep (6 PLUME variants + the new
  ``cnn_qant`` Q.ANT-primitive CNN baseline -- ``cnn``/``lstm`` are excluded,
  they have no Q.ANT backend at all, see docs/paper1/),
- runs across every seed for which a checkpoint is available, so the
  bfloat16-vs-float32 delta can be compared directly against the seed-to-seed
  (statistical) spread already collected by ``run_multiseed_sweep.sh``,
- always evaluates both backends on CPU, on the *same* batches, so the only
  source of difference is bfloat16 quantization, not device nondeterminism.

Usage:
    uv run python quantify_bfloat16_error.py --task task_3-1 \\
        --config /src/config/config_model.yaml --split random \\
        --max-batches 8 --output results/bfloat16_error_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

try:
    from globals import REPO_ROOT
except ImportError:
    from .globals import REPO_ROOT

from MAST_tools.utils.path_utils import (
    RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
)
from tokamark.data import initialize_MAST_dataset, initialize_TokaMark_dataset
from tokamark.data_split import get_train_test_val_shots
from tokamark.tasks import get_task_config, get_task_metadata
from tokamark.tools.path import (
    RANDOM_SPLIT_SIGNALS_STATS_FILE,
    RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
    TEMPORAL_SPLIT_SIGNALS_STATS_FILE,
    TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
)
from tokamark.tools.transforms.compose_transform import ComposeTransforms
from tokamark.tools.utils import get_device

from src.model_factory import PLUME_MODEL_CHOICES, create_model
from src.model_transform import ModelTransform_1, ModelTransform_2
from src.plume_qant_layers import qant_available, set_qant_backend
from src.plume_tokamark_adapter import unwrap_predictions
from src.trainer import model_collate_fn

# cnn/lstm have no Q.ANT primitives at all (see docs/paper1/README.md);
# cnn_qant is the Q.ANT-primitive CNN baseline added alongside this script.
QANT_CAPABLE_MODELS = (*PLUME_MODEL_CHOICES, "cnn_qant")


class _ChannelAccumulator:
    def __init__(self) -> None:
        self.squared_error = 0.0
        self.count = 0
        self.target_squared = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        error = (prediction - target).double()
        self.squared_error += error.square().sum().item()
        self.target_squared += target.double().square().sum().item()
        self.count += error.numel()

    def rmse(self) -> float:
        return float(np.sqrt(self.squared_error / self.count)) if self.count else float("nan")

    def target_std(self) -> float:
        return float(np.sqrt(self.target_squared / self.count)) if self.count else float("nan")

    def nrmse(self) -> float:
        std = self.target_std()
        return self.rmse() / std if std else float("nan")


def _build_test_dataloader(task: str, split: str, config: dict, config_task: dict, batch_size: int):
    if split == "random":
        data_split = RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        outlier_file = RANDOM_SPLIT_OUTLIER_METADATA_FILE
        signal_stats = RANDOM_SPLIT_SIGNALS_STATS_FILE
    elif split == "temporal":
        data_split = TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE
        outlier_file = TEMPORAL_SPLIT_OUTLIER_METADATA_FILE
        signal_stats = TEMPORAL_SPLIT_SIGNALS_STATS_FILE
    else:
        raise ValueError(f"Unknown split: {split}")

    _, test_shots, _ = get_train_test_val_shots(
        max_index=config["subset_of_shots"], shuffle=config["shuffle"], data_splits_file_path=data_split
    )

    test_mast_dataset = initialize_MAST_dataset(
        config_task=config_task,
        shots_list=test_shots,
        local_flag=config["local"],
        use_std_scaling=True,
        stats_metadata_file_path=signal_stats,
        use_nan_filling=False,
        remove_outliers=True,
        outlier_metadata_file=outlier_file,
        remove_bad_efit_rating=True,
        store_manager_settings=config["store_manager_settings"],
        verbose=False,
    )

    dict_task_metadata = get_task_metadata(config_task=config_task, verbose=False)
    model_specific_transform = ComposeTransforms(
        [ModelTransform_1(dict_task_metadata | config_task), ModelTransform_2(dict_task_metadata | config_task)]
    )
    test_dataset = initialize_TokaMark_dataset(
        dataset=test_mast_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=config_task,
        custom_transform=model_specific_transform,
        test_mode=True,
        shuffle_windows=False,
    )
    dataloader = DataLoader(
        dataset=test_dataset, collate_fn=model_collate_fn, batch_size=batch_size, num_workers=0
    )
    return dataloader, dict_task_metadata


def _checkpoint_path(config: dict, split: str, model_name: str, task_name: str, seed: int) -> Path:
    base = config["paths"]["data_output_directory"]
    return Path(REPO_ROOT + base + f"/{split}/{model_name}/{task_name}/seed_{seed}/best_model.pt")


def _discover_seeds(config: dict, split: str, model_name: str, task_name: str) -> list[int]:
    base = config["paths"]["data_output_directory"]
    model_dir = Path(REPO_ROOT + base + f"/{split}/{model_name}/{task_name}")
    if not model_dir.exists():
        return []
    seeds = []
    for child in model_dir.iterdir():
        if child.name.startswith("seed_") and (child / "best_model.pt").exists():
            seeds.append(int(child.name.removeprefix("seed_")))
    return sorted(seeds)


def _evaluate_backend(model, dataloader, backend: str, max_batches: int) -> tuple[_ChannelAccumulator, _ChannelAccumulator]:
    set_qant_backend(model, backend)
    model.eval()
    te_acc, ne_acc = _ChannelAccumulator(), _ChannelAccumulator()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            if batch is None:
                continue
            _, _, inputs, targets = batch
            inputs = [x.float().cpu() for x in inputs]
            predictions = unwrap_predictions(model(*inputs))
            te_pred, ne_pred = predictions[0], predictions[1]
            te_target, ne_target = targets[0].float().cpu(), targets[1].float().cpu()
            te_acc.update(te_pred, te_target)
            ne_acc.update(ne_pred, ne_target)
    return te_acc, ne_acc


def quantify_model_seed(
    model_name: str,
    seed: int,
    dataloader,
    metadata: dict,
    config: dict,
    task_name: str,
    max_batches: int,
) -> dict:
    model = create_model(
        model_name=model_name, dataloader=dataloader, metadata=metadata, config=config, task_name=task_name
    ).cpu()
    checkpoint = _checkpoint_path(config, config.get("_split", "random"), model_name, task_name, seed)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    te_torch, ne_torch = _evaluate_backend(model, dataloader, "torch", max_batches)
    te_qant, ne_qant = _evaluate_backend(model, dataloader, "qant", max_batches)

    def _rel_delta(a: float, b: float) -> float:
        return (b - a) / a if a else float("nan")

    return {
        "model": model_name,
        "seed": seed,
        "n_windows": te_torch.count,
        "te_nrmse_torch": te_torch.nrmse(),
        "te_nrmse_qant": te_qant.nrmse(),
        "te_nrmse_rel_delta": _rel_delta(te_torch.nrmse(), te_qant.nrmse()),
        "ne_nrmse_torch": ne_torch.nrmse(),
        "ne_nrmse_qant": ne_qant.nrmse(),
        "ne_nrmse_rel_delta": _rel_delta(ne_torch.nrmse(), ne_qant.nrmse()),
        "combined_nrmse_torch": (te_torch.nrmse() + ne_torch.nrmse()) / 2,
        "combined_nrmse_qant": (te_qant.nrmse() + ne_qant.nrmse()) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task_3-1")
    parser.add_argument("--config", default="/src/config/config_model.yaml")
    parser.add_argument("--split", default="random", choices=["random", "temporal"])
    parser.add_argument("--max-batches", type=int, default=8, help="Test batches per backend (qant is ~6.5x slower)")
    parser.add_argument(
        "--models", nargs="*", default=list(QANT_CAPABLE_MODELS), choices=list(QANT_CAPABLE_MODELS)
    )
    parser.add_argument("--output", default="results/bfloat16_error_report.json")
    args = parser.parse_args()

    if not qant_available():
        raise RuntimeError("Native Q.ANT SDK not available in this environment; cannot quantify bfloat16 error.")

    with open(REPO_ROOT + args.config, "r") as f:
        config = yaml.safe_load(f)
    config["_split"] = args.split
    config_task = get_task_config(task_name=args.task)

    dataloader, metadata = _build_test_dataloader(
        args.task, args.split, config, config_task, batch_size=config["dataloader_setting"]["batch_size"]
    )
    metadata = metadata | config_task

    results = []
    for model_name in args.models:
        seeds = _discover_seeds(config, args.split, model_name, args.task)
        if not seeds:
            print(f"[skip] no checkpoints found for {model_name}")
            continue
        for seed in seeds:
            print(f"[run] {model_name} seed={seed} ...", flush=True)
            try:
                result = quantify_model_seed(
                    model_name, seed, dataloader, metadata, config, args.task, args.max_batches
                )
            except Exception as exc:  # noqa: BLE001 - report and continue across models/seeds
                print(f"[error] {model_name} seed={seed}: {exc}")
                continue
            print(f"  combined NRMSE torch={result['combined_nrmse_torch']:.4f} "
                  f"qant={result['combined_nrmse_qant']:.4f} "
                  f"(Te delta={result['te_nrmse_rel_delta']:+.2%}, Ne delta={result['ne_nrmse_rel_delta']:+.2%})")
            results.append(result)

    output_path = Path(REPO_ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    if results:
        te_deltas = [r["te_nrmse_rel_delta"] for r in results if not np.isnan(r["te_nrmse_rel_delta"])]
        ne_deltas = [r["ne_nrmse_rel_delta"] for r in results if not np.isnan(r["ne_nrmse_rel_delta"])]
        combined_torch = [r["combined_nrmse_torch"] for r in results]
        print(f"\nAcross {len(results)} (model, seed) pairs:")
        print(f"  Te NRMSE relative delta:  mean={np.mean(te_deltas):+.2%}  std={np.std(te_deltas):.2%}")
        print(f"  Ne NRMSE relative delta:  mean={np.mean(ne_deltas):+.2%}  std={np.std(ne_deltas):.2%}")
        if len(combined_torch) > 1:
            seed_spread = (max(combined_torch) - min(combined_torch)) / np.mean(combined_torch)
            print(f"  Seed-to-seed NRMSE spread (torch backend only): {seed_spread:.2%} "
                  "(compare against the bfloat16 deltas above)")


if __name__ == "__main__":
    main()
