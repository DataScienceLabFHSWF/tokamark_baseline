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
from tokamark.tasks import get_signals_metadata, get_task_config, get_task_metadata
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
    def __init__(self, output_scale: float, metric_scale: float) -> None:
        self.output_scale = output_scale
        self.metric_scale = metric_scale
        self.windows_by_shot: dict[int, list[tuple[float, float]]] = {}

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        shot_ids: torch.Tensor,
    ) -> None:
        prediction = prediction.reshape(prediction.shape[0], -1).double()
        target = target.reshape(target.shape[0], -1).double()
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shapes differ after flattening: "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}"
            )

        for index, shot_id in enumerate(shot_ids.tolist()):
            valid = torch.isfinite(target[index])
            if not valid.any():
                continue
            if not torch.isfinite(prediction[index, valid]).all():
                raise ValueError(f"Non-finite prediction for valid target bins in shot {shot_id}")
            error = (prediction[index, valid] - target[index, valid]) * self.output_scale
            rmse = torch.sqrt(error.square().mean()).item()
            mae = error.abs().mean().item()
            self.windows_by_shot.setdefault(int(shot_id), []).append((rmse**2, mae))

    def metrics(self) -> dict[str, float | int]:
        if not self.windows_by_shot:
            return {"n_shots": 0, "nrmse": float("nan"), "nmae": float("nan")}

        shot_nrmse = []
        shot_nmae = []
        for windows in self.windows_by_shot.values():
            shot_rmse = float(np.sqrt(np.mean([rmse_squared for rmse_squared, _ in windows])))
            shot_mae = float(np.mean([mae for _, mae in windows]))
            shot_nrmse.append(shot_rmse / self.metric_scale)
            shot_nmae.append(shot_mae / self.metric_scale)
        return {
            "n_shots": len(shot_nrmse),
            "nrmse": float(np.mean(shot_nrmse)),
            "nmae": float(np.mean(shot_nmae)),
        }


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


def _evaluate_backends(
    model,
    dataloader,
    metadata: dict,
    max_batches: int,
) -> tuple[dict[str, _ChannelAccumulator], dict[str, _ChannelAccumulator]]:
    signal_metadata = get_signals_metadata()
    output_names = metadata["sources_and_signals"]["output_name"]
    feature_names = [f"{source}-{signal}" for source, signal in output_names]
    torch_accumulators = {
        name: _ChannelAccumulator(metadata["output"][name]["std"], signal_metadata[name]["std"])
        for name in feature_names
    }
    qant_accumulators = {
        name: _ChannelAccumulator(metadata["output"][name]["std"], signal_metadata[name]["std"])
        for name in feature_names
    }

    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            if batch is None:
                continue
            shot_ids, _, inputs, targets = batch
            inputs = [x.float().cpu() for x in inputs]
            targets = [target.float().cpu() for target in targets]

            set_qant_backend(model, "torch")
            torch_predictions = unwrap_predictions(model(*inputs))
            set_qant_backend(model, "qant")
            qant_predictions = unwrap_predictions(model(*inputs))

            for index, feature_name in enumerate(feature_names):
                torch_accumulators[feature_name].update(
                    torch_predictions[index], targets[index], shot_ids
                )
                qant_accumulators[feature_name].update(
                    qant_predictions[index], targets[index], shot_ids
                )
    return torch_accumulators, qant_accumulators


def _combine_channel_metrics(accumulators: dict[str, _ChannelAccumulator]) -> dict[str, float | int]:
    shot_nrmse: dict[int, list[float]] = {}
    shot_nmae: dict[int, list[float]] = {}
    for accumulator in accumulators.values():
        for shot_id, windows in accumulator.windows_by_shot.items():
            rmse = float(np.sqrt(np.mean([rmse_squared for rmse_squared, _ in windows])))
            mae = float(np.mean([mae for _, mae in windows]))
            shot_nrmse.setdefault(shot_id, []).append(rmse / accumulator.metric_scale)
            shot_nmae.setdefault(shot_id, []).append(mae / accumulator.metric_scale)
    task_nrmse = [float(np.mean(values)) for values in shot_nrmse.values()]
    task_nmae = [float(np.mean(values)) for values in shot_nmae.values()]
    return {
        "n_shots": len(task_nrmse),
        "combined_nrmse": float(np.mean(task_nrmse)) if task_nrmse else float("nan"),
        "combined_nmae": float(np.mean(task_nmae)) if task_nmae else float("nan"),
    }


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

    torch_accumulators, qant_accumulators = _evaluate_backends(
        model, dataloader, metadata, max_batches
    )
    torch_metrics = _combine_channel_metrics(torch_accumulators)
    qant_metrics = _combine_channel_metrics(qant_accumulators)

    def _rel_delta(a: float, b: float) -> float:
        return (b - a) / a if a else float("nan")

    return {
        "model": model_name,
        "seed": seed,
        "n_shots": torch_metrics["n_shots"],
        "nrmse_torch_by_signal": {name: metric.metrics()["nrmse"] for name, metric in torch_accumulators.items()},
        "nrmse_qant_by_signal": {name: metric.metrics()["nrmse"] for name, metric in qant_accumulators.items()},
        "nrmse_rel_delta_by_signal": {
            name: _rel_delta(
                float(torch_accumulators[name].metrics()["nrmse"]),
                float(qant_accumulators[name].metrics()["nrmse"]),
            )
            for name in torch_accumulators
        },
        "combined_nrmse_torch": torch_metrics["combined_nrmse"],
        "combined_nrmse_qant": qant_metrics["combined_nrmse"],
        "combined_nrmse_rel_delta": _rel_delta(
            float(torch_metrics["combined_nrmse"]), float(qant_metrics["combined_nrmse"])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task_3-1")
    parser.add_argument("--config", default="/src/config/config_model.yaml")
    parser.add_argument("--split", default="random", choices=["random", "temporal"])
    parser.add_argument("--batch-size", type=int, default=8, help="Small CPU batch for the Q.ANT SDK backend")
    parser.add_argument("--max-batches", type=int, default=8, help="Test batches per backend (qant is ~6.5x slower)")
    parser.add_argument(
        "--models", nargs="*", default=list(QANT_CAPABLE_MODELS), choices=list(QANT_CAPABLE_MODELS)
    )
    parser.add_argument("--seeds", nargs="*", type=int, help="Optional subset of checkpoint seeds")
    parser.add_argument("--output", default="results/bfloat16_error_report.json")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_batches < 1:
        parser.error("--batch-size and --max-batches must both be positive")

    if not qant_available():
        raise RuntimeError("Native Q.ANT SDK not available in this environment; cannot quantify bfloat16 error.")

    with open(REPO_ROOT + args.config, "r") as f:
        config = yaml.safe_load(f)
    config["_split"] = args.split
    config_task = get_task_config(task_name=args.task)

    dataloader, metadata = _build_test_dataloader(
        args.task, args.split, config, config_task, batch_size=args.batch_size
    )
    metadata = metadata | config_task

    results = []
    for model_name in args.models:
        seeds = _discover_seeds(config, args.split, model_name, args.task)
        if args.seeds is not None:
            seeds = [seed for seed in seeds if seed in args.seeds]
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
            print(
                f"  combined NRMSE torch={result['combined_nrmse_torch']:.4f} "
                f"qant={result['combined_nrmse_qant']:.4f} "
                f"(delta={result['combined_nrmse_rel_delta']:+.2%})"
            )
            results.append(result)

    output_path = Path(REPO_ROOT) / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {output_path}")

    if results:
        deltas = [r["combined_nrmse_rel_delta"] for r in results if not np.isnan(r["combined_nrmse_rel_delta"])]
        combined_torch = [r["combined_nrmse_torch"] for r in results]
        print(f"\nAcross {len(results)} (model, seed) pairs:")
        print(f"  Combined NRMSE relative delta: mean={np.mean(deltas):+.2%} std={np.std(deltas):.2%}")
        if len(combined_torch) > 1:
            seed_spread = (max(combined_torch) - min(combined_torch)) / np.mean(combined_torch)
            print(f"  Seed-to-seed NRMSE spread (torch backend only): {seed_spread:.2%} "
                  "(compare against the bfloat16 deltas above)")


if __name__ == "__main__":
    main()
