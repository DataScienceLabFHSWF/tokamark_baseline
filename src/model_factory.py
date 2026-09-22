"""Single model factory shared by training and evaluation."""

from __future__ import annotations

from math import prod
from typing import Any

import torch

from src.jepa_models import TokaMarkDirectWindowJEPA, TokaMarkJEPARollout
from src.multi_conv_lstm_model import create_lstm_architecture
from src.multi_conv_mlp_model import create_cnn_architecture


MODEL_CHOICES = (
    "cnn",
    "lstm",
    "jepa_rollout",
    "jepa_direct_mse",
    "jepa_direct",
)


def _first_valid_window(dataloader) -> dict[str, Any]:
    for window in dataloader.dataset:
        if window is not None:
            return window
    raise RuntimeError("Could not find a valid dataset window for shape inference")


def _create_jepa(
    model_name: str,
    dataloader,
    config: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    window = _first_valid_window(dataloader)
    input_tensors = window["input"]
    future_tensors = window["exogenous"]
    targets = window["y"]

    if len(targets) != 2:
        raise ValueError("Task 3-1 JEPA expects exactly Te and ne targets")
    if len(future_tensors) == 0:
        raise ValueError("Task 3-1 JEPA requires future actuator inputs")

    horizon = int(targets[0].shape[0])
    profile_bins = int(targets[0].shape[-1])
    future_start_index = len(input_tensors)

    # Each exogenous sample is [horizon, ...].  Concatenating its remaining
    # dimensions reproduces the complete actuator waveform for one model step.
    action_step_dim = sum(prod(tensor.shape[1:]) for tensor in future_tensors)

    jepa_config = config.get("jepa", {})
    common = {
        "profile_bins": profile_bins,
        "action_step_dim": action_step_dim,
        "latent_dim": int(jepa_config.get("latent_dim", 128)),
        "hidden_dim": int(jepa_config.get("hidden_dim", 256)),
        "future_start_index": future_start_index,
        "dropout": float(jepa_config.get("dropout", 0.0)),
        "horizon": horizon,
    }

    if model_name == "jepa_rollout":
        model = TokaMarkJEPARollout(**common)
    else:
        latent_loss_weight = (
            0.0
            if model_name == "jepa_direct_mse"
            else float(jepa_config.get("latent_loss_weight", 1.0))
        )
        model = TokaMarkDirectWindowJEPA(
            **common,
            latent_loss_weight=latent_loss_weight,
            target_ema=float(jepa_config.get("target_ema", 0.996)),
        )

    return model.to(device)


def create_model(
    model_name: str,
    dataloader,
    dict_metadata: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> torch.nn.Module:
    if model_name == "cnn":
        return create_cnn_architecture(
            dataloader_=dataloader,
            dict_metadata=dict_metadata,
            verbose=verbose,
        )
    if model_name == "lstm":
        return create_lstm_architecture(
            dataloader_=dataloader,
            dict_metadata=dict_metadata,
            verbose=verbose,
        )
    if model_name in {"jepa_rollout", "jepa_direct_mse", "jepa_direct"}:
        task_name = dict_metadata.get("task_name")
        if task_name != "task_3-1":
            raise ValueError(f"{model_name} currently supports task_3-1 only")
        return _create_jepa(model_name, dataloader, config, device)
    raise ValueError(f"Unknown model: {model_name}")

