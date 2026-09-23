"""Central model construction for baseline and PLUME Task 3-1 models."""

from __future__ import annotations

from src.multi_conv_lstm_model import create_lstm_architecture
from src.multi_conv_mlp_model import create_cnn_architecture
from src.plume_tokamark_adapter import TokaMarkPLUMEAdapter

PLUME_MODEL_CHOICES = (
    "plume_vanilla_mse",
    "plume_vanilla_jepa",
    "plume_koopman_mse",
    "plume_koopman_jepa",
    "plume_direct_mse",
    "plume_direct_jepa",
)
MODEL_CHOICES = ("cnn", "lstm", *PLUME_MODEL_CHOICES)


def create_model(
    model_name,
    dataloader,
    metadata,
    config,
    task_name,
    verbose=False,
):
    if model_name == "cnn":
        return create_cnn_architecture(
            dataloader_=dataloader,
            dict_metadata=metadata,
            verbose=verbose,
        )
    if model_name == "lstm":
        return create_lstm_architecture(
            dataloader_=dataloader,
            dict_metadata=metadata,
            verbose=verbose,
        )
    if model_name not in PLUME_MODEL_CHOICES:
        raise ValueError(f"Unknown model: {model_name}")
    if task_name != "task_3-1":
        raise ValueError(
            f"{model_name} is a profile model for task_3-1, not {task_name}"
        )

    dynamics, objective = model_name.removeprefix("plume_").rsplit("_", 1)
    plume_config = config.get("plume", {}).get("model", {})
    return TokaMarkPLUMEAdapter(
        dynamics=dynamics,
        objective=objective,
        profile_bins=plume_config.get("profile_bins", 120),
        action_dim=plume_config.get("action_dim", 4),
        latent_dim=plume_config.get("latent_dim", 128),
        horizon=plume_config.get("horizon", 10),
        qant_backend=plume_config.get("qant_backend", "torch"),
    )


def get_loss_weights(model_name, config):
    """Make the ``*_mse`` versus ``*_jepa`` distinction explicit."""

    if not model_name.startswith("plume_") or model_name.endswith("_mse"):
        return {
            "profile": 1.0,
            "latent": 0.0,
            "reconstruction": 0.0,
        }

    loss_config = config.get("plume", {}).get("loss", {})
    return {
        "profile": float(loss_config.get("profile_weight", 1.0)),
        "latent": float(loss_config.get("latent_weight", 1.0)),
        "reconstruction": float(loss_config.get("reconstruction_weight", 0.3)),
    }
