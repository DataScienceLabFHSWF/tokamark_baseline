from __future__ import annotations

import copy

import pytest
import torch

from src.model_factory import PLUME_MODEL_CHOICES, create_model, get_loss_weights
from src.plume_qant_layers import qant_available
from src.plume_tokamark_adapter import unwrap_predictions
from src.trainer import ForecastLoss


def _task31_batch(batch_size=2, horizon=10, profile_bins=120):
    profiles = [
        torch.randn(batch_size, 1, 1, 1, profile_bins),
        torch.randn(batch_size, 1, 1, 1, profile_bins),
    ]
    past_actions = [torch.randn(batch_size, 1, 1, 20) for _ in range(4)]
    future_actions = [torch.randn(batch_size, horizon, 1, 20) for _ in range(4)]
    targets = [
        torch.randn(batch_size, horizon, profile_bins),
        torch.randn(batch_size, horizon, profile_bins),
    ]
    return profiles + past_actions + future_actions, targets


def _config(backend="torch"):
    return {
        "plume": {
            "model": {
                "profile_bins": 120,
                "action_dim": 4,
                "latent_dim": 8,
                "horizon": 10,
                "qant_backend": backend,
            },
            "loss": {
                "profile_weight": 1.0,
                "latent_weight": 0.5,
                "reconstruction_weight": 0.25,
            },
        }
    }


@pytest.mark.parametrize("model_name", PLUME_MODEL_CHOICES)
def test_all_plume_choices_match_baseline_output_contract(model_name):
    model = create_model(
        model_name=model_name,
        dataloader=None,
        metadata={},
        config=_config(),
        task_name="task_3-1",
    )
    inputs, targets = _task31_batch()
    output = model(*inputs, targets=targets)
    predictions = unwrap_predictions(output)

    assert len(predictions) == 2
    assert predictions[0].shape == (2, 10, 120)
    assert predictions[1].shape == (2, 10, 120)
    if model_name.endswith("_jepa"):
        assert set(output["aux_losses"]) == {"latent", "reconstruction"}
        assert all(torch.isfinite(value) for value in output["aux_losses"].values())
    else:
        assert output["aux_losses"] == {}


@pytest.mark.parametrize("model_name", PLUME_MODEL_CHOICES)
def test_composite_loss_backpropagates(model_name):
    config = _config()
    model = create_model(
        model_name=model_name,
        dataloader=None,
        metadata={},
        config=config,
        task_name="task_3-1",
    )
    inputs, targets = _task31_batch(batch_size=1)
    output = model(*inputs, targets=targets)
    criterion = ForecastLoss(get_loss_weights(model_name, config))
    loss = criterion(output, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())
    expected_aux_weight = 0.0 if model_name.endswith("_mse") else 0.5
    assert criterion.loss_weights["latent"] == expected_aux_weight


def test_eval_keeps_torch_backend_and_checkpoint_roundtrip(tmp_path):
    config = _config(backend="torch")
    model = create_model(
        model_name="plume_vanilla_jepa",
        dataloader=None,
        metadata={},
        config=config,
        task_name="task_3-1",
    )
    inputs, _ = _task31_batch(batch_size=1)
    model.eval()
    with torch.no_grad():
        expected = unwrap_predictions(model(*inputs))

    checkpoint = tmp_path / "best_model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = copy.deepcopy(model)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = unwrap_predictions(restored(*inputs))

    assert torch.allclose(expected[0], actual[0])
    assert torch.allclose(expected[1], actual[1])


def test_explicit_qant_backend_fails_clearly_without_sdk():
    if qant_available():
        pytest.skip("Native Q.ANT SDK is installed in this environment")
    model = create_model(
        model_name="plume_koopman_mse",
        dataloader=None,
        metadata={},
        config=_config(backend="qant"),
        task_name="task_3-1",
    )
    inputs, _ = _task31_batch(batch_size=1)
    model.eval()
    with pytest.raises(RuntimeError, match="Q.ANT backend requested"):
        model(*inputs)


def test_plume_models_reject_non_profile_task():
    with pytest.raises(ValueError, match="task_3-1"):
        create_model(
            model_name="plume_direct_mse",
            dataloader=None,
            metadata={},
            config=_config(),
            task_name="task_1-1",
        )
