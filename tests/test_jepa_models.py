import torch

from src.jepa_models import TokaMarkDirectWindowJEPA, TokaMarkJEPARollout


def canonical_task31_batch(batch_size=3, horizon=10, bins=120, chunk=20):
    inputs = [
        torch.randn(batch_size, 1, 1, 1, bins),
        torch.randn(batch_size, 1, 1, 1, bins),
    ]
    inputs += [torch.randn(batch_size, 1, 1, chunk) for _ in range(4)]
    inputs += [torch.randn(batch_size, horizon, 1, chunk) for _ in range(4)]
    targets = [
        torch.randn(batch_size, horizon, bins),
        torch.randn(batch_size, horizon, bins),
    ]
    return inputs, targets


def common_kwargs():
    return {
        "profile_bins": 120,
        "action_step_dim": 80,
        "latent_dim": 32,
        "hidden_dim": 64,
        "future_start_index": 6,
        "dropout": 0.0,
        "horizon": 10,
    }


def assert_outputs(outputs):
    assert len(outputs) == 2
    assert outputs[0].shape == (3, 10, 120)
    assert outputs[1].shape == (3, 10, 120)


def test_rollout_shapes_and_backward():
    inputs, targets = canonical_task31_batch()
    model = TokaMarkJEPARollout(**common_kwargs())
    outputs = model(*inputs)
    assert_outputs(outputs)
    sum((pred - target).square().mean() for pred, target in zip(outputs, targets)).backward()


def test_direct_mse_has_no_auxiliary_loss():
    inputs, targets = canonical_task31_batch()
    model = TokaMarkDirectWindowJEPA(
        **common_kwargs(), latent_loss_weight=0.0, target_ema=0.996
    )
    outputs, auxiliary = model.training_forward(inputs, targets)
    assert_outputs(outputs)
    assert auxiliary == {}


def test_direct_jepa_has_latent_loss_and_frozen_target_encoder():
    inputs, targets = canonical_task31_batch()
    model = TokaMarkDirectWindowJEPA(
        **common_kwargs(), latent_loss_weight=1.0, target_ema=0.996
    )
    outputs, auxiliary = model.training_forward(inputs, targets)
    assert_outputs(outputs)
    assert set(auxiliary) == {"latent_mse"}
    assert not any(parameter.requires_grad for parameter in model.target_encoder.parameters())
    auxiliary["latent_mse"].backward()

