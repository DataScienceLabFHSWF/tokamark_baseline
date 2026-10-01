import torch
import numpy as np

from src.controlled_edmd_qant import TokaMarkControlledEDMD
from src.plume_tokamark_adapter import unwrap_predictions
from src.trainer import edmd_collate_fn


def _batch(batch_size=3, horizon=10, bins=4):
    profiles_te = torch.randn(batch_size, 1, 1, 1, bins)
    profiles_ne = torch.randn(batch_size, 1, 1, 1, bins)
    past = [torch.randn(batch_size, 1, 1, 20) for _ in range(4)]
    future = [torch.randn(batch_size, horizon, 1, 20) for _ in range(4)]
    targets = [torch.randn(batch_size, horizon, bins) for _ in range(2)]
    inputs = [profiles_te, profiles_ne, *past, *future]
    return torch.zeros(batch_size), torch.zeros(batch_size), inputs, targets


def test_controlled_edmd_fits_and_matches_profile_contract():
    model = TokaMarkControlledEDMD(profile_bins=4)
    batches = [_batch()]
    count = model.fit_batches(batches)

    assert count == 30
    output = model(*batches[0][2], targets=batches[0][3])
    predictions = unwrap_predictions(output)
    assert predictions[0].shape == (3, 10, 4)
    assert predictions[1].shape == (3, 10, 4)
    assert torch.isfinite(predictions[0]).all()
    assert model.spectral_radius() >= 0


def test_bilinear_controlled_edmd_fits_and_rolls_out():
    model = TokaMarkControlledEDMD(profile_bins=4, input_structure="bilinear")
    batch = _batch(batch_size=2)
    batch[2][0].zero_()
    batch[2][1].zero_()
    for branch in batch[2][-4:]:
        branch.zero_()
    for target in batch[3]:
        target.zero_()
    batches = [batch]
    model.fit_batches(batches)

    output = model(*batches[0][2], targets=batches[0][3])
    predictions = unwrap_predictions(output)
    assert predictions[0].shape == (2, 10, 4)
    assert model.bilinear is not None
    assert torch.isfinite(predictions[1]).all()


def test_edmd_collate_preserves_missing_measurements():
    batch = [{
        "shot_id": 1,
        "window_index": 2,
        "input": [np.array([[np.nan]], dtype=np.float32)],
        "exogenous": [],
        "y": [np.array([[np.nan]], dtype=np.float32)],
    }]
    _, _, inputs, targets = edmd_collate_fn(batch)
    assert torch.isnan(inputs[0]).any()
    assert torch.isnan(targets[0]).any()


def test_edmd_tiny_ridge_handles_rank_deficient_snapshots():
    model = TokaMarkControlledEDMD(profile_bins=4, ridge=1e-8)
    states = torch.zeros(32, model.state_dim)
    controls = torch.zeros(32, model.action_dim)
    next_states = torch.zeros_like(states)

    model.fit(states, controls, next_states)

    assert torch.isfinite(model.operator.weight).all()
    assert torch.isfinite(model.control.weight).all()
    assert torch.isfinite(model.decoder.weight).all()
    assert model.fit_condition_number < float("inf")


def test_pod_quadratic_observables_fit_declared_basis():
    model = TokaMarkControlledEDMD(
        profile_bins=4,
        pod_components=3,
        observable_type="pod_quadratic",
        ridge=1e-5,
    )
    generator = torch.Generator().manual_seed(11)
    states = torch.randn(128, model.state_dim, generator=generator)
    controls = torch.randn(128, model.action_dim, generator=generator)
    next_states = 0.8 * states + 0.1 * controls[:, :1].expand_as(states)

    model.fit(states, controls, next_states)

    assert model.feature_dim == 1 + 2 * 3
    assert model.pod_basis.shape == (model.state_dim, 3)
    assert torch.allclose(
        model.pod_basis.T @ model.pod_basis,
        torch.eye(3),
        atol=1e-5,
    )
    assert model.diagnostics()["observable_type"] == "pod_quadratic"


def test_delay_coordinates_use_profile_context_and_keep_output_contract():
    model = TokaMarkControlledEDMD(profile_bins=4, delay_steps=3, ridge=1e-5)
    batch = _batch(batch_size=2)
    batch[2][0].zero_()
    batch[2][1].zero_()
    for branch in batch[2][-4:]:
        branch.zero_()
    for target in batch[3]:
        target.zero_()

    state_history, controls, next_history = model._snapshot_batch(batch)
    assert state_history.shape == (20, 24)
    assert next_history.shape == (20, 24)
    assert controls.shape == (20, 4)
    model.fit_batches([batch])

    output = model(*batch[2], targets=batch[3])
    predictions = unwrap_predictions(output)
    assert predictions[0].shape == (2, 10, 4)
    assert torch.isfinite(predictions[1]).all()


def test_bilinear_effective_spectrum_is_control_conditioned():
    model = TokaMarkControlledEDMD(profile_bins=4, input_structure="bilinear")
    batch = _batch(batch_size=2)
    for branch in batch[2]:
        branch.zero_()
    for target in batch[3]:
        target.zero_()
    model.fit_batches([batch])

    radii = model.effective_spectral_radii(torch.randn(5, 4))
    assert radii.shape == (5,)
    assert torch.isfinite(radii).all()
