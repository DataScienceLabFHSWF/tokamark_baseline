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
