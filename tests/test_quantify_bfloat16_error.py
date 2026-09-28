import pytest
import torch

from quantify_bfloat16_error import _ChannelAccumulator


def test_accumulator_ignores_missing_target_bins_and_aggregates_per_shot():
    accumulator = _ChannelAccumulator(output_scale=2.0, metric_scale=2.0)
    predictions = torch.tensor([[2.0, 99.0, 100.0], [6.0, 2.0, 99.0]])
    targets = torch.tensor([[1.0, float("nan"), 99.0], [5.0, 1.0, float("nan")]])

    accumulator.update(predictions, targets, torch.tensor([17, 17]))

    metrics = accumulator.metrics()
    assert metrics["n_shots"] == 1
    assert metrics["nrmse"] == pytest.approx(1.0)
    assert metrics["nmae"] == pytest.approx(1.0)


def test_accumulator_rejects_nonfinite_prediction_at_valid_target():
    accumulator = _ChannelAccumulator(output_scale=1.0, metric_scale=1.0)

    with pytest.raises(ValueError, match="Non-finite prediction"):
        accumulator.update(
            torch.tensor([[float("nan")]]),
            torch.tensor([[1.0]]),
            torch.tensor([17]),
        )


def test_accumulator_returns_nan_when_every_target_bin_is_missing():
    accumulator = _ChannelAccumulator(output_scale=1.0, metric_scale=1.0)
    accumulator.update(
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[float("nan"), float("nan")]]),
        torch.tensor([17]),
    )

    metrics = accumulator.metrics()
    assert metrics["n_shots"] == 0
    assert torch.isnan(torch.tensor(metrics["nrmse"]))
