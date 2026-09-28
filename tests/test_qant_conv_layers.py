"""Correctness tests for the Q.ANT-primitive conv/pool/norm layers.

Two properties are checked for every layer:
1. The ``torch`` backend (default) must be bit-identical to the equivalent
   plain ``torch.nn`` layer -- these are drop-in replacements when Q.ANT
   dispatch is disabled.
2. When the native Q.ANT SDK is installed, the ``qant`` backend must match
   the ``torch`` backend within bfloat16 quantization tolerance (a few
   permille relative error) on a fixed random input. This is a regression
   test for the 1D->2D lifting workaround (padding/output_padding handling)
   in ``src/qant_conv_layers.py`` -- a naive implementation silently produces
   large (>10%) errors instead of raising, which earlier caused two real bugs
   caught here: (a) non-zero padding on the fake height axis corrupting
   values, and (b) transposed-conv output_padding being zero-padded instead
   of keeping genuine extended values.
"""

from __future__ import annotations

import pytest
import torch

from src.plume_qant_layers import qant_available
from src.qant_conv_layers import (
    QBatchNorm1d,
    QBatchNorm2d,
    QConv1d,
    QConv2d,
    QConvTranspose1d,
    QConvTranspose2d,
    QMaxPool1d,
    QMaxPool2d,
)

QANT_SDK_REQUIRED = pytest.mark.skipif(
    not qant_available(), reason="Native Q.ANT SDK not installed in this environment"
)

# A few permille: expected bfloat16 rounding noise, not a correctness bug.
BF16_REL_TOLERANCE = 0.01


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-8)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


class TestTorchBackendMatchesPlainTorch:
    """qant_backend='torch' (the default) must be numerically identical to nn.*."""

    def test_qconv1d(self):
        layer = QConv1d(4, 8, kernel_size=3, stride=3, padding=1)
        ref = torch.nn.Conv1d(4, 8, kernel_size=3, stride=3, padding=1)
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 4, 20)
        assert torch.allclose(layer(x), ref(x))

    def test_qconv2d(self):
        layer = QConv2d(4, 8, kernel_size=3, stride=3, padding=1)
        ref = torch.nn.Conv2d(4, 8, kernel_size=3, stride=3, padding=1)
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 4, 15, 17)
        assert torch.allclose(layer(x), ref(x))

    def test_qbatchnorm1d(self):
        layer = QBatchNorm1d(8)
        ref = torch.nn.BatchNorm1d(8)
        layer.eval()
        ref.eval()
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 8, 20)
        assert torch.allclose(layer(x), ref(x), atol=1e-6)

    def test_qmaxpool1d(self):
        layer = QMaxPool1d(2, padding=1)
        ref = torch.nn.MaxPool1d(2, padding=1)
        x = torch.randn(2, 8, 20)
        assert torch.allclose(layer(x), ref(x))

    def test_qconvtranspose1d(self):
        layer = QConvTranspose1d(8, 4, kernel_size=3, stride=3, padding=1, output_padding=1)
        ref = torch.nn.ConvTranspose1d(8, 4, kernel_size=3, stride=3, padding=1, output_padding=1)
        ref.load_state_dict(layer.state_dict())
        x = torch.randn(2, 8, 7)
        assert torch.allclose(layer(x), ref(x))


@QANT_SDK_REQUIRED
class TestQantBackendMatchesTorchWithinBf16Tolerance:
    """qant_backend='qant' must be close to 'torch', not merely non-crashing."""

    def test_qconv1d(self):
        layer = QConv1d(4, 8, kernel_size=3, stride=3, padding=1)
        layer.eval()
        x = torch.randn(2, 4, 20)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qconv2d(self):
        layer = QConv2d(4, 8, kernel_size=3, stride=3, padding=1)
        layer.eval()
        x = torch.randn(2, 4, 15, 17)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qbatchnorm1d(self):
        layer = QBatchNorm1d(8)
        layer.eval()
        x = torch.randn(2, 8, 20)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qbatchnorm2d(self):
        layer = QBatchNorm2d(8)
        layer.eval()
        x = torch.randn(2, 8, 5, 7)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qmaxpool1d(self):
        layer = QMaxPool1d(2, padding=1)
        layer.eval()
        x = torch.randn(2, 8, 20)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qmaxpool2d(self):
        layer = QMaxPool2d(2, padding=1)
        layer.eval()
        x = torch.randn(2, 8, 5, 7)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qconvtranspose1d(self):
        # Regression test: output_padding used to be zero-padded instead of
        # keeping genuine raw values, producing a >0.5 relative error on the
        # last output_padding elements.
        layer = QConvTranspose1d(8, 4, kernel_size=3, stride=3, padding=1, output_padding=1)
        layer.eval()
        x = torch.randn(2, 8, 7)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE

    def test_qconvtranspose2d(self):
        # Regression test: native SDK raises "output_padding != 0 not
        # implemented yet"; must be emulated via host-side cropping.
        layer = QConvTranspose2d(8, 4, kernel_size=3, stride=3, padding=1, output_padding=1)
        layer.eval()
        x = torch.randn(2, 8, 5, 7)
        expected = layer(x)
        layer.set_qant_backend("qant")
        actual = layer(x)
        assert _rel_err(expected, actual) < BF16_REL_TOLERANCE
