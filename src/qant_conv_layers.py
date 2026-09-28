"""Q.ANT-primitive convolutional building blocks for the CNN baseline.

These wrap ``torch.nn`` conv/pool/norm layers exactly like ``QLinear`` /
``QFourier`` in :mod:`src.plume_qant_layers`: the PyTorch forward path is used
during training and whenever ``qant_backend == "torch"``; the native Q.ANT
SDK (bfloat16, CPU-only) path is used in eval mode when ``qant_backend ==
"qant"``. This lets the existing CNN baseline (``src/multi_conv_mlp_model.py``)
run through the same NPU-emulation path as the PLUME variants, enabling a
genuine apples-to-apples bfloat16 error comparison.

Native Q.ANT conv/pool/transpose-conv primitives only expose a single
(isotropic) stride/padding/dilation int applied to both spatial dimensions of
a real 2D op. For the 1D encoders/decoders (used for actuator time series) we
lift the 1D tensor to a 2D tensor with a singleton height dimension. Applying
the real padding/stride to that fake height dimension via the native int
parameters would corrupt values (e.g. non-zero padding on a height-1 axis
shifts which row of data the kernel sees), so for the 1D layers below we
instead perform any padding/cropping needed for the fake height axis
ourselves on the host (exact, lossless -- it is pure zero-padding/slicing of
already bfloat16-cast data) and always call the native primitive with
padding=0 (conv/pool) or padding=0, output_padding=0 (transpose-conv) so the
height axis passes through unchanged at size 1. The real (width) axis padding
is applied host-side before dispatch, so all of the actual multiply-
accumulate compute still runs through the bfloat16-emulated NPU path.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.plume_qant_layers import (
    _BackendControlled,
    _assert_cpu,
    _from_bf16,
    _to_bf16,
    QANTBackend,
    qant,
)

# Sentinel used to emulate max-pooling's implicit "-inf" padding; representable
# exactly enough in bfloat16 to never win a max() against real profile data.
_NEG_INF_SENTINEL = -1e30


def _assert_height_pinned(x: torch.Tensor, expected: int = 1) -> None:
    if x.shape[-2] != expected:
        raise RuntimeError(
            "1D->2D Q.ANT conv/pool lifting invariant violated: expected height "
            f"dimension {expected}, got {x.shape[-2]}."
        )


def _batchnorm2d_fprop_unbatched(features_bf16, mean, var, weight, bias, eps):
    """Loop over the batch dimension: the installed backend rejects batched input."""
    outputs = [
        qant.ai.batchnorm2d_fprop(sample, mean, var, weight, bias, eps)
        for sample in features_bf16
    ]
    return np.stack(outputs, axis=0)


def _maxpool2d_fprop_unbatched(features_bf16, kernel_size, stride, padding):
    """Loop over the batch dimension: the installed backend rejects batched input."""
    outputs = [
        qant.ai.maxpool2d_fprop(sample, kernel_size, stride, padding)
        for sample in features_bf16
    ]
    return np.stack(outputs, axis=0)


def _conv_transpose_fprop_unbatched(features_bf16, kernels_bf16, padding, stride, dilation, output_padding):
    """Loop over the batch dimension: the installed backend rejects batched input."""
    outputs = [
        qant.ai.conv_transpose_fprop(sample, kernels_bf16, padding, stride, dilation, output_padding)
        for sample in features_bf16
    ]
    return np.stack(outputs, axis=0)


class QReLU(_BackendControlled, nn.ReLU):
    def __init__(self, backend: QANTBackend = "torch") -> None:
        super().__init__()
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return F.relu(x)
        _assert_cpu(x, "QReLU")
        return _from_bf16(qant.ai.relu_fprop(_to_bf16(x)))


class QBatchNorm2d(_BackendControlled, nn.BatchNorm2d):
    def __init__(self, num_features: int, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QBatchNorm2d")
        output = _batchnorm2d_fprop_unbatched(
            _to_bf16(x),
            _to_bf16(self.running_mean),
            _to_bf16(self.running_var),
            _to_bf16(self.weight),
            _to_bf16(self.bias),
            self.eps,
        )
        return _from_bf16(output)


class QBatchNorm1d(_BackendControlled, nn.BatchNorm1d):
    """Lifts (B, C, L) -> (B, C, 1, L) to reuse the native 2D batchnorm kernel.

    Batchnorm has no kernel/stride, so the height=1 lift needs no host-side
    padding workaround -- only the batching-loop workaround above.
    """

    def __init__(self, num_features: int, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(num_features, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QBatchNorm1d")
        x2d = x.unsqueeze(-2)
        output = _batchnorm2d_fprop_unbatched(
            _to_bf16(x2d),
            _to_bf16(self.running_mean),
            _to_bf16(self.running_var),
            _to_bf16(self.weight),
            _to_bf16(self.bias),
            self.eps,
        )
        result = _from_bf16(output)
        _assert_height_pinned(result)
        return result.squeeze(-2)


class QConv2d(_BackendControlled, nn.Conv2d):
    def __init__(self, *args, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QConv2d")
        padding = self.padding[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        output = qant.ai.conv_fprop(
            _to_bf16(x), _to_bf16(self.weight), padding, stride, dilation
        )
        output = _from_bf16(output)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


class QConv1d(_BackendControlled, nn.Conv1d):
    """Lifts (B, C, L) -> (B, C, 1, L) to reuse the native 2D conv kernel.

    Padding is applied to the (real) width axis on the host before dispatch;
    the native call always uses padding=0 so the fake height axis is a no-op.
    """

    def __init__(self, *args, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QConv1d")
        padding = self.padding[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        x_padded = F.pad(x, (padding, padding))
        x2d = x_padded.unsqueeze(-2)
        weight2d = self.weight.unsqueeze(-2)  # (out, in, 1, k)
        output = qant.ai.conv_fprop(_to_bf16(x2d), _to_bf16(weight2d), 0, stride, dilation)
        output = _from_bf16(output)
        _assert_height_pinned(output)
        output = output.squeeze(-2)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


class QConvTranspose2d(_BackendControlled, nn.ConvTranspose2d):
    """The native SDK does not implement ``output_padding != 0``, so (like the
    1D lift below) we always call it with padding=0, output_padding=0 and
    apply the real padding/output_padding ourselves on the host: crop
    ``padding`` off both sides of each spatial axis, but ``padding -
    output_padding`` off the trailing side, keeping genuine raw values on the
    extended side rather than zero-padding.
    """

    def __init__(self, *args, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QConvTranspose2d")
        padding = self.padding[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        output_padding = self.output_padding[0]
        output = _conv_transpose_fprop_unbatched(
            _to_bf16(x), _to_bf16(self.weight), 0, stride, dilation, 0
        )
        output = _from_bf16(output)
        right_trim = padding - output_padding
        if padding > 0 or right_trim > 0:
            h_end = output.shape[-2] - right_trim if right_trim > 0 else output.shape[-2]
            w_end = output.shape[-1] - right_trim if right_trim > 0 else output.shape[-1]
            output = output[:, :, padding:h_end, padding:w_end]
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output


class QConvTranspose1d(_BackendControlled, nn.ConvTranspose1d):
    """Lifts (B, C, L) -> (B, C, 1, L) to reuse the native 2D transpose-conv kernel.

    The native call always uses padding=0, output_padding=0 (fake height axis
    stays a no-op); the real padding/output_padding are applied to the width
    axis afterward on the host (exact crop / zero-extend).
    """

    def __init__(self, *args, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QConvTranspose1d")
        padding = self.padding[0]
        stride = self.stride[0]
        dilation = self.dilation[0]
        output_padding = self.output_padding[0]
        x2d = x.unsqueeze(-2)
        weight2d = self.weight.unsqueeze(-2)  # (in, out, 1, k)
        output = _conv_transpose_fprop_unbatched(_to_bf16(x2d), _to_bf16(weight2d), 0, stride, dilation, 0)
        output = _from_bf16(output)
        _assert_height_pinned(output)
        output = output.squeeze(-2)
        # output_padding keeps real values from the raw (padding=0) computation
        # rather than zero-padding: it reduces how much the right side is
        # cropped, it does not add synthetic zeros (verified against nn.ConvTranspose1d).
        right_trim = padding - output_padding
        if right_trim > 0:
            output = output[:, :, padding : output.shape[-1] - right_trim]
        else:
            output = output[:, :, padding:]
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1)
        return output


class QMaxPool2d(_BackendControlled, nn.MaxPool2d):
    def __init__(self, kernel_size, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(kernel_size, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QMaxPool2d")
        kernel = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        padding = self.padding if isinstance(self.padding, int) else self.padding[0]
        output = _maxpool2d_fprop_unbatched(_to_bf16(x), kernel, stride, padding)
        return _from_bf16(output)


class QMaxPool1d(_BackendControlled, nn.MaxPool1d):
    """Lifts (B, C, L) -> (B, C, 1, L) to reuse the native 2D maxpool kernel.

    Padding is emulated on the host with a -inf sentinel (matching PyTorch's
    implicit max-pool padding semantics) on the width axis; the native call
    always uses padding=0 with kernel_size=(1, k) so the fake height axis is
    a no-op regardless of stride.
    """

    def __init__(self, kernel_size, backend: QANTBackend = "torch", **kwargs) -> None:
        super().__init__(kernel_size, **kwargs)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return super().forward(x)
        _assert_cpu(x, "QMaxPool1d")
        kernel = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        padding = self.padding if isinstance(self.padding, int) else self.padding[0]
        x_padded = F.pad(x, (padding, padding), value=_NEG_INF_SENTINEL) if padding > 0 else x
        x2d = x_padded.unsqueeze(-2)
        output = _maxpool2d_fprop_unbatched(_to_bf16(x2d), (1, kernel), stride, 0)
        output = _from_bf16(output)
        _assert_height_pinned(output)
        return output.squeeze(-2)
