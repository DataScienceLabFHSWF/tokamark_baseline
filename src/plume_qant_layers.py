"""Minimal PLUME Q.ANT primitives used by the TokaMark profile models.

The mathematical PyTorch paths are vendored from PLUME.  Backend selection is
explicit instead of being coupled to ``Module.training``: calling
``model.eval()`` therefore remains safe on ordinary CPU/GPU machines.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

QANTBackend = Literal["torch", "qant", "auto"]

try:
    import qant_native_computing_toolkit as qant
    from ml_dtypes import bfloat16 as _bf16

    _QANT_AVAILABLE = True
except ImportError:
    _QANT_AVAILABLE = False
    _bf16 = None
    qant = None


def qant_available() -> bool:
    """Return whether the optional native Q.ANT SDK can be imported."""

    return _QANT_AVAILABLE


def _validate_backend(backend: str) -> QANTBackend:
    if backend not in {"torch", "qant", "auto"}:
        raise ValueError(f"Unsupported Q.ANT backend: {backend}")
    return backend  # type: ignore[return-value]


class _BackendControlled:
    qant_backend: QANTBackend

    def _init_backend(self, backend: QANTBackend = "torch") -> None:
        self.qant_backend = _validate_backend(backend)

    def set_qant_backend(self, backend: QANTBackend) -> None:
        self.qant_backend = _validate_backend(backend)

    def _use_qant(self) -> bool:
        # Native Q.ANT inference is non-differentiable, so training always uses
        # the original PyTorch computation.  Evaluation uses the explicit flag.
        if self.training or self.qant_backend == "torch":
            return False
        if self.qant_backend == "qant" and not _QANT_AVAILABLE:
            raise RuntimeError(
                "Q.ANT backend requested, but qant_native_computing_toolkit "
                "and ml_dtypes are not installed. Use qant_backend='torch'."
            )
        return _QANT_AVAILABLE


def set_qant_backend(module: nn.Module, backend: QANTBackend) -> None:
    """Set one explicit execution backend on every vendored Q.ANT layer."""

    _validate_backend(backend)
    for child in module.modules():
        if isinstance(child, _BackendControlled):
            child.set_qant_backend(backend)


def _to_bf16(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(_bf16)


def _from_bf16(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array.astype(np.float32))


def _assert_cpu(x: torch.Tensor, layer_name: str) -> None:
    if x.device.type != "cpu":
        raise RuntimeError(
            f"{layer_name} Q.ANT execution requires CPU tensors; got {x.device}."
        )


def _q_tanh(x_np: np.ndarray, device_id: int = 0) -> np.ndarray:
    shape = x_np.shape
    flat = x_np.reshape(-1)
    two = np.full_like(flat, 2.0, dtype=_bf16)
    doubled = qant.native.mul_elementwise(flat, two, device_id)
    sigmoid = qant.ai.sigmoid_fprop(doubled.reshape(shape), device_id)
    two_again = np.full_like(sigmoid, 2.0, dtype=_bf16)
    one = np.full_like(sigmoid, 1.0, dtype=_bf16)
    result = qant.native.mul_elementwise(
        sigmoid.reshape(-1), two_again.reshape(-1), device_id
    )
    return result.reshape(shape) - one


class QSigmoid(_BackendControlled, nn.Module):
    def __init__(self, backend: QANTBackend = "torch") -> None:
        super().__init__()
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return torch.sigmoid(x)
        _assert_cpu(x, "QSigmoid")
        return _from_bf16(qant.ai.sigmoid_fprop(_to_bf16(x)))


class QTanh(_BackendControlled, nn.Module):
    def __init__(self, backend: QANTBackend = "torch") -> None:
        super().__init__()
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return torch.tanh(x)
        _assert_cpu(x, "QTanh")
        return _from_bf16(_q_tanh(_to_bf16(x)))


class QElementwiseMul(_BackendControlled, nn.Module):
    def __init__(self, backend: QANTBackend = "torch") -> None:
        super().__init__()
        self._init_backend(backend)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return a * b
        _assert_cpu(a, "QElementwiseMul")
        a_np, b_np = _to_bf16(a), _to_bf16(b)
        shape = a_np.shape
        output = qant.native.mul_elementwise(a_np.reshape(-1), b_np.reshape(-1))
        return _from_bf16(output.reshape(shape))


class QLinear(_BackendControlled, nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend: QANTBackend = "torch",
    ) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self._init_backend(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            return F.linear(x, self.weight, self.bias)
        _assert_cpu(x, "QLinear")
        output = qant.ai.linear_fprop(_to_bf16(x), _to_bf16(self.weight))
        if self.bias is not None:
            output = qant.ai.add_bias_fprop(output, _to_bf16(self.bias))
        return _from_bf16(output)


class QFourier(_BackendControlled, nn.Module):
    """PLUME/Q.ANT Fourier KAN layer with an explicit inference backend."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        add_bias: bool = True,
        noise_std: float | None = 0.02,
        backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        self._init_backend(backend)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.noise_std = noise_std

        amplitude_std = math.sqrt(2.0 / ((in_features + out_features) * grid_size))
        self.amplitude = nn.Parameter(
            torch.randn(out_features, in_features, grid_size) * amplitude_std
        )
        self.phase = nn.Parameter(
            torch.empty(out_features, in_features, grid_size).uniform_(
                -math.pi / 2, math.pi / 2
            )
        )
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        self.register_buffer("k", torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._use_qant():
            output = torch.sum(
                self.amplitude[None]
                * torch.cos(
                    x[:, None, :, None] * self.k[None, None, None, :] + self.phase[None]
                ),
                dim=(2, 3),
            )
            if self.training and self.noise_std is not None:
                output = output + torch.randn_like(output) * self.noise_std
            if self.bias is not None:
                output = output + self.bias[None, :]
            return output

        _assert_cpu(x, "QFourier")
        output = qant.ai.calc_kan_layer_fprop(
            _to_bf16(x),
            _to_bf16(self.phase),
            _to_bf16(self.amplitude),
            _to_bf16(self.k),
        )
        if self.bias is not None:
            output = qant.ai.add_bias_fprop(output, _to_bf16(self.bias))
        return _from_bf16(output)
