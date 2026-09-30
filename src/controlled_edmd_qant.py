"""Controlled EDMD baseline for TokaMark task 3-1.

The model uses the measured profile coordinates as a fixed observable basis.
Ridge fitting is performed offline with PyTorch; fitted inference and decoding
can be dispatched through the vendored Q.ANT linear primitive.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

from src.plume_qant_layers import QANTBackend, QElementwiseMul, QLinear, set_qant_backend


class TokaMarkControlledEDMD(nn.Module):
    """Degree-one controlled EDMD model with a TokaMark output contract."""

    supports_auxiliary_losses = False

    def __init__(
        self,
        profile_bins: int = 120,
        action_dim: int = 4,
        horizon: int = 10,
        ridge: float = 1e-6,
        input_structure: str = "affine",
        qant_backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        if profile_bins < 1 or action_dim < 1 or horizon < 1:
            raise ValueError("profile_bins, action_dim, and horizon must be positive")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if input_structure not in {"affine", "bilinear"}:
            raise ValueError("input_structure must be 'affine' or 'bilinear'")

        self.profile_bins = profile_bins
        self.state_dim = profile_bins * 2
        self.action_dim = action_dim
        self.horizon = horizon
        self.ridge = ridge
        self.input_structure = input_structure
        self.feature_dim = self.state_dim + 1
        self.qant_backend = qant_backend

        self.operator = QLinear(self.feature_dim, self.feature_dim, bias=False)
        self.control = QLinear(action_dim, self.feature_dim, bias=False)
        self.bilinear = (
            QLinear(self.feature_dim * action_dim, self.feature_dim, bias=False)
            if input_structure == "bilinear"
            else None
        )
        self.multiply = QElementwiseMul() if input_structure == "bilinear" else None
        self.decoder = QLinear(self.feature_dim, self.state_dim, bias=False)
        self.register_buffer("imputation_mean", torch.zeros(self.state_dim))
        self.register_buffer("is_fitted", torch.tensor(False))
        self.fit_condition_number = float("nan")
        self.fit_observed_fraction = float("nan")
        self.fit_snapshot_count = 0
        self.set_qant_backend(qant_backend)

    def set_qant_backend(self, backend: QANTBackend) -> None:
        self.qant_backend = backend
        set_qant_backend(self, backend)

    def _observables(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.state_dim:
            raise ValueError(
                f"expected state dimension {self.state_dim}, got {states.shape[-1]}"
            )
        return torch.cat([torch.ones_like(states[..., :1]), states], dim=-1)

    def fit(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        next_states: torch.Tensor,
        ridge: float | None = None,
    ) -> "TokaMarkControlledEDMD":
        """Fit the controlled lifted operator and physical decoder."""
        states, controls, next_states = self._validate_snapshots(
            states, controls, next_states
        )
        fit_device = self.operator.weight.device
        fit_dtype = torch.float64
        states = states.to(device=fit_device, dtype=fit_dtype)
        controls = controls.to(device=fit_device, dtype=fit_dtype)
        next_states = next_states.to(device=fit_device, dtype=fit_dtype)
        ridge = self.ridge if ridge is None else ridge
        if ridge < 0:
            raise ValueError("ridge must be non-negative")

        all_states = torch.cat([states, next_states], dim=0)
        finite = torch.isfinite(all_states)
        self.fit_snapshot_count = states.shape[0]
        self.fit_observed_fraction = finite.float().mean().item()
        counts = finite.sum(dim=0).clamp_min(1)
        means = torch.where(
            finite,
            all_states,
            torch.zeros_like(all_states),
        ).sum(dim=0) / counts
        states = torch.where(torch.isfinite(states), states, means)
        next_states = torch.where(torch.isfinite(next_states), next_states, means)
        self.imputation_mean.copy_(means)

        phi = self._observables(states)
        phi_next = self._observables(next_states)
        regression = torch.cat([phi, controls], dim=-1)
        if self.input_structure == "bilinear":
            interaction = (
                phi.unsqueeze(-1) * controls.unsqueeze(-2)
            ).flatten(start_dim=-2)
            regression = torch.cat([regression, interaction], dim=-1)
        gram = regression.T @ regression
        regularizer = ridge * torch.eye(
            regression.shape[-1], dtype=regression.dtype, device=regression.device
        )
        regularized_gram = gram + regularizer
        try:
            self.fit_condition_number = torch.linalg.cond(regularized_gram).item()
        except torch.linalg.LinAlgError:
            self.fit_condition_number = float("inf")
        try:
            coefficients = torch.linalg.solve(
                regularized_gram,
                regression.T @ phi_next,
            )
        except torch.linalg.LinAlgError:
            coefficients = torch.linalg.pinv(regularized_gram) @ (regression.T @ phi_next)
        control_start = self.feature_dim
        bilinear_start = control_start + self.action_dim
        with torch.no_grad():
            self.operator.weight.copy_(coefficients[: self.feature_dim].T)
            self.control.weight.copy_(coefficients[control_start:bilinear_start].T)
            if self.bilinear is not None:
                self.bilinear.weight.copy_(coefficients[bilinear_start:].T)

        decoder_gram = phi.T @ phi
        decoder_regularizer = ridge * torch.eye(
            self.feature_dim, dtype=phi.dtype, device=phi.device
        )
        decoder_system = decoder_gram + decoder_regularizer
        try:
            decoder = torch.linalg.solve(decoder_system, phi.T @ states)
        except torch.linalg.LinAlgError:
            decoder = torch.linalg.pinv(decoder_system) @ (phi.T @ states)
        with torch.no_grad():
            self.decoder.weight.copy_(decoder.T)
            self.is_fitted.fill_(True)
        return self

    def snapshots_from_batches(
        self, batches: Iterable[tuple]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract snapshot tensors while preserving missing values as NaNs."""
        states_list = []
        controls_list = []
        next_states_list = []
        for batch in batches:
            if batch is None:
                continue
            _, _, inputs, targets = batch
            initial = self._profile_state(inputs[0], inputs[1])[:, -1]
            future = torch.cat(
                [self._profile_sequence(targets[0]), self._profile_sequence(targets[1])],
                dim=-1,
            )
            actions = torch.stack(
                [self._actuator_sequence(branch) for branch in inputs[-self.action_dim :]],
                dim=-1,
            )
            if future.shape[1] != self.horizon or actions.shape[1] != self.horizon:
                raise ValueError("batch horizon does not match the EDMD model")
            states_list.append(torch.cat([initial.unsqueeze(1), future[:, :-1]], dim=1))
            next_states_list.append(future)
            controls_list.append(actions)

        if not states_list:
            raise ValueError("no batches supplied for EDMD fitting")
        states = torch.cat(states_list, dim=0).reshape(-1, self.state_dim)
        next_states = torch.cat(next_states_list, dim=0).reshape(-1, self.state_dim)
        controls = torch.cat(controls_list, dim=0).reshape(-1, self.action_dim)
        return states, controls, next_states

    def fit_batches(self, batches: Iterable[tuple], ridge: float | None = None) -> int:
        """Fit from canonical baseline batches and return snapshot count."""
        states, controls, next_states = self.snapshots_from_batches(batches)
        self.fit(states, controls, next_states, ridge=ridge)
        return states.shape[0]

    def forward(self, *inputs: torch.Tensor, targets=None) -> dict[str, object]:
        if not bool(self.is_fitted):
            raise RuntimeError("fit the EDMD model before inference")
        profiles_te = self._profile_sequence(inputs[0])
        profiles_ne = self._profile_sequence(inputs[1])
        initial = self._profile_state(profiles_te, profiles_ne)[:, -1]
        actions = torch.stack(
            [self._actuator_sequence(branch) for branch in inputs[-self.action_dim :]],
            dim=-1,
        )
        predictions = self.rollout(initial, actions)
        return {
            "predictions": [
                predictions[..., : self.profile_bins],
                predictions[..., self.profile_bins :],
            ],
            "aux_losses": {},
        }

    @torch.no_grad()
    def rollout(self, initial_state: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        self._check_shapes(initial_state, controls)
        state = torch.where(torch.isfinite(initial_state), initial_state, self.imputation_mean)
        lifted = self._observables(state)
        predictions = []
        for control in controls.unbind(dim=1):
            lifted = self._lifted_step(lifted, control)
            predictions.append(self.decoder(lifted))
        return torch.stack(predictions, dim=1)

    @torch.no_grad()
    def closure_residual(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-snapshot observable closure residuals."""
        self._check_fitted()
        states, controls, next_states = self._validate_snapshots(
            states, controls, next_states
        )
        all_states = torch.cat([states, next_states], dim=0)
        finite = torch.isfinite(all_states)
        means = torch.where(
            finite,
            all_states,
            torch.zeros_like(all_states),
        ).sum(dim=0) / finite.sum(dim=0).clamp_min(1)
        states = torch.where(torch.isfinite(states), states, means)
        next_states = torch.where(torch.isfinite(next_states), next_states, means)
        lifted_next = self._lifted_step(
            self._observables(states), controls
        )
        return self._observables(next_states) - lifted_next

    def diagnostics(
        self,
        snapshots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Return fit and optional held-out closure diagnostics."""
        self._check_fitted()
        result = {
            "input_structure": self.input_structure,
            "snapshot_count": float(self.fit_snapshot_count),
            "observed_fraction": self.fit_observed_fraction,
            "condition_number": self.fit_condition_number,
            "spectral_radius": self.spectral_radius(),
        }
        if snapshots is not None:
            residual = self.closure_residual(*snapshots)
            result["closure_rmse"] = torch.sqrt(residual.square().mean()).item()
            result["closure_max_abs"] = residual.abs().max().item()
        return result

    def _lifted_step(
        self, lifted: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        lifted_next = self.operator(lifted) + self.control(control)
        if self.bilinear is not None:
            lifted_next = lifted_next + self.bilinear(
                self._bilinear_features(lifted, control)
            )
        return lifted_next

    def spectral_radius(self) -> float:
        self._check_fitted()
        return torch.linalg.eigvals(self.operator.weight.detach().cpu()).abs().max().item()

    @staticmethod
    def _profile_sequence(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 3:
            raise ValueError(f"profile tensor must be at least rank 3, got {tensor.shape}")
        if tensor.ndim == 3:
            return tensor
        reduction_dims = tuple(range(2, tensor.ndim - 1))
        return tensor.mean(dim=reduction_dims) if reduction_dims else tensor

    def _profile_state(self, profiles_te: torch.Tensor, profiles_ne: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self._profile_sequence(profiles_te), self._profile_sequence(profiles_ne)], dim=-1
        )

    @staticmethod
    def _actuator_sequence(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 2:
            return tensor
        if tensor.ndim < 3:
            raise ValueError(f"actuator tensor must be rank 2 or higher, got {tensor.shape}")
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)[..., 0]

    def _validate_snapshots(self, states, controls, next_states):
        if states.ndim != 2 or controls.ndim != 2 or next_states.ndim != 2:
            raise ValueError("snapshot tensors must be rank 2")
        if states.shape != next_states.shape:
            raise ValueError("states and next_states must have equal shapes")
        if states.shape[0] != controls.shape[0]:
            raise ValueError("snapshot row counts must match")
        if states.shape[1] != self.state_dim:
            raise ValueError(f"expected state dimension {self.state_dim}")
        if controls.shape[1] != self.action_dim:
            raise ValueError(f"expected control dimension {self.action_dim}")
        return states.float(), controls.float(), next_states.float()

    def _check_fitted(self) -> None:
        if not bool(self.is_fitted):
            raise RuntimeError("fit the EDMD model before using it")

    def _check_shapes(self, states: torch.Tensor, controls: torch.Tensor) -> None:
        if states.shape[-1] != self.state_dim or controls.shape[-1] != self.action_dim:
            raise ValueError("rollout tensors have incompatible feature dimensions")
        if controls.ndim != 3:
            raise ValueError("controls must have shape (batch, horizon, action_dim)")

    def _bilinear_features(
        self, lifted: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        if self.multiply is None:
            raise RuntimeError("bilinear features requested for an affine model")
        lifted_expanded = lifted.unsqueeze(-1).expand(*lifted.shape, self.action_dim)
        control_expanded = control.unsqueeze(-2).expand_as(lifted_expanded)
        return self.multiply(lifted_expanded, control_expanded).flatten(start_dim=-2)
