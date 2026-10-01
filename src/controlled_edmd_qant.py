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
        observable_type: str = "physical",
        pod_components: int = 16,
        delay_steps: int = 1,
        qant_backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        if profile_bins < 1 or action_dim < 1 or horizon < 1:
            raise ValueError("profile_bins, action_dim, and horizon must be positive")
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if input_structure not in {"affine", "bilinear"}:
            raise ValueError("input_structure must be 'affine' or 'bilinear'")
        if observable_type not in {"physical", "pod_quadratic"}:
            raise ValueError("observable_type must be 'physical' or 'pod_quadratic'")
        if delay_steps < 1:
            raise ValueError("delay_steps must be positive")
        history_dim = profile_bins * 2 * delay_steps
        if observable_type == "pod_quadratic" and not 1 <= pod_components <= history_dim:
            raise ValueError("pod_components must be between 1 and the state dimension")

        self.profile_bins = profile_bins
        self.state_dim = profile_bins * 2
        self.delay_steps = delay_steps
        self.history_dim = self.state_dim * delay_steps
        self.action_dim = action_dim
        self.horizon = horizon
        self.ridge = ridge
        self.input_structure = input_structure
        self.observable_type = observable_type
        self.pod_components = pod_components
        self.feature_dim = (
            self.history_dim + 1
            if observable_type == "physical"
            else 1 + 2 * pod_components
        )
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
        self.register_buffer("imputation_mean", torch.zeros(self.history_dim))
        self.register_buffer(
            "pod_basis", torch.zeros(self.history_dim, pod_components)
        )
        self.register_buffer("is_fitted", torch.tensor(False))
        self.fit_condition_number = float("nan")
        self.fit_observed_fraction = float("nan")
        self.fit_snapshot_count = 0
        self.set_qant_backend(qant_backend)

    def set_qant_backend(self, backend: QANTBackend) -> None:
        self.qant_backend = backend
        set_qant_backend(self, backend)

    def _observables(self, states: torch.Tensor) -> torch.Tensor:
        if states.shape[-1] != self.history_dim:
            raise ValueError(
                f"expected history dimension {self.history_dim}, got {states.shape[-1]}"
            )
        if self.observable_type == "physical":
            features = states
        else:
            mean = self.imputation_mean.to(device=states.device, dtype=states.dtype)
            basis = self.pod_basis.to(device=states.device, dtype=states.dtype)
            coordinates = (states - mean) @ basis
            features = torch.cat([coordinates, coordinates.square()], dim=-1)
        return torch.cat([torch.ones_like(states[..., :1]), features], dim=-1)

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
        fit_device = torch.device("cpu")
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
        if self.observable_type == "pod_quadratic":
            centered = torch.cat([states, next_states], dim=0) - means
            covariance = centered.T @ centered
            self._set_pod_basis(covariance)

        phi = self._observables(states)
        phi_next = self._observables(next_states)
        regression = torch.cat([phi, controls], dim=-1)
        if self.input_structure == "bilinear":
            interaction = (
                phi.unsqueeze(-1) * controls.unsqueeze(-2)
            ).flatten(start_dim=-2)
            regression = torch.cat([regression, interaction], dim=-1)
        gram = regression.T @ regression
        rhs = regression.T @ phi_next
        decoder_gram = phi.T @ phi
        decoder_rhs = phi.T @ next_states[:, : self.state_dim]
        self._fit_from_moments(gram, rhs, decoder_gram, decoder_rhs, ridge)
        self.fit_snapshot_count = states.shape[0]
        self.fit_observed_fraction = finite.float().mean().item()
        return self

    def _fit_from_moments(
        self,
        gram: torch.Tensor,
        rhs: torch.Tensor,
        decoder_gram: torch.Tensor,
        decoder_rhs: torch.Tensor,
        ridge: float,
    ) -> None:
        feature_scale = torch.diagonal(gram).clamp_min(1e-24).sqrt()
        scaled_gram = gram / feature_scale[:, None] / feature_scale[None, :]
        regularizer = ridge * torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        regularized_gram = scaled_gram + regularizer
        try:
            self.fit_condition_number = torch.linalg.cond(regularized_gram).item()
        except (torch.linalg.LinAlgError, RuntimeError):
            self.fit_condition_number = float("inf")
        try:
            coefficients = torch.linalg.solve(
                regularized_gram,
                rhs / feature_scale[:, None],
            )
        except (torch.linalg.LinAlgError, RuntimeError):
            coefficients = torch.linalg.lstsq(
                regularized_gram,
                rhs / feature_scale[:, None],
            ).solution
        coefficients = coefficients / feature_scale[:, None]
        control_start = self.feature_dim
        bilinear_start = control_start + self.action_dim
        with torch.no_grad():
            self.operator.weight.copy_(coefficients[: self.feature_dim].T)
            self.control.weight.copy_(coefficients[control_start:bilinear_start].T)
            if self.bilinear is not None:
                self.bilinear.weight.copy_(coefficients[bilinear_start:].T)

        decoder_scale = torch.diagonal(decoder_gram).clamp_min(1e-24).sqrt()
        scaled_decoder_gram = (
            decoder_gram / decoder_scale[:, None] / decoder_scale[None, :]
        )
        decoder_regularizer = ridge * torch.eye(
            self.feature_dim, dtype=decoder_gram.dtype, device=decoder_gram.device
        )
        decoder_system = scaled_decoder_gram + decoder_regularizer
        try:
            decoder = torch.linalg.solve(
                decoder_system,
                decoder_rhs / decoder_scale[:, None],
            )
        except (torch.linalg.LinAlgError, RuntimeError):
            decoder = torch.linalg.lstsq(
                decoder_system,
                decoder_rhs / decoder_scale[:, None],
            ).solution
        decoder = decoder / decoder_scale[:, None]
        with torch.no_grad():
            self.decoder.weight.copy_(decoder.T)
            self.is_fitted.fill_(True)

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
            states, controls, next_states = self._snapshot_batch(batch)
            states_list.append(states)
            next_states_list.append(next_states)
            controls_list.append(controls)

        if not states_list:
            raise ValueError("no batches supplied for EDMD fitting")
        states = torch.cat(states_list, dim=0).reshape(-1, self.history_dim)
        next_states = torch.cat(next_states_list, dim=0).reshape(-1, self.history_dim)
        controls = torch.cat(controls_list, dim=0).reshape(-1, self.action_dim)
        return states, controls, next_states

    def fit_batches(self, batches: Iterable[tuple], ridge: float | None = None) -> int:
        """Fit in two passes using CPU float64 sufficient statistics."""
        ridge = self.ridge if ridge is None else ridge
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        device = torch.device("cpu")
        state_sum = torch.zeros(self.history_dim, dtype=torch.float64)
        state_count = torch.zeros(self.history_dim, dtype=torch.float64)
        observed_count = 0
        total_count = 0
        snapshot_count = 0

        for batch in batches:
            if batch is None:
                continue
            states, _, next_states = self._snapshot_batch(batch)
            all_states = torch.cat([states, next_states], dim=0).to(device=device, dtype=torch.float64)
            finite = torch.isfinite(all_states)
            observed_count += int(finite.sum())
            total_count += finite.numel()
            state_sum += torch.where(finite, all_states, 0.0).sum(dim=0)
            state_count += finite.sum(dim=0)
            snapshot_count += states.shape[0]

        if snapshot_count == 0:
            raise ValueError("no batches supplied for EDMD fitting")
        means = state_sum / state_count.clamp_min(1)
        with torch.no_grad():
            self.imputation_mean.copy_(means)

        if self.observable_type == "pod_quadratic":
            covariance = torch.zeros(self.history_dim, self.history_dim, dtype=torch.float64)
            for batch in batches:
                if batch is None:
                    continue
                states, _, next_states = self._snapshot_batch(batch)
                states = torch.where(torch.isfinite(states), states, means).to(dtype=torch.float64)
                next_states = torch.where(
                    torch.isfinite(next_states), next_states, means
                ).to(dtype=torch.float64)
                centered = torch.cat([states, next_states], dim=0) - means
                covariance += centered.T @ centered
            self._set_pod_basis(covariance)

        regression_dim = self.feature_dim + self.action_dim
        if self.input_structure == "bilinear":
            regression_dim += self.feature_dim * self.action_dim
        gram = torch.zeros(regression_dim, regression_dim, dtype=torch.float64)
        rhs = torch.zeros(regression_dim, self.feature_dim, dtype=torch.float64)
        decoder_gram = torch.zeros(self.feature_dim, self.feature_dim, dtype=torch.float64)
        decoder_rhs = torch.zeros(self.feature_dim, self.state_dim, dtype=torch.float64)

        for batch in batches:
            if batch is None:
                continue
            states, controls, next_states = self._snapshot_batch(batch)
            states = torch.where(torch.isfinite(states), states, means).to(dtype=torch.float64)
            next_states = torch.where(torch.isfinite(next_states), next_states, means).to(dtype=torch.float64)
            controls = torch.nan_to_num(controls).to(dtype=torch.float64)
            phi = self._observables(states)
            phi_next = self._observables(next_states)
            design = torch.cat([phi, controls], dim=-1)
            if self.input_structure == "bilinear":
                design = torch.cat([
                    design,
                    (phi.unsqueeze(-1) * controls.unsqueeze(-2)).flatten(start_dim=-2),
                ], dim=-1)
            gram += design.T @ design
            rhs += design.T @ phi_next
            decoder_gram += phi.T @ phi
            decoder_rhs += phi.T @ next_states[:, : self.state_dim]

        self._fit_from_moments(gram, rhs, decoder_gram, decoder_rhs, ridge)
        self.fit_snapshot_count = snapshot_count
        self.fit_observed_fraction = observed_count / max(total_count, 1)
        return snapshot_count

    def _snapshot_batch(self, batch):
        _, _, inputs, targets = batch
        context = self._profile_state(inputs[0], inputs[1])
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
        sequence = torch.cat([context, future], dim=1)
        current_start = context.shape[1] - 1
        state_history = []
        next_history = []
        for step in range(self.horizon):
            current_index = current_start + step
            state_history.append(self._history_at(sequence, current_index))
            next_history.append(self._history_at(sequence, current_index + 1))
        states = torch.stack(state_history, dim=1)
        next_states = torch.stack(next_history, dim=1)
        return (
            states.reshape(-1, self.history_dim),
            torch.nan_to_num(actions).reshape(-1, self.action_dim),
            next_states.reshape(-1, self.history_dim),
        )

    def _history_at(self, sequence: torch.Tensor, index: int) -> torch.Tensor:
        history = [sequence[:, max(0, index - lag)] for lag in range(self.delay_steps)]
        return torch.cat(history, dim=-1)

    def _initial_history(self, context: torch.Tensor) -> torch.Tensor:
        return self._history_at(context, context.shape[1] - 1)

    def _set_pod_basis(self, covariance: torch.Tensor) -> None:
        covariance = (covariance + covariance.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -self.pod_components :].flip(dims=[1])
        with torch.no_grad():
            self.pod_basis.copy_(basis)

    def forward(self, *inputs: torch.Tensor, targets=None) -> dict[str, object]:
        if not bool(self.is_fitted):
            raise RuntimeError("fit the EDMD model before inference")
        profile_context = self._profile_state(inputs[0], inputs[1])
        initial = self._initial_history(profile_context)
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
        model_device = self.operator.weight.device
        states = states.to(model_device)
        controls = controls.to(model_device)
        next_states = next_states.to(model_device)
        all_states = torch.cat([states, next_states], dim=0)
        means = self.imputation_mean.to(device=model_device, dtype=states.dtype)
        states = torch.where(torch.isfinite(states), states, means)
        next_states = torch.where(torch.isfinite(next_states), next_states, means)
        lifted_next = self._lifted_step(
            self._observables(states), controls
        )
        return self._observables(next_states) - lifted_next

    @torch.no_grad()
    def closure_diagnostics_batches(self, batches: Iterable[tuple]) -> dict[str, float]:
        """Stream closure residual summaries without materializing a full split."""
        squared_error = 0.0
        element_count = 0
        max_abs = 0.0
        snapshot_count = 0
        for batch in batches:
            if batch is None:
                continue
            states, controls, next_states = self._snapshot_batch(batch)
            residual = self.closure_residual(states, controls, next_states)
            squared_error += residual.square().sum().item()
            element_count += residual.numel()
            max_abs = max(max_abs, residual.abs().max().item())
            snapshot_count += states.shape[0]
        if element_count == 0:
            raise ValueError("no validation snapshots supplied for closure diagnostics")
        return {
            "closure_rmse": (squared_error / element_count) ** 0.5,
            "closure_max_abs": max_abs,
            "validation_snapshot_count": float(snapshot_count),
        }

    def diagnostics(
        self,
        snapshots: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Return fit and optional held-out closure diagnostics."""
        self._check_fitted()
        result = {
            "input_structure": self.input_structure,
            "observable_type": self.observable_type,
            "observable_dimension": float(self.feature_dim),
            "pod_components": float(self.pod_components),
            "delay_steps": float(self.delay_steps),
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

    def effective_spectral_radii(self, controls: torch.Tensor) -> torch.Tensor:
        """Return spectral radii of the controlled operator over input samples."""
        self._check_fitted()
        if self.bilinear is None:
            return torch.full(
                (controls.reshape(-1, self.action_dim).shape[0],),
                self.spectral_radius(),
                dtype=torch.float64,
            )
        controls = controls.reshape(-1, self.action_dim).detach().cpu().double()
        bilinear_blocks = self.bilinear.weight.detach().cpu().double().reshape(
            self.feature_dim, self.action_dim, self.feature_dim
        )
        base = self.operator.weight.detach().cpu().double()
        radii = []
        for control in controls:
            effective = base + torch.einsum("j,ijq->iq", control, bilinear_blocks)
            radii.append(torch.linalg.eigvals(effective).abs().max())
        return torch.stack(radii)

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
        if states.shape[1] != self.history_dim:
            raise ValueError(f"expected history dimension {self.history_dim}")
        if controls.shape[1] != self.action_dim:
            raise ValueError(f"expected control dimension {self.action_dim}")
        return states.float(), controls.float(), next_states.float()

    def _check_fitted(self) -> None:
        if not bool(self.is_fitted):
            raise RuntimeError("fit the EDMD model before using it")

    def _check_shapes(self, states: torch.Tensor, controls: torch.Tensor) -> None:
        if states.shape[-1] != self.history_dim or controls.shape[-1] != self.action_dim:
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
