"""Adapter from canonical TokaMark baseline batches to PLUME profile models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from src.plume_profile_models import (
    TokaMarkDirectWindowJEPA,
    TokaMarkProfileJEPA,
)
from src.plume_qant_layers import QANTBackend


def unwrap_predictions(output):
    """Return the baseline-compatible output list from any supported model."""

    if isinstance(output, dict) and "predictions" in output:
        return output["predictions"]
    return output


class TokaMarkPLUMEAdapter(nn.Module):
    """Expose PLUME Task 3-1 models through the baseline ``model(*x)`` API.

    The first two input branches are T_e and n_e.  The final four branches are
    the future actuator windows. Past actuator branches remain in the canonical
    loader output but are deliberately not used by PLUME's controlled latent
    transition.
    """

    supports_auxiliary_losses = True

    def __init__(
        self,
        dynamics: str,
        objective: str,
        profile_bins: int = 120,
        action_dim: int = 4,
        latent_dim: int = 128,
        horizon: int = 10,
        qant_backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        if dynamics not in {"vanilla", "koopman", "direct"}:
            raise ValueError(f"Unsupported PLUME dynamics: {dynamics}")
        if objective not in {"mse", "jepa"}:
            raise ValueError(f"Unsupported PLUME objective: {objective}")

        self.dynamics = dynamics
        self.objective = objective
        self.profile_bins = profile_bins
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.qant_backend = qant_backend

        if dynamics == "direct":
            self.plume_model = TokaMarkDirectWindowJEPA(
                profile_bins=profile_bins,
                action_dim=action_dim,
                latent_dim=latent_dim,
                horizon=horizon,
                qant_backend=qant_backend,
            )
        else:
            self.plume_model = TokaMarkProfileJEPA(
                model_type=dynamics,
                profile_bins=profile_bins,
                action_dim=action_dim,
                latent_dim=latent_dim,
                qant_backend=qant_backend,
            )

    def set_qant_backend(self, backend: QANTBackend) -> None:
        self.qant_backend = backend
        self.plume_model.set_qant_backend(backend)

    def _profile_sequence(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim < 3 or tensor.shape[-1] != self.profile_bins:
            raise ValueError(
                f"{name} must end in {self.profile_bins} profile bins; got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.ndim == 3:
            return tensor
        reduction_dims = tuple(range(2, tensor.ndim - 1))
        return tensor.mean(dim=reduction_dims) if reduction_dims else tensor

    @staticmethod
    def _actuator_sequence(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim < 2:
            raise ValueError(f"Actuator tensor must be batched; got {tensor.shape}")
        if tensor.ndim == 2:
            return tensor
        # ModelTransform_2 groups the 4 kHz actuator trace into ten 5 ms
        # windows. Selecting the first sample of each window reproduces
        # PLUME's original actions[20::20] Task 3-1 downsampling rule.
        return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)[..., 0]

    def _extract_inputs(
        self, inputs: Sequence[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(inputs) < 2 + self.action_dim:
            raise ValueError(
                f"Task 3-1 requires two profiles and {self.action_dim} future "
                f"actuators; received {len(inputs)} branches"
            )
        profiles_te = self._profile_sequence(inputs[0], "T_e")
        profiles_ne = self._profile_sequence(inputs[1], "n_e")
        action_branches = inputs[-self.action_dim :]
        actions = torch.stack(
            [self._actuator_sequence(branch) for branch in action_branches], dim=-1
        )
        if actions.shape[1] != self.horizon:
            raise ValueError(
                f"Expected {self.horizon} future action steps, got {actions.shape[1]}"
            )
        return profiles_te, profiles_ne, actions

    def _predict_latents(
        self, initial_latent: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        if self.dynamics == "direct":
            return self.plume_model.predict_window(initial_latent, actions)

        predicted = []
        latent = initial_latent
        for step in range(actions.shape[1]):
            latent = self.plume_model.latent_step(latent, actions[:, step])
            predicted.append(latent)
        return torch.stack(predicted, dim=1)

    def forward(
        self,
        *inputs: torch.Tensor,
        targets: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, object]:
        profiles_te, profiles_ne, actions = self._extract_inputs(inputs)
        observed_latents = self.plume_model.encode(profiles_te, profiles_ne)
        initial_latent = observed_latents[:, -1]
        predicted_latents = self._predict_latents(initial_latent, actions)
        decoded = self.plume_model.decoder(predicted_latents)

        output: dict[str, object] = {
            "predictions": [
                decoded["profiles_te"],
                decoded["profiles_ne"],
            ],
            "aux_losses": {},
        }

        if self.objective == "jepa" and targets is not None:
            if len(targets) != 2:
                raise ValueError(
                    f"Task 3-1 requires two target profiles; got {len(targets)}"
                )
            target_te = torch.nan_to_num(targets[0])
            target_ne = torch.nan_to_num(targets[1])
            target_latents = self.plume_model.encode(target_te, target_ne).detach()

            reconstruction = self.plume_model.decoder(observed_latents)
            reconstruction_loss = (
                F.mse_loss(reconstruction["profiles_te"], profiles_te)
                + F.mse_loss(reconstruction["profiles_ne"], profiles_ne)
            ) / 2.0
            output["aux_losses"] = {
                "latent": F.mse_loss(predicted_latents, target_latents),
                "reconstruction": reconstruction_loss,
            }

        return output
