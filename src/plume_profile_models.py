"""PLUME profile JEPA models adapted for the TokaMark Task 3-1 benchmark.

The encoder, decoder, vanilla predictor, and Koopman transition are the PLUME
implementations.  ``TokaMarkDirectWindowPredictor`` is the only new model
component; it is intentionally named as a direct-window extension.
"""

from __future__ import annotations

import torch
from torch import nn

from src.plume_qant_layers import (
    QANTBackend,
    QElementwiseMul,
    QFourier,
    QLinear,
    QSigmoid,
    QTanh,
    set_qant_backend,
)


class TokaMarkProfileEncoder(nn.Module):
    def __init__(self, profile_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            QFourier(profile_dim, latent_dim, noise_std=None),
            QLinear(latent_dim, latent_dim),
        )

    def forward(
        self, profiles_te: torch.Tensor, profiles_ne: torch.Tensor
    ) -> torch.Tensor:
        profiles = torch.cat([profiles_te, profiles_ne], dim=-1)
        batch_size, steps, feature_dim = profiles.shape
        latent = self.network(profiles.reshape(batch_size * steps, feature_dim))
        return latent.reshape(batch_size, steps, -1)


class TokaMarkProfileDecoder(nn.Module):
    def __init__(self, latent_dim: int, profile_dim: int) -> None:
        super().__init__()
        self.profile_dim = profile_dim
        self.network = nn.Sequential(
            QFourier(latent_dim, latent_dim, noise_std=None),
            QLinear(latent_dim, profile_dim),
        )

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, steps, latent_dim = latent.shape
        profiles = self.network(latent.reshape(batch_size * steps, latent_dim))
        profiles = profiles.reshape(batch_size, steps, self.profile_dim)
        split = self.profile_dim // 2
        return {
            "profiles_te": profiles[..., :split],
            "profiles_ne": profiles[..., split:],
        }


class TokaMarkVanillaPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.backbone = QFourier(latent_dim + action_dim, latent_dim, noise_std=None)
        self.gate = nn.Sequential(QLinear(latent_dim, latent_dim), QSigmoid())
        self.candidate = nn.Sequential(QLinear(latent_dim, latent_dim), QTanh())
        self.multiply = QElementwiseMul()

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        leading_shape = latent.shape[:-1]
        features = torch.cat([latent, actions], dim=-1).reshape(
            -1, self.latent_dim + self.action_dim
        )
        hidden = self.backbone(features)
        gate = self.gate(hidden)
        candidate = self.candidate(hidden)
        previous = latent.reshape(-1, self.latent_dim)
        predicted = self.multiply(gate, candidate) + self.multiply(1.0 - gate, previous)
        return predicted.reshape(*leading_shape, self.latent_dim)


class TokaMarkProfileJEPA(nn.Module):
    """The PLUME TokaMark profile model with vanilla or Koopman dynamics."""

    def __init__(
        self,
        model_type: str,
        profile_bins: int = 120,
        action_dim: int = 4,
        latent_dim: int = 128,
        qant_backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        if model_type not in {"koopman", "vanilla"}:
            raise ValueError(f"Unsupported TokaMark profile model type: {model_type}")

        self.model_type = model_type
        self.profile_bins = profile_bins
        self.profile_dim = profile_bins * 2
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.scalar_dim = 0
        self.device_dim = 0
        self.control_dim = action_dim

        self.encoder = TokaMarkProfileEncoder(self.profile_dim, latent_dim)
        self.decoder = TokaMarkProfileDecoder(latent_dim, self.profile_dim)
        if model_type == "koopman":
            self.koopman_k = QLinear(latent_dim, latent_dim, bias=False)
            self.koopman_b = QLinear(action_dim, latent_dim, bias=False)
            nn.init.eye_(self.koopman_k.weight)
            nn.init.zeros_(self.koopman_b.weight)
        else:
            self.predictor = TokaMarkVanillaPredictor(latent_dim, action_dim)
        self.set_qant_backend(qant_backend)

    def set_qant_backend(self, backend: QANTBackend) -> None:
        set_qant_backend(self, backend)

    def encode(
        self, profiles_te: torch.Tensor, profiles_ne: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(profiles_te, profiles_ne)

    def latent_step(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if self.model_type == "koopman":
            return self.koopman_k(latent) + self.koopman_b(actions)
        return self.predictor(latent, actions)

    def forward(
        self,
        profiles_te: torch.Tensor,
        profiles_ne: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        latent = self.encode(profiles_te, profiles_ne)
        if actions.shape[1] != latent.shape[1] - 1:
            raise ValueError(
                f"Expected one action per transition, got {actions.shape[1]} "
                f"actions for {latent.shape[1]} states"
            )
        reconstruction = self.decoder(latent)
        latent_next = latent[:, 1:]
        latent_next_pred = self.latent_step(latent[:, :-1], actions)
        prediction = self.decoder(latent_next_pred)
        return {
            "z": latent,
            "z_next_true": latent_next,
            "z_next_pred": latent_next_pred,
            "recon_profiles_te": reconstruction["profiles_te"],
            "recon_profiles_ne": reconstruction["profiles_ne"],
            "pred_profiles_te": prediction["profiles_te"],
            "pred_profiles_ne": prediction["profiles_ne"],
        }


class TokaMarkDirectWindowPredictor(nn.Module):
    """Single-pass extension that predicts the complete future latent window."""

    def __init__(self, latent_dim: int, action_dim: int, horizon: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.horizon = horizon
        context_dim = latent_dim + horizon * action_dim
        self.network = nn.Sequential(
            QFourier(context_dim, latent_dim, noise_std=None),
            QLinear(latent_dim, horizon * latent_dim),
        )

    def forward(
        self, initial_latent: torch.Tensor, future_actions: torch.Tensor
    ) -> torch.Tensor:
        if future_actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(
                "Expected future actions with shape "
                f"[batch, {self.horizon}, {self.action_dim}], got "
                f"{tuple(future_actions.shape)}"
            )
        context = torch.cat(
            [initial_latent, future_actions.flatten(start_dim=1)], dim=-1
        )
        residual = self.network(context).reshape(
            initial_latent.shape[0], self.horizon, self.latent_dim
        )
        return initial_latent.unsqueeze(1) + residual


class TokaMarkDirectWindowJEPA(nn.Module):
    """PLUME encoder/decoder plus the direct-window predictor extension."""

    def __init__(
        self,
        profile_bins: int = 120,
        action_dim: int = 4,
        latent_dim: int = 128,
        horizon: int = 10,
        qant_backend: QANTBackend = "torch",
    ) -> None:
        super().__init__()
        self.model_type = "direct"
        self.profile_bins = profile_bins
        self.profile_dim = profile_bins * 2
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.horizon = horizon
        self.encoder = TokaMarkProfileEncoder(self.profile_dim, latent_dim)
        self.decoder = TokaMarkProfileDecoder(latent_dim, self.profile_dim)
        self.predictor = TokaMarkDirectWindowPredictor(latent_dim, action_dim, horizon)
        self.set_qant_backend(qant_backend)

    def set_qant_backend(self, backend: QANTBackend) -> None:
        set_qant_backend(self, backend)

    def encode(
        self, profiles_te: torch.Tensor, profiles_ne: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(profiles_te, profiles_ne)

    def predict_window(
        self, initial_latent: torch.Tensor, future_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.predictor(initial_latent, future_actions)
