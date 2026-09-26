"""Task 3-1 JEPA variants compatible with the canonical TokaMark baseline.

All models consume the exact tensor list produced by ``ModelTransform_1``,
``ModelTransform_2`` and ``model_collate_fn``.  Their public ``forward``
returns ``[pred_t_e, pred_n_e]`` so the upstream evaluator and
``MultiOutputMSELoss`` can be reused without a second data pipeline.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ProfileEncoder(nn.Module):
    def __init__(
        self,
        profile_bins: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.profile_dim = 2 * profile_bins
        self.encoder = MLP(
            self.profile_dim,
            latent_dim,
            hidden_dim,
            dropout,
        )

    def forward(
        self,
        profiles_te: torch.Tensor,
        profiles_ne: torch.Tensor,
    ) -> torch.Tensor:
        profiles = torch.cat([profiles_te, profiles_ne], dim=-1)
        batch_size, steps, feature_dim = profiles.shape
        latent = self.encoder(profiles.reshape(batch_size * steps, feature_dim))
        return latent.reshape(batch_size, steps, -1)


class ProfileDecoder(nn.Module):
    def __init__(
        self,
        profile_bins: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.profile_bins = profile_bins
        self.decoder = MLP(
            latent_dim,
            2 * profile_bins,
            hidden_dim,
            dropout,
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, steps, latent_dim = latent.shape
        profiles = self.decoder(latent.reshape(batch_size * steps, latent_dim))
        profiles = profiles.reshape(batch_size, steps, 2 * self.profile_bins)
        return (
            profiles[..., : self.profile_bins],
            profiles[..., self.profile_bins :],
        )


def _sequence_features(tensor: torch.Tensor) -> torch.Tensor:
    """Convert canonical ``[B, T, ...]`` tensors to ``[B, T, features]``."""
    if tensor.ndim < 3:
        raise ValueError(f"Expected [B, T, ...], received {tuple(tensor.shape)}")
    return tensor.reshape(tensor.shape[0], tensor.shape[1], -1)


class Task31JEPAInputAdapter(nn.Module):
    """Common profile/action input handling for Task 3-1."""

    def __init__(
        self,
        profile_bins: int,
        action_step_dim: int,
        latent_dim: int,
        hidden_dim: int,
        future_start_index: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.profile_bins = profile_bins
        self.action_step_dim = action_step_dim
        self.latent_dim = latent_dim
        self.future_start_index = future_start_index

        self.profile_encoder = ProfileEncoder(
            profile_bins=profile_bins,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.action_encoder = MLP(
            in_dim=action_step_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.context_fusion = MLP(
            in_dim=2 * latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.profile_decoder = ProfileDecoder(
            profile_bins=profile_bins,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def _prepare_inputs(
        self,
        inputs: Sequence[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(inputs) <= self.future_start_index:
            raise ValueError(
                f"Expected profile, past-action and future-action tensors; "
                f"received {len(inputs)} tensors"
            )

        profiles_te = _sequence_features(inputs[0])
        profiles_ne = _sequence_features(inputs[1])

        if profiles_te.shape[-1] != self.profile_bins:
            raise ValueError(
                f"Expected {self.profile_bins} Te bins, got {profiles_te.shape[-1]}"
            )
        if profiles_ne.shape[-1] != self.profile_bins:
            raise ValueError(
                f"Expected {self.profile_bins} ne bins, got {profiles_ne.shape[-1]}"
            )

        past_actions = torch.cat(
            [_sequence_features(x) for x in inputs[2 : self.future_start_index]],
            dim=-1,
        )
        future_actions = torch.cat(
            [_sequence_features(x) for x in inputs[self.future_start_index :]],
            dim=-1,
        )

        if past_actions.shape[-1] != self.action_step_dim:
            raise ValueError(
                f"Expected {self.action_step_dim} past-action features, "
                f"got {past_actions.shape[-1]}"
            )
        if future_actions.shape[-1] != self.action_step_dim:
            raise ValueError(
                f"Expected {self.action_step_dim} future-action features, "
                f"got {future_actions.shape[-1]}"
            )

        return profiles_te, profiles_ne, past_actions, future_actions

    def _encode_conditioning(
        self,
        profiles_te: torch.Tensor,
        profiles_ne: torch.Tensor,
        past_actions: torch.Tensor,
        future_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        profile_latent = self.profile_encoder(profiles_te, profiles_ne)[:, -1]

        batch_size, past_steps, _ = past_actions.shape
        past_latent = self.action_encoder(
            past_actions.reshape(batch_size * past_steps, -1)
        ).reshape(batch_size, past_steps, self.latent_dim)

        initial_latent = self.context_fusion(
            torch.cat([profile_latent, past_latent.mean(dim=1)], dim=-1)
        )

        batch_size, future_steps, _ = future_actions.shape
        future_action_latents = self.action_encoder(
            future_actions.reshape(batch_size * future_steps, -1)
        ).reshape(batch_size, future_steps, self.latent_dim)

        return initial_latent, future_action_latents

    def _predict_latents(self, *inputs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, *inputs: torch.Tensor) -> list[torch.Tensor]:
        predicted_latents = self._predict_latents(*inputs)
        pred_te, pred_ne = self.profile_decoder(predicted_latents)
        return [pred_te, pred_ne]


class TokaMarkJEPARollout(Task31JEPAInputAdapter):
    """Recursive latent rollout; output MSE is supplied by the baseline trainer."""

    def __init__(self, horizon: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.horizon = horizon
        self.transition = nn.GRUCell(self.latent_dim, self.latent_dim)

    def _predict_latents(self, *inputs: torch.Tensor) -> torch.Tensor:
        profiles_te, profiles_ne, past_actions, future_actions = self._prepare_inputs(inputs)
        latent, action_latents = self._encode_conditioning(
            profiles_te,
            profiles_ne,
            past_actions,
            future_actions,
        )

        if action_latents.shape[1] != self.horizon:
            raise ValueError(
                f"Expected horizon {self.horizon}, got {action_latents.shape[1]}"
            )

        predictions = []
        for step in range(self.horizon):
            latent = self.transition(action_latents[:, step], latent)
            predictions.append(latent)

        return torch.stack(predictions, dim=1)


class DirectWindowPredictor(nn.Module):
    """Predict all future latent states in one non-recurrent forward pass."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        horizon: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.horizon = horizon

        self.action_window_encoder = MLP(
            in_dim=horizon * latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, horizon, latent_dim) * 0.02
        )
        self.predictor = MLP(
            in_dim=4 * latent_dim,
            out_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        initial_latent: torch.Tensor,
        future_action_latents: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, horizon, latent_dim = future_action_latents.shape
        if horizon != self.horizon:
            raise ValueError(f"Expected horizon {self.horizon}, got {horizon}")

        action_window = self.action_window_encoder(
            future_action_latents.reshape(batch_size, horizon * latent_dim)
        )

        features = torch.cat(
            [
                initial_latent.unsqueeze(1).expand(-1, horizon, -1),
                future_action_latents,
                action_window.unsqueeze(1).expand(-1, horizon, -1),
                self.horizon_embedding.expand(batch_size, -1, -1),
            ],
            dim=-1,
        )

        predictions = self.predictor(features.reshape(batch_size * horizon, -1))
        return predictions.reshape(batch_size, horizon, latent_dim)


class TokaMarkDirectWindowJEPA(Task31JEPAInputAdapter):
    """Direct-window predictor with optional EMA target-encoder JEPA loss."""

    def __init__(
        self,
        horizon: int,
        hidden_dim: int,
        dropout: float,
        latent_loss_weight: float,
        target_ema: float,
        **kwargs,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, **kwargs)
        self.horizon = horizon
        self.latent_loss_weight = latent_loss_weight
        self.target_ema = target_ema
        self.direct_predictor = DirectWindowPredictor(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            dropout=dropout,
        )

        if latent_loss_weight > 0.0:
            self.target_encoder = copy.deepcopy(self.profile_encoder)
            self.target_encoder.requires_grad_(False)
        else:
            self.target_encoder = None

    def _predict_latents(self, *inputs: torch.Tensor) -> torch.Tensor:
        profiles_te, profiles_ne, past_actions, future_actions = self._prepare_inputs(inputs)
        initial_latent, future_action_latents = self._encode_conditioning(
            profiles_te,
            profiles_ne,
            past_actions,
            future_actions,
        )
        return self.direct_predictor(initial_latent, future_action_latents)

    def training_forward(
        self,
        inputs: Sequence[torch.Tensor],
        targets: Sequence[torch.Tensor],
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        predicted_latents = self._predict_latents(*inputs)
        pred_te, pred_ne = self.profile_decoder(predicted_latents)
        outputs = [pred_te, pred_ne]

        if self.target_encoder is None:
            return outputs, {}

        target_te = torch.nan_to_num(targets[0], nan=0.0)
        target_ne = torch.nan_to_num(targets[1], nan=0.0)
        with torch.no_grad():
            target_latents = self.target_encoder(target_te, target_ne)

        latent_loss = F.mse_loss(predicted_latents, target_latents.detach())
        return outputs, {"latent_mse": self.latent_loss_weight * latent_loss}

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        if self.target_encoder is None:
            return

        for online_parameter, target_parameter in zip(
            self.profile_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_parameter.data.mul_(self.target_ema)
            target_parameter.data.add_(
                online_parameter.data,
                alpha=1.0 - self.target_ema,
            )

