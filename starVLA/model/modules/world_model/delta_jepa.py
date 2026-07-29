# Copyright 2025 starVLA community. All rights reserved.
# Delta-JEPA utilities: control-aware latent displacement + candidate action encoding.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_vjepa_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Mean-pool V-JEPA patch tokens and L2-normalize. tokens: [B, N, D] -> [B, D]."""
    pooled = tokens.mean(dim=1)
    return F.normalize(pooled, dim=-1)


def pool_last_temporal_frame(tokens: torch.Tensor, tokens_per_frame: int) -> torch.Tensor:
    """Pool the last temporal slice of V-JEPA tokens."""
    return pool_vjepa_tokens(tokens[:, -tokens_per_frame:, :])


def cosine_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Cosine distance in [0, 2]. z1, z2: [B, D] (normalized or not)."""
    z1_n = F.normalize(z1, dim=-1)
    z2_n = F.normalize(z2, dim=-1)
    return 1.0 - (z1_n * z2_n).sum(dim=-1)


def latent_progress_score(z_current: torch.Tensor, z_predicted: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Predicted progress toward goal: d(current, goal) - d(predicted, goal). Higher is better."""
    return cosine_distance(z_current, z_goal) - cosine_distance(z_predicted, z_goal)


class CandidateActionEncoder(nn.Module):
    """Encode continuous action chunks into predictor conditioning tokens."""

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_output_tokens: int,
    ) -> None:
        super().__init__()
        self.num_output_tokens = num_output_tokens
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [B, T, action_dim]
        Returns:
            action tokens: [B, num_output_tokens, output_dim]
        """
        feat = self.action_mlp(actions)
        if feat.shape[1] != self.num_output_tokens:
            feat = F.interpolate(
                feat.transpose(1, 2),
                size=self.num_output_tokens,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return self.token_proj(feat)


class LatentInverseDynamics(nn.Module):
    """Decode action from latent displacement and proprioception (Delta-JEPA control-aware head)."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, delta_z: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_z: [B, latent_dim]
            state: [B, state_dim]
        Returns:
            predicted action: [B, action_dim]
        """
        if state.dim() == 3:
            state = state.squeeze(1)
        return self.net(torch.cat([delta_z, state], dim=-1))


class ActionDynamicsPrior(nn.Module):
    """Predict the next normalized action from the preceding action sequence."""

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.input_proj = nn.Sequential(
            nn.Linear(action_dim + state_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal_model = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )

    def _expand_state(self, state: torch.Tensor | None, actions: torch.Tensor) -> torch.Tensor:
        batch_size, horizon = actions.shape[:2]
        if state is None:
            return actions.new_zeros(batch_size, horizon, self.state_dim)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.shape[1] == 1:
            state = state.expand(-1, horizon, -1)
        elif state.shape[1] != horizon:
            state = F.interpolate(
                state.transpose(1, 2),
                size=horizon,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return state.to(device=actions.device, dtype=actions.dtype)

    def predict_next(self, actions: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Return predictions for actions[:, 1:] from actions[:, :-1]."""
        if actions.shape[1] < 2:
            return actions[:, :0]
        prior_param = next(self.parameters())
        action_inputs = actions[:, :-1].to(
            device=prior_param.device,
            dtype=prior_param.dtype,
        )
        state_inputs = self._expand_state(state, action_inputs)
        hidden = self.input_proj(torch.cat([action_inputs, state_inputs], dim=-1))
        hidden, _ = self.temporal_model(hidden)
        return self.action_head(hidden)

    def loss(self, actions: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        if actions.shape[1] < 2:
            return actions.new_zeros(())
        prediction = self.predict_next(actions, state)
        target = actions[:, 1:].to(device=prediction.device, dtype=prediction.dtype)
        return F.mse_loss(prediction, target)

    def energy(self, actions: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """Per-sample action-prior error used to rank candidate action chunks."""
        if actions.shape[1] < 2:
            return actions.new_zeros(actions.shape[0])
        prediction = self.predict_next(actions, state)
        target = actions[:, 1:].to(device=prediction.device, dtype=prediction.dtype)
        error = (prediction - target).square()
        return error.mean(dim=(1, 2))


class GoalConditionedActionProposal(nn.Module):
    """Amortized action proposal conditioned on current and goal JEPA states."""

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_horizon = action_horizon
        self.context_net = nn.Sequential(
            nn.Linear(latent_dim * 3 + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.time_embedding = nn.Parameter(
            torch.empty(action_horizon, hidden_dim)
        )
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        proposal_param = next(self.parameters())
        z_current = z_current.to(
            device=proposal_param.device,
            dtype=proposal_param.dtype,
        )
        z_goal = z_goal.to(device=proposal_param.device, dtype=proposal_param.dtype)
        if state is None:
            state = z_current.new_zeros(z_current.shape[0], self.state_dim)
        else:
            if state.dim() == 3:
                state = state[:, -1]
            state = state.to(device=proposal_param.device, dtype=proposal_param.dtype)

        context = torch.cat(
            [z_current, z_goal, z_goal - z_current, state],
            dim=-1,
        )
        context = self.context_net(context)
        temporal_features = context.unsqueeze(1) + self.time_embedding.unsqueeze(0)
        return self.action_head(temporal_features)

    def loss(
        self,
        z_current: torch.Tensor,
        z_goal: torch.Tensor,
        state: torch.Tensor | None,
        target_actions: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self(z_current, z_goal, state)
        target_actions = target_actions.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        return F.mse_loss(prediction, target_actions)
