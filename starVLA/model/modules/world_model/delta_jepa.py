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
