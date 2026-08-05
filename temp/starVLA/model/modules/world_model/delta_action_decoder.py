"""Multi-step action decoding from latent displacements.

This module implements the action-grounding objective used by the one-stage
privileged LaWAM variant.  It follows Delta-JEPA's multi-step LDAD design: a
single long-horizon latent displacement conditions a bank of temporal action
queries through adaptive layer normalization, and the decoder reconstructs the
complete continuous action chunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _DeltaAdaLNBlock(nn.Module):
    """Transformer block conditioned on a per-transition latent displacement."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        ffn_expansion: float = 2.0,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm_ffn = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        ffn_dim = int(hidden_dim * ffn_expansion)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )

        # Identity initialization keeps the auxiliary decoder stable at the
        # beginning of joint training.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, queries: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = self.modulation(
            condition
        ).chunk(6, dim=-1)

        attn_input = _modulate(self.norm_attn(queries), shift_attn, scale_attn)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        queries = queries + gate_attn.unsqueeze(1) * attn_output

        ffn_input = _modulate(self.norm_ffn(queries), shift_ffn, scale_ffn)
        return queries + gate_ffn.unsqueeze(1) * self.ffn(ffn_input)


class MultiStepDeltaActionDecoder(nn.Module):
    """Decode a full action chunk from one long-horizon latent displacement.

    Args:
        latent_dim: Dimension of the current/future visual latent.
        action_dim: Number of continuous controls per timestep.
        action_horizon: Number of timesteps in the reconstructed chunk.
        state_dim: Proprioceptive state dimension.  State conditioning is
            optional and disabled by default so the decoder cannot bypass the
            visual displacement through a proprioceptive shortcut.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.0,
        state_dim: int = 0,
        use_state: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}."
            )
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}.")

        self.action_horizon = int(action_horizon)
        self.use_state = bool(use_state)
        self.state_dim = int(state_dim)
        if self.use_state and self.state_dim <= 0:
            raise ValueError("state_dim must be positive when use_state=true.")

        condition_dim = int(latent_dim) + (self.state_dim if self.use_state else 0)
        self.condition_encoder = nn.Sequential(
            nn.LayerNorm(condition_dim),
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_queries = nn.Parameter(
            torch.empty(1, self.action_horizon, hidden_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _DeltaAdaLNBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, int(action_dim)),
        )

    def forward(
        self,
        delta_latent: torch.Tensor,
        state_0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if delta_latent.dim() != 2:
            raise ValueError(
                f"delta_latent must have shape [B,D], got {tuple(delta_latent.shape)}."
            )

        condition_parts = [delta_latent]
        if self.use_state:
            if state_0 is None:
                raise ValueError("state_0 is required when use_state=true.")
            if state_0.dim() == 3 and state_0.shape[1] == 1:
                state_0 = state_0[:, 0]
            if state_0.dim() != 2 or state_0.shape[-1] != self.state_dim:
                raise ValueError(
                    "state_0 must have shape "
                    f"[B,{self.state_dim}] or [B,1,{self.state_dim}], got {tuple(state_0.shape)}."
                )
            condition_parts.append(
                state_0.to(device=delta_latent.device, dtype=delta_latent.dtype)
            )

        condition = self.condition_encoder(torch.cat(condition_parts, dim=-1))
        queries = self.action_queries.expand(delta_latent.shape[0], -1, -1).to(
            device=delta_latent.device,
            dtype=condition.dtype,
        )
        # Inject the displacement directly as well as through AdaLN.  The
        # modulation gates are deliberately zero-initialized for stability;
        # this residual path keeps the decoder conditioned (and gives the
        # displacement encoder useful gradients) from the first update.
        queries = queries + condition.unsqueeze(1)
        for block in self.blocks:
            queries = block(queries, condition)
        return self.output_head(queries)
