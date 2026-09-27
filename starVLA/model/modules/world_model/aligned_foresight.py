"""Time-aligned endpoint conditioning and command-group prior controls."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from .delta_jepa import CandidateActionEncoder, ActionDynamicsPrior


def validate_horizons(h, K, H):
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (h, K, H)):
        raise ValueError('h, K, H must be explicit integers')
    if not (1 <= K <= h <= H and h >= 2):
        raise ValueError('Require 1 <= K <= h <= H and h >= 2')


def endpoint_video_pair(videos, horizon, repeat_frames):
    """Raw [B,V,h+1,H,W,C] -> two disjoint repeated-image batches."""
    if videos.ndim != 6 or videos.shape[2] != horizon + 1:
        raise ValueError('Endpoint training requires exactly h+1 timestamp-aligned frames')
    return (np.repeat(videos[:, :, :1], repeat_frames, axis=2),
            np.repeat(videos[:, :, horizon:horizon+1], repeat_frames, axis=2))


class EndpointActionEncoder(CandidateActionEncoder):
    """One token per evaluated command, with its order explicitly represented."""
    def forward(self, actions):
        if actions.shape[1] != self.num_output_tokens:
            raise ValueError('Endpoint condition must contain exactly h commands')
        tokens = self.token_proj(self.action_mlp(actions))
        position = torch.arange(tokens.shape[1], device=tokens.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, tokens.shape[2], 2, device=tokens.device,
                                          dtype=torch.float32) * (-math.log(10000.) / tokens.shape[2]))
        pos = torch.zeros(tokens.shape[1:], device=tokens.device, dtype=torch.float32)
        pos[:, 0::2] = torch.sin(position * frequency)
        pos[:, 1::2] = torch.cos(position * frequency[:pos[:, 1::2].shape[1]])
        return tokens + pos.to(tokens.dtype)[None]


def prior_parameter_count(action_dim, state_dim, width, layers):
    # Input linear, equal-width GRU layers, LayerNorm, linear output.
    return ((action_dim + state_dim) * width + width
            + layers * (6 * width * width + 6 * width)
            + 2 * width + action_dim * width + action_dim)


def matched_factorized_width(groups, state_dim, joint_width, layers):
    target = prior_parameter_count(sum(groups), state_dim, joint_width, layers)
    return min(range(1, joint_width + 1), key=lambda w: abs(
        sum(prior_parameter_count(d, state_dim, w, layers) for d in groups) - target))


class FactorizedActionPrior(nn.Module):
    """Same state; each group sees only its own within-candidate history."""
    def __init__(self, groups, state_dim, hidden_dim=512, num_layers=2):
        super().__init__()
        self.groups = tuple(groups)
        self.hidden_dim = matched_factorized_width(self.groups, state_dim, hidden_dim, num_layers)
        self.group_models = nn.ModuleList([
            ActionDynamicsPrior(d, state_dim, self.hidden_dim, num_layers) for d in self.groups])

    def predict_next(self, actions, state=None):
        if actions.shape[-1] != sum(self.groups):
            raise ValueError('Action dimensions do not match configured command groups')
        parts = actions.split(self.groups, dim=-1)
        return torch.cat([m.predict_next(a, state) for m, a in zip(self.group_models, parts)], dim=-1)


def prior_group_energies(prior, actions, state, groups):
    """Returns group mean squared residuals [B,G], using one joint forward pass."""
    if actions.shape[1] < 2 or actions.shape[-1] != sum(groups):
        raise ValueError('Need at least two commands and a complete group partition')
    prediction = prior.predict_next(actions, state)
    error = (prediction - actions[:, 1:].to(prediction)).square()
    return torch.stack([e.mean(dim=(1, 2)) for e in error.split(tuple(groups), dim=-1)], dim=-1)


def weighted_prior_energy(prior, actions, state, groups, weights):
    components = prior_group_energies(prior, actions, state, groups)
    w = components.new_tensor(weights)
    if len(weights) != len(groups) or not torch.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Command-group weights must be finite, nonnegative, and nonzero')
    return (components * (w / w.sum())).sum(dim=-1), components
