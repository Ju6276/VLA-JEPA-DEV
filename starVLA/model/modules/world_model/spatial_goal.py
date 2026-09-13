"""Spatial goal prediction and task-conditioned reads of frozen JEPA features.

The supervision space is a fixed spatial grid. Learned attention only decides
which original features to read; it cannot rotate its values into an easier
prediction or scoring space. No boxes, segmentation masks, or object IDs are
needed by these modules.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _square_side(token_count: int) -> int:
    side = math.isqrt(token_count)
    if token_count < 1 or side * side != token_count:
        raise ValueError(f"Expected one square spatial grid, got {token_count} tokens")
    return side


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def fixed_spatial_grid(tokens: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Pool one raw patch grid [B, P, D] to a fixed [B, grid_size**2, D] grid.

    Pooling happens before normalization, preserving the frozen encoder's
    feature space. The caller must select a single temporal slice and camera.
    """
    _positive_int("grid_size", grid_size)
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [B, P, D]")
    side = _square_side(tokens.shape[1])
    if grid_size > side:
        raise ValueError(f"grid_size {grid_size} exceeds the patch grid size {side}")
    spatial = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
    return F.adaptive_avg_pool2d(spatial, (grid_size, grid_size)).flatten(2).transpose(1, 2)


def _coordinates(token_count: int, reference: torch.Tensor) -> torch.Tensor:
    """Fixed, row-major cell positions shared by current, goal, and future grids."""
    side = _square_side(token_count)
    locations = (torch.arange(side, device=reference.device, dtype=torch.float32) + 0.5) / side
    y, x = torch.meshgrid(locations, locations, indexing="ij")
    positions = torch.stack(
        [x, y, torch.sin(math.pi * x), torch.cos(math.pi * x),
         torch.sin(math.pi * y), torch.cos(math.pi * y)], dim=-1,
    )
    return positions.reshape(token_count, 6).to(dtype=reference.dtype)


def _task_and_state(
    task_tokens: torch.Tensor,
    state: torch.Tensor | None,
    reference: torch.Tensor,
    task_dim: int,
    state_dim: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if task_tokens.ndim != 3 or task_tokens.shape[0] != batch_size or task_tokens.shape[2] != task_dim:
        raise ValueError(f"task_tokens must have shape [B, T, {task_dim}]")
    if task_tokens.shape[1] == 0:
        raise ValueError("task_tokens must contain at least one token")
    task = task_tokens.to(device=reference.device, dtype=reference.dtype)
    if state is None:
        proprioception = reference.new_zeros(batch_size, state_dim)
    else:
        if state.ndim == 3:
            if state.shape[1] == 0:
                raise ValueError("state history must contain at least one step")
            state = state[:, -1]
        if state.ndim != 2 or state.shape != (batch_size, state_dim):
            raise ValueError(f"state must have shape [B, {state_dim}] or [B, T, {state_dim}]")
        proprioception = state.to(device=reference.device, dtype=reference.dtype)
    return task, proprioception


class SpatialGoalPredictor(nn.Module):
    """Predict raw future grid features from current deployable observations.

    Each future spatial cell has its own query. Current and historical cells
    carry their fixed image position and elapsed observation age. Historical
    inputs are real past observations; their ages are positive seconds.
    """

    def __init__(
        self,
        latent_dim: int,
        task_dim: int,
        state_dim: int,
        grid_size: int = 8,
        hidden_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("task_dim", task_dim),
                            ("grid_size", grid_size), ("hidden_dim", hidden_dim), ("num_heads", num_heads)):
            _positive_int(name, value)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if state_dim < 0:
            raise ValueError("state_dim must be nonnegative")
        self.latent_dim = latent_dim
        self.task_dim = task_dim
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.feature_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim))
        self.position_projection = nn.Linear(6, hidden_dim)
        self.age_projection = nn.Sequential(nn.Linear(2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.task_projection = nn.Linear(task_dim, hidden_dim)
        self.condition_projection = nn.Linear(task_dim + state_dim, hidden_dim)
        self.queries = nn.Parameter(torch.empty(grid_size * grid_size, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.0, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, latent_dim)

    def _age_embedding(self, ages: torch.Tensor) -> torch.Tensor:
        return self.age_projection(torch.stack([torch.log1p(ages), 1.0 / (1.0 + ages)], dim=-1))

    def forward(
        self,
        current_grid: torch.Tensor,
        task_tokens: torch.Tensor,
        state: torch.Tensor | None,
        history_grid: torch.Tensor | None = None,
        history_valid: torch.Tensor | None = None,
        history_ages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        count = self.grid_size * self.grid_size
        if current_grid.ndim != 3 or current_grid.shape[1:] != (count, self.latent_dim):
            raise ValueError(f"current_grid must have shape [B, {count}, {self.latent_dim}]")
        parameter = self.queries
        current = current_grid.to(device=parameter.device, dtype=parameter.dtype)
        batch = current.shape[0]
        task, proprioception = _task_and_state(task_tokens, state, parameter, self.task_dim, self.state_dim, batch)
        position = self.position_projection(_coordinates(count, current))
        condition = self.condition_projection(torch.cat([task.mean(dim=1), proprioception], dim=-1))
        query = self.queries.unsqueeze(0) + position.unsqueeze(0) + condition.unsqueeze(1)
        current_memory = self.feature_projection(current) + position.unsqueeze(0)
        current_memory = current_memory + self._age_embedding(current.new_zeros(batch, 1))
        memory_parts = [current_memory, self.task_projection(task), condition.unsqueeze(1)]
        padding_parts = [torch.zeros(batch, count + task.shape[1] + 1, device=current.device, dtype=torch.bool)]

        if history_grid is None:
            if history_valid is not None or history_ages is not None:
                raise ValueError("history_valid/history_ages require history_grid")
        else:
            if history_grid.ndim != 4 or history_grid.shape[0] != batch or history_grid.shape[2:] != (count, self.latent_dim):
                raise ValueError(f"history_grid must have shape [B, M, {count}, {self.latent_dim}]")
            history_count = history_grid.shape[1]
            if history_count:
                if history_valid is None:
                    valid = torch.ones(batch, history_count, device=current.device, dtype=torch.bool)
                else:
                    if history_valid.shape != (batch, history_count) or history_valid.dtype != torch.bool:
                        raise ValueError("history_valid must be a boolean [B, M] tensor")
                    valid = history_valid.to(device=current.device)
                if history_ages is None or history_ages.shape != (batch, history_count):
                    raise ValueError("history_ages must provide elapsed seconds with shape [B, M]")
                ages = history_ages.to(device=current.device, dtype=current.dtype)
                if torch.any(valid & (~torch.isfinite(ages) | (ages <= 0))):
                    raise ValueError("Valid historical observations must have finite positive ages")
                ages = torch.where(valid, ages, torch.zeros_like(ages))
                history = history_grid.to(device=current.device, dtype=current.dtype)
                # A key padding mask alone cannot prevent masked NaNs from
                # contaminating matrix products. Sanitize before projection.
                history = torch.where(valid[:, :, None, None], history, torch.zeros_like(history))
                history_memory = self.feature_projection(history) + position[None, None]
                history_memory = history_memory + self._age_embedding(ages).unsqueeze(2)
                memory_parts.append(history_memory.flatten(1, 2))
                padding_parts.append((~valid).unsqueeze(-1).expand(-1, -1, count).flatten(1, 2))

        memory = self.memory_norm(torch.cat(memory_parts, dim=1))
        padding_mask = torch.cat(padding_parts, dim=1)
        attended, _ = self.attention(self.query_norm(query), memory, memory, key_padding_mask=padding_mask, need_weights=False)
        hidden = query + attended
        hidden = hidden + self.feed_forward(self.output_norm(hidden))
        # Prediction remains in the raw frozen JEPA feature space. The current
        # grid is a residual reference, not an assumption of object alignment.
        return current + self.output_projection(self.output_norm(hidden))


class TaskSpatialReader(nn.Module):
    """Read original spatial features using task/state-conditioned queries.

    Reuse the exact output of make_query() for current, goal, and every candidate
    future. Attention weights can move spatially; the task queries stay shared.
    There is deliberately no learned value projection.
    """

    def __init__(
        self,
        latent_dim: int,
        task_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
        num_queries: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("task_dim", task_dim), ("hidden_dim", hidden_dim),
                            ("num_queries", num_queries), ("num_heads", num_heads)):
            _positive_int(name, value)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if state_dim < 0:
            raise ValueError("state_dim must be nonnegative")
        self.latent_dim, self.task_dim, self.state_dim = latent_dim, task_dim, state_dim
        self.hidden_dim, self.num_queries, self.num_heads = hidden_dim, num_queries, num_heads
        self.query_embedding = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.query_embedding, std=0.02)
        self.condition_projection = nn.Sequential(nn.Linear(task_dim + state_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, hidden_dim))
        self.position_projection = nn.Linear(6, hidden_dim)

    def make_query(self, task_tokens: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        task, proprioception = _task_and_state(
            task_tokens, state, self.query_embedding, self.task_dim, self.state_dim, task_tokens.shape[0],
        )
        condition = self.condition_projection(torch.cat([task.mean(dim=1), proprioception], dim=-1))
        return self.query_norm(self.query_embedding.unsqueeze(0) + condition.unsqueeze(1))

    def forward(self, grid: torch.Tensor, query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if grid.ndim != 3 or grid.shape[2] != self.latent_dim:
            raise ValueError(f"grid must have shape [B, P, {self.latent_dim}]")
        _square_side(grid.shape[1])
        batch, count, _ = grid.shape
        if query.shape != (batch, self.num_queries, self.hidden_dim):
            raise ValueError(f"query must have shape [B, {self.num_queries}, {self.hidden_dim}]")
        values = grid.to(device=self.query_embedding.device, dtype=self.query_embedding.dtype)
        query = query.to(device=values.device, dtype=values.dtype)
        keys = self.key_projection(values) + self.position_projection(_coordinates(count, values)).unsqueeze(0)
        head_dim = self.hidden_dim // self.num_heads
        q = self.query_projection(query).reshape(batch, self.num_queries, self.num_heads, head_dim).transpose(1, 2)
        k = keys.reshape(batch, count, self.num_heads, head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)).float() / math.sqrt(head_dim)
        attention = logits.softmax(dim=-1).mean(dim=1)
        local = torch.matmul(attention.to(dtype=values.dtype), values)
        local = F.normalize(local.float(), dim=-1).to(dtype=local.dtype)
        return local, attention


class SpatialActionAdapter(nn.Module):
    """Add a learned spatial residual to the existing full action proposal."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 256,
        num_queries: int = 4,
    ) -> None:
        super().__init__()
        for name, value in (("latent_dim", latent_dim), ("action_dim", action_dim),
                            ("action_horizon", action_horizon), ("hidden_dim", hidden_dim), ("num_queries", num_queries)):
            _positive_int(name, value)
        self.latent_dim, self.num_queries = latent_dim, num_queries
        self.action_horizon = action_horizon
        self.context = nn.Sequential(nn.Linear(3 * num_queries * latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.time_embedding = nn.Parameter(torch.empty(action_horizon, hidden_dim))
        nn.init.trunc_normal_(self.time_embedding, std=0.02)
        self.action_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))
        # Small but nonzero weights allow gradients into attention immediately.
        nn.init.normal_(self.action_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.action_head[-1].bias)

    def forward(self, current_local: torch.Tensor, goal_local: torch.Tensor) -> torch.Tensor:
        if current_local.ndim != 3 or current_local.shape[1:] != (self.num_queries, self.latent_dim):
            raise ValueError(f"current_local must have shape [B, {self.num_queries}, {self.latent_dim}]")
        if goal_local.shape != current_local.shape:
            raise ValueError("goal_local must have the same shape as current_local")
        current = current_local.to(device=self.time_embedding.device, dtype=self.time_embedding.dtype)
        goal = goal_local.to(device=current.device, dtype=current.dtype)
        context = torch.cat([current.flatten(1), goal.flatten(1), (goal - current).flatten(1)], dim=-1)
        hidden = self.context(context).unsqueeze(1) + self.time_embedding.unsqueeze(0)
        return self.action_head(hidden)
