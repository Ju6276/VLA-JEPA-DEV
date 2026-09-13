"""Spatial feature invariants without pretrained models or region annotations."""

from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F

from starVLA.model.modules.world_model.spatial_goal import (
    SpatialActionAdapter,
    SpatialGoalPredictor,
    TaskSpatialReader,
    fixed_spatial_grid,
)


def _inputs():
    torch.manual_seed(17)
    return torch.randn(2, 4, 6), torch.randn(2, 3, 8), torch.randn(2, 2)


def _goal():
    return SpatialGoalPredictor(6, 8, 2, grid_size=2, hidden_dim=16, num_heads=4)


def _reader():
    return TaskSpatialReader(6, 8, 2, hidden_dim=16, num_queries=3, num_heads=4)


def test_fixed_grid_keeps_original_scale_and_spatial_order():
    # Every original 2x2 quadrant has a different mean and feature magnitude.
    patches = torch.arange(32.0).reshape(1, 16, 2).requires_grad_()
    actual = fixed_spatial_grid(patches, 2)
    image = patches.reshape(1, 4, 4, 2)
    expected = torch.stack([
        image[:, :2, :2].mean((1, 2)), image[:, :2, 2:].mean((1, 2)),
        image[:, 2:, :2].mean((1, 2)), image[:, 2:, 2:].mean((1, 2)),
    ], dim=1)
    torch.testing.assert_close(actual, expected)
    assert actual.norm(dim=-1).max() > 1
    actual.sum().backward()
    torch.testing.assert_close(patches.grad, torch.full_like(patches, 0.25))


@pytest.mark.parametrize("patches,grid_size", [(6, 2), (4, 3), (4, 0), (0, 1)])
def test_fixed_grid_rejects_ambiguous_or_upsampled_grids(patches, grid_size):
    with pytest.raises(ValueError):
        fixed_spatial_grid(torch.ones(1, patches, 3), grid_size)


def test_goal_preserves_raw_feature_space_and_has_task_state_gradients():
    current, task, state = _inputs()
    current = (current + 20).requires_grad_()
    task.requires_grad_()
    state.requires_grad_()
    model = _goal()
    prediction = model(current, task, state)
    assert prediction.shape == current.shape
    assert prediction.norm(dim=-1).min() > 10
    F.mse_loss(prediction, torch.randn_like(prediction)).backward()
    for value in (current, task, state):
        assert value.grad is not None and value.grad.abs().sum() > 0
    assert model.queries.grad.abs().sum() > 0
    assert model.position_projection.weight.grad.abs().sum() > 0


def test_invalid_history_is_ignored_even_with_nan_padding():
    current, task, state = _inputs()
    model = _goal().eval()
    plain = model(current, task, state)
    invalid = torch.full((2, 3, 4, 6), float("nan"))
    masked = model(
        current, task, state, invalid,
        torch.zeros(2, 3, dtype=torch.bool), torch.full((2, 3), float("nan")),
    )
    assert torch.isfinite(masked).all()
    torch.testing.assert_close(masked, plain, rtol=1e-5, atol=1e-6)


def test_history_receives_gradients_only_at_valid_observations():
    current, task, state = _inputs()
    model = _goal()
    history = torch.randn(2, 2, 4, 6, requires_grad=True)
    valid = torch.tensor([[True, False], [False, True]])
    ages = torch.tensor([[0.3, 0.0], [0.0, 0.6]])
    actual = model(current, task, state, history, valid, ages)
    no_history = model(current, task, state)
    assert not torch.allclose(actual, no_history)
    actual.square().mean().backward()
    assert history.grad[valid].abs().sum() > 0
    assert torch.count_nonzero(history.grad[~valid]) == 0


def test_history_age_changes_prediction_but_joint_permutation_does_not():
    current, task, state = _inputs()
    model = _goal().eval()
    history = torch.randn(2, 2, 4, 6)
    valid = torch.ones(2, 2, dtype=torch.bool)
    ages = torch.tensor([[0.2, 0.8], [0.3, 1.2]])
    original = model(current, task, state, history, valid, ages)
    changed_ages = model(current, task, state, history, valid, ages + 2)
    reordered = model(current, task, state, history.flip(1), valid.flip(1), ages.flip(1))
    assert not torch.allclose(original, changed_ages)
    torch.testing.assert_close(original, reordered, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("age", [0.0, -1.0, float("inf"), float("nan")])
def test_future_or_invalid_timestamps_cannot_enter_valid_history(age):
    current, task, state = _inputs()
    with pytest.raises(ValueError, match="positive ages"):
        _goal()(current, task, state, current[:, None], torch.ones(2, 1, dtype=torch.bool), torch.full((2, 1), age))


def test_history_requires_explicit_ages():
    current, task, state = _inputs()
    with pytest.raises(ValueError, match="history_ages"):
        _goal()(current, task, state, current[:, None])


def test_reader_returns_attention_weighted_original_features():
    grid, task, state = _inputs()
    reader = _reader()
    query = reader.make_query(task, state)
    local, attention = reader(grid, query)
    assert local.shape == (2, 3, 6)
    assert attention.shape == (2, 3, 4)
    torch.testing.assert_close(attention.sum(dim=-1), torch.ones(2, 3))
    expected = F.normalize(attention @ grid, dim=-1)
    torch.testing.assert_close(local, expected)
    torch.testing.assert_close(local.norm(dim=-1), torch.ones(2, 3))


def test_shared_query_is_reused_without_mutation_across_candidates():
    grid, task, state = _inputs()
    reader = _reader()
    query = reader.make_query(task, state)
    saved = query.detach().clone()
    first, first_weights = reader(grid, query)
    other, _ = reader(grid.flip(1), query)
    repeated, repeated_weights = reader(grid, query)
    torch.testing.assert_close(query, saved)
    torch.testing.assert_close(first, repeated)
    torch.testing.assert_close(first_weights, repeated_weights)
    # Image position matters even when the set of appearance features is equal.
    assert not torch.allclose(first, other)


def test_action_supervision_trains_reader_from_the_first_step():
    grid, task, state = _inputs()
    reader = _reader()
    adapter = SpatialActionAdapter(6, 5, 7, hidden_dim=16, num_queries=3)
    query = reader.make_query(task, state)
    current, _ = reader(grid.detach(), query)
    goal, _ = reader((grid + torch.randn_like(grid)).detach(), query)
    action = adapter(current, goal)
    assert action.shape == (2, 7, 5)
    assert action.abs().max() < 0.1
    F.mse_loss(action, torch.randn_like(action)).backward()
    for parameter in (reader.query_embedding, reader.query_projection.weight,
                      reader.key_projection[-1].weight, reader.position_projection.weight,
                      adapter.context[0].weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0


def test_state_history_uses_last_observation_and_none_means_zero():
    grid, task, state = _inputs()
    goal, reader = _goal(), _reader()
    states = torch.stack([state + 50, state], dim=1)
    torch.testing.assert_close(goal(grid, task, states), goal(grid, task, state))
    torch.testing.assert_close(reader.make_query(task, states), reader.make_query(task, state))
    torch.testing.assert_close(goal(grid, task, None), goal(grid, task, torch.zeros_like(state)))


@pytest.mark.parametrize("convert_parameters", [False, True])
def test_bfloat16_modules_and_autocast_are_finite_with_backward(convert_parameters):
    grid, task, state = _inputs()
    predictor, reader = _goal(), _reader()
    adapter = SpatialActionAdapter(6, 5, 7, hidden_dim=16, num_queries=3)
    if convert_parameters:
        predictor, reader, adapter = predictor.bfloat16(), reader.bfloat16(), adapter.bfloat16()
    context = nullcontext() if convert_parameters else torch.autocast("cpu", dtype=torch.bfloat16)
    # Some CPU oneDNN builds implement BF16 forward but not backward. Exercise
    # the same dtype/gradient path with PyTorch's portable CPU kernels instead.
    with torch.backends.mkldnn.flags(enabled=False):
        with context:
            goal_grid = predictor(grid, task, state, grid[:, None], torch.ones(2, 1, dtype=torch.bool), torch.ones(2, 1))
            query = reader.make_query(task, state)
            current, weights = reader(grid, query)
            goal, _ = reader(goal_grid, query)
            actions = adapter(current, goal)
            loss = goal_grid.float().square().mean() + actions.float().square().mean()
        assert torch.isfinite(weights).all() and torch.isfinite(actions).all()
        loss.backward()
    for model in (predictor, reader, adapter):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(grad).all() for grad in gradients)


def test_module_checkpoint_roundtrip():
    grid, task, state = _inputs()
    model = _goal().eval()
    restored = _goal().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(grid, task, state), model(grid, task, state))


@pytest.mark.parametrize("constructor", [_goal, _reader])
def test_bad_task_and_state_shapes_raise_clear_errors(constructor):
    grid, task, state = _inputs()
    module = constructor()
    invoke = (lambda t, s: module(grid, t, s)) if isinstance(module, SpatialGoalPredictor) else module.make_query
    with pytest.raises(ValueError, match="task_tokens"):
        invoke(task[:, :, :4], state)
    with pytest.raises(ValueError, match="state"):
        invoke(task, state[:, :1])
