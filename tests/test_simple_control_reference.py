"""SIMPLE deployment math matches the XXY adapter, including boundary behavior.

Reference: origin/vlajepa-xxy at
0d7d15f39c524563310fe7196d85d863494aafb3,
deployment/model_server/simple_g1_adapter.py:18-54.
The adapter was last changed by 8f8e52d59cfa5ef9a953ab8b5c6c5bb5c36f948c.
The source project's pyproject.toml declares the MIT License. These small
NumPy reference functions preserve its formulas without importing its model
loader or depending on a Git remote/ref being available at test time.
"""

from copy import deepcopy

import numpy as np
import pytest

from deployment.model_server.control_transforms import normalize_simple_state
from starVLA.model.framework.base_framework import baseframework


def _xxy_normalize_state(raw_state, state_stats):
    state_min = np.array(state_stats["min"], dtype=np.float32)
    state_max = np.array(state_stats["max"], dtype=np.float32)
    denom = state_max - state_min
    denom = np.where(denom < 1e-8, 1.0, denom)
    return (2.0 * (raw_state - state_min) / denom - 1.0).astype(np.float32)


def _xxy_unnormalize_actions(raw_actions, action_stats):
    action_min = np.array(action_stats["min"], dtype=np.float32)
    action_max = np.array(action_stats["max"], dtype=np.float32)
    action_mean = np.array(action_stats["mean"], dtype=np.float32)
    action_std = np.array(action_stats["std"], dtype=np.float32)
    result = np.empty_like(raw_actions, dtype=np.float32)
    clipped = np.clip(raw_actions[..., :32], -1.0, 1.0)
    result[..., :32] = 0.5 * (clipped + 1.0) * (action_max[:32] - action_min[:32]) + action_min[:32]
    result[..., 32:36] = raw_actions[..., 32:36] * action_std[32:36] + action_mean[32:36]
    return result


def _action_statistics():
    low = np.linspace(-2.0, 1.0, 36, dtype=np.float32)
    high = low + np.linspace(0.5, 3.0, 36, dtype=np.float32)
    low[5] = high[5] = 0.125  # A constant joint still restores its measured value.
    std = np.linspace(0.2, 1.2, 36, dtype=np.float32)
    std[-1] = 0.0
    return {
        "min": low.tolist(), "max": high.tolist(),
        "mean": np.linspace(-0.75, 1.25, 36, dtype=np.float32).tolist(),
        "std": std.tolist(),
    }


@pytest.mark.parametrize("tag", ["g1_handover", "g1_pick_between_tables"])
@pytest.mark.parametrize("shape", [(36,), (4, 36), (2, 3, 36)])
def test_simple_action_decoding_matches_xxy_clipping_and_continuous_channels(tag, shape):
    stats = _action_statistics()
    original_stats = deepcopy(stats)
    stored_stats = {tag: {"action": stats}}
    action_stats = baseframework.get_action_stats(unnorm_key=tag, norm_stats=stored_stats)
    assert action_stats["normalization_modes"] == ["min_max"] * 32 + ["mean_std"] * 4

    normalized = np.linspace(-3.0, 3.0, np.prod(shape), dtype=np.float32).reshape(shape)
    normalized[..., 0] = -2.5
    normalized[..., 6] = 0.25  # This is a finger joint, with no binary threshold.
    normalized[..., 31] = 2.5
    normalized[..., 32:36] = [-4.25, 2.0, 5.5, -6.0]
    original_actions = normalized.copy()

    expected = _xxy_unnormalize_actions(normalized, stats)
    actual = baseframework.unnormalize_actions(normalized, action_stats)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[..., 0], np.float32(stats["min"][0]))
    np.testing.assert_array_equal(actual[..., 31], np.float32(stats["max"][31]))
    # torso_vx/vy/vyaw/target_yaw retain the unbounded mean/std semantics.
    clipped_tail = np.clip(normalized[..., 32:36], -1, 1) * np.asarray(stats["std"][32:36]) + stats["mean"][32:36]
    assert not np.allclose(actual[..., 32:36], clipped_tail)
    assert np.all((actual[..., 6] != 0.0) & (actual[..., 6] != 1.0))
    np.testing.assert_array_equal(normalized, original_actions)
    assert stats == original_stats


@pytest.mark.parametrize("shape", [(32,), (1, 1, 32), (2, 3, 32)])
def test_simple_state_normalization_matches_xxy_at_constant_and_epsilon_boundaries(shape):
    low = np.zeros(32, dtype=np.float32)
    high = np.ones(32, dtype=np.float32)
    low[0] = high[0] = 0.25
    epsilon = np.float32(1e-8)
    high[1] = np.nextafter(epsilon, np.float32(0))
    high[2] = epsilon
    high[3] = np.nextafter(epsilon, np.float32(np.inf))
    stats = {"min": low.tolist(), "max": high.tolist()}
    original_stats = deepcopy(stats)
    state = np.broadcast_to(np.linspace(-0.5, 1.5, 32, dtype=np.float32), shape).copy()
    state[..., 0] = low[0]
    state[..., 1:4] = high[1:4]
    state[..., 4] = 2.0
    state[..., 5] = -1.0
    original_state = state.copy()

    expected = _xxy_normalize_state(state, stats)
    actual = normalize_simple_state(state, stats)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    assert actual.shape == shape
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[..., 0], -1.0)
    np.testing.assert_array_equal(actual[..., 1], -1.0)
    np.testing.assert_array_equal(actual[..., 2:4], 1.0)
    # The reference applies no clipping to normalized observations.
    np.testing.assert_array_equal(actual[..., 4], 3.0)
    np.testing.assert_array_equal(actual[..., 5], -3.0)
    np.testing.assert_array_equal(state, original_state)
    assert stats == original_stats
