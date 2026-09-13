"""SONIC deployment parity against an independent, pinned client reference.

Source: https://github.com/BlackOtters/SonicStar
Branch: dev, commit: 82b395f67ec56805d465b9d1b6043c6d6cdfb06a
File: starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py
StarVLAPolicyAdapter._normalize_state and _unnormalize_actions, lines 159-179.
The source file SHA-256 is
7c148885cc995db3feb62ec7bbc5e2e0bb498b82a462888d2a0226673997ff9b.

Only the two pure NumPy methods are reproduced below. Tests require neither
the external checkout nor its robot, camera, network, or policy dependencies.
The reference deliberately differs from the training inverse: deployment
clips normalized actions, while state normalization remains unclipped.

Source license notices (SonicStar/LICENSE and SonicStar/starVLA/LICENSE):

MIT License
Copyright (c) 2026 Black Otter
Copyright (c) StarVLA Team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

Rebases are allowed for forks and feature branches. When rebasing from upstream
StarVLA, use descriptive commit messages, e.g., "chore: clone from StarVLA".
Preserve attribution: keep at least the two latest upstream StarVLA commits as
separate.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from copy import deepcopy

import numpy as np
import pytest

from deployment.model_server.control_transforms import normalize_sonic_state
from starVLA.dataloader.gr00t_lerobot.data_config import SonicLatentDataConfig
from starVLA.model.framework.base_framework import baseframework


class _SonicStarNumericReference:
    # These method bodies retain the source operations and their order.
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        state_min = np.asarray(self.state_stats["min"], dtype=np.float32)
        state_max = np.asarray(self.state_stats["max"], dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        mask = state_min != state_max
        normalized = np.zeros_like(state, dtype=np.float32)
        normalized[..., mask] = 2.0 * (state[..., mask] - state_min[mask]) / (
            state_max[mask] - state_min[mask]
        ) - 1.0
        return normalized

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        mask = self.action_stats.get("mask", np.ones_like(self.action_stats["min"], dtype=bool))
        action_high = np.asarray(self.action_stats["max"], dtype=np.float32)
        action_low = np.asarray(self.action_stats["min"], dtype=np.float32)
        normalized_actions = np.clip(normalized_actions, -1, 1)
        return np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        ).astype(np.float32)


def _statistics(dimensions):
    low = np.linspace(-0.8, 0.2, dimensions, dtype=np.float32)
    high = low + np.linspace(0.5, 1.5, dimensions, dtype=np.float32)
    return {"min": low.tolist(), "max": high.tolist()}


@pytest.mark.parametrize("shape", [(46,), (4, 46), (2, 3, 46)])
def test_sonic_state_matches_source_without_epsilon_or_clip(shape):
    stats = _statistics(46)
    stats["min"][9] = stats["max"][9] = 0.75
    stats["min"][10], stats["max"][10] = 0.0, 1e-9
    low, high = (np.asarray(stats[key], dtype=np.float32) for key in ("min", "max"))
    fractions = np.linspace(-0.5, 1.5, np.prod(shape), dtype=np.float32).reshape(shape)
    raw = low + fractions * (high - low)
    raw[..., 0] = high[0] + (high[0] - low[0])
    raw[..., 9] = 7.0  # A constant channel maps to zero even away from its observed value.
    raw[..., 10] = 1.5e-9  # An epsilon or a small-span fallback changes this result.
    before, stats_before = raw.copy(), deepcopy(stats)
    reference = _SonicStarNumericReference()
    reference.state_stats = stats

    actual = normalize_sonic_state(raw, stats)

    np.testing.assert_array_equal(actual, reference._normalize_state(raw))
    np.testing.assert_array_equal(actual[..., 0], np.float32(3.0))
    np.testing.assert_array_equal(actual[..., 9], np.float32(0.0))
    np.testing.assert_allclose(actual[..., 10], 2.0, rtol=1e-6, atol=0)
    np.testing.assert_array_equal(raw, before)
    assert stats == stats_before
    assert actual.shape == shape
    assert actual.dtype == np.float32


@pytest.mark.parametrize("tag", ["sonic_humanoid", "garbage", "new_embodiment"])
@pytest.mark.parametrize("shape", [(78,), (4, 78), (2, 3, 78)])
@pytest.mark.parametrize("mixed_mask", [False, True])
def test_sonic_action_matches_source_clipping_and_continuous_channels(tag, shape, mixed_mask):
    stats = _statistics(78)
    stats["min"][64] = stats["max"][64] = 0.35
    stats["min"][71] = stats["max"][71] = -0.2
    if mixed_mask:
        stats["mask"] = [True] * 78
        stats["mask"][65] = stats["mask"][72] = False
    normalized = np.linspace(-2.5, 2.5, np.prod(shape), dtype=np.float32).reshape(shape)
    normalized[..., 0], normalized[..., 1] = -1.0, 1.0
    normalized[..., 6] = 0.25
    normalized[..., 65], normalized[..., 72] = -1.6, 2.0
    before, stats_before = normalized.copy(), deepcopy(stats)
    reference = _SonicStarNumericReference()
    reference.action_stats = stats
    # The original SonicStar config saves NEW_EMBODIMENT statistics. Its
    # semantics are selected explicitly, not inferred from the action width.
    interface = {"control_interface": "sonic"} if tag == "new_embodiment" else {}
    action_stats = baseframework.get_action_stats(norm_stats={tag: {"action": stats}}, **interface)

    actual = baseframework.unnormalize_actions(normalized, action_stats)

    np.testing.assert_array_equal(actual, reference._unnormalize_actions(normalized))
    # Index 6 belongs to the 64-dimensional continuous motion token, not a gripper.
    expected_six = np.float32(0.625) * (
        np.float32(stats["max"][6]) - np.float32(stats["min"][6])
    ) + np.float32(stats["min"][6])
    np.testing.assert_array_equal(actual[..., 6], expected_six)
    assert expected_six not in (0.0, 1.0)
    np.testing.assert_array_equal(actual[..., 64], np.float32(0.35))
    np.testing.assert_array_equal(actual[..., 71], np.float32(-0.2))
    if mixed_mask:
        # SONIC clips mask=False channels too, then leaves them unscaled.
        np.testing.assert_array_equal(actual[..., 65], np.float32(-1.0))
        np.testing.assert_array_equal(actual[..., 72], np.float32(1.0))
    np.testing.assert_array_equal(normalized, before)
    assert stats == stats_before
    assert actual.shape == shape
    assert actual.dtype == np.float32


def test_sonic_training_field_order_matches_source_control_interface():
    # Source data_registry/data_config.py:14-27, same pinned SonicStar commit.
    config = SonicLatentDataConfig(observation_indices=[0], action_indices=[0])
    assert config.state_keys == [
        "state.left_leg", "state.right_leg", "state.waist", "state.left_arm",
        "state.left_hand", "state.right_arm", "state.right_hand", "state.projected_gravity",
    ]
    assert config.action_keys == [
        "action.motion_token", "action.left_hand_joints", "action.right_hand_joints",
    ]
    modes = {key: mode for transform in config.transform().transforms
             for key, mode in getattr(transform, "normalization_modes", {}).items()}
    assert all(modes[key] == "min_max" for key in config.state_keys + config.action_keys)
