import base64

import numpy as np
import pytest

from examples.HumanoidArena.humanoidarena_protocol import (
    canonical_instruction,
    denormalize_action,
    normalize_state,
    parse_payload,
)


def payload(state_dim=64):
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "observation": {
            "images": {
                "front": {
                    "shape": list(image.shape),
                    "dtype": str(image.dtype),
                    "data_b64": base64.b64encode(image.tobytes()).decode("ascii"),
                }
            },
            "state": np.zeros(state_dim, dtype=np.float32).tolist(),
        },
        "task": "HSI_open_door",
    }


def test_input_and_prompt_contract():
    image, state, instruction = parse_payload(payload())
    assert image.shape == (480, 640, 3)
    assert state.shape == (64,)
    assert instruction == "Open the door."
    assert canonical_instruction("HOI_football") == "Kick the soccer ball into the goal."


def test_rejects_wrong_state_width():
    with pytest.raises(ValueError, match="64"):
        parse_payload(payload(43))


def test_state_normalization():
    stats = {"min": np.zeros(64), "max": np.full(64, 2)}
    np.testing.assert_allclose(normalize_state(np.ones(64), stats), 0)


def test_action_shape_order_and_binary_hands():
    stats = {"min": np.zeros(40), "max": np.full(40, 2)}
    normalized = np.zeros((1, 30, 40), dtype=np.float32)
    normalized[:, :, 38] = 0.4
    normalized[:, :, 39] = 0.6
    action = denormalize_action(normalized, stats)
    assert action.shape == (30, 40)
    np.testing.assert_allclose(action[:, :38], 1)
    np.testing.assert_array_equal(action[:, 38], 0)
    np.testing.assert_array_equal(action[:, 39], 1)
    with pytest.raises(ValueError, match="30,40"):
        denormalize_action(np.zeros((30, 78)), stats)
