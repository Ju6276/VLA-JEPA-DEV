"""HumanoidArena HTTP protocol shared by VLA-JEPA serving and tests."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np


STATE_DIM = 64
ACTION_DIM = 40
ACTION_HORIZON = 30

TASK_INSTRUCTIONS = {
    "doubledesk": "Put the hammer from the right table into the basket on the left table.",
    "football": "Kick the soccer ball into the goal.",
    "ppbox": "Move the box from the table onto the shelf.",
    "visionnavi": "Avoid obstacles and move to the yellow marked area.",
    "opendoor": "Open the door.",
    "sitsofa": "Sit on the sofa.",
    "boxing": "Strike the green markers on the punching bag.",
}


def _normalized(value: Any) -> str:
    return "".join(char for char in str(value or "").lower() if char.isalnum())


def canonical_instruction(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = _normalized(raw)
    for alias, instruction in TASK_INSTRUCTIONS.items():
        if alias in normalized or normalized == _normalized(instruction):
            return instruction
    return raw or "perform the humanoid task"


def parse_payload(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, str]:
    observation = payload["observation"]
    spec = observation["images"]["front"]
    shape = tuple(int(value) for value in spec["shape"])
    dtype = np.dtype(spec["dtype"])
    image = np.frombuffer(base64.b64decode(spec["data_b64"]), dtype=dtype).reshape(shape)
    if dtype != np.uint8 or image.shape != (480, 640, 3):
        raise ValueError(f"front image must be uint8 [480,640,3], got {dtype} {shape}")
    state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,):
        raise ValueError(f"state must be [{STATE_DIM}], got {state.shape}")
    instruction = canonical_instruction(payload.get("task", payload.get("task_name")))
    return np.ascontiguousarray(image), state, instruction


def normalize_state(state: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    mask = high != low
    result = np.zeros_like(state, dtype=np.float32)
    result[mask] = 2 * (state[mask] - low[mask]) / (high[mask] - low[mask]) - 1
    return np.clip(result, -1, 1)


def denormalize_action(action: Any, stats: dict[str, Any]) -> np.ndarray:
    normalized = np.asarray(action, dtype=np.float32)
    if normalized.ndim == 3 and normalized.shape[0] == 1:
        normalized = normalized[0]
    if normalized.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"VLA-JEPA action must be [{ACTION_HORIZON},{ACTION_DIM}], "
            f"got {normalized.shape}"
        )
    low = np.asarray(stats["min"], dtype=np.float32)
    high = np.asarray(stats["max"], dtype=np.float32)
    clipped = np.clip(normalized, -1, 1)
    result = 0.5 * (clipped + 1) * (high - low) + low
    result[:, 38:40] = (normalized[:, 38:40] > 0.5).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("VLA-JEPA action contains non-finite values")
    return result
