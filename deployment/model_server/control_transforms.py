# Copyright (c) 2025 StarVLA Team.
# Copyright (c) 2026 Black Otter.
# SPDX-License-Identifier: MIT
"""Client state transforms matching the reference control repositories.

SIMPLE: Ju6276/VLA-JEPA-DEV, vlajepa-xxy at 0d7d15f39c524563310fe7196d85d863494aafb3,
deployment/model_server/simple_g1_adapter.py::normalize_state.
SONIC: SonicStar, dev at 82b395f67ec56805d465b9d1b6043c6d6cdfb06a,
starVLA/examples/SonicLatent/eval_files/run_starvla_inference.py,
StarVLAPolicyAdapter._normalize_state.

The interfaces retain their respective handling of constant state channels.
These functions run on the client before sending normalized state to the server.
"""

import numpy as np


def normalize_simple_state(raw_state: np.ndarray, state_stats: dict) -> np.ndarray:
    """Normalize SIMPLE's 32 state channels using the xxy deployment rule."""
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim == 0 or state.shape[-1] != 32:
        raise ValueError("SIMPLE state must have 32 channels on the final axis.")
    state_min = np.asarray(state_stats["min"], dtype=np.float32)
    state_max = np.asarray(state_stats["max"], dtype=np.float32)
    denom = state_max - state_min
    denom = np.where(denom < 1e-8, 1.0, denom)
    return (2.0 * (state - state_min) / denom - 1.0).astype(np.float32)


def normalize_sonic_state(raw_state: np.ndarray, state_stats: dict) -> np.ndarray:
    """Normalize SONIC's 46 state channels; constant channels become zero."""
    state = np.asarray(raw_state, dtype=np.float32)
    if state.ndim == 0 or state.shape[-1] != 46:
        raise ValueError("SONIC state must have 46 channels on the final axis.")
    state_min = np.asarray(state_stats["min"], dtype=np.float32)
    state_max = np.asarray(state_stats["max"], dtype=np.float32)
    mask = state_min != state_max
    normalized = np.zeros_like(state, dtype=np.float32)
    normalized[..., mask] = 2.0 * (state[..., mask] - state_min[mask]) / (
        state_max[mask] - state_min[mask]
    ) - 1.0
    return normalized
