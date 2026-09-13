"""Time-based sampling of past observations, shared by train and deployment."""

from collections.abc import Sequence

import numpy as np


def validate_history_options(offsets_seconds: Sequence[float], tolerance_seconds: float):
    offsets = np.asarray(offsets_seconds, dtype=np.float64)
    if offsets.ndim != 1 or not np.all(np.isfinite(offsets)):
        raise ValueError("history_offsets_seconds must be a finite one-dimensional sequence")
    if np.any(offsets >= 0) or np.any(np.diff(offsets) <= 0):
        raise ValueError("history_offsets_seconds must be strictly increasing negative seconds")
    if not np.isfinite(tolerance_seconds) or tolerance_seconds < 0:
        raise ValueError("history_tolerance_seconds must be finite and nonnegative")
    return offsets


def select_history_indices(
    timestamps: Sequence[float],
    current_index: int,
    offsets_seconds: Sequence[float],
    tolerance_seconds: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Choose the latest frame at/before each requested past time.

    The selected frame must precede the current observation and be no more than
    ``tolerance_seconds`` earlier than the requested time. Missing history has
    index -1, valid=False and age=0; callers supply a masked current-frame
    placeholder. Sampling never clips to a future frame or another episode.
    """
    offsets = validate_history_options(offsets_seconds, tolerance_seconds)
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("History requires finite one-dimensional episode timestamps")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Episode timestamps must increase strictly")
    if isinstance(current_index, (bool, np.bool_)) or not isinstance(current_index, (int, np.integer)):
        raise ValueError("current_index must be an integer episode row")
    if not 0 <= current_index < len(times):
        raise ValueError("current_index is outside the episode")
    targets = times[current_index] + offsets
    # Search only real past observations, even when timestamps round to the
    # same value after adding a very small offset.
    past = times[:current_index]
    roundoff = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(targets)) * 8
    indices = np.searchsorted(past, targets + roundoff, side="right") - 1
    valid = indices >= 0
    ages = np.zeros(len(offsets), dtype=np.float32)
    if len(past):
        selected_times = past[np.maximum(indices, 0)]
        valid &= targets - selected_times <= tolerance_seconds + 1e-7
        ages[valid] = times[current_index] - selected_times[valid]
    indices[~valid] = -1
    return indices.astype(np.int64), valid, ages
