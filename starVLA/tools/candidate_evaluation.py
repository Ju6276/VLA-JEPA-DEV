"""Offline candidate identity and evaluation; never labels unexecuted actions."""
from __future__ import annotations
import numpy as np


def convert_normalization(actions, source_center, source_scale, target_center, target_scale):
    """Affine controller-command conversion, NOT motion-token -> joint decoding.

    Supply the actual affine maps for each checkpoint; do not infer them from
    checkpoint names. The same controller/decoder must interpret both banks.
    """
    a = np.asarray(actions, dtype=np.float64)
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (source_center, source_scale, target_center, target_scale)]
    if any(x.shape != (a.shape[-1],) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Normalization maps must match all command coordinates')
    sc, ss, tc, ts = arrays
    if not np.isfinite(a).all() or np.any(ss <= 0) or np.any(ts <= 0):
        raise ValueError('Finite commands and strictly positive affine scales required')
    canonical = a * ss + sc
    normalized = (canonical - tc) / ts
    return canonical, normalized  # Do not clip: clipping changes the candidate.


def selection_metrics(scores, observed_returns):
    """One decision, measured Q for every candidate; regret is bank-relative."""
    s, q = np.asarray(scores, float), np.asarray(observed_returns, float)
    if s.ndim != 1 or s.size == 0 or q.shape != s.shape or not np.isfinite(s).all() or not np.isfinite(q).all():
        raise ValueError('Need finite scores and independently measured outcomes for all candidates')
    selected = int(s.argmax())
    return {'selected_index': selected, 'selected_return': float(q[selected]),
            'candidate_set_regret': float(q.max() - q[selected]),
            'oracle_return': float(q.max())}


def crossed_pairs(first, second, split):
    a, b = np.asarray(first), np.asarray(second)
    if a.shape != b.shape or a.ndim != 2 or not 0 < split < a.shape[-1]:
        raise ValueError('Need two [h,d] commands and a valid group split')
    return np.concatenate((a[:, :split], b[:, split:]), -1), np.concatenate((b[:, :split], a[:, split:]), -1)
