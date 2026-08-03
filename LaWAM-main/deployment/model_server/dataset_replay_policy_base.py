from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from starVLA.model.tools import read_mode_config


def pick_unnorm_key(norm_stats: dict, requested_key: Optional[str]) -> str:
    if requested_key is None:
        if len(norm_stats) != 1:
            raise ValueError(
                "Checkpoint contains multiple dataset statistics. "
                f"Please pass --unnorm-key from: {list(norm_stats.keys())}"
            )
        return next(iter(norm_stats.keys()))
    if requested_key not in norm_stats:
        raise KeyError(f"Unknown --unnorm-key `{requested_key}`. Available: {list(norm_stats.keys())}")
    return requested_key


def load_action_stats_from_checkpoint(
    ckpt_path: str | Path,
    unnorm_key: Optional[str] = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    resolved_ckpt = Path(ckpt_path).expanduser().resolve()
    model_config, norm_stats = read_mode_config(resolved_ckpt)
    picked_key = pick_unnorm_key(norm_stats, unnorm_key)
    dataset_stats = norm_stats[picked_key]
    if "action" not in dataset_stats:
        raise KeyError(f"Checkpoint stats for `{picked_key}` do not contain `action`.")
    return resolved_ckpt, model_config, norm_stats, picked_key, dataset_stats["action"]


def normalize_actions_with_minmax(
    raw_actions: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    raw_actions = np.asarray(raw_actions, dtype=np.float32)
    low = np.asarray(low, dtype=np.float32)
    high = np.asarray(high, dtype=np.float32)
    denom = np.maximum(high - low, 1e-6)
    normalized = 2.0 * (raw_actions - low) / denom - 1.0
    if clip:
        normalized = np.clip(normalized, -1.0, 1.0)
    return normalized.astype(np.float32, copy=False)


class ChunkedDatasetReplayPolicyBase(ABC):
    """Reusable replay-policy skeleton for dataset-backed benchmark servers.

    Subclasses provide:
    - how a request maps to replay sources / episodes
    - how to load raw action trajectories for a source
    - how to convert raw action chunks into the client-facing normalized chunk
    """

    def __init__(self, *, action_chunk_size: int, loop: bool) -> None:
        self.action_chunk_size = int(action_chunk_size)
        self.loop = bool(loop)
        if self.action_chunk_size <= 0:
            raise ValueError(f"action_chunk_size must be > 0, got {self.action_chunk_size}")

        self._batch_signature: Any = None
        self._batch_sources: list[Any] = []
        self._batch_cursors: list[int] = []
        self.warmup()

    def predict_action(self, examples: Optional[list[dict]] = None, **query: dict) -> dict[str, np.ndarray]:
        batch_size = self.resolve_batch_size(examples=examples, **query)
        self._ensure_batch_state(batch_size=batch_size, examples=examples, **query)
        chunks = [self._next_chunk(batch_idx) for batch_idx in range(batch_size)]
        return {"normalized_actions": np.stack(chunks, axis=0)}

    def resolve_batch_size(self, examples: Optional[list[dict]] = None, **_: dict) -> int:
        return len(examples) if examples is not None else 1

    def warmup(self) -> None:
        for source in self.warmup_sources():
            self.get_raw_actions(source)

    def warmup_sources(self) -> Sequence[Any]:
        return ()

    def get_batch_signature(self, *, batch_size: int, examples: Optional[list[dict]] = None, **query: dict) -> Any:
        del examples, query
        return batch_size

    @abstractmethod
    def build_batch_sources(
        self,
        *,
        batch_size: int,
        examples: Optional[list[dict]] = None,
        **query: dict,
    ) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def get_raw_actions(self, source: Any) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_chunk(self, raw_chunk: np.ndarray, *, source: Any) -> np.ndarray:
        raise NotImplementedError

    def _ensure_batch_state(self, *, batch_size: int, examples: Optional[list[dict]] = None, **query: dict) -> None:
        signature = self.get_batch_signature(batch_size=batch_size, examples=examples, **query)
        if signature == self._batch_signature and len(self._batch_sources) == batch_size:
            return

        sources = list(self.build_batch_sources(batch_size=batch_size, examples=examples, **query))
        if len(sources) != batch_size:
            raise ValueError(
                f"`build_batch_sources` returned {len(sources)} sources for batch_size={batch_size}."
            )
        self._batch_signature = signature
        self._batch_sources = sources
        self._batch_cursors = [0] * batch_size

    def _next_chunk(self, batch_idx: int) -> np.ndarray:
        source = self._batch_sources[batch_idx]
        raw_actions = np.asarray(self.get_raw_actions(source), dtype=np.float32)
        if raw_actions.ndim != 2:
            raise ValueError(f"Expected raw actions with shape [T, D], got {tuple(raw_actions.shape)}.")
        if int(raw_actions.shape[0]) <= 0:
            raise RuntimeError(f"Replay source {source!r} has zero action steps.")

        start = self._batch_cursors[batch_idx]
        end = min(start + self.action_chunk_size, int(raw_actions.shape[0]))
        chunk = np.asarray(raw_actions[start:end], dtype=np.float32)

        if self.loop:
            if chunk.shape[0] < self.action_chunk_size:
                needed = self.action_chunk_size - int(chunk.shape[0])
                extra = raw_actions[:needed]
                chunk = np.concatenate([chunk, extra], axis=0)
                self._batch_cursors[batch_idx] = needed % int(raw_actions.shape[0])
            else:
                self._batch_cursors[batch_idx] = end % int(raw_actions.shape[0])
        else:
            self._batch_cursors[batch_idx] = end
            if chunk.shape[0] < self.action_chunk_size:
                pad = np.repeat(raw_actions[-1:], repeats=self.action_chunk_size - int(chunk.shape[0]), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)

        encoded = np.asarray(self.encode_chunk(chunk, source=source), dtype=np.float32)
        if encoded.ndim != 2 or int(encoded.shape[0]) != self.action_chunk_size:
            raise ValueError(
                "`encode_chunk` must return shape [action_chunk_size, D], "
                f"got {tuple(encoded.shape)} with action_chunk_size={self.action_chunk_size}."
            )
        return encoded
