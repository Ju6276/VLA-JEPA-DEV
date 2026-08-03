from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import cv2 as cv
import numpy as np
import torch

STARVLA_ROOT = Path(__file__).resolve().parents[3]

if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.tools import read_mode_config

logger = logging.getLogger(__name__)


def invert_gripper_value(gripper: np.ndarray) -> np.ndarray:
    return 1.0 - np.asarray(gripper, dtype=np.float32)


def normalize_quaternion_value(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat[None, :]

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    normalized = np.zeros_like(quat, dtype=np.float32)
    valid_rows = np.isfinite(norm[:, 0]) & (norm[:, 0] > 1e-8)
    if np.any(valid_rows):
        normalized[valid_rows] = quat[valid_rows] / norm[valid_rows]
    invalid = ~valid_rows
    if np.any(invalid):
        normalized[invalid, 0] = 1.0
    return normalized


_ROBOTWIN_JOINT_DATA_MIXES = {"robotwin_joint"}
_ROBOTWIN_EEF_DATA_MIXES = {
    "robotwin_eef": 50.0,
    "robotwin_eef_all": 50.0,
    "robotwin_eef_30hz": 30.0,
}


@dataclass(frozen=True)
class RobotwinControlSpec:
    data_mix: str
    mode: str
    env_action_type: str
    action_hz: float
    state_gripper_indices: tuple[int, ...]
    passthrough_indices: tuple[int, ...]
    gripper_is_binary: bool


def resolve_robotwin_control_from_data_mix(data_mix: Any) -> RobotwinControlSpec:
    normalized = str(data_mix).strip()
    if normalized in _ROBOTWIN_JOINT_DATA_MIXES:
        return RobotwinControlSpec(
            data_mix=normalized,
            mode="joint",
            env_action_type="qpos",
            action_hz=30.0,
            state_gripper_indices=(6, 13),
            passthrough_indices=(),
            gripper_is_binary=True,
        )
    if normalized in _ROBOTWIN_EEF_DATA_MIXES:
        return RobotwinControlSpec(
            data_mix=normalized,
            mode="eef",
            env_action_type="ee",
            action_hz=_ROBOTWIN_EEF_DATA_MIXES[normalized],
            state_gripper_indices=(7, 15),
            passthrough_indices=(3, 4, 5, 6, 11, 12, 13, 14),
            gripper_is_binary=False,
        )
    raise ValueError(
        "Unsupported Robotwin data_mix for eval mode resolution: "
        f"{normalized!r}. Expected one of {sorted(_ROBOTWIN_JOINT_DATA_MIXES | set(_ROBOTWIN_EEF_DATA_MIXES))}."
    )


def _extract_robotwin_data_mix(model_config: dict[str, Any]) -> str:
    datasets_cfg = model_config.get("datasets", {})
    if not isinstance(datasets_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets` dict; cannot resolve Robotwin action mode.")
    vla_data_cfg = datasets_cfg.get("vla_data", {})
    if not isinstance(vla_data_cfg, dict):
        raise ValueError("Checkpoint config is missing `datasets.vla_data`; cannot resolve Robotwin action mode.")
    data_mix = vla_data_cfg.get("data_mix", None)
    if data_mix is None:
        raise ValueError("Checkpoint config is missing `datasets.vla_data.data_mix`; cannot resolve Robotwin action mode.")
    return str(data_mix)


def flatten_robotwin_endpose_state(observation: dict[str, Any]) -> np.ndarray:
    endpose = observation.get("endpose", None)
    if not isinstance(endpose, dict):
        raise KeyError("Robotwin EEF eval requires observation['endpose'].")

    required_keys = (
        "left_endpose",
        "left_gripper",
        "right_endpose",
        "right_gripper",
    )
    missing = [key for key in required_keys if key not in endpose]
    if missing:
        raise KeyError(f"Robotwin EEF eval observation is missing endpose keys: {missing}")

    state = np.concatenate(
        [
            np.asarray(endpose["left_endpose"], dtype=np.float32).reshape(-1),
            np.asarray([endpose["left_gripper"]], dtype=np.float32).reshape(-1),
            np.asarray(endpose["right_endpose"], dtype=np.float32).reshape(-1),
            np.asarray([endpose["right_gripper"]], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    )
    return np.asarray(state, dtype=np.float32)


def build_robotwin_example(
    task_description: str,
    observation: dict[str, Any],
    *,
    robotwin_mode: str = "joint",
) -> dict[str, Any]:
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    if robotwin_mode == "joint":
        state = observation.get("joint_action", {}).get("vector", None)
    elif robotwin_mode == "eef":
        state = flatten_robotwin_endpose_state(observation)
    else:
        raise ValueError(f"Unsupported Robotwin mode: {robotwin_mode!r}")
    return {
        "lang": str(task_description),
        "image": [head_img, left_img, right_img],
        "state": state,
    }


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _parse_bool_like(value: Any, *, default: bool = False) -> bool:
    if _is_none_like(value):
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean-like value, got {value!r}.")


def _resolve_precision_dtype(mixed_precision: Any) -> Optional[torch.dtype]:
    precision = str(mixed_precision or "bf16").strip().lower()
    if precision == "no":
        return None
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    raise ValueError(
        f"Unsupported mixed_precision={mixed_precision!r}. Expected one of ['no', 'fp16', 'bf16']."
    )


class LocalStarVLARobotwinPolicy:
    _DEFAULT_IMAGE_SIZE = (256, 256)
    _DEFAULT_BINARY_THRESHOLD = 0.49

    @staticmethod
    def _bounded_indices(dim: int, indices: tuple[int, ...] | list[int] | np.ndarray) -> np.ndarray:
        resolved = np.asarray(indices, dtype=np.int64).reshape(-1)
        return resolved[(resolved >= 0) & (resolved < int(dim))]

    @staticmethod
    def _infer_binary_indices(state_norm_stats: Optional[dict]) -> tuple[int, ...]:
        if state_norm_stats is None:
            return ()
        mask = state_norm_stats.get("mask", None)
        if mask is not None:
            low_key = "min" if "min" in state_norm_stats else "q01"
            state_low = np.asarray(state_norm_stats[low_key], dtype=np.float32)
            dim = int(state_low.shape[0])
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] == dim:
                return tuple(np.where(~mask_arr)[0].tolist())
        return ()

    @classmethod
    def _resolve_state_binary_indices(
        cls,
        state_norm_stats: Optional[dict],
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
        gripper_is_binary: bool,
    ) -> tuple[int, ...]:
        inferred = cls._infer_binary_indices(state_norm_stats)
        if robotwin_mode in {"joint", "eef"} and gripper_is_binary:
            return tuple(sorted(set(inferred) | set(state_gripper_indices)))
        return inferred

    @staticmethod
    def _resolve_state_invert_indices(
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if robotwin_mode in {"joint", "eef"}:
            return tuple(state_gripper_indices)
        return ()

    @staticmethod
    def _resolve_action_invert_indices(
        robotwin_mode: str,
        action_gripper_indices: tuple[int, ...],
    ) -> tuple[int, ...]:
        if robotwin_mode in {"joint", "eef"}:
            return tuple(action_gripper_indices)
        return ()

    @staticmethod
    def _legacy_action_gripper_indices(robotwin_mode: str) -> tuple[int, ...]:
        if robotwin_mode == "joint":
            return (12, 13)
        if robotwin_mode == "eef":
            return (14, 15)
        raise ValueError(f"Unsupported Robotwin mode for legacy layout detection: {robotwin_mode}")

    @classmethod
    def _resolve_action_binary_indices(
        cls,
        action_norm_stats: dict,
        robotwin_mode: str,
        state_gripper_indices: tuple[int, ...],
        gripper_is_binary: bool,
    ) -> tuple[int, ...]:
        expected = tuple(int(idx) for idx in state_gripper_indices)
        inferred = cls._infer_binary_indices(action_norm_stats)
        legacy = cls._legacy_action_gripper_indices(robotwin_mode)
        if inferred == legacy:
            raise ValueError(
                "Robotwin checkpoint statistics use legacy action order. "
                f"mode={robotwin_mode}, legacy_gripper_indices={legacy}, expected={expected}. "
                "Regenerate `dataset_statistics.json` before running benchmark."
            )
        if gripper_is_binary and inferred in {(), expected}:
            return expected
        if not gripper_is_binary and inferred in {(), expected}:
            return inferred
        raise ValueError(
            "Robotwin checkpoint statistics have unexpected action.mask layout. "
            f"mode={robotwin_mode}, inferred_gripper_indices={inferred}, expected={expected}. "
            "Regenerate `dataset_statistics.json` before running benchmark."
        )

    @classmethod
    def normalize_state(
        cls,
        state: np.ndarray,
        state_norm_stats: dict[str, np.ndarray],
        binary_indices: tuple[int, ...],
        passthrough_indices: tuple[int, ...] = (),
        invert_indices: tuple[int, ...] = (),
        binary_threshold: float = _DEFAULT_BINARY_THRESHOLD,
    ) -> np.ndarray:
        high_key = "max" if "max" in state_norm_stats else "q99"
        low_key = "min" if "min" in state_norm_stats else "q01"
        state_high = np.asarray(state_norm_stats[high_key], dtype=np.float32)
        state_low = np.asarray(state_norm_stats[low_key], dtype=np.float32)

        normalized = np.asarray(state, dtype=np.float32).copy()
        stats_dim = int(state_high.shape[0])
        if normalized.shape[-1] != stats_dim:
            if normalized.shape[-1] < stats_dim:
                raise ValueError(
                    "`state` dim does not match checkpoint statistics: "
                    f"state_dim={normalized.shape[-1]}, stats_dim={stats_dim}."
                )
            normalized = normalized[..., :stats_dim]

        binary_idx = cls._bounded_indices(normalized.shape[-1], binary_indices)
        passthrough_idx = cls._bounded_indices(normalized.shape[-1], passthrough_indices)
        continuous_mask = np.ones(stats_dim, dtype=bool)
        if binary_idx.size > 0:
            continuous_mask[binary_idx] = False
        if passthrough_idx.size > 0:
            continuous_mask[passthrough_idx] = False

        denom = state_high - state_low
        valid = np.abs(denom) > 1e-12
        valid &= continuous_mask
        if np.any(valid):
            normalized[..., valid] = (normalized[..., valid] - state_low[valid]) / denom[valid] * 2.0 - 1.0
            normalized[..., valid] = np.clip(normalized[..., valid], -1.0, 1.0)
        zero_mask = ~valid & continuous_mask
        if np.any(zero_mask):
            normalized[..., zero_mask] = 0.0
        if binary_idx.size > 0:
            normalized[..., binary_idx] = (normalized[..., binary_idx] > binary_threshold).astype(np.float32)
        invert_idx = cls._bounded_indices(normalized.shape[-1], invert_indices)
        if invert_idx.size > 0:
            binary_invert_idx = np.intersect1d(invert_idx, binary_idx, assume_unique=False)
            continuous_invert_idx = np.setdiff1d(invert_idx, binary_idx, assume_unique=False)
            if binary_invert_idx.size > 0:
                normalized[..., binary_invert_idx] = invert_gripper_value(
                    normalized[..., binary_invert_idx]
                )
            if continuous_invert_idx.size > 0:
                normalized[..., continuous_invert_idx] = -normalized[..., continuous_invert_idx]
        return normalized

    @classmethod
    def unnormalize_actions(
        cls,
        normalized_actions: np.ndarray,
        action_norm_stats: dict[str, np.ndarray],
        gripper_indices: tuple[int, ...] = (),
        passthrough_indices: tuple[int, ...] = (),
        invert_indices: tuple[int, ...] = (),
        binary_threshold: float = _DEFAULT_BINARY_THRESHOLD,
    ) -> np.ndarray:
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"
        action_high = np.asarray(action_norm_stats[high_key], dtype=np.float32)
        action_low = np.asarray(action_norm_stats[low_key], dtype=np.float32)
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))

        normalized_actions = np.array(normalized_actions, dtype=np.float32, copy=True)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        stats_dim = int(action_high.shape[0])
        if normalized_actions.shape[-1] != stats_dim:
            if normalized_actions.shape[-1] < stats_dim:
                raise ValueError(
                    "`normalized_actions` dim does not match checkpoint statistics: "
                    f"action_dim={normalized_actions.shape[-1]}, stats_dim={stats_dim}."
                )
            normalized_actions = normalized_actions[..., :stats_dim]

        gripper_idx = cls._bounded_indices(stats_dim, gripper_indices)
        passthrough_idx = cls._bounded_indices(stats_dim, passthrough_indices)
        denorm_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if passthrough_idx.size > 0:
            denorm_mask[passthrough_idx] = False
        if gripper_idx.size > 0:
            normalized_actions[:, gripper_idx] = np.where(
                normalized_actions[:, gripper_idx] < binary_threshold,
                0.0,
                1.0,
            )

        actions = normalized_actions.copy()
        if np.any(denorm_mask):
            clipped = np.clip(normalized_actions[:, denorm_mask], -1.0, 1.0)
            actions[:, denorm_mask] = (
                0.5 * (clipped + 1.0) * (action_high[denorm_mask] - action_low[denorm_mask])
                + action_low[denorm_mask]
            )
        if passthrough_idx.size > 0:
            actions[:, passthrough_idx] = normalized_actions[:, passthrough_idx]
        invert_idx = cls._bounded_indices(stats_dim, invert_indices)
        if invert_idx.size > 0:
            actions[:, invert_idx] = invert_gripper_value(actions[:, invert_idx])
        cls._normalize_robotwin_eef_quaternions_(actions, passthrough_indices)
        return actions

    @staticmethod
    def _normalize_robotwin_eef_quaternions_(
        actions: np.ndarray,
        passthrough_indices: tuple[int, ...] = (),
    ) -> None:
        passthrough_set = set(int(idx) for idx in passthrough_indices)
        quaternion_groups = ((3, 4, 5, 6), (11, 12, 13, 14))
        for group in quaternion_groups:
            if not set(group).issubset(passthrough_set):
                continue
            if actions.shape[-1] <= group[-1]:
                continue
            actions[:, group[0] : group[-1] + 1] = normalize_quaternion_value(
                actions[:, group[0] : group[-1] + 1]
            )

    def __init__(
        self,
        *,
        policy_ckpt_path: str,
        unnorm_key: Optional[str] = None,
        image_size: Optional[Sequence[int]] = None,
        replan_steps: Optional[int | str] = None,
        action_ensemble: Optional[bool | str] = None,
        action_ensemble_alpha: Optional[float | str] = None,
        action_reorder: Optional[Sequence[int] | str] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        mixed_precision: Any = "bf16",
        device: str = "cuda",
    ) -> None:
        self.policy_ckpt_path = Path(str(policy_ckpt_path)).expanduser().resolve()
        if not self.policy_ckpt_path.exists():
            raise FileNotFoundError(f"Policy checkpoint not found: {self.policy_ckpt_path}")

        self.model_config, self.norm_stats = read_mode_config(self.policy_ckpt_path)
        self.framework_name = str(self.model_config.get("framework", {}).get("name", ""))
        self.data_mix = _extract_robotwin_data_mix(self.model_config)
        self.control_spec = resolve_robotwin_control_from_data_mix(self.data_mix)
        self.robotwin_mode = self.control_spec.mode
        self.env_action_type = self.control_spec.env_action_type
        self.action_hz = float(self.control_spec.action_hz)
        self.state_gripper_indices = self.control_spec.state_gripper_indices
        self.passthrough_indices = self.control_spec.passthrough_indices
        self.gripper_is_binary = bool(self.control_spec.gripper_is_binary)

        flow_cfg = self.model_config.get("framework", {}).get("action_model", {}).get("flow_cfg", {})
        self.use_state = bool(flow_cfg.get("use_state", True))

        self.unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)
        self.dataset_stats = self.norm_stats[self.unnorm_key]
        self.action_norm_stats = self.dataset_stats["action"]
        self.state_norm_stats = self.dataset_stats.get("state", None)
        self.action_dim = int(np.asarray(self.action_norm_stats["min"], dtype=np.float32).shape[0])
        self.action_binary_indices = self._resolve_action_binary_indices(
            self.action_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_binary_indices = self._resolve_state_binary_indices(
            self.state_norm_stats,
            self.robotwin_mode,
            self.state_gripper_indices,
            gripper_is_binary=self.gripper_is_binary,
        )
        self.state_invert_indices = self._resolve_state_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )
        self.action_invert_indices = self._resolve_action_invert_indices(
            self.robotwin_mode,
            self.state_gripper_indices,
        )

        resolved_image_size = self._resolve_image_size(self.model_config, image_size)
        self.image_size = [int(resolved_image_size[0]), int(resolved_image_size[1])]
        self.replan_steps = self._resolve_replan_steps(replan_steps)
        self.action_ensemble = _parse_bool_like(action_ensemble, default=False)
        self.action_ensemble_alpha = float(_parse_optional_float(action_ensemble_alpha) or 0.0)
        if self.action_ensemble and self.replan_steps is None:
            raise ValueError("`action_ensemble` requires a positive `replan_steps` value.")
        self.action_reorder = self._resolve_action_reorder(action_reorder, action_dim=self.action_dim)
        self.guidance_scale = _parse_optional_float(guidance_scale)
        self.num_inference_steps = _parse_optional_int(num_inference_steps)
        self.device = str(device)
        self.policy = self._load_policy(mixed_precision=mixed_precision, device=self.device)
        self.pending_actions: list[np.ndarray] = []
        self.action_cursor = 0
        self.executed_steps = 0
        self.action_ensemble_history: list[dict[str, Any]] = []
        self.current_task_description: Optional[str] = None

        logger.info(
            "Initialized LocalStarVLARobotwinPolicy | ckpt=%s | framework=%s | data_mix=%s | "
            "mode=%s | action_hz=%.1f | replan_steps=%s | action_ensemble=%s | "
            "action_ensemble_alpha=%s | image_size=%s | action_reorder=%s",
            self.policy_ckpt_path,
            self.framework_name,
            self.data_mix,
            self.robotwin_mode,
            self.action_hz,
            self.replan_steps,
            self.action_ensemble,
            self.action_ensemble_alpha,
            self.image_size,
            self.action_reorder,
        )

    def _load_policy(self, *, mixed_precision: Any, device: str):
        policy = baseframework.from_pretrained(str(self.policy_ckpt_path))
        cast_dtype = _resolve_precision_dtype(mixed_precision)
        if cast_dtype is not None:
            policy = policy.to(dtype=cast_dtype)
        policy = policy.to(device)
        return policy.eval()

    def reset(self, task_description: Optional[str] = None) -> None:
        self.pending_actions = []
        self.action_cursor = 0
        self.executed_steps = 0
        self.action_ensemble_history.clear()
        self.current_task_description = task_description

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    def _prepare_state(self, state: Any) -> Optional[np.ndarray]:
        if state is None or not self.use_state:
            return None
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 2 and state_array.shape[0] == 1:
            state_array = state_array[0]
        if state_array.ndim != 1:
            raise ValueError(f"`state` must have shape [D], got {state_array.shape}.")
        if self.state_norm_stats is None:
            return state_array
        return self.normalize_state(
            state=state_array,
            state_norm_stats=self.state_norm_stats,
            binary_indices=self.state_binary_indices,
            passthrough_indices=self.passthrough_indices,
            invert_indices=self.state_invert_indices,
            binary_threshold=self._DEFAULT_BINARY_THRESHOLD,
        )

    def _prepare_example(self, example: dict[str, Any]) -> dict[str, Any]:
        images = example.get("image", None)
        if not isinstance(images, (list, tuple)) or len(images) == 0:
            raise ValueError("Robotwin example must contain non-empty `image` list.")
        return {
            "lang": str(example.get("lang", "")),
            "image": [self._resize_image(np.asarray(image)) for image in images],
            "state": self._prepare_state(example.get("state", None)),
        }

    def _build_infer_example(self, example: dict[str, Any]) -> dict[str, Any]:
        images = list(example["image"])
        infer_example: dict[str, Any] = {
            "lang": str(example["lang"]),
            "primary_image": [images[0]],
            "embodiment_id": 1,
            "action_hz": float(self.action_hz),
        }
        wrist_images = list(images[1:])
        if wrist_images:
            infer_example["wrist_image"] = wrist_images
        if example.get("state", None) is not None:
            infer_example["state"] = np.asarray(example["state"], dtype=np.float32)
        return infer_example

    def _truncate_action_chunk(self, raw_actions: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"`raw_actions` must have shape [T, D], got {action_chunk.shape}.")
        if self.replan_steps is None:
            return action_chunk
        return np.asarray(action_chunk[: int(self.replan_steps)], dtype=np.float32)

    def _infer_action_chunk(self, task_description: str, observation: dict[str, Any]) -> np.ndarray:
        prepared = self._prepare_example(
            build_robotwin_example(task_description, observation, robotwin_mode=self.robotwin_mode)
        )
        infer_example = self._build_infer_example(prepared)
        predict_kwargs = {"examples": [infer_example]}
        if self.guidance_scale is not None:
            predict_kwargs["guidance_scale"] = float(self.guidance_scale)
        if self.num_inference_steps is not None:
            predict_kwargs["num_inference_steps"] = int(self.num_inference_steps)

        with torch.inference_mode():
            output = self.policy.predict_action(**predict_kwargs)
        normalized_actions = np.asarray(output["normalized_actions"], dtype=np.float32)
        if normalized_actions.ndim == 3:
            normalized_actions = normalized_actions[0]
        elif normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        elif normalized_actions.ndim != 2:
            raise ValueError(
                f"Unexpected `normalized_actions` shape from policy: {tuple(normalized_actions.shape)}."
            )

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions,
            action_norm_stats=self.action_norm_stats,
            gripper_indices=self.action_binary_indices,
            passthrough_indices=self.passthrough_indices,
            invert_indices=self.action_invert_indices,
        )
        if self.action_ensemble:
            return self._build_action_ensemble_cache(raw_actions)
        return self._truncate_action_chunk(raw_actions)

    def _remap_action_for_robotwin_env(self, action: np.ndarray) -> np.ndarray:
        action_array = np.asarray(action, dtype=np.float32)
        if self.action_reorder is None:
            return action_array
        return np.asarray(action_array[list(self.action_reorder)], dtype=np.float32)

    def _build_action_ensemble_cache(self, raw_actions: np.ndarray) -> np.ndarray:
        action_chunk = np.asarray(raw_actions, dtype=np.float32)
        if action_chunk.ndim != 2:
            raise ValueError(f"`raw_actions` must have shape [T, D], got {action_chunk.shape}.")
        if int(action_chunk.shape[0]) <= 0:
            raise RuntimeError("Inference returned empty action chunk.")
        if self.replan_steps is None:
            raise ValueError("`action_ensemble` requires a positive `replan_steps` value.")
        query_interval = int(self.replan_steps)
        if int(action_chunk.shape[0]) < query_interval:
            raise ValueError(
                "`action_ensemble` requires model action chunk length to be >= replan_steps, "
                f"got chunk_len={int(action_chunk.shape[0])}, replan_steps={query_interval}."
            )

        current_step = int(self.executed_steps)
        self.action_ensemble_history = [
            item
            for item in self.action_ensemble_history
            if int(item["start_step"]) + int(item["actions"].shape[0]) > current_step
        ]
        self.action_ensemble_history.append({"start_step": current_step, "actions": action_chunk})
        return np.asarray(
            [self._ensemble_action_for_step(current_step + offset) for offset in range(query_interval)],
            dtype=np.float32,
        )

    def _ensemble_action_for_step(self, step: int) -> np.ndarray:
        relevant_preds = []
        for item in self.action_ensemble_history:
            idx = int(step) - int(item["start_step"])
            actions = np.asarray(item["actions"], dtype=np.float32)
            if 0 <= idx < int(actions.shape[0]):
                relevant_preds.append(actions[idx])
        if not relevant_preds:
            raise ValueError(f"Step {step}: no actions available for action ensemble.")

        curr_act_preds = np.stack(relevant_preds, axis=0)
        if int(curr_act_preds.shape[0]) == 1:
            final_action = np.asarray(curr_act_preds[0], dtype=np.float32).copy()
        else:
            ref = curr_act_preds[-1]
            dot_product = np.sum(curr_act_preds * ref, axis=1)
            norm_previous_pred = np.linalg.norm(curr_act_preds, axis=1)
            norm_ref = np.linalg.norm(ref)
            cos_similarity = dot_product / (norm_previous_pred * norm_ref + 1e-7)
            weights = np.exp(float(self.action_ensemble_alpha) * cos_similarity)
            weights = weights / weights.sum()
            final_action = np.sum(weights[:, None] * curr_act_preds, axis=0).astype(np.float32)

        binary_idx = self._bounded_indices(final_action.shape[-1], self.action_binary_indices)
        if binary_idx.size > 0:
            final_action[binary_idx] = (final_action[binary_idx] > self._DEFAULT_BINARY_THRESHOLD).astype(np.float32)
        final_action = np.asarray(final_action[None, :], dtype=np.float32)
        self._normalize_robotwin_eef_quaternions_(final_action, self.passthrough_indices)
        return final_action[0]

    def _next_cached_action(self) -> np.ndarray:
        if self.action_cursor >= len(self.pending_actions):
            raise RuntimeError("No cached actions available.")
        action = np.asarray(self.pending_actions[self.action_cursor], dtype=np.float32)
        self.action_cursor += 1
        self.executed_steps += 1
        return self._remap_action_for_robotwin_env(action)

    def step(self, observation: dict[str, Any], task_description: str) -> np.ndarray:
        resolved_task = str(task_description)
        if resolved_task != self.current_task_description:
            self.reset(task_description=resolved_task)
        if self.action_cursor >= len(self.pending_actions):
            self.pending_actions = list(self._infer_action_chunk(resolved_task, observation))
            self.action_cursor = 0
            if len(self.pending_actions) == 0:
                raise RuntimeError("Inference returned empty action chunk.")
        return self._next_cached_action()

    @staticmethod
    def _check_unnorm_key(norm_stats: dict, unnorm_key: Optional[str]) -> str:
        if unnorm_key is None or unnorm_key not in norm_stats:
            return next(iter(norm_stats.keys()))
        return str(unnorm_key)

    @classmethod
    def _resolve_image_size(
        cls,
        model_config: dict[str, Any],
        image_size: Optional[Sequence[int] | str],
    ) -> tuple[int, int]:
        if image_size is not None and not _is_none_like(image_size):
            if isinstance(image_size, str):
                stripped = image_size.strip()
                if "," in stripped:
                    parts = [part.strip() for part in stripped.split(",")]
                else:
                    parts = [part.strip() for part in stripped.split("x")]
                if len(parts) != 2:
                    raise ValueError(f"`image_size` must look like 'W,H' or 'WxH', got {image_size!r}.")
                return int(parts[0]), int(parts[1])
            if len(image_size) != 2:
                raise ValueError(f"`image_size` must have length 2, got {image_size}.")
            return int(image_size[0]), int(image_size[1])

        datasets_cfg = model_config.get("datasets", {})
        if isinstance(datasets_cfg, dict):
            vla_data_cfg = datasets_cfg.get("vla_data", {})
            if isinstance(vla_data_cfg, dict):
                image_resolution = vla_data_cfg.get("image_resolution", None)
                if image_resolution is not None:
                    size = int(image_resolution)
                    if size > 0:
                        return size, size
                cfg_image_size = vla_data_cfg.get("image_size", None)
                if isinstance(cfg_image_size, (list, tuple)) and len(cfg_image_size) == 2:
                    return int(cfg_image_size[0]), int(cfg_image_size[1])

        return cls._DEFAULT_IMAGE_SIZE

    @staticmethod
    def _resolve_replan_steps(replan_steps: Optional[int | str]) -> Optional[int]:
        if replan_steps is None or _is_none_like(replan_steps):
            return None
        resolved = int(replan_steps)
        if resolved <= 0:
            return None
        return resolved

    @staticmethod
    def _resolve_action_reorder(
        action_reorder: Optional[Sequence[int] | str],
        *,
        action_dim: int,
    ) -> Optional[tuple[int, ...]]:
        if action_reorder is None or _is_none_like(action_reorder):
            return None
        if isinstance(action_reorder, str):
            resolved = tuple(int(part.strip()) for part in action_reorder.split(","))
        else:
            resolved = tuple(int(part) for part in action_reorder)
        if len(resolved) != int(action_dim):
            raise ValueError(
                f"`action_reorder` must contain exactly {action_dim} indices, got {resolved}."
            )
        if tuple(sorted(resolved)) != tuple(range(int(action_dim))):
            raise ValueError(
                "`action_reorder` must be a permutation of action indices "
                f"[0, {action_dim - 1}], got {resolved}."
            )
        return resolved


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if _is_none_like(policy_ckpt_path):
        raise ValueError("policy_ckpt_path must be provided in config/overrides")

    return LocalStarVLARobotwinPolicy(
        policy_ckpt_path=str(policy_ckpt_path),
        unnorm_key=usr_args.get("unnorm_key", None),
        image_size=usr_args.get("image_size", None),
        replan_steps=usr_args.get("replan_steps", os.getenv("ROBOTWIN_REPLAN_STEPS")),
        action_ensemble=usr_args.get("action_ensemble", os.getenv("ROBOTWIN_ACTION_ENSEMBLE")),
        action_ensemble_alpha=usr_args.get("action_ensemble_alpha", os.getenv("ROBOTWIN_ACTION_ENSEMBLE_ALPHA")),
        action_reorder=usr_args.get("action_reorder", None),
        guidance_scale=usr_args.get("guidance_scale", None),
        num_inference_steps=usr_args.get("num_inference_steps", None),
        mixed_precision=usr_args.get("mixed_precision", "bf16"),
        device=str(usr_args.get("device", "cuda")),
    )


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    action = model.step(observation=observation, task_description=str(instruction))
    TASK_ENV.take_action(action, action_type=model.env_action_type)
