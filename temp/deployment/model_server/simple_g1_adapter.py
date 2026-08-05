"""
SIMPLE / G1 handover serving adapter for VLA-JEPA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.__init__ import build_framework
from starVLA.model.framework.share_tools import dict_to_namespace, read_mode_config


def normalize_state(raw_state: np.ndarray, state_stats: dict[str, Any]) -> np.ndarray:
    """Min-max normalize raw 32D state to [-1, 1] to match training preprocessing."""
    state_min = np.array(state_stats["min"], dtype=np.float32)
    state_max = np.array(state_stats["max"], dtype=np.float32)
    denom = state_max - state_min
    denom = np.where(denom < 1e-8, 1.0, denom)
    return (2.0 * (raw_state - state_min) / denom - 1.0).astype(np.float32)


def unnormalize_actions(raw_actions: np.ndarray, action_stats: dict[str, Any]) -> np.ndarray:
    """Custom de-normalization for G1 actions.

    We intentionally do not call baseframework.unnormalize_actions because its
    default implementation hard-codes a binary threshold at global index 6 for
    gripper actions, which is not valid for the G1 handover action layout.

    Action layout (36D):
      [0:32]  min_max normalized — left_hand(7) + right_hand(7) + left_arm(7) +
              right_arm(7) + rpy(3) + height(1)
      [32:36] mean_std normalized — torso_vx(1) + torso_vy(1) + torso_vyaw(1) +
              target_yaw(1)
    """
    action_min = np.array(action_stats["min"], dtype=np.float32)
    action_max = np.array(action_stats["max"], dtype=np.float32)
    action_mean = np.array(action_stats["mean"], dtype=np.float32)
    action_std = np.array(action_stats["std"], dtype=np.float32)

    result = np.empty_like(raw_actions, dtype=np.float32)

    # dims 0-31: min_max — model outputs in [-1, 1], map back to [min, max]
    clipped = np.clip(raw_actions[..., :32], -1.0, 1.0)
    result[..., :32] = 0.5 * (clipped + 1.0) * (action_max[:32] - action_min[:32]) + action_min[:32]

    # dims 32-35: mean_std — model outputs zero-mean unit-variance, map back
    result[..., 32:36] = raw_actions[..., 32:36] * action_std[32:36] + action_mean[32:36]

    return result


class SimpleG1PolicyAdapter:
    def __init__(
        self,
        ckpt_path: str | Path,
        device: str,
        use_bf16: bool = False,
        base_vlm_path: str | Path | None = None,
        base_encoder_path: str | Path | None = None,
    ):
        self._ckpt_path = Path(ckpt_path).resolve()
        self._device = torch.device(device)

        model_config, norm_stats = read_mode_config(str(self._ckpt_path))
        config = dict_to_namespace(model_config)
        config.trainer.pretrained_checkpoint = None
        if hasattr(config.framework, "privileged_latent"):
            config.framework.privileged_latent.load_vjepa = False
            config.framework.privileged_latent.delta_action_grounding = False

        repo_root = Path(__file__).resolve().parents[2]
        override_base_vlm = self._resolve_base_vlm_path(
            config,
            repo_root=repo_root,
            explicit_base_vlm_path=base_vlm_path,
        )
        if override_base_vlm is not None and hasattr(config.framework, "qwenvl"):
            config.framework.qwenvl.base_vlm = str(override_base_vlm)

        if not (hasattr(config.framework, "privileged_latent") and not config.framework.privileged_latent.load_vjepa):
            override_vjepa_encoder = self._resolve_vjepa_encoder_path(
                config,
                repo_root=repo_root,
                explicit_base_encoder_path=base_encoder_path,
            )
            if override_vjepa_encoder is not None and hasattr(config.framework, "vj2_model"):
                config.framework.vj2_model.base_encoder = str(override_vjepa_encoder)

        self._model = build_framework(cfg=config)
        self._model.norm_stats = norm_stats
        model_state_dict = torch.load(self._ckpt_path, map_location="cpu")
        if hasattr(config.framework, "privileged_latent") and not config.framework.privileged_latent.load_vjepa:
            training_only_prefixes = (
                "vj_encoder.",
                "teacher_encoder.",
                "delta_action_decoder.",
            )
            model_state_dict = {
                key: value
                for key, value in model_state_dict.items()
                if not key.startswith(training_only_prefixes)
            }
        self._model.load_state_dict(model_state_dict, strict=True)
        if use_bf16:
            self._model = self._model.to(torch.bfloat16)
        self._model = self._model.to(self._device).eval()
        self._stats_key = self._resolve_stats_key(norm_stats)
        self._action_stats = norm_stats[self._stats_key]["action"]
        self._state_stats = norm_stats[self._stats_key]["state"]

    @staticmethod
    def _resolve_stats_key(norm_stats: dict[str, Any]) -> str:
        if len(norm_stats) == 1:
            return next(iter(norm_stats.keys()))
        if "g1_handover" in norm_stats:
            return "g1_handover"
        raise ValueError(
            "Checkpoint contains multiple dataset statistic keys. "
            f"Available keys: {list(norm_stats.keys())}"
        )

    @staticmethod
    def _resolve_base_vlm_path(
        config,
        repo_root: Path,
        explicit_base_vlm_path: str | Path | None,
    ) -> Path | None:
        if explicit_base_vlm_path is not None:
            explicit_path = Path(explicit_base_vlm_path).expanduser().resolve()
            if not explicit_path.exists():
                raise FileNotFoundError(f"Provided --base_vlm_path does not exist: {explicit_path}")
            return explicit_path

        if not hasattr(config.framework, "qwenvl"):
            return None

        configured_model_id = config.framework.qwenvl.get("base_vlm", None)
        if configured_model_id is None:
            return None

        configured_path = Path(str(configured_model_id)).expanduser()
        if configured_path.exists():
            return configured_path.resolve()

        local_candidates = [
            repo_root / configured_path.name,
            repo_root / "Qwen3-VL-2B-Instruct",
            repo_root / "Qwen3-VL-4B-Instruct",
            repo_root / "Qwen2.5-VL-3B-Instruct",
        ]
        for candidate in local_candidates:
            if candidate.exists():
                return candidate.resolve()

        return None

    @staticmethod
    def _resolve_vjepa_encoder_path(
        config,
        repo_root: Path,
        explicit_base_encoder_path: str | Path | None,
    ) -> Path | None:
        if explicit_base_encoder_path is not None:
            explicit_path = Path(explicit_base_encoder_path).expanduser().resolve()
            if not explicit_path.exists():
                raise FileNotFoundError(f"Provided --base_encoder_path does not exist: {explicit_path}")
            return explicit_path

        if not hasattr(config.framework, "vj2_model"):
            return None

        configured_encoder = config.framework.vj2_model.get("base_encoder", None)
        if configured_encoder is None:
            return None

        configured_path = Path(str(configured_encoder)).expanduser()
        if configured_path.exists():
            return configured_path.resolve()

        local_candidates = [
            repo_root / configured_path.name,
            repo_root / "vjepa2-vitl-fpc64-256",
            Path(__file__).resolve().parents[2] / configured_path.name,
            Path(__file__).resolve().parents[2] / "vjepa2-vitl-fpc64-256",
        ]
        for candidate in local_candidates:
            if candidate.exists():
                return candidate.resolve()

        return None

    def predict_action(
        self,
        batch_images,
        instructions,
        state: np.ndarray,
        reset: bool = False,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        instruction = instructions[0]
        pil_image = batch_images[0][0]
        if isinstance(pil_image, np.ndarray):
            pil_image = Image.fromarray(pil_image.astype(np.uint8), mode="RGB")

        raw_state = np.asarray(state, dtype=np.float32)
        normed_state = normalize_state(raw_state, self._state_stats)

        output = self._model.predict_action(
            batch_images=[[pil_image]],
            instructions=[instruction],
            state=normed_state,
        )
        normalized_actions = output["normalized_actions"]
        actions = unnormalize_actions(normalized_actions, self._action_stats)
        return {"actions": actions[0]}

    @property
    def stats_key(self) -> str:
        return self._stats_key
