"""
Base framework abstraction providing:
- Pretrained loading (config + normalization stats + weights)
- Action space utilities (dimension, stats, (un)normalization)
- Trainable module discovery helper
Note: No device placement or optimizer concerns handled here (delegated to trainer).
"""

import torch.nn as nn
from typing import List
from functools import partial, update_wrapper

from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from typing import List

from pathlib import Path
from typing import Dict, List
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from omegaconf import OmegaConf
import numpy as np
from starVLA.model.tools import auto_get_trainable_modules

from starVLA.model.framework.share_tools import read_mode_config
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.framework.share_tools import dict_to_namespace
from starVLA.model.framework.__init__ import build_framework

logger = initialize_overwatch(__name__)


class _optional_instance_method:
    """Bind an instance when available, retaining explicit-statistics class calls."""

    def __init__(self, function):
        self.function = function
        update_wrapper(self, function)

    def __get__(self, instance, owner):
        return partial(self.function, instance)


# Flat action schemas corresponding to G1HandoverDataConfig and
# SonicLatentDataConfig. These tags identify semantics, not merely dimensions.
_ACTION_NORMALIZATION_MODES = {
    "g1_handover": ["min_max"] * 32 + ["mean_std"] * 4,
    "g1_pick_between_tables": ["min_max"] * 32 + ["mean_std"] * 4,
    "sonic_humanoid": ["min_max"] * 78,
    "garbage": ["min_max"] * 78,
}


# PreTrainedModel, AutoModel, PretrainedConfig,  are so good, find sometime to study them
# TODO @JinhuiYE find sometime to merge yaml config with transformer config

class baseframework(PreTrainedModel):
    """
    Lightweight base class for higher-level VLA model assemblies.
    Subclasses are expected to:
      - Accept a structured config
      - Register components in __init__
      - Use provided helpers for action normalization handling
    """

    def __init__(
        self,
        hf_config = PretrainedConfig()
    ) -> None:
        """
        Initialize base nn.Module. Subclasses add components.
        """
        
        super().__init__(hf_config)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: str,
        config_overrides: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Restore a model instance from a saved checkpoint.

        Workflow:
            1. Resolve checkpoint path
            2. Load config + dataset normalization statistics
            3. Build model with loaded config
            4. Load state_dict strictly (reports missing/unexpected keys)
            5. Attach normalization stats for later un-normalization

        Args:
            pretrained_checkpoint: Path to .pt file inside run/checkpoints directory.
            config_overrides: Optional dotted config paths mapped to replacement
                values, applied before constructing any model components.
            **kwargs: Extra constructor overrides passed to subclass.

        Returns:
            baseframework: Instantiated model (left on CPU; caller decides device).

        Raises:
            RuntimeError: If state_dict key mismatch occurs under strict=True.
            FileNotFoundError: If underlying files are missing (surfaced earlier).
        """
        pretrained_checkpoint = Path(pretrained_checkpoint)
        model_config, norm_stats = read_mode_config(pretrained_checkpoint)  # read config and norm_stats

        config = dict_to_namespace(model_config)
        for key, value in (config_overrides or {}).items():
            OmegaConf.update(config, key, value, merge=False, force_add=True)
        model_config = config
        model_config.trainer.pretrained_checkpoint = None
        # FrameworkModel = cls(config=model_config, **kwargs) # TODO find cls by config
        FrameworkModel = build_framework(cfg=model_config)
        # set for action un-norm
        FrameworkModel.norm_stats = norm_stats
        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")
        # logger.info(f"Loading model weights from `{pretrained_checkpoint}`")
        model_keys = set(FrameworkModel.state_dict().keys())
        checkpoint_keys = set(model_state_dict.keys())
        try:
            FrameworkModel.load_state_dict(model_state_dict, strict=True)
        except RuntimeError as e:
            # must keep all keys matched
            common_keys = model_keys.intersection(checkpoint_keys)
            missing_keys = model_keys - common_keys
            unexpected_keys = checkpoint_keys - common_keys
            if missing_keys:
                logger.warning(f"Missing keys in state_dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys in state_dict: {unexpected_keys}")

            raise e

        # **ensure model is on GPU**
        FrameworkModel = FrameworkModel
        return FrameworkModel

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Infer or validate the dataset stats key used for un-normalization.

        Args:
            norm_stats: Dict[str, dict] mapping dataset key -> stats block.
            unnorm_key: Optional explicit dataset key.

        Returns:
            str: Resolved key.

        Raises:
            AssertionError: If multiple datasets present and key not provided,
                            or provided key not found.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    @_optional_instance_method
    def get_action_stats(self, unnorm_key=None, norm_stats=None, *, control_interface=None):
        """
        Retrieve action statistics and the known interface normalization schema.

        Args:
            unnorm_key: Optional dataset stats key.
            norm_stats: Explicit statistics; required for calls on the class.
            control_interface: Optional "simple" or "sonic" to select reference
                deployment semantics for checkpoints with a different stats tag.

        Returns:
            dict: A shallow copy of the action statistics. Known SIMPLE/SONIC
                tags include per-dimension ``normalization_modes``. An explicit
                schema already stored in the statistics takes precedence unless
                ``control_interface`` explicitly selects a reference interface.
        """
        if norm_stats is None:
            if self is None or not hasattr(self, "norm_stats"):
                raise ValueError("Action statistics are unavailable; load a checkpoint or pass norm_stats explicitly.")
            norm_stats = self.norm_stats
        unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key)
        action_stats = dict(norm_stats[unnorm_key]["action"])
        interface_tags = {"simple": "g1_handover", "sonic": "sonic_humanoid"}
        if control_interface is not None and control_interface not in interface_tags:
            raise ValueError("control_interface must be 'simple' or 'sonic'.")
        schema_key = interface_tags.get(control_interface, unnorm_key)
        if (
            control_interface is not None or "normalization_modes" not in action_stats
        ) and schema_key in _ACTION_NORMALIZATION_MODES:
            modes = _ACTION_NORMALIZATION_MODES[schema_key]
            if len(action_stats["min"]) != len(modes):
                raise ValueError(f"Action statistics for {unnorm_key} must contain {len(modes)} dimensions.")
            action_stats["normalization_modes"] = list(modes)
            if control_interface is not None:
                action_stats["normalization_clip_mask"] = [mode == "min_max" for mode in modes]
            if schema_key in {"sonic_humanoid", "garbage"}:
                # SonicStar clips every channel, then scales only mask=True
                # channels. SIMPLE's xxy adapter does not use this mask.
                mask = np.asarray(action_stats.get("mask", [True] * len(modes)), dtype=bool)
                if mask.shape != (len(modes),):
                    raise ValueError("SONIC action mask must contain 78 values.")
                action_stats["normalization_modes"] = np.where(mask, modes, "identity").tolist()
                action_stats.setdefault("normalization_clip_mask", [True] * len(modes))
        return action_stats

    @property
    def trainable_module_keys(self, max_depth=1) -> List[str]:
        """
        Enumerate trainable submodule names up to a depth.

        Args:
            max_depth: Descent depth when traversing module tree.

        Returns:
            List[str]: Module path names considered trainable.
        """
        keys = auto_get_trainable_modules(self, max_depth=max_depth)  # auto check which modules are trainable
        return keys

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict[str, np.ndarray],
        normalization_modes=None,
    ) -> np.ndarray:
        """
        Restore actions using explicit per-dimension normalization semantics.

        A mode string applies to every channel; a list specifies each channel.
        Modes are min_max, mean_std, q99, binary, and identity. Explicit modes
        take precedence over ``action_norm_stats["normalization_modes"]``.
        With neither schema, legacy q01/q99 scaling and its mask are used.
        Min-max and q99 channels are clipped before scaling, matching the
        SIMPLE xxy and SonicStar deployment adapters. Mean/std channels retain
        their unbounded values. ``normalization_clip_mask`` can specify the
        clipping channels explicitly (SONIC also clips unscaled mask=False
        channels). Binary conversion requires binary mode.

        Args:
            normalized_actions: Array with action channels on the final axis,
                including [D], [T,D], or [B,T,D]. The input is not modified.
            action_norm_stats: Per-channel statistics required by the selected
                modes. Legacy mask=False channels pass through unchanged.

        Returns:
            np.ndarray: Unnormalized actions (same shape as input).
        """
        values = np.asarray(normalized_actions)
        if values.ndim == 0 or not np.issubdtype(values.dtype, np.number):
            raise ValueError("Actions must be a numeric array with channels on the final axis.")
        values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
        dimensions = values.shape[-1]
        modes = normalization_modes
        if modes is None:
            modes = action_norm_stats.get("normalization_modes")
        if modes is None:
            mask = np.asarray(action_norm_stats.get("mask", np.ones(dimensions, dtype=bool)), dtype=bool)
            if mask.shape != (dimensions,):
                raise ValueError("Action normalization mask must match the final action dimension.")
            modes = np.where(mask, "q99", "identity")
        elif isinstance(modes, str):
            modes = [modes] * dimensions
        modes = np.asarray(modes)
        if modes.shape != (dimensions,):
            raise ValueError("normalization_modes must contain one mode per action dimension.")
        unknown = set(modes.tolist()) - {"min_max", "mean_std", "q99", "binary", "identity"}
        if unknown:
            raise ValueError(f"Unknown action normalization modes: {sorted(unknown)}")
        clip_mask = np.asarray(
            action_norm_stats.get("normalization_clip_mask", np.isin(modes, ["min_max", "q99"])),
            dtype=bool,
        )
        if clip_mask.shape != (dimensions,):
            raise ValueError("normalization_clip_mask must match the final action dimension.")
        values = np.where(clip_mask, np.clip(values, -1, 1), values)

        def statistics(key):
            stats = np.asarray(action_norm_stats[key], dtype=values.dtype)
            if stats.shape != (dimensions,):
                raise ValueError(f"Action statistic {key} must contain {dimensions} values.")
            return stats

        actions = values.copy()
        for mode in set(modes.tolist()):
            selected = modes == mode
            if mode in {"min_max", "q99"}:
                lower_key, upper_key = ("min", "max") if mode == "min_max" else ("q01", "q99")
                low, high = statistics(lower_key)[selected], statistics(upper_key)[selected]
                normalized = values[..., selected]
                actions[..., selected] = (normalized + 1) / 2 * (high - low) + low
            elif mode == "mean_std":
                actions[..., selected] = values[..., selected] * statistics("std")[selected] + statistics("mean")[selected]
            elif mode == "binary":
                actions[..., selected] = values[..., selected] > 0.5
        return actions
