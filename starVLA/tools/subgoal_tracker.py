# Copyright 2025 starVLA community. All rights reserved.
"""Online visual subgoal tracking for World-Verified VLA-JEPA."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import torch
from PIL import Image

from starVLA.model.modules.world_model.delta_jepa import cosine_distance


class SubgoalTracker:
    """Track demo-derived visual subgoals during deployment."""

    def __init__(
        self,
        subgoal_images: Sequence[Image.Image],
        z_goals: Optional[torch.Tensor] = None,
        mode: str = "sequential",
        epsilon: float = 0.15,
    ) -> None:
        if len(subgoal_images) == 0:
            raise ValueError("SubgoalTracker requires at least one subgoal image")

        self.subgoal_images = [image.convert("RGB") for image in subgoal_images]
        self.z_goals = z_goals
        self.mode = mode
        self.epsilon = epsilon
        self.current_index = 0

    @property
    def num_subgoals(self) -> int:
        return len(self.subgoal_images)

    def reset(self) -> None:
        self.current_index = 0

    def current_subgoal_image(self) -> Image.Image:
        return self.subgoal_images[min(self.current_index, self.num_subgoals - 1)]

    def current_subgoal_images(self, batch_size: int) -> List[Image.Image]:
        image = self.current_subgoal_image()
        return [image for _ in range(batch_size)]

    def maybe_precompute_latents(self, encode_fn: Callable[[List[Image.Image]], torch.Tensor]) -> None:
        if self.z_goals is None:
            self.z_goals = encode_fn(self.subgoal_images)

    def update(self, z_current: torch.Tensor) -> int:
        """Advance subgoal index based on current observation latent."""
        if self.z_goals is None:
            return self.current_index

        if z_current.dim() == 1:
            z_current = z_current.unsqueeze(0)

        if self.mode == "nearest":
            distances = cosine_distance(
                z_current,
                self.z_goals.to(z_current.device, dtype=z_current.dtype),
            )
            j_near = int(torch.argmin(distances).item())
            self.current_index = min(j_near + 1, self.num_subgoals - 1)
            return self.current_index

        z_goal = self.z_goals[self.current_index].to(z_current.device, dtype=z_current.dtype).unsqueeze(0)
        if cosine_distance(z_current, z_goal).item() < self.epsilon:
            self.current_index = min(self.current_index + 1, self.num_subgoals - 1)
        return self.current_index

    def state_dict(self) -> dict:
        return {
            "current_index": self.current_index,
            "num_subgoals": self.num_subgoals,
            "mode": self.mode,
            "epsilon": self.epsilon,
        }

    @classmethod
    def from_path(
        cls,
        subgoals_path: str,
        mode: str = "sequential",
        epsilon: float = 0.15,
        encode_fn: Optional[Callable[[List[Image.Image]], torch.Tensor]] = None,
    ) -> "SubgoalTracker":
        path = Path(subgoals_path)
        if path.is_dir():
            return cls.from_directory(path, mode=mode, epsilon=epsilon, encode_fn=encode_fn)
        if path.suffix == ".pkl":
            return cls.from_pickle(path, mode=mode, epsilon=epsilon, encode_fn=encode_fn)
        raise FileNotFoundError(f"Unsupported subgoals path: {subgoals_path}")

    @classmethod
    def from_pickle(
        cls,
        pickle_path: Path,
        mode: str = "sequential",
        epsilon: float = 0.15,
        encode_fn: Optional[Callable[[List[Image.Image]], torch.Tensor]] = None,
    ) -> "SubgoalTracker":
        with open(pickle_path, "rb") as f:
            payload = pickle.load(f)
        frames = payload.get("frames")
        if frames is None:
            raise ValueError(f"Invalid subgoals pickle: {pickle_path}")
        tracker = cls(subgoal_images=frames, mode=mode, epsilon=epsilon)
        if encode_fn is not None:
            tracker.maybe_precompute_latents(encode_fn)
        return tracker

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        mode: str = "sequential",
        epsilon: float = 0.15,
        encode_fn: Optional[Callable[[List[Image.Image]], torch.Tensor]] = None,
    ) -> "SubgoalTracker":
        pickle_path = directory / "subgoals.pkl"
        if pickle_path.exists():
            return cls.from_pickle(pickle_path, mode=mode, epsilon=epsilon, encode_fn=encode_fn)

        manifest_path = directory / "subgoals.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            image_paths = [entry["path"] for entry in manifest["subgoals"]]
            frames = [Image.open(path).convert("RGB") for path in image_paths]
            tracker = cls(subgoal_images=frames, mode=mode, epsilon=epsilon)
            if encode_fn is not None:
                tracker.maybe_precompute_latents(encode_fn)
            return tracker

        image_paths = sorted(directory.glob("subgoal_*.png"))
        if not image_paths:
            raise FileNotFoundError(f"No subgoal assets found in {directory}")
        frames = [Image.open(path).convert("RGB") for path in image_paths]
        tracker = cls(subgoal_images=frames, mode=mode, epsilon=epsilon)
        if encode_fn is not None:
            tracker.maybe_precompute_latents(encode_fn)
        return tracker
