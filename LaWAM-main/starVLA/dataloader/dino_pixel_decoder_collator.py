"""
Lightweight collator for DinoPixelDecoder training.

Extracts a single frame per sample from the VLA data pipeline,
applies ImageNet normalization, and returns a simple batch dict.

Input:  list of raw samples with "primary_videos" [V, T, C, H, W] uint8
Output: {"images": [B, C, H, W] float32 ImageNet-normalized}
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoPixelDecoderCollator:

    def __init__(self, *, image_resolution: int = 256) -> None:
        self.image_resolution = image_resolution
        self.training = True

    def train(self) -> "DinoPixelDecoderCollator":
        self.training = True
        return self

    def eval(self) -> "DinoPixelDecoderCollator":
        self.training = False
        return self

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(features) == 0:
            raise ValueError("DinoPixelDecoderCollator received an empty feature list.")

        images = []
        for sample in features:
            primary_videos = sample["primary_videos"]  # [V, T, C, H, W] uint8
            frame = primary_videos[0, 0].contiguous()  # [C, H, W] uint8, first view first frame
            images.append(frame)

        batch_images = torch.stack(images, dim=0).to(dtype=torch.float32).div_(255.0)  # [B, C, H, W] [0,1]

        mean = batch_images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = batch_images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        batch_images.sub_(mean).div_(std)

        return {"images": batch_images}
