from __future__ import annotations

"""
Deprecated training-side image preprocessing helpers.

LAM and latent-world training now use deterministic spatial preprocessing from
`data_config` directly, so these PIL/NHWC helpers are retained only for legacy
offline scripts and should not be used in the training data path.
"""

import numpy as np
from PIL import Image
import torch

TARGET_IMAGE_WIDTH = 256
TARGET_IMAGE_HEIGHT = 256
TARGET_IMAGE_WH = (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT)
MAX_SOURCE_ASPECT = 4.0 / 3.0


def _to_uint8_array(frame: np.ndarray) -> np.ndarray:
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        max_val = float(np.max(frame)) if frame.size > 0 else 0.0
        if max_val <= 1.0 + 1e-6:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def preprocess_pil_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"`image` must be PIL.Image.Image, got {type(image)}")

    width, height = image.size
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    if (width / height) > MAX_SOURCE_ASPECT:
        crop_width = max(1, int(np.floor(height * MAX_SOURCE_ASPECT)))
        left = (width - crop_width) // 2
        image = image.crop((left, 0, left + crop_width, height))

    return image.resize(TARGET_IMAGE_WH, resample=Image.BILINEAR)


def preprocess_video_frames_nhwc(frames: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"Expected [T,H,W,C] frames, got shape {frames.shape}")
    if frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB channels in last dim, got shape {frames.shape}")

    processed = []
    for frame in frames:
        frame_u8 = _to_uint8_array(np.asarray(frame))
        processed_img = preprocess_pil_image(Image.fromarray(frame_u8))
        processed.append(np.asarray(processed_img, dtype=np.uint8))
    return np.stack(processed, axis=0)
