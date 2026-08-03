# Copyright 2025 starVLA community.
"""
V-JEPA 2.1 vision encoder adapter.

The installed `transformers` does not support the V-JEPA 2.1 architecture
(corrected RoPE / modality embeddings / deep-supervision norms / proj_context),
and Meta ships 2.1 only as a raw `.pt` via PyTorch Hub / CDN. This module vendors
Meta's official 2.1 encoder code (under `vjepa21_vendor/`) and wraps it so the
rest of the framework can use it exactly like a HuggingFace `VJEPA2Model`, i.e.
via `encoder.config.{tubelet_size,image_size,hidden_size}` and
`encoder.get_vision_features(pixel_values_videos=...)`.

Reference build recipe: facebookresearch/vjepa2 -> src/hub/backbones.py
`vjepa2_1_vit_large_384` (checkpoint key = "ema_encoder").
"""
import os
import sys
import types

import torch
import torch.nn as nn

from transformers.models.vjepa2.video_processing_vjepa2 import VJEPA2VideoProcessor

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vjepa21_vendor")


def _ensure_vendor_on_path():
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)


def _clean_backbone_key(state_dict):
    # Mirrors src/hub/backbones.py::_clean_backbone_key
    for key, val in state_dict.copy().items():
        _ = state_dict.pop(key)
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        state_dict[key] = val
    return state_dict


class _EncoderConfig(types.SimpleNamespace):
    """Minimal stand-in for HF `VJEPA2Config` (only fields the framework reads)."""


class VJEPA21VisionEncoder(nn.Module):
    """Wraps Meta's V-JEPA 2.1 ViT encoder with a HF-compatible surface."""

    def __init__(self, encoder: nn.Module, img_size: int, tubelet_size: int, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.config = _EncoderConfig(
            image_size=img_size,
            tubelet_size=tubelet_size,
            hidden_size=hidden_size,
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values_videos: [B, T, C, H, W] (HF layout, same as the framework feeds).
        Returns:
            tokens: [B, N, hidden_size] (final-layer normed patch tokens).
        """
        # Meta encoder expects [B, C, T, H, W].
        x = pixel_values_videos.permute(0, 2, 1, 3, 4).contiguous()
        return self.encoder(x, training=False)


def load_vjepa21_encoder(
    checkpoint_path: str,
    arch: str = "vit_large",
    img_size: int = 384,
    patch_size: int = 16,
    tubelet_size: int = 2,
    num_frames: int = 64,
    checkpoint_key: str = "ema_encoder",
) -> VJEPA21VisionEncoder:
    """Build the V-JEPA 2.1 encoder and load weights from a Meta `.pt` checkpoint."""
    _ensure_vendor_on_path()
    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    encoder_kwargs = dict(
        patch_size=patch_size,
        img_size=(img_size, img_size),
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    encoder = vit_encoder.__dict__[arch](**encoder_kwargs)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint_key not in state_dict:
        raise KeyError(
            f"checkpoint_key '{checkpoint_key}' not found in {checkpoint_path}; "
            f"available top-level keys: {list(state_dict.keys())}"
        )
    encoder_state_dict = _clean_backbone_key(state_dict[checkpoint_key])
    missing, unexpected = encoder.load_state_dict(encoder_state_dict, strict=False)
    if missing:
        # pos_embed is expected to be missing because the model uses RoPE.
        non_pos_missing = [k for k in missing if "pos_embed" not in k]
        if non_pos_missing:
            raise RuntimeError(f"Unexpected missing keys when loading 2.1 encoder: {non_pos_missing}")

    return VJEPA21VisionEncoder(
        encoder=encoder,
        img_size=img_size,
        tubelet_size=tubelet_size,
        hidden_size=encoder.embed_dim,
    )


def build_vjepa21_processor(img_size: int = 384) -> VJEPA2VideoProcessor:
    """ImageNet-normalized video processor producing img_size x img_size clips."""
    return VJEPA2VideoProcessor(
        do_resize=True,
        size={"shortest_edge": img_size},
        do_center_crop=True,
        crop_size={"height": img_size, "width": img_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        data_format="channels_first",
    )
