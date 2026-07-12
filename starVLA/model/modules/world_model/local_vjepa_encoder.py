from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.world_model.vj2_modules import Block


class PatchEmbed3D(nn.Module):
    def __init__(self, embed_dim: int, tubelet_size: int, patch_size: int):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        x = self.proj(x)
        _, _, t, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, t, h, w


class VJEPA2LocalEncoderConfig:
    def __init__(
        self,
        hidden_size: int = 1024,
        tubelet_size: int = 2,
        patch_size: int = 16,
        image_size: int = 384,
    ):
        self.hidden_size = hidden_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.image_size = image_size


class VJEPA2LocalBackbone(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        tubelet_size: int = 2,
        patch_size: int = 16,
        image_size: int = 384,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.config = VJEPA2LocalEncoderConfig(
            hidden_size=hidden_size,
            tubelet_size=tubelet_size,
            patch_size=patch_size,
            image_size=image_size,
        )
        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.patch_embed = PatchEmbed3D(hidden_size, tubelet_size, patch_size)
        self.patch_embed_img = PatchEmbed3D(hidden_size, 1, patch_size)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    use_rope=True,
                    grid_size=image_size // patch_size,
                )
                for _ in range(depth)
            ]
        )
        self.norms_block = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(4)])

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        if pixel_values_videos.dim() != 5:
            raise ValueError("VJEPA2LocalBackbone expects pixel_values_videos with shape [B, T, C, H, W].")
        x = pixel_values_videos.permute(0, 2, 1, 3, 4)
        x, t, h, w = self.patch_embed(x)
        x = x + self.video_mod_embed.to(dtype=x.dtype, device=x.device)
        for block in self.blocks:
            x = block(x, T=t, H_patches=h, W_patches=w)
        return self.norms_block[-1](x)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.get_vision_features(pixel_values_videos)


class VJEPA2LocalEncoderModel(nn.Module):
    """Wrapper matching policy checkpoint keys under `vj_encoder.encoder.*`."""

    def __init__(self, **kwargs):
        super().__init__()
        self.encoder = VJEPA2LocalBackbone(**kwargs)
        self.config = self.encoder.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_vision_features(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.encoder.get_vision_features(pixel_values_videos)

    def forward(self, pixel_values_videos: torch.Tensor) -> torch.Tensor:
        return self.get_vision_features(pixel_values_videos)


class LocalVJEPAVideoProcessor:
    def __init__(self, size: int = 384):
        self.size = size
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def __call__(self, videos, return_tensors: str = "pt"):
        video = torch.as_tensor(np.array(videos), dtype=torch.float32)
        if video.dim() != 4:
            raise ValueError("LocalVJEPAVideoProcessor expects video with shape [T, C, H, W] or [T, H, W, C].")
        if video.shape[-1] == 3:
            video = video.permute(0, 3, 1, 2)
        if video.shape[1] != 3:
            raise ValueError("LocalVJEPAVideoProcessor expects 3-channel RGB video.")

        video = video.unsqueeze(0) / 255.0
        video = F.interpolate(
            video.flatten(0, 1),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        ).view(1, -1, 3, self.size, self.size)
        mean = self.mean.to(device=video.device, dtype=video.dtype)
        std = self.std.to(device=video.device, dtype=video.dtype)
        return {"pixel_values_videos": (video - mean) / std}


def resolve_local_vjepa_checkpoint(model_path: str | Path | None) -> Path | None:
    if model_path is None:
        return None

    path = Path(str(model_path)).expanduser()
    if path.is_file() and path.suffix == ".pt":
        return path.resolve()
    if not path.is_dir():
        return None

    preferred = path / "vjepa2_1_vitl_dist_vitG_384.pt"
    if preferred.exists():
        return preferred.resolve()

    pt_files = sorted(path.glob("*.pt"))
    if len(pt_files) == 1:
        return pt_files[0].resolve()
    return None


def load_local_vjepa_encoder_checkpoint(model: VJEPA2LocalEncoderModel, checkpoint_path: str | Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "encoder" not in checkpoint:
        raise KeyError(f"V-JEPA checkpoint `{checkpoint_path}` does not contain an `encoder` state_dict.")

    converted_state = {}
    for key, value in checkpoint["encoder"].items():
        if key.startswith("module.backbone."):
            converted_state[key.removeprefix("module.backbone.")] = value
        elif key.startswith("backbone."):
            converted_state[key.removeprefix("backbone.")] = value
        elif key.startswith("module."):
            converted_state[key.removeprefix("module.")] = value
        else:
            converted_state[key] = value

    incompatible = model.encoder.load_state_dict(converted_state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Failed to load V-JEPA encoder checkpoint after prefix conversion. "
            f"Missing keys: {incompatible.missing_keys[:20]} "
            f"Unexpected keys: {incompatible.unexpected_keys[:20]} "
            f"Checkpoint: {checkpoint_path}"
        )
