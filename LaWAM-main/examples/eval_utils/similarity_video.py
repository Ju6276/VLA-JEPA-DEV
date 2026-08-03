from __future__ import annotations

from pathlib import Path
from typing import Sequence

import imageio
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _to_cpu_float_tensor(value: np.ndarray | torch.Tensor, *, name: str) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        array = value
        if not array.flags.writeable:
            array = np.array(array, copy=True)
        tensor = torch.from_numpy(array)
    elif torch.is_tensor(value):
        tensor = value.detach()
    else:
        raise TypeError(f"`{name}` must be np.ndarray or torch.Tensor, got {type(value)}.")
    return tensor.to(device="cpu", dtype=torch.float32)


def _validate_vision_tokens_hw(vision_tokens_hw: tuple[int, int]) -> tuple[int, int]:
    if len(vision_tokens_hw) != 2:
        raise ValueError(f"`vision_tokens_hw` must have 2 entries, got {vision_tokens_hw}.")
    grid_h = int(vision_tokens_hw[0])
    grid_w = int(vision_tokens_hw[1])
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"`vision_tokens_hw` must be positive, got {(grid_h, grid_w)}.")
    return grid_h, grid_w


def extract_anchor_feature(
    tokens: np.ndarray | torch.Tensor,
    anchor_row: int,
    anchor_col: int,
    vision_tokens_hw: tuple[int, int],
) -> torch.Tensor:
    grid_h, grid_w = _validate_vision_tokens_hw(vision_tokens_hw)
    row = int(anchor_row)
    col = int(anchor_col)
    if row < 0 or col < 0:
        raise ValueError(f"`anchor_row` and `anchor_col` must be >= 0, got {(row, col)}.")
    if row >= grid_h or col >= grid_w:
        raise ValueError(
            f"Anchor patch {(row, col)} is out of bounds for grid {(grid_h, grid_w)}."
        )

    tokens_t = _to_cpu_float_tensor(tokens, name="tokens")
    if tokens_t.ndim != 2:
        raise ValueError(f"`tokens` must have shape [K, D], got {tuple(tokens_t.shape)}.")
    num_tokens = int(tokens_t.shape[0])
    expected_tokens = int(grid_h * grid_w)
    if num_tokens != expected_tokens:
        raise ValueError(
            f"`tokens` length mismatch: got K={num_tokens}, expected grid_h*grid_w={expected_tokens}."
        )

    tokens_grid = tokens_t.view(grid_h, grid_w, -1).permute(2, 0, 1).contiguous()
    tokens_grid = F.normalize(tokens_grid, p=2, dim=0)
    return tokens_grid[:, row, col].contiguous()


def render_similarity_overlay(
    dst_tokens: np.ndarray | torch.Tensor,
    anchor_feature: np.ndarray | torch.Tensor,
    vision_tokens_hw: tuple[int, int],
    target_image: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    alpha: float,
    cmap: str,
) -> np.ndarray:
    if float(vmax) <= float(vmin):
        raise ValueError(f"`vmax` must be > `vmin`, got vmin={vmin}, vmax={vmax}.")
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"`alpha` must be in [0, 1], got {alpha}.")
    if not isinstance(target_image, np.ndarray):
        raise TypeError(f"`target_image` must be np.ndarray, got {type(target_image)}.")
    if target_image.ndim != 3 or int(target_image.shape[2]) != 3:
        raise ValueError(f"`target_image` must have shape [H, W, 3], got {target_image.shape}.")
    if target_image.dtype != np.uint8:
        raise TypeError(f"`target_image` must be uint8, got {target_image.dtype}.")

    grid_h, grid_w = _validate_vision_tokens_hw(vision_tokens_hw)
    dst_tokens_t = _to_cpu_float_tensor(dst_tokens, name="dst_tokens")
    anchor_t = _to_cpu_float_tensor(anchor_feature, name="anchor_feature").reshape(-1)

    if dst_tokens_t.ndim != 2:
        raise ValueError(f"`dst_tokens` must have shape [K, D], got {tuple(dst_tokens_t.shape)}.")
    if anchor_t.ndim != 1:
        raise ValueError(f"`anchor_feature` must have shape [D], got {tuple(anchor_t.shape)}.")

    num_tokens = int(dst_tokens_t.shape[0])
    expected_tokens = int(grid_h * grid_w)
    if num_tokens != expected_tokens:
        raise ValueError(
            f"`dst_tokens` length mismatch: got K={num_tokens}, expected grid_h*grid_w={expected_tokens}."
        )
    if int(dst_tokens_t.shape[1]) != int(anchor_t.shape[0]):
        raise ValueError(
            f"Feature dim mismatch between `dst_tokens` and `anchor_feature`: "
            f"{tuple(dst_tokens_t.shape)} vs {tuple(anchor_t.shape)}."
        )

    dst_grid = dst_tokens_t.view(grid_h, grid_w, -1).permute(2, 0, 1).contiguous()
    dst_grid = F.normalize(dst_grid, p=2, dim=0)
    anchor_t = F.normalize(anchor_t.unsqueeze(0), p=2, dim=-1).squeeze(0)

    sim_map = (dst_grid * anchor_t.view(-1, 1, 1)).sum(dim=0)
    sim_map = F.interpolate(
        sim_map.unsqueeze(0).unsqueeze(0),
        size=(int(target_image.shape[0]), int(target_image.shape[1])),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)

    sim_np = sim_map.numpy()
    sim_norm = np.clip((sim_np - float(vmin)) / float(vmax - vmin), 0.0, 1.0)
    heatmap = plt.get_cmap(str(cmap))(sim_norm)
    heatmap_rgb = (heatmap[..., :3] * 255.0).round().astype(np.uint8)

    overlay = (
        (1.0 - float(alpha)) * target_image.astype(np.float32)
        + float(alpha) * heatmap_rgb.astype(np.float32)
    )
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def fit_token_rgb_projection(
    token_batches: Sequence[np.ndarray | torch.Tensor],
) -> dict[str, np.ndarray]:
    flattened: list[np.ndarray] = []
    for batch in token_batches:
        arr = _to_cpu_float_tensor(batch, name="token_batch").numpy()
        if arr.ndim == 2:
            flattened.append(arr)
        elif arr.ndim == 3:
            flattened.append(arr.reshape(-1, arr.shape[-1]))
        else:
            raise ValueError(f"Expected token batch [K,D] or [S,K,D], got {tuple(arr.shape)}.")
    if not flattened:
        raise ValueError("Cannot fit RGB projection on empty token batches.")

    stacked = np.concatenate(flattened, axis=0).astype(np.float32, copy=False)
    if stacked.ndim != 2 or stacked.shape[0] == 0:
        raise ValueError(f"Expected stacked token features [N,D], got {tuple(stacked.shape)}.")

    mean = stacked.mean(axis=0, keepdims=True)
    centered = stacked - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = np.zeros((stacked.shape[1], 3), dtype=np.float32)
    if vh.size > 0:
        transposed = vh[: min(3, vh.shape[0])].T.astype(np.float32, copy=False)
        components[:, : transposed.shape[1]] = transposed
    source_projected = centered @ components
    projected_scale = source_projected.std(axis=0, keepdims=True).astype(np.float32, copy=False)
    projected_scale = np.maximum(projected_scale, 1e-6)
    return {
        "mean": mean.astype(np.float32, copy=False),
        "components": components,
        "projected_scale": projected_scale,
    }


def render_pca_image(
    tokens: np.ndarray | torch.Tensor,
    projection: dict[str, np.ndarray],
    vision_tokens_hw: tuple[int, int],
    *,
    target_hw: tuple[int, int] | None = None,
    scale: float = 1.5,
) -> np.ndarray:
    grid_h, grid_w = _validate_vision_tokens_hw(vision_tokens_hw)
    arr = _to_cpu_float_tensor(tokens, name="tokens").numpy()
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected [1,K,D] or [K,D], got {tuple(arr.shape)}.")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected tokens [K,D], got {tuple(arr.shape)}.")
    expected_tokens = int(grid_h * grid_w)
    if int(arr.shape[0]) != expected_tokens:
        raise ValueError(
            f"`tokens` length mismatch: got K={int(arr.shape[0])}, expected grid_h*grid_w={expected_tokens}."
        )

    projected = (arr - projection["mean"]) @ projection["components"]
    projected = projected / projection["projected_scale"]
    projected = 1.0 / (1.0 + np.exp(-projected * float(scale)))
    pca_rgb = projected.reshape(grid_h, grid_w, 3).clip(0.0, 1.0).astype(np.float32)
    if target_hw is not None:
        pca_rgb = torch.from_numpy(pca_rgb).permute(2, 0, 1).unsqueeze(0)
        pca_rgb = F.interpolate(
            pca_rgb,
            size=(int(target_hw[0]), int(target_hw[1])),
            mode="nearest",
        ).squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(pca_rgb * 255.0, 0.0, 255.0).round().astype(np.uint8)


def save_video(frames: Sequence[np.ndarray], path: str | Path, fps: int) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path_obj.as_posix(), fps=int(fps)) as writer:
        for frame in frames:
            frame_arr = np.asarray(frame)
            if frame_arr.ndim != 3 or int(frame_arr.shape[2]) != 3:
                raise ValueError(f"Each frame must have shape [H, W, 3], got {frame_arr.shape}.")
            if frame_arr.dtype != np.uint8:
                raise TypeError(f"Each frame must be uint8, got {frame_arr.dtype}.")
            writer.append_data(frame_arr)
