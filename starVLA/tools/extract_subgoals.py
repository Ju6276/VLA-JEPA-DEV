#!/usr/bin/env python3
"""Extract demo-derived visual subgoals for World-Verified VLA-JEPA."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoVideoProcessor

from starVLA.dataloader.gr00t_lerobot.video import get_frames_by_indices
from starVLA.model.modules.world_model.delta_jepa import cosine_distance, pool_vjepa_tokens


def load_encoder(encoder_path: str, device: str = "cuda"):
    if str(encoder_path).endswith(".pt"):
        from starVLA.model.modules.world_model.vjepa21_encoder import (
            build_vjepa21_processor,
            load_vjepa21_encoder,
        )

        encoder = load_vjepa21_encoder(checkpoint_path=encoder_path, img_size=384)
        processor = build_vjepa21_processor(img_size=384)
    else:
        encoder = AutoModel.from_pretrained(encoder_path)
        processor = AutoVideoProcessor.from_pretrained(encoder_path)
    encoder = encoder.to(device).eval()
    return encoder, processor


@torch.inference_mode()
def encode_image(encoder, processor, image: Image.Image, device: str = "cuda") -> torch.Tensor:
    frame = np.array(image.convert("RGB"), dtype=np.uint8)
    video = np.stack([frame] * 8, axis=0)
    pixel_values = processor(videos=[video], return_tensors="pt")["pixel_values_videos"].to(device)
    tokens = encoder.get_vision_features(pixel_values_videos=pixel_values)
    return pool_vjepa_tokens(tokens).squeeze(0).cpu()


def extract_uniform_subgoals(num_frames: int, num_subgoals: int) -> List[int]:
    if num_subgoals <= 0:
        return []
    indices = [
        min(num_frames - 1, max(0, int(round((j + 1) * num_frames / num_subgoals) - 1)))
        for j in range(num_subgoals)
    ]
    return sorted(set(indices))


def extract_jepa_change_subgoals(
    frames: Sequence[Image.Image],
    encoder,
    processor,
    num_subgoals: int,
    device: str = "cuda",
) -> List[int]:
    if len(frames) <= 1 or num_subgoals <= 0:
        return [0]

    latents = [encode_image(encoder, processor, frame, device=device) for frame in frames]
    deltas = [
        cosine_distance(latents[i].unsqueeze(0), latents[i - 1].unsqueeze(0)).item()
        for i in range(1, len(latents))
    ]
    cumulative = np.cumsum(deltas)
    if cumulative[-1] <= 1e-8:
        return extract_uniform_subgoals(len(frames), num_subgoals)

    targets = [(j + 1) / num_subgoals * cumulative[-1] for j in range(num_subgoals)]
    indices = [int(np.searchsorted(cumulative, target)) for target in targets]
    return [min(len(frames) - 1, idx + 1) for idx in indices]


def load_demo_frames(demo_dir: Path, image_key: str = "rs_view") -> List[Image.Image]:
    candidates = [
        demo_dir / "frames",
        demo_dir / "images" / image_key,
        demo_dir,
    ]
    for folder in candidates:
        if not folder.exists():
            continue
        frame_paths = sorted(
            list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
        )
        if frame_paths:
            return [Image.open(path).convert("RGB") for path in frame_paths]
    raise FileNotFoundError(f"No frames found under {demo_dir}")


def find_lerobot_episode_video(dataset_root: Path, episode_index: int, video_key: str) -> Path:
    episode_name = f"episode_{episode_index:06d}.mp4"
    patterns = [
        f"videos/chunk-*/{video_key}/{episode_name}",
        f"videos/chunk-*/observation.images.{video_key}/{episode_name}",
        f"videos/chunk-*/observation.images.{video_key.replace('video.', '')}/{episode_name}",
    ]
    for pattern in patterns:
        matches = sorted(dataset_root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"Could not find episode video for index={episode_index}, video_key={video_key} under {dataset_root}"
    )


def load_lerobot_episode_frames(
    dataset_root: Path,
    episode_index: int,
    video_key: str = "observation.images.rs_view",
    frame_stride: int = 5,
    video_backend: str = "decord",
) -> List[Image.Image]:
    video_path = find_lerobot_episode_video(dataset_root, episode_index, video_key)

    if video_backend == "decord":
        import decord

        vr = decord.VideoReader(str(video_path), num_threads=1)
        total_frames = len(vr)
        frame_indices = list(range(0, total_frames, max(1, frame_stride)))
        frames_np = vr.get_batch(frame_indices).asnumpy()
    else:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = list(range(0, total_frames, max(1, frame_stride)))
        frames_np = get_frames_by_indices(
            str(video_path),
            frame_indices,
            video_backend=video_backend,
            video_backend_kwargs={"num_threads": 1},
        )
        cap.release()

    return [Image.fromarray(frame).convert("RGB") for frame in frames_np]


def save_subgoals(
    frames: Sequence[Image.Image],
    indices: Sequence[int],
    output_dir: Path,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for rank, frame_idx in enumerate(indices):
        image = frames[frame_idx]
        out_path = output_dir / f"subgoal_{rank:02d}_frame_{frame_idx:05d}.png"
        image.save(out_path)
        saved.append({"rank": rank, "frame_idx": int(frame_idx), "path": str(out_path)})

    with open(output_dir / "subgoals.json", "w", encoding="utf-8") as f:
        json.dump({"subgoals": saved, **metadata}, f, indent=2)

    with open(output_dir / "subgoals.pkl", "wb") as f:
        pickle.dump({"frames": [frames[i] for i in indices], "indices": list(indices), **metadata}, f)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract visual subgoals from successful demos.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo_dir", type=str, help="Directory containing one successful demo.")
    source.add_argument("--lerobot_dataset", type=str, help="LeRobot dataset root directory.")
    parser.add_argument("--episode_index", type=int, default=0, help="Episode index for --lerobot_dataset.")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save subgoal images/metadata.")
    parser.add_argument("--encoder_path", type=str, required=True, help="V-JEPA encoder path (HF dir or .pt).")
    parser.add_argument("--num_subgoals", type=int, default=5)
    parser.add_argument(
        "--method",
        type=str,
        default="jepa_change",
        choices=["uniform", "jepa_change"],
    )
    parser.add_argument("--image_key", type=str, default="observation.images.rs_view")
    parser.add_argument("--frame_stride", type=int, default=5, help="Subsample stride for long LeRobot episodes.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.lerobot_dataset:
        dataset_root = Path(args.lerobot_dataset)
        frames = load_lerobot_episode_frames(
            dataset_root=dataset_root,
            episode_index=args.episode_index,
            video_key=args.image_key,
            frame_stride=args.frame_stride,
        )
        source_desc = f"lerobot:{dataset_root}:episode_{args.episode_index}"
    else:
        demo_dir = Path(args.demo_dir)
        frames = load_demo_frames(demo_dir, image_key=args.image_key.split(".")[-1])
        source_desc = str(demo_dir)

    if args.method == "uniform":
        indices = extract_uniform_subgoals(len(frames), args.num_subgoals)
    else:
        encoder, processor = load_encoder(args.encoder_path, device=args.device)
        indices = extract_jepa_change_subgoals(
            frames=frames,
            encoder=encoder,
            processor=processor,
            num_subgoals=args.num_subgoals,
            device=args.device,
        )

    metadata = {
        "method": args.method,
        "num_subgoals": args.num_subgoals,
        "source": source_desc,
        "encoder_path": args.encoder_path,
        "image_key": args.image_key,
    }
    save_subgoals(frames, indices, output_dir, metadata)
    print(f"Saved {len(indices)} subgoals to {output_dir}")


if __name__ == "__main__":
    main()
