#!/usr/bin/env python3
"""Extract a subset of episodes from a LeRobot v2.1 dataset into a new dataset.

Episode indices are remapped to a contiguous range [0, N). Parquet fields
``episode_index`` / ``index`` and meta files are updated accordingly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


COPY_META_FILES = (
    "modality.json",
    "tasks.jsonl",
    "lang_map.json",
    "relative_stats.json",
    "stats_psi0.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"),
        help="Source LeRobot dataset directory",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=None,
        help="Output dataset directory (default: <src>-<N>eps)",
    )
    parser.add_argument("--num-episodes", type=int, default=75, help="Number of episodes to keep")
    parser.add_argument(
        "--strategy",
        choices=("random", "first", "last"),
        default="random",
        help="How to select episodes when --episode-ids is not set",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for random selection")
    parser.add_argument(
        "--episode-ids",
        type=str,
        default=None,
        help="Comma-separated source episode indices, e.g. 0,1,5,10",
    )
    parser.add_argument(
        "--symlink-videos",
        action="store_true",
        help="Symlink videos instead of copying (saves disk, keeps dependency on src)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove destination directory if it already exists",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_episode_ids(
    all_ids: list[int],
    num_episodes: int,
    strategy: str,
    seed: int,
    episode_ids: str | None,
) -> list[int]:
    if episode_ids is not None:
        selected = [int(x.strip()) for x in episode_ids.split(",") if x.strip()]
        missing = sorted(set(selected) - set(all_ids))
        if missing:
            raise ValueError(f"Unknown episode ids: {missing}")
        # keep user order, drop duplicates
        seen = set()
        ordered = []
        for ep in selected:
            if ep not in seen:
                ordered.append(ep)
                seen.add(ep)
        return ordered

    if num_episodes > len(all_ids):
        raise ValueError(
            f"Requested {num_episodes} episodes, but source only has {len(all_ids)}"
        )

    if strategy == "first":
        return all_ids[:num_episodes]
    if strategy == "last":
        return all_ids[-num_episodes:]

    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_ids, size=num_episodes, replace=False)
    return sorted(int(x) for x in chosen)


def compute_dataset_statistics(parquet_paths: list[Path]) -> dict:
    frames = []
    for path in tqdm(parquet_paths, desc="Computing dataset statistics"):
        frames.append(pd.read_parquet(path))
    data = pd.concat(frames, axis=0, ignore_index=True)

    stats = {}
    for col in data.columns:
        if col.startswith("annotation."):
            continue
        values = np.vstack([np.asarray(x, dtype=np.float32) for x in data[col]])
        stats[col] = {
            "mean": np.mean(values, axis=0).tolist(),
            "std": np.std(values, axis=0).tolist(),
            "min": np.min(values, axis=0).tolist(),
            "max": np.max(values, axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(),
            "q99": np.quantile(values, 0.99, axis=0).tolist(),
        }
    return stats


def remap_parquet(
    src_parquet: Path,
    dst_parquet: Path,
    new_episode_index: int,
    global_index_start: int,
) -> int:
    df = pd.read_parquet(src_parquet)
    n = len(df)
    df = df.copy()
    df["episode_index"] = new_episode_index
    df["index"] = np.arange(global_index_start, global_index_start + n, dtype=np.int64)
    if "frame_index" in df.columns:
        df["frame_index"] = np.arange(n, dtype=np.int64)
    dst_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_parquet, index=False)
    return n


def main() -> None:
    args = parse_args()
    src = args.src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"Source dataset not found: {src}")

    info = json.loads((src / "meta" / "info.json").read_text())
    episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    episodes_by_id = {ep["episode_index"]: ep for ep in episodes}
    all_ids = sorted(episodes_by_id)

    selected_ids = select_episode_ids(
        all_ids=all_ids,
        num_episodes=args.num_episodes,
        strategy=args.strategy,
        seed=args.seed,
        episode_ids=args.episode_ids,
    )
    n_eps = len(selected_ids)

    dst = args.dst
    if dst is None:
        dst = src.parent / f"{src.name}-{n_eps}eps"
    dst = dst.resolve()

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Destination already exists: {dst}. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Optional source episodes_stats for remapping
    src_ep_stats = {}
    ep_stats_path = src / "meta" / "episodes_stats.jsonl"
    if ep_stats_path.exists():
        for row in load_jsonl(ep_stats_path):
            src_ep_stats[row["episode_index"]] = row

    data_path_tmpl = info["data_path"]
    video_path_tmpl = info["video_path"]
    chunk_size = info.get("chunks_size", 1000)

    new_episodes = []
    new_ep_stats = []
    parquet_paths = []
    global_index = 0
    total_frames = 0

    print(f"Selecting {n_eps}/{len(all_ids)} episodes from {src}")
    print(f"Strategy: {args.strategy if args.episode_ids is None else 'explicit'}, seed={args.seed}")
    print(f"Source episode ids: {selected_ids}")
    print(f"Writing to: {dst}")

    for new_idx, old_idx in enumerate(tqdm(selected_ids, desc="Copying episodes")):
        ep = episodes_by_id[old_idx]
        length = int(ep["length"])
        old_chunk = old_idx // chunk_size
        new_chunk = new_idx // chunk_size

        src_parquet = src / data_path_tmpl.format(
            episode_chunk=old_chunk, episode_index=old_idx
        )
        dst_parquet = dst / data_path_tmpl.format(
            episode_chunk=new_chunk, episode_index=new_idx
        )
        if not src_parquet.exists():
            raise FileNotFoundError(f"Missing parquet: {src_parquet}")

        n = remap_parquet(src_parquet, dst_parquet, new_idx, global_index)
        if n != length:
            print(
                f"Warning: episode {old_idx} meta length={length}, parquet rows={n}; "
                "using parquet length"
            )
            length = n

        # video
        src_video = src / video_path_tmpl.format(
            episode_chunk=old_chunk, episode_index=old_idx
        )
        dst_video = dst / video_path_tmpl.format(
            episode_chunk=new_chunk, episode_index=new_idx
        )
        if not src_video.exists():
            raise FileNotFoundError(f"Missing video: {src_video}")
        dst_video.parent.mkdir(parents=True, exist_ok=True)
        if args.symlink_videos:
            if dst_video.exists() or dst_video.is_symlink():
                dst_video.unlink()
            dst_video.symlink_to(src_video)
        else:
            shutil.copy2(src_video, dst_video)

        new_ep = dict(ep)
        new_ep["episode_index"] = new_idx
        new_ep["length"] = length
        new_ep["dataset_from_index"] = global_index
        new_ep["dataset_to_index"] = global_index + length - 1
        new_episodes.append(new_ep)

        if old_idx in src_ep_stats:
            stats_row = dict(src_ep_stats[old_idx])
            stats_row["episode_index"] = new_idx
            new_ep_stats.append(stats_row)

        parquet_paths.append(dst_parquet)
        global_index += length
        total_frames += length

    # meta files
    meta_dst = dst / "meta"
    meta_dst.mkdir(parents=True, exist_ok=True)

    write_jsonl(meta_dst / "episodes.jsonl", new_episodes)
    if new_ep_stats:
        write_jsonl(meta_dst / "episodes_stats.jsonl", new_ep_stats)

    for name in COPY_META_FILES:
        src_file = src / "meta" / name
        if src_file.exists():
            shutil.copy2(src_file, meta_dst / name)

    new_info = dict(info)
    new_info["total_episodes"] = n_eps
    new_info["total_frames"] = total_frames
    new_info["total_videos"] = n_eps
    new_info["total_chunks"] = (n_eps - 1) // chunk_size + 1
    (meta_dst / "info.json").write_text(json.dumps(new_info, indent=4) + "\n")

    # Save selection manifest for reproducibility
    manifest = {
        "source": str(src),
        "destination": str(dst),
        "num_episodes": n_eps,
        "strategy": args.strategy if args.episode_ids is None else "explicit",
        "seed": args.seed,
        "source_episode_ids": selected_ids,
        "total_frames": total_frames,
        "symlink_videos": bool(args.symlink_videos),
    }
    (meta_dst / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("Recomputing normalization statistics...")
    stats = compute_dataset_statistics(parquet_paths)
    (meta_dst / "stats.json").write_text(json.dumps(stats, indent=4) + "\n")
    (meta_dst / "stats_gr00t.json").write_text(json.dumps(stats, indent=4) + "\n")

    print("Done.")
    print(f"  episodes: {n_eps}")
    print(f"  frames:   {total_frames}")
    print(f"  output:   {dst}")
    print("Note: do not copy steps_*.pkl from the source; the dataloader will rebuild it.")


if __name__ == "__main__":
    main()
