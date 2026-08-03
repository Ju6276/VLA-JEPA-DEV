#!/usr/bin/env python3
"""
Precompute dataset-side caches used by the current StarVLA training pipeline.

This script intentionally follows the same dataset factory path as training:
`starVLA.dataloader.lerobot_datasets.make_LeRobotSingleDataset`.

It is only responsible for eagerly triggering dataset initialization so that:
1. `meta/stats_gr00t.json` is generated when missing.
2. Optional disk-backed video-frame cache is built and reused.
3. Video-cache prebuild automatically honors `LAM_VIDEO_CACHE_BUILD_WORKERS`.

Examples:
    python precompute_dataset_cache.py --config starVLA/config/training/starvla_train_oxe.yaml
    python precompute_dataset_cache.py --data_root_dir /data/lerobot --data_mix bridge_rt_1
    python precompute_dataset_cache.py --data_root_dir /data/lerobot --dataset_name foo --robot_type widowx
"""

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import yaml

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _extract_vla_data_config(config: dict[str, Any]) -> dict[str, Any]:
    datasets_cfg = config.get("datasets")
    if isinstance(datasets_cfg, dict):
        vla_cfg = datasets_cfg.get("vla_data")
        if isinstance(vla_cfg, dict):
            return dict(vla_cfg)

    legacy_data_cfg = config.get("data")
    if isinstance(legacy_data_cfg, dict):
        warnings.warn(
            "Legacy `data` config format is deprecated. Prefer `datasets.vla_data` from the current training pipeline.",
            DeprecationWarning,
            stacklevel=2,
        )
        init_args = legacy_data_cfg.get("init_args")
        if isinstance(init_args, dict):
            return dict(init_args)
        return dict(legacy_data_cfg)

    return dict(config)


def _normalize_data_cfg(
    config_path: Path | None,
    data_root_dir: Path | None,
    data_mix: str | None,
    video_backend: str | None,
    enable_video_frame_cache: bool,
) -> dict[str, Any]:
    data_cfg: dict[str, Any] = {}
    if config_path is not None:
        print(f"Loading config from: {config_path}")
        data_cfg.update(_extract_vla_data_config(_load_yaml(config_path)))

    if data_root_dir is not None:
        data_cfg["data_root_dir"] = str(data_root_dir)
    if data_mix is not None:
        data_cfg["data_mix"] = data_mix
    if video_backend is not None:
        data_cfg["video_backend"] = video_backend
    if enable_video_frame_cache:
        data_cfg["enable_video_frame_cache"] = True
    else:
        data_cfg.setdefault("enable_video_frame_cache", False)

    return data_cfg


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _summarize_dataset(dataset: Any, dataset_path: Path, elapsed_sec: float) -> None:
    stats_path = dataset_path / "meta" / "stats_gr00t.json"
    print(f"\n{'=' * 80}")
    print("Dataset initialization finished")
    print(f"Elapsed: {elapsed_sec:.2f}s")
    print(f"Dataset length: {len(dataset)}")
    print(f"Trajectory count: {len(dataset.trajectory_ids)}")
    print(f"Stats cache: {stats_path} ({'exists' if stats_path.exists() else 'missing'})")

    cache_dir = getattr(dataset, "_video_cache_dir", None)
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        total_size = sum(path.stat().st_size for path in cache_dir.rglob("*") if path.is_file())
        print(f"Video frame cache: {cache_dir}")
        print(f"Video cache size: {_format_bytes(total_size)}")
    print(f"{'=' * 80}\n")


def precompute_single_dataset(
    data_root_dir: Path,
    dataset_name: str,
    robot_type: str,
    data_cfg: dict[str, Any],
) -> bool:
    dataset_path = data_root_dir / dataset_name

    print(f"\n{'=' * 80}")
    print(f"Precomputing dataset: {dataset_name}")
    print(f"Robot type: {robot_type}")
    print(f"Dataset path: {dataset_path}")
    print(f"{'=' * 80}\n")

    if not dataset_path.exists():
        print(f"Warning: dataset path does not exist: {dataset_path}")
        return False

    try:
        start = time.time()
        dataset = make_LeRobotSingleDataset(
            data_root_dir=data_root_dir,
            data_name=dataset_name,
            robot_type=robot_type,
            mode="train",
            data_cfg=data_cfg,
        )
        _summarize_dataset(dataset, dataset_path=dataset_path, elapsed_sec=time.time() - start)
        return True
    except Exception as exc:
        print(f"Error while processing dataset `{dataset_name}`:")
        print(f"  {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return False


def precompute_mixture(data_root_dir: Path, data_mix: str, data_cfg: dict[str, Any]) -> None:
    print(f"\n{'#' * 80}")
    print(f"# Precomputing mixture: {data_mix}")
    print(f"# Data root: {data_root_dir}")
    print(f"{'#' * 80}\n")

    if data_mix not in DATASET_NAMED_MIXTURES:
        print(f"Error: unknown data_mix `{data_mix}`")
        print(f"Available mixes: {sorted(DATASET_NAMED_MIXTURES.keys())}")
        return

    seen: set[tuple[str, str]] = set()
    datasets_to_process: list[tuple[str, str, float]] = []
    for dataset_name, weight, robot_type in DATASET_NAMED_MIXTURES[data_mix]:
        dedupe_key = (dataset_name, robot_type)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        datasets_to_process.append((dataset_name, robot_type, weight))

    print(f"Unique datasets: {len(datasets_to_process)}\n")
    for dataset_name, robot_type, weight in datasets_to_process:
        print(f"  - {dataset_name:50s} (robot_type={robot_type:20s}, weight={weight:.4f})")
    print()

    success_count = 0
    failed: list[str] = []
    for index, (dataset_name, robot_type, _weight) in enumerate(datasets_to_process, start=1):
        print(f"Progress: [{index}/{len(datasets_to_process)}]")
        ok = precompute_single_dataset(
            data_root_dir=data_root_dir,
            dataset_name=dataset_name,
            robot_type=robot_type,
            data_cfg=data_cfg,
        )
        if ok:
            success_count += 1
        else:
            failed.append(dataset_name)

    print(f"\n{'#' * 80}")
    print("# Precompute finished")
    print(f"# Success: {success_count}/{len(datasets_to_process)}")
    if failed:
        print("# Failed datasets:")
        for dataset_name in failed:
            print(f"#   - {dataset_name}")
    print(f"{'#' * 80}\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute current StarVLA dataset caches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python precompute_dataset_cache.py --config starVLA/config/training/starvla_train_oxe.yaml
  python precompute_dataset_cache.py --data_root_dir /data/lerobot --data_mix bridge_rt_1
  python precompute_dataset_cache.py --data_root_dir /data/lerobot --dataset_name foo --robot_type widowx
  python precompute_dataset_cache.py --config starVLA/config/training/starvla_train_libero.yaml --enable-video-frame-cache
        """,
    )
    parser.add_argument("--config", type=Path, help="Training YAML. Prefer current `datasets.vla_data` format.")
    parser.add_argument("--data_root_dir", type=Path, help="Dataset root directory.")
    parser.add_argument("--data_mix", type=str, help="Named mixture from DATASET_NAMED_MIXTURES.")
    parser.add_argument("--dataset_name", type=str, help="Single dataset name.")
    parser.add_argument("--robot_type", type=str, help="Robot type for `--dataset_name`.")
    parser.add_argument(
        "--video_backend",
        type=str,
        choices=["pyav"],
        help="Optional override for video decoding backend.",
    )
    parser.add_argument(
        "--enable-video-frame-cache",
        action="store_true",
        help="Force-enable disk-backed video-frame cache during precompute.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.data_mix and args.dataset_name:
        parser.error("Use either --data_mix or --dataset_name, not both.")

    data_cfg = _normalize_data_cfg(
        config_path=args.config,
        data_root_dir=args.data_root_dir,
        data_mix=args.data_mix,
        video_backend=args.video_backend,
        enable_video_frame_cache=args.enable_video_frame_cache,
    )

    data_root_value = data_cfg.get("data_root_dir")
    if not data_root_value:
        parser.error("Missing data root. Provide --data_root_dir or set datasets.vla_data.data_root_dir in --config.")
    data_root_dir = Path(data_root_value)

    if args.dataset_name:
        if not args.robot_type:
            parser.error("When using --dataset_name, --robot_type is required.")
        precompute_single_dataset(
            data_root_dir=data_root_dir,
            dataset_name=args.dataset_name,
            robot_type=args.robot_type,
            data_cfg=data_cfg,
        )
        return

    data_mix = data_cfg.get("data_mix")
    if not data_mix:
        parser.error(
            "Missing data mix. Provide --data_mix or set datasets.vla_data.data_mix in --config. "
            "For a single dataset use --dataset_name plus --robot_type."
        )
    precompute_mixture(data_root_dir=data_root_dir, data_mix=str(data_mix), data_cfg=data_cfg)


if __name__ == "__main__":
    main()
