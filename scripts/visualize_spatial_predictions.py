#!/usr/bin/env python3
"""Visualize future JEPA grids with one training-only PCA and shared error scales.

Colors in the latent columns represent PCA coordinates, not generated RGB or
TaskSpatialReader attention. Test samples are selected by episode/frame order.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2
import decord
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_training_pca(train, max_tokens):
    """Use deterministic evenly spaced tokens from train current+target grids."""
    current, target = train["current"], train["target"]
    if current.shape != target.shape or current.ndim != 3:
        raise ValueError("Training current/target features must share [N,P,D] shape")
    n, p, dim = current.shape
    token_count = n * p
    indices = torch.linspace(0, 2 * token_count - 1, min(max_tokens, 2 * token_count)).round().long().unique()
    fit = torch.empty(len(indices), dim, dtype=torch.float64)
    first = indices < token_count
    fit[first] = current.reshape(-1, dim)[indices[first]].double()
    fit[~first] = target.reshape(-1, dim)[indices[~first] - token_count].double()
    mean = fit.mean(0)
    centered = fit - mean
    eigenvalues, eigenvectors = torch.linalg.eigh(centered.T @ centered / max(1, len(fit) - 1))
    components = eigenvectors[:, -3:].flip(1).T.contiguous()
    # Resolve each eigenvector's sign deterministically for color reproducibility.
    for row in components:
        if row[row.abs().argmax()] < 0:
            row.neg_()
    scores = centered @ components.T
    low = torch.quantile(scores, 0.01, dim=0)
    high = torch.quantile(scores, 0.99, dim=0)
    high = torch.maximum(high, low + 1e-12)
    projector = {
        "fit_split": "train", "fit_sources": ["current", "target"],
        "selection": "Evenly spaced flattened token indices over train current followed by train target",
        "fitted_tokens": len(indices), "available_tokens": 2 * token_count,
        "fit_flat_token_indices": indices.tolist(),
        "mean": mean.tolist(), "components": components.tolist(),
        "color_percentiles": [1, 99], "color_low": low.tolist(), "color_high": high.tolist(),
        "sign_rule": "Largest-absolute loading of each component is positive",
        "explained_variance_ratio": (eigenvalues[-3:].flip(0) / eigenvalues.clamp_min(0).sum()).tolist(),
    }

    def colorize(tokens):
        transformed = (tokens.double() - mean) @ components.T
        return ((transformed - low) / (high - low)).clamp(0, 1).numpy()

    return colorize, projector


def select_samples(cache, ranks):
    test = cache["splits"]["test"]
    rows = {row["sample_id"]: row for row in cache["metadata"]["samples"]["test"]}
    episodes = sorted(set(test["episode_ids"]))
    if any(rank < 0 or rank >= len(episodes) for rank in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError(f"Episode ranks must be unique values between 0 and {len(episodes) - 1}")
    selected = []
    for rank in ranks:
        episode = episodes[rank]
        indices = [index for index, value in enumerate(test["episode_ids"]) if value == episode]
        indices.sort(key=lambda index: rows[test["sample_ids"][index]]["frame_index"])
        middle = len(indices) // 2
        index = indices[middle]
        row = rows[test["sample_ids"][index]]
        if row["episode_id"] != episode:
            raise ValueError("Cache tensor/manifest episode mismatch")
        selected.append({**row, "test_tensor_index": index, "sorted_episode_rank": rank,
                         "sorted_anchor_rank": middle, "anchors_in_episode": len(indices)})
    return selected


def load_rgb_pair(dataset_path, metadata, row, size):
    info = json.loads((dataset_path / "meta/info.json").read_text())
    modality = json.loads((dataset_path / "meta/modality.json").read_text())
    cameras = list(modality["video"].values())
    if len(cameras) != 1:
        raise ValueError("This figure requires a single ego camera")
    episode = row["episode_index"]
    path = dataset_path / info["video_path"].format(
        episode_index=episode, episode_chunk=episode // info.get("chunks_size", 1000),
        video_key=cameras[0]["original_key"],
    )
    reader = decord.VideoReader(str(path), num_threads=1)
    timestamps = np.asarray(reader.get_frame_timestamp(range(len(reader))))[:, 0]
    requested = np.asarray([row["timestamp"], row["future_timestamp"]], dtype=np.float64)
    indices = np.abs(timestamps[:, None] - requested[None]).argmin(0)
    frames = reader.get_batch(indices).asnumpy()
    # Production dataset preprocessing uses RGB decord frames and OpenCV resize.
    aligned = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) for frame in frames]
    decoding = {
        "video_path": str(path), "video_sha256": sha256(path), "backend": "decord",
        "requested_timestamps": requested.tolist(), "decoded_frame_indices": indices.tolist(),
        "decoded_timestamps": timestamps[indices].tolist(),
        "rgb_alignment": f"RGB square resize to {size}x{size}, OpenCV INTER_LINEAR",
    }
    return aligned, decoding


def run(args):
    torch.set_num_threads(args.cpu_threads)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "visualization_metadata.json"
    if metadata_path.exists():
        raise ValueError("Use a new output directory to preserve previous visualization metadata")
    cache = torch.load(args.features, map_location="cpu", weights_only=True, mmap=True)
    predictions = torch.load(args.predictions, map_location="cpu", weights_only=True, mmap=True)
    if cache.get("schema_version") != 1:
        raise ValueError("Expected feature cache schema_version=1")
    test = cache["splits"]["test"]
    head = predictions[args.prediction_key]
    if head.shape != test["target"].shape:
        raise ValueError("Predictions do not match the cached test grid shape")
    if "copy_current" in predictions and not torch.equal(predictions["copy_current"].float(), test["current"].float()):
        raise ValueError("Predictions and cached test inputs differ; check their sample ordering")
    episode_count = len(set(test["episode_ids"]))
    if not episode_count:
        raise ValueError("Need at least one test episode")
    ranks = args.episode_ranks
    if ranks is None:
        ranks = np.rint(np.linspace(0, episode_count - 1, min(4, episode_count))).astype(int).tolist()
    selected = select_samples(cache, ranks)
    indices = torch.tensor([row["test_tensor_index"] for row in selected])
    current, target, predicted = (value[indices].float() for value in (test["current"], test["target"], head))
    if not all(torch.isfinite(value).all() for value in (current, target, predicted)):
        raise ValueError("Selected features/predictions must be finite")
    side = math.isqrt(current.shape[1])
    if side * side != current.shape[1]:
        raise ValueError("Spatial tokens must form one square grid")
    colorize, projector = fit_training_pca(cache["splits"]["train"], args.pca_tokens)
    colored = [colorize(value).reshape(len(selected), side, side, 3) for value in (current, target, predicted)]
    copy_error = (current - target).abs().mean(-1).reshape(len(selected), side, side).numpy()
    head_error = (predicted - target).abs().mean(-1).reshape(len(selected), side, side).numpy()
    vmax = float(max(copy_error.max(), head_error.max(), 1e-12))
    norm = Normalize(vmin=0, vmax=vmax)
    dataset_path = Path(args.dataset_root or cache["metadata"]["dataset_path"])
    config = cache["metadata"]["config"]
    image_size = int(config["framework"]["vj2_model"].get("image_size", 384))

    titles = ["Current RGB", "True future RGB", "Current latent\n(training PCA)",
              "True future latent\n(same PCA)", "Predicted future latent\n(same PCA)",
              "Copy-current L1", "Spatial-head L1"]
    fig, axes = plt.subplots(len(selected), 7, figsize=(20, 3.1 * len(selected) + 1.2),
                             squeeze=False, layout="constrained")
    image = None
    for row_index, selected_row in enumerate(selected):
        rgb, decode_metadata = load_rgb_pair(dataset_path, cache["metadata"], selected_row, image_size)
        selected_row["rgb_decoding"] = decode_metadata
        for column in range(7):
            ax = axes[row_index, column]
            if row_index == 0:
                ax.set_title(titles[column], fontsize=11, pad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            if column < 2:
                ax.imshow(rgb[column])
            elif column < 5:
                ax.imshow(colored[column - 2][row_index], interpolation="nearest")
            else:
                error = copy_error[row_index] if column == 5 else head_error[row_index]
                image = ax.imshow(error, cmap="inferno", norm=norm, interpolation="nearest")
                ax.set_xlabel(f"Mean L1 = {error.mean():.4f}", fontsize=10)
        axes[row_index, 0].set_ylabel(
            f"Episode {selected_row['source_episode_index']}\n"
            f"frame {selected_row['frame_index']} → {selected_row['future_frame_index']}\n"
            f"{selected_row['timestamp']:.2f} → {selected_row['future_timestamp']:.2f} s",
            fontsize=10, labelpad=10,
        )
        selected_row["displayed_sample_metrics"] = {
            "copy_current_l1": float(copy_error[row_index].mean()),
            "spatial_head_l1": float(head_error[row_index].mean()),
        }
    colorbar = fig.colorbar(image, ax=axes[:, -2:].ravel().tolist(), shrink=0.8, fraction=0.04, pad=0.02)
    colorbar.set_label("Mean absolute feature error per cell (shared scale)", fontsize=10)
    fig.suptitle(
        "Future spatial-grid prediction: fixed held-out episodes\n"
        "Latent colors are a shared training-only PCA projection, not generated RGB or task attention",
        fontsize=15,
    )
    png = args.output / "spatial_future_predictions.png"
    pdf = args.output / "spatial_future_predictions.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "features": str(args.features.resolve()), "features_sha256": sha256(args.features),
        "predictions": str(args.predictions.resolve()), "predictions_sha256": sha256(args.predictions),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(__file__),
        "prediction_key": args.prediction_key, "dataset_path": str(dataset_path.resolve()),
        "selection_rule": "Sort source episode IDs; select requested zero-based ranks; within each sort frame indices and take index len//2",
        "episode_ranks": ranks, "selected_test_samples": selected,
        "selection_uses_errors": False, "grid_size": side, "pca": projector,
        "error_scale": {"metric": "mean absolute raw JEPA feature error over channels per grid cell",
                        "vmin": 0.0, "vmax": vmax, "shared_over": "both methods and all displayed samples", "colormap": "inferno"},
        "notes": ["PCA is fitted exclusively to a deterministic sample of training current/target features.",
                  "All three latent columns use the same PCA components and train-derived 1st/99th percentile RGB scales.",
                  "Only real current and future camera images are RGB; colored latent grids are not image reconstruction.",
                  "This figure does not display TaskSpatialReader attention and does not establish object localization.",
                  "Displayed samples were selected by IDs and middle anchor, without inspecting prediction errors.",
                  "PCA shows three feature directions; quantitative L1 uses every feature dimension."],
        "artifacts": {"png": str(png), "pdf": str(pdf)},
    }
    with metadata_path.open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"png": str(png), "pdf": str(pdf), "metadata": str(metadata_path),
                      "selected_episodes": [row["source_episode_index"] for row in selected],
                      "sample_metrics": [row["displayed_sample_metrics"] for row in selected]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--prediction-key", default="spatial_predictor")
    parser.add_argument("--dataset-root", type=Path, help="Override the recorded single dataset directory")
    parser.add_argument("--episode-ranks", type=int, nargs="+", help="Zero-based sorted test episode ranks; defaults to up to four evenly spaced episodes")
    parser.add_argument("--pca-tokens", type=int, default=8192)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.pca_tokens < 3 or args.cpu_threads < 1:
        parser.error("pca-tokens must be at least 3; cpu-threads must be positive")
    run(args)
