#!/usr/bin/env python3
"""Train only SpatialGoalPredictor on frozen features and held-out episodes.

Feature extraction is separate from this small-head experiment. No action
expert, world predictor, visual encoder, or Qwen parameters are optimized.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starVLA.model.modules.world_model.spatial_goal import SpatialGoalPredictor


TENSOR_FIELDS = ("current", "target", "task", "state", "history", "valid", "ages")


def validate_cache(cache):
    """Reject inconsistent tensors and episode/sample overlap before training."""
    if cache.get("schema_version") != 1:
        raise ValueError("Expected feature cache schema_version=1")
    kwargs = cache["model_kwargs"]
    splits = cache["splits"]
    seen_episodes, seen_samples = set(), set()
    for name in ("train", "val", "test"):
        split = splits[name]
        count = len(split["episode_ids"])
        if not count or len(split["sample_ids"]) != count:
            raise ValueError(f"{name}: need nonempty aligned episode_ids/sample_ids")
        episodes, samples = set(split["episode_ids"]), set(split["sample_ids"])
        if len(samples) != count or samples & seen_samples or episodes & seen_episodes:
            raise ValueError(f"{name}: duplicate samples or overlapping episodes across splits")
        seen_episodes.update(episodes)
        seen_samples.update(samples)
        for field in TENSOR_FIELDS:
            value = split[field]
            if not isinstance(value, torch.Tensor) or value.shape[0] != count:
                raise ValueError(f"{name}.{field}: expected batch dimension {count}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name}.{field}: non-finite features")
        grid_shape = (count, kwargs["grid_size"] ** 2, kwargs["latent_dim"])
        if split["current"].shape != grid_shape or split["target"].shape != grid_shape:
            raise ValueError(f"{name}: wrong current/target grid shape")
        if split["task"].ndim != 3 or split["task"].shape[1] < 1 or split["task"].shape[-1] != kwargs["task_dim"]:
            raise ValueError(f"{name}: wrong task token shape")
        if split["state"].shape != (count, kwargs["state_dim"]):
            raise ValueError(f"{name}: wrong state shape")
        history = split["history"]
        if history.ndim != 4 or history.shape[2:] != grid_shape[1:]:
            raise ValueError(f"{name}: wrong history grid shape")
        valid, ages = split["valid"], split["ages"]
        if valid.dtype != torch.bool or valid.shape != history.shape[:2] or ages.shape != valid.shape:
            raise ValueError(f"{name}: wrong history mask/age shape or dtype")
        if (ages[valid] <= 0).any():
            raise ValueError(f"{name}: valid history must precede current observations")
    return cache


def index_split(split, indices):
    return {key: (value[indices] if isinstance(value, torch.Tensor) else
                  [value[i] for i in indices.tolist()]) for key, value in split.items()}


def model_forward(model, split, indices, device, history_dropout=0.0, intervention=None):
    batch = {key: split[key][indices].to(device=device, dtype=(torch.bool if key == "valid" else torch.float32))
             for key in TENSOR_FIELDS}
    if intervention == "no_history":
        batch["valid"] = torch.zeros_like(batch["valid"])
    elif history_dropout:
        batch["valid"] = batch["valid"] & (torch.rand_like(batch["ages"]) >= history_dropout)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        predicted = model(batch["current"], batch["task"], batch["state"],
                          batch["history"], batch["valid"], batch["ages"])
    return predicted.float(), batch["target"]


@torch.inference_mode()
def predict(model, split, device, batch_size=32, intervention=None):
    model.eval()
    outputs = []
    for start in range(0, len(split["episode_ids"]), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(split["episode_ids"])))
        prediction, _ = model_forward(model, split, indices, device, intervention=intervention)
        outputs.append(prediction.cpu())
    return torch.cat(outputs)


def prediction_metrics(predicted, target, current, episode_ids):
    """All predictors share a target-defined top-change mask, used only in evaluation."""
    predicted, target, current = predicted.float(), target.float(), current.float()
    if predicted.shape != target.shape or current.shape != target.shape or predicted.ndim != 3:
        raise ValueError("Metrics need matching [N,P,D] tensors")
    if len(episode_ids) != predicted.shape[0] or not episode_ids:
        raise ValueError("episode_ids must contain one ID per prediction")
    if not all(torch.isfinite(x).all() for x in (predicted, target, current)):
        raise ValueError("Cannot report metrics for non-finite predictions")
    error = (predicted - target).abs().mean(-1)
    change = (target - current).abs().mean(-1)
    top = change.topk(max(1, math.ceil(change.shape[1] * 0.25)), dim=1).indices
    sample_values = {
        "l1": error.mean(-1),
        "cosine_error": (1 - F.cosine_similarity(predicted, target, dim=-1)).mean(-1),
        "top_change_l1": error.gather(1, top).mean(-1),
    }
    per_episode = {}
    for episode in sorted(set(episode_ids)):
        mask = torch.tensor([item == episode for item in episode_ids])
        per_episode[episode] = {key: float(value[mask].mean()) for key, value in sample_values.items()}
        per_episode[episode]["samples"] = int(mask.sum())
    return {
        "samples": len(episode_ids), "episodes": len(per_episode),
        "sample_mean": {key: float(value.mean()) for key, value in sample_values.items()},
        "episode_mean": {key: float(np.mean([episode[key] for episode in per_episode.values()]))
                         for key in sample_values},
        "per_episode": per_episode,
    }


def baseline_predictions(train, split):
    # Train-only coordinatewise median minimizes constant-residual training L1.
    median_delta = (train["target"].float() - train["current"].float()).median(dim=0).values
    return {"copy_current": split["current"].float(),
            "training_median_delta": split["current"].float() + median_delta}


def train_head(model_kwargs, train, validation, device, *, seed=42, steps=2000,
               batch_size=32, learning_rate=3e-4, history_dropout=0.25, eval_every=100,
               progress_path=None):
    """Only validation episode L1 selects the final head; test data is not accepted."""
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    model = SpatialGoalPredictor(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=learning_rate * 0.05)
    curve, best_weights, best_value, best_step = [], None, float("inf"), 0
    started = time.perf_counter()

    def evaluate(step):
        nonlocal best_weights, best_value, best_step
        train_metrics = prediction_metrics(predict(model, train, device, batch_size), train["target"],
                                            train["current"], train["episode_ids"])
        val_metrics = prediction_metrics(predict(model, validation, device, batch_size), validation["target"],
                                          validation["current"], validation["episode_ids"])
        row = {"step": step, "train_l1": train_metrics["episode_mean"]["l1"],
               "validation_l1": val_metrics["episode_mean"]["l1"],
               "validation_top_change_l1": val_metrics["episode_mean"]["top_change_l1"],
               "seconds": time.perf_counter() - started}
        curve.append(row)
        if row["validation_l1"] < best_value:
            best_value, best_step = row["validation_l1"], step
            best_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress_path:
            with Path(progress_path).open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    evaluate(0)
    for step in range(1, steps + 1):
        model.train()
        indices = torch.randint(len(train["episode_ids"]), (min(batch_size, len(train["episode_ids"])),), generator=generator)
        predicted, target = model_forward(model, train, indices, device, history_dropout)
        loss = F.l1_loss(predicted, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite spatial loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % eval_every == 0 or step == steps:
            evaluate(step)
    model.load_state_dict(best_weights, strict=True)
    return model, {"best_step": best_step, "best_validation_l1": best_value, "curve": curve,
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "seconds": time.perf_counter() - started}


def plot_results(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    curve = report["holdout_fit"]["curve"]
    axes[0].plot([x["step"] for x in curve], [x["train_l1"] for x in curve], label="Train episodes")
    axes[0].plot([x["step"] for x in curve], [x["validation_l1"] for x in curve], label="Validation episodes")
    axes[0].axhline(report["validation_baselines"]["copy_current"]["episode_mean"]["l1"],
                    color="gray", linestyle="--", label="Copy current (validation)")
    axes[0].set(xlabel="Optimizer updates", ylabel="Raw spatial feature L1", title="Head-only training")
    axes[0].legend(fontsize=8)
    names = ["copy_current", "training_median_delta", "spatial_predictor"]
    x = np.arange(len(names))
    for offset, metric, label in [(-0.18, "l1", "All patches"), (0.18, "top_change_l1", "Top 25% changing patches")]:
        axes[1].bar(x + offset, [report["test"][name]["episode_mean"][metric] for name in names], 0.36, label=label)
    axes[1].set_xticks(x, ["Copy current", "Median delta", "Spatial head"])
    axes[1].set(ylabel="Episode mean L1 (lower is better)", title="Held-out test episodes")
    axes[1].legend(fontsize=8)
    fig.savefig(Path(output) / "prediction_probe.png", dpi=170)
    plt.close(fig)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.set_num_threads(args.cpu_threads)
    cache = validate_cache(torch.load(args.features, map_location="cpu", weights_only=True))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to keep prior experiment results")
    output.mkdir(parents=True, exist_ok=True)
    train, val, test = (cache["splits"][name] for name in ("train", "val", "test"))
    kwargs = dict(seed=args.seed, learning_rate=args.learning_rate, eval_every=args.eval_every)
    # Spread the tiny fit across training episodes. No held-out inputs are used.
    tiny_indices = torch.linspace(0, len(train["episode_ids"]) - 1, min(args.overfit_samples, len(train["episode_ids"]))).round().long().unique()
    tiny = index_split(train, tiny_indices)
    tiny_model, tiny_fit = train_head(cache["model_kwargs"], tiny, tiny, device,
        steps=args.overfit_steps, batch_size=len(tiny_indices), history_dropout=0.0,
        progress_path=output / "overfit_curve.jsonl", **kwargs)
    tiny_predictions = {**baseline_predictions(tiny, tiny), "spatial_predictor": predict(tiny_model, tiny, device)}
    tiny_metrics = {name: prediction_metrics(value, tiny["target"], tiny["current"], tiny["episode_ids"])
                    for name, value in tiny_predictions.items()}
    del tiny_model
    model, fit = train_head(cache["model_kwargs"], train, val, device, steps=args.steps,
        batch_size=args.batch_size, history_dropout=args.history_dropout,
        progress_path=output / "fit_curve.jsonl", **kwargs)
    validation_baselines = {name: prediction_metrics(value, val["target"], val["current"], val["episode_ids"])
                            for name, value in baseline_predictions(train, val).items()}
    # Only now access held-out test targets. They never select steps or hyperparameters.
    predictions = {**baseline_predictions(train, test), "spatial_predictor": predict(model, test, device, args.batch_size),
                   "predictor_no_history": predict(model, test, device, args.batch_size, "no_history")}
    test_metrics = {name: prediction_metrics(value, test["target"], test["current"], test["episode_ids"])
                    for name, value in predictions.items()}
    report = {
        "scope": "Isolated spatial head with frozen Qwen/JEPA features; no policy training or robot rollout",
        "arguments": vars(args), "feature_metadata": cache.get("metadata", {}),
        "model_kwargs": cache["model_kwargs"],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256((Path(__file__).resolve().parents[1] / "starVLA/model/modules/world_model/spatial_goal.py").read_bytes()).hexdigest(),
        "overfit": {"fit": tiny_fit, "metrics": tiny_metrics}, "holdout_fit": fit,
        "validation_baselines": validation_baselines, "test": test_metrics,
        "test_comparisons": {},
        "notes": ["Small episode split and one seed are a feasibility probe, not a full benchmark.",
                  "No-history inference is an input intervention, not a separately trained ablation.",
                  "Qwen context includes current image information; this probe does not isolate language understanding.",
                  "Top-changing patches are selected from ground truth only for evaluation and shared by all methods."],
    }
    for baseline in ("copy_current", "training_median_delta"):
        report["test_comparisons"][baseline] = {
            metric: 100 * (test_metrics[baseline]["episode_mean"][metric] - test_metrics["spatial_predictor"]["episode_mean"][metric]) /
                    max(test_metrics[baseline]["episode_mean"][metric], 1e-12)
            for metric in ("l1", "top_change_l1")}
    with (output / "report.json").open("w") as stream:
        json.dump(report, stream, indent=2)
    torch.save({"model_kwargs": cache["model_kwargs"], "state_dict": model.state_dict(),
                "best_step": fit["best_step"], "seed": args.seed}, output / "spatial_goal_head.pt")
    torch.save(predictions, output / "test_predictions.pt")
    plot_results(report, output)
    print(json.dumps({"report": str(output / "report.json"), "test_comparisons": report["test_comparisons"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, help="Frozen feature cache from extract_spatial_goal_features.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--overfit-steps", type=int, default=1000)
    parser.add_argument("--overfit-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--history-dropout", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.steps, args.overfit_steps, args.overfit_samples, args.batch_size, args.eval_every, args.cpu_threads) < 1:
        parser.error("Step counts, sample counts, and batch sizes must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or not 0 <= args.history_dropout <= 1:
        parser.error("Need positive finite learning rate and history dropout in [0,1]")
    run(args)


if __name__ == "__main__":
    main()
