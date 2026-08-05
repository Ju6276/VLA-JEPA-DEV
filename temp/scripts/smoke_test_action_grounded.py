#!/usr/bin/env python3
"""Run one real-data forward/backward pass through privileged VLA-JEPA.

This is intentionally a local smoke test rather than a training launcher.  It
uses batch size one and can shrink only the Action Head depth to make the check
fit on a single development GPU; the teacher, Knowledge Insulation, world
decoder, Delta-JEPA auxiliary decoder, and loss routing remain enabled.
"""

from __future__ import annotations

import argparse
import math

import torch
from accelerate import PartialState
from omegaconf import OmegaConf

from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.framework import build_framework


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="scripts/config/vlajepa_merged_dataset_001_e2e.yaml",
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--vjepa-encoder", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--action-layers",
        type=int,
        default=None,
        help="Optional smoke-only override for Action Head Transformer depth.",
    )
    parser.add_argument(
        "--freeze-vlm",
        action="store_true",
        help="Freeze Qwen during the smoke backward pass to reduce memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}.")
    if not torch.cuda.is_available():
        raise RuntimeError("This full-model smoke test requires CUDA.")
    # Repository modules use Accelerate's process-aware logger even outside the
    # full trainer, so initialize its lightweight singleton first.
    PartialState()

    cfg = OmegaConf.load(args.config)
    cfg.framework.qwenvl.base_vlm = args.base_vlm
    cfg.framework.vj2_model.base_encoder = args.vjepa_encoder
    cfg.datasets.vla_data.data_root_dir = args.data_root
    cfg.datasets.vla_data.per_device_batch_size = args.batch_size
    cfg.datasets.vla_data.num_workers = 0
    cfg.framework.action_model.repeated_diffusion_steps = 1
    cfg.trainer.pretrained_checkpoint = None
    cfg.trainer.resume_step = None
    if args.action_layers is not None:
        cfg.framework.action_model.diffusion_model_cfg.num_layers = args.action_layers

    dataset = get_vla_dataset(
        data_cfg=cfg.datasets.vla_data,
        action_horizon=cfg.framework.action_model.action_horizon,
        video_horizon=cfg.framework.vj2_model.num_frames,
        delete_pause_frame=cfg.datasets.vla_data.get("delete_pause_frame", False),
    )
    examples = [dataset[index] for index in range(args.batch_size)]

    model = build_framework(cfg).cuda().train()
    if args.freeze_vlm:
        model.qwen_vl_interface.requires_grad_(False)
    model._freeze_vjepa21()

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(examples)
        loss = outputs["loss_total"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.detach().float().item()}")
    loss.backward()

    expected_losses = {
        "action_loss",
        "current_align_loss",
        "student_wm_loss",
        "teacher_wm_loss",
        "distill_loss",
        "ldad_gt_loss",
        "ldad_pred_loss",
    }
    missing = sorted(expected_losses.difference(outputs))
    if missing:
        raise RuntimeError(f"Smoke output is missing component losses: {missing}")

    grad_modules = {
        "student_current_adapter": model.student_current_adapter,
        "student_predictor": model.student_predictor,
        "shared_world_decoder": model.shared_world_decoder,
        "delta_action_decoder": model.delta_action_decoder,
        "action_model": model.action_model,
    }
    grad_norms = {}
    for name, module in grad_modules.items():
        squared_norm = sum(
            parameter.grad.detach().float().square().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        grad_norms[name] = math.sqrt(squared_norm)
        if not math.isfinite(grad_norms[name]) or grad_norms[name] == 0.0:
            raise RuntimeError(f"Invalid gradient norm for {name}: {grad_norms[name]}")

    print("SMOKE_TEST_OK")
    print(f"batch_size={args.batch_size}")
    print(f"loss_total={loss.detach().float().item():.6f}")
    for key in sorted(expected_losses):
        print(f"{key}={outputs[key].detach().float().item():.6f}")
    for name, norm in grad_norms.items():
        print(f"grad_norm.{name}={norm:.6f}")
    print(f"peak_cuda_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f}")


if __name__ == "__main__":
    main()
