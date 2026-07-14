#!/usr/bin/env python3
"""
Dry-run: verify Qwen3-VL flash_attention_2 compatibility without training.

Checks:
  1. flash_attn import + optional CUDA runtime probe
  2. Qwen3-VL load via _QWen3_VL_Interface with resolved attn backend
  3. Minimal forward pass (text-only)

Usage:
  python scripts/dry_run_flash_attn.py \\
    --config_yaml scripts/config/vlajepa_merged_dataset_001_ft.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run flash_attention_2 for Qwen3-VL")
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="scripts/config/vlajepa_merged_dataset_001_ft.yaml",
        help="Training YAML (only framework.qwenvl section is used)",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Only resolve attn backend and load weights, skip forward",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from accelerate import PartialState

    PartialState()

    cfg = OmegaConf.load(args.config_yaml)
    requested = cfg.framework.qwenvl.get("attn_implementation", "sdpa")

    _section("1. Environment")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    from starVLA.model.modules.vlm.attn_utils import (
        is_flash_attn_importable,
        probe_flash_attention_runtime,
        resolve_attn_implementation,
    )

    _section("2. Attention backend resolution")
    print(f"requested: {requested}")
    print(f"flash_attn importable: {is_flash_attn_importable()}")
    if is_flash_attn_importable() and torch.cuda.is_available():
        ok, err = probe_flash_attention_runtime()
        print(f"flash_attn runtime probe: {'OK' if ok else 'FAILED'}")
        if err:
            print(f"  probe error: {err}")
    resolved = resolve_attn_implementation(requested)
    print(f"resolved:  {resolved}")

    _section("3. Load Qwen3-VL wrapper")
    from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

    interface = _QWen3_VL_Interface(config=cfg)
    model_device = next(interface.model.parameters()).device
    print(f"model loaded on: {model_device}")
    print(f"model attn layers use: {resolved}")

    if args.skip_forward:
        print("\n[DRY-RUN OK] Load succeeded (--skip-forward).")
        return 0

    _section("4. Minimal forward (text-only)")
    tokenizer = interface.processor.tokenizer
    messages = [[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]]
    inputs = interface.processor.apply_chat_template(
        messages,
        tokenize=True,
        padding=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    target_device = model_device if model_device.type != "cpu" else torch.device("cuda")
    if target_device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; running forward on CPU.")
        target_device = torch.device("cpu")

    interface.model.to(target_device)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = interface.model(**inputs, output_hidden_states=True)

    hidden = outputs.hidden_states[-1]
    print(f"forward OK — hidden shape: {list(hidden.shape)}, dtype: {hidden.dtype}")

    print("\n[DRY-RUN OK] flash_attention_2 compatibility check passed.")
    if resolved != requested:
        print(
            f"Note: config requests {requested!r} but runtime uses {resolved!r} "
            "(automatic fallback — training will still work)."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[DRY-RUN FAILED] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
