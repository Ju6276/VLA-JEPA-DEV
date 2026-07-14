# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").

"""Attention backend resolution for Qwen-VL wrappers."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_SUPPORTED_ATTN = frozenset({"sdpa", "eager", "flash_attention_2"})


def is_flash_attn_importable() -> bool:
    try:
        import flash_attn  # noqa: F401

        return True
    except ImportError:
        return False


def probe_flash_attention_runtime() -> tuple[bool, Optional[str]]:
    """Run a tiny FA2 kernel on CUDA in a subprocess; return (ok, error_message)."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    import subprocess
    import sys

    probe_code = """
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 1, 1, 64, device="cuda", dtype=torch.float16)
flash_attn_func(q, q, q)
torch.cuda.synchronize()
print("OK")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return False, "flash_attn runtime probe timed out"

    if result.returncode == 0 and "OK" in result.stdout:
        return True, None

    err = (result.stderr or result.stdout or "").strip()
    if not err:
        err = f"flash_attn probe exited with code {result.returncode}"
    return False, err


def resolve_attn_implementation(
    requested: Optional[str],
    *,
    probe_runtime: bool = True,
) -> str:
    """
    Resolve the attention backend for Qwen-VL loading.

    When ``flash_attention_2`` is requested but unavailable or fails a runtime
    probe (e.g. CUDA driver too old), fall back to ``sdpa``.
    """
    requested = (requested or "sdpa").strip().lower()
    if requested not in _SUPPORTED_ATTN:
        logger.warning(
            "Unknown attn_implementation '%s'; falling back to sdpa.",
            requested,
        )
        return "sdpa"

    if requested != "flash_attention_2":
        return requested

    if not is_flash_attn_importable():
        logger.warning(
            "flash_attn is not installed; falling back to sdpa. "
            "Install with: pip install flash-attn --no-build-isolation"
        )
        return "sdpa"

    if probe_runtime:
        ok, err = probe_flash_attention_runtime()
        if not ok:
            logger.warning(
                "flash_attention_2 runtime probe failed (%s); falling back to sdpa.",
                err,
            )
            return "sdpa"

    return "flash_attention_2"


def resolve_qwen_device_map(qwenvl_config: dict) -> Optional[str]:
    """
    Decide ``device_map`` for ``from_pretrained``.

    Default is ``None`` so Accelerate / DeepSpeed can place weights during
    ``accelerator.prepare``. Set ``framework.qwenvl.device_map: cuda`` for
    single-GPU inference-style loading.
    """
    if "device_map" not in qwenvl_config:
        return None

    device_map = qwenvl_config.get("device_map")
    if device_map is None or str(device_map).lower() in {"null", "none"}:
        return None
    return str(device_map)
