# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 

import logging
import socket
import argparse
import time
from pathlib import Path
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


def load_policy_from_checkpoint(
    ckpt_path: str,
    *,
    use_bf16: bool = False,
    device: str = "cuda",
):
    total_start = time.perf_counter()
    logging.info("Loading policy checkpoint into CPU memory from %s", ckpt_path)
    vla = baseframework.from_pretrained(ckpt_path)
    logging.info(
        "Checkpoint materialization completed in %.1fs",
        time.perf_counter() - total_start,
    )
    if use_bf16:
        cast_start = time.perf_counter()
        logging.info("Casting policy to bfloat16")
        vla = vla.to(torch.bfloat16)
        logging.info("Policy cast to bfloat16 in %.1fs", time.perf_counter() - cast_start)
    move_start = time.perf_counter()
    logging.info("Moving policy to device=%s", device)
    vla = vla.to(device)
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    logging.info(
        "Policy moved to %s in %.1fs (startup total %.1fs)",
        device,
        time.perf_counter() - move_start,
        time.perf_counter() - total_start,
    )
    return vla.eval()


def build_policy_server_metadata(
    policy,
    *,
    ckpt_path: str | Path,
    server_type: str,
    env: str = "generic",
    supported_eval_envs: list[str] | None = None,
    extra_metadata: dict | None = None,
) -> dict:
    framework_cfg = _cfg_get(getattr(policy, "config", None), "framework", None)
    framework_name = _cfg_get(framework_cfg, "name", policy.__class__.__name__)
    metadata = {
        "env": env,
        "server_type": server_type,
        "supported_eval_envs": supported_eval_envs or [],
        "ckpt_path": str(Path(ckpt_path).expanduser().resolve()),
        "framework_name": str(framework_name),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = load_policy_from_checkpoint(
        args.ckpt_path,
        use_bf16=bool(args.use_bf16),
        device="cuda",
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    metadata = build_policy_server_metadata(
        vla,
        ckpt_path=args.ckpt_path,
        server_type="starvla_websocket",
        env="generic",
        supported_eval_envs=["simpler_env", "libero", "robocasa365", "robocasa_tabletop", "robotwin"],
    )

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=metadata,
    )
    logging.info("Websocket policy server initialized; entering serve loop on 0.0.0.0:%d", args.port)
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    main(args)
