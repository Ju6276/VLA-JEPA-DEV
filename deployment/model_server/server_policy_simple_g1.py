# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

from __future__ import annotations

import argparse
import logging
import os
import socket

import torch

from deployment.model_server.simple_g1_adapter import SimpleG1PolicyAdapter
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


DEFAULT_OPENFAUCET_CKPT = (
    "/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/"
    "steps_40000_pytorch_model.pt"
)
DEFAULT_BASE_VLM_PATH = "/home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct"
DEFAULT_BASE_ENCODER_PATH = "/home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256"


def main(args) -> None:
    device = f"cuda:{str(args.cuda)}" if torch.cuda.is_available() else "cpu"
    policy = SimpleG1PolicyAdapter(
        ckpt_path=args.ckpt_path,
        device=device,
        use_bf16=args.use_bf16,
        base_vlm_path=args.base_vlm_path,
        base_encoder_path=args.base_encoder_path,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating SIMPLE/G1 server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "env": "simple_g1_handover",
            "checkpoint": args.ckpt_path,
            "stats_key": policy.stats_key,
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=DEFAULT_OPENFAUCET_CKPT,
    )
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument(
        "--base_vlm_path",
        type=str,
        default=DEFAULT_BASE_VLM_PATH,
        help="Optional local path overriding config.framework.qwenvl.base_vlm",
    )
    parser.add_argument(
        "--base_encoder_path",
        type=str,
        default=DEFAULT_BASE_ENCODER_PATH,
        help="Optional local path overriding config.framework.vj2_model.base_encoder",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    for key in (
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(key, None)

    parser = build_argparser()
    args = parser.parse_args()
    main(args)
