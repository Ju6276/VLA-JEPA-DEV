# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import torch, os


def main(args) -> None:
    vla = baseframework.from_pretrained(
        args.ckpt_path,
    )

    device = torch.device(f"cuda:{str(args.cuda)}")

    if args.use_bf16:
        vla = vla.to(torch.bfloat16)
    vla = vla.to(device).eval()

    subgoals_path = args.subgoals_path
    if subgoals_path is None:
        delta_cfg = getattr(getattr(vla, "config", None), "framework", None)
        if delta_cfg is not None and hasattr(delta_cfg, "delta_jepa"):
            subgoals_path = delta_cfg.delta_jepa.get("subgoals_path", None)

    if subgoals_path and hasattr(vla, "load_subgoal_tracker"):
        vla.load_subgoal_tracker(subgoals_path)
        logging.info("Loaded subgoals from %s", subgoals_path)

    if args.use_verifier and hasattr(vla, "use_verifier_default"):
        vla.use_verifier_default = True
        logging.info("Enabled latent action verifier by default")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        metadata={
            "env": "g1_humanoid",
            "use_verifier": bool(getattr(vla, "use_verifier_default", False)),
            "num_subgoals": getattr(getattr(vla, "subgoal_tracker", None), "num_subgoals", 0),
        },
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--cuda", default=0)
    parser.add_argument("--subgoals_path", type=str, default=None, help="Path to extracted subgoals dir/pkl")
    parser.add_argument("--use_verifier", action="store_true", help="Enable best-of-N latent verification")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10091))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10091 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)
