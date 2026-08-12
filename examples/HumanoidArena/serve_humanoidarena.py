#!/usr/bin/env python3
"""Serve an official VLA-JEPA-based SONIC40 checkpoint to HumanoidArena."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch

# HumanoidArena launches the server with cwd set to this script's directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.HumanoidArena.humanoidarena_protocol import (
    ACTION_DIM,
    ACTION_HORIZON,
    STATE_DIM,
    denormalize_action,
    normalize_state,
    parse_payload,
)
from starVLA.model.framework.base_framework import baseframework


class HumanoidArenaVLAJEPAPolicy:
    def __init__(self, checkpoint: Path, device: str):
        self.device = torch.device(device)
        self.model = baseframework.from_pretrained(str(checkpoint))
        config = self.model.config.framework.action_model
        expected = (config.state_dim, config.action_dim, config.action_horizon)
        if expected != (STATE_DIM, ACTION_DIM, ACTION_HORIZON):
            raise ValueError(
                "Checkpoint contract must be state=64, action=40, horizon=30; "
                f"got {expected}"
            )
        self.model.to(self.device).eval()
        if len(self.model.norm_stats) != 1:
            raise ValueError(f"Expected one normalization tag, got {self.model.norm_stats.keys()}")
        tag_stats = next(iter(self.model.norm_stats.values()))
        self.state_stats = tag_stats["state"]
        self.action_stats = tag_stats["action"]
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        image, state, instruction = parse_payload(payload)
        normalized_state = normalize_state(state, self.state_stats)
        prediction = self.model.predict_action(
            batch_images=[[Image.fromarray(image)]],
            instructions=[instruction],
            state=[normalized_state[None]],
        )["normalized_actions"]
        action = denormalize_action(prediction, self.action_stats)
        return {"action_chunk": action.tolist()}


def make_handler(policy: HumanoidArenaVLAJEPAPolicy):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length)) if length else {}
                if self.path == "/infer":
                    self._reply(200, policy.infer(payload))
                elif self.path == "/reset":
                    policy.reset()
                    self._reply(200, {"ok": True, "seed": payload.get("seed")})
                else:
                    self._reply(404, {"error": f"unknown endpoint: {self.path}"})
            except Exception as exc:  # noqa: BLE001
                self._reply(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[VLA-JEPA-HumanoidArena] {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "--policy-path", dest="checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    policy = HumanoidArenaVLAJEPAPolicy(args.checkpoint, args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(
        f"VLA-JEPA HumanoidArena server listening on {args.host}:{args.port}; "
        f"state={STATE_DIM}, action={ACTION_HORIZON}x{ACTION_DIM} semantic_v3"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
