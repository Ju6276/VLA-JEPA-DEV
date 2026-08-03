#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


class DatasetReplayHTTPServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dataset_root = Path(args.dataset_root).expanduser().resolve()
        self.episode_index = int(args.episode_index)
        self.chunk_size = int(args.action_chunk_size)
        self.loop = bool(args.loop)
        if self.chunk_size <= 0:
            raise ValueError(f"`action_chunk_size` must be > 0, got {self.chunk_size}.")

        self.raw_actions = self._load_episode_actions(self.episode_index)
        if self.raw_actions.ndim != 2 or int(self.raw_actions.shape[1]) < 7:
            raise ValueError(
                f"Expected action trajectory with shape [T, >=7], got {tuple(self.raw_actions.shape)}."
            )
        self.raw_actions = np.asarray(self.raw_actions[:, :7], dtype=np.float32)
        self.cursor = 0
        self._lock = threading.Lock()
        self.metadata = {
            "server_type": "dataset_replay_http",
            "dataset_root": str(self.dataset_root),
            "episode_index": int(self.episode_index),
            "action_chunk_size": int(self.chunk_size),
            "loop": bool(self.loop),
            "num_actions": int(self.raw_actions.shape[0]),
            "action_dim": int(self.raw_actions.shape[1]),
        }
        LOGGER.info(
            "Loaded replay trajectory | dataset=%s | episode=%d | steps=%d | chunk=%d | loop=%s",
            self.dataset_root,
            self.episode_index,
            int(self.raw_actions.shape[0]),
            self.chunk_size,
            self.loop,
        )

    def _load_episode_actions(self, episode_index: int) -> np.ndarray:
        episode_path = self.dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        if not episode_path.exists():
            raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
        df = pd.read_parquet(episode_path, columns=["actions"])
        if "actions" not in df.columns:
            raise KeyError(f"`actions` column not found in {episode_path}")
        return np.stack(df["actions"].to_numpy())

    def _next_chunk(self) -> np.ndarray:
        if int(self.raw_actions.shape[0]) == 0:
            raise RuntimeError("Replay trajectory contains zero actions.")

        start = int(self.cursor)
        end = min(start + self.chunk_size, int(self.raw_actions.shape[0]))
        chunk = np.asarray(self.raw_actions[start:end], dtype=np.float32)

        if self.loop:
            if int(chunk.shape[0]) < self.chunk_size:
                need = self.chunk_size - int(chunk.shape[0])
                chunk = np.concatenate([chunk, self.raw_actions[:need]], axis=0)
                self.cursor = need % int(self.raw_actions.shape[0])
            else:
                self.cursor = end % int(self.raw_actions.shape[0])
        else:
            self.cursor = end
            if int(chunk.shape[0]) < self.chunk_size:
                pad = np.repeat(self.raw_actions[-1:], repeats=self.chunk_size - int(chunk.shape[0]), axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)

        return chunk

    def predict(self) -> dict[str, Any]:
        with self._lock:
            chunk = self._next_chunk()
            cursor = int(self.cursor)
        return {
            "status": "ok",
            "actions": chunk,
            "episode_index": int(self.episode_index),
            "cursor": cursor,
        }

    def serve_forever(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = urlparse(self.path).path
                if path == "/health":
                    self._send_json({"status": "ok"})
                    return
                if path == "/metadata":
                    self._send_json(server.metadata)
                    return
                self._send_json({"status": "error", "message": f"Unknown path: {path}"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path != "/act":
                    self._send_json(
                        {"status": "error", "message": f"Unknown path: {path}"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                try:
                    self._send_json(server.predict())
                except Exception as exc:
                    LOGGER.exception("HTTP /act replay failed")
                    self._send_json(
                        {"status": "error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )

            def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
                data = json.dumps(_to_jsonable(payload)).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args) -> None:
                LOGGER.info("%s - %s", self.address_string(), fmt % args)

        httpd = ThreadingHTTPServer((self.args.host, self.args.port), Handler)
        LOGGER.info("Dataset replay HTTP server listening on http://%s:%d", self.args.host, self.args.port)
        LOGGER.info("Endpoints: POST /act | GET /health | GET /metadata")
        httpd.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay expert actions from a dataset for the Franka HTTP client.")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument("--loop", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    DatasetReplayHTTPServer(args).serve_forever()


if __name__ == "__main__":
    main()
