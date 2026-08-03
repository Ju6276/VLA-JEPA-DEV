#!/usr/bin/env python3

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import threading
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy

try:
    import json_numpy  # type: ignore
except ImportError:
    json_numpy = None
else:
    json_numpy.patch()


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


def _decode_special_arrays(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode_special_arrays(item) for item in value]
    if not isinstance(value, dict):
        return value

    decoded = {key: _decode_special_arrays(item) for key, item in value.items()}
    keys = set(decoded.keys())
    if "__numpy__" in keys and "dtype" in keys:
        data = decoded["__numpy__"]
        dtype = np.dtype(decoded["dtype"])
        shape = tuple(decoded.get("shape", ()))
        arr = (
            np.frombuffer(base64.b64decode(data), dtype=dtype)
            if isinstance(data, str)
            else np.asarray(data, dtype=dtype)
        )
        return arr.reshape(shape) if shape else arr
    if "__ndarray__" in keys and "dtype" in keys:
        data = decoded["__ndarray__"]
        dtype = np.dtype(decoded["dtype"])
        shape = tuple(decoded.get("shape", ()))
        arr = (
            np.frombuffer(base64.b64decode(data), dtype=dtype)
            if isinstance(data, str)
            else np.asarray(data, dtype=dtype)
        )
        return arr.reshape(shape) if shape else arr
    if {"data", "dtype", "shape"}.issubset(keys):
        data = decoded["data"]
        dtype = np.dtype(decoded["dtype"])
        shape = tuple(decoded["shape"])
        arr = (
            np.frombuffer(base64.b64decode(data), dtype=dtype)
            if isinstance(data, str)
            else np.asarray(data, dtype=dtype)
        )
        return arr.reshape(shape) if shape else arr
    return decoded


class IsaacGR00THTTPPolicyServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.model_path = Path(args.model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        self._serving_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
        self.serving_model_path = self._resolve_serving_model_path(self.model_path)

        self.embodiment_tag = EmbodimentTag(args.embodiment_tag)
        self.policy = Gr00tPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=str(self.serving_model_path),
            device=args.device,
            strict=bool(args.strict),
        )
        self.modality_configs = self.policy.get_modality_config()
        self.video_keys = list(self.modality_configs["video"].modality_keys)
        self.state_keys = list(self.modality_configs["state"].modality_keys)
        self.action_keys = list(self.modality_configs["action"].modality_keys)
        self.language_key = list(self.modality_configs["language"].modality_keys)[0]
        self.action_horizon = len(self.modality_configs["action"].delta_indices)
        self._policy_lock = threading.Lock()
        self.metadata = {
            "status": "ok",
            "server_type": "isaac_groot_http_franka",
            "framework": "isaac_groot",
            "dataset_contract": "pp_filtered_real_merged_v3.0",
            "model_path": str(self.model_path),
            "serving_model_path": str(self.serving_model_path),
            "embodiment_tag": self.embodiment_tag.value,
            "device": str(args.device),
            "video_keys": self.video_keys,
            "state_keys": self.state_keys,
            "action_keys": self.action_keys,
            "language_key": self.language_key,
            "action_chunk_len": int(self.action_horizon),
        }

    def _resolve_serving_model_path(self, model_path: Path) -> Path:
        required_processor_files = [
            "processor_config.json",
            "statistics.json",
            "embodiment_id.json",
        ]
        if all((model_path / name).exists() for name in required_processor_files):
            return model_path

        processor_dir = model_path / "processor"
        if not processor_dir.is_dir():
            return model_path
        if not all((processor_dir / name).exists() for name in required_processor_files):
            return model_path

        temp_dir = tempfile.TemporaryDirectory(prefix="isaac_groot_serving_")
        temp_path = Path(temp_dir.name)
        for src in model_path.iterdir():
            dst = temp_path / src.name
            if src.is_dir():
                continue
            os.symlink(src, dst)
        for name in required_processor_files:
            os.symlink(processor_dir / name, temp_path / name)
        self._serving_dir_ctx = temp_dir
        LOGGER.info(
            "Created temporary Isaac-GR00T serving dir %s with processor files from %s",
            temp_path,
            processor_dir,
        )
        return temp_path

    @staticmethod
    def _extract_instruction_from_payload(payload: dict[str, Any]) -> str:
        for key in ("instruction", "lang", "task_description", "instructions"):
            value = payload.get(key)
            if value is not None and str(value).strip():
                return str(value)
        raise KeyError("Request must contain `instruction` (or `lang` / `task_description`).")

    @staticmethod
    def _coerce_image_list(raw_images: Any, *, field_name: str) -> list[np.ndarray]:
        if raw_images is None:
            return []
        if isinstance(raw_images, np.ndarray):
            if raw_images.ndim == 3:
                images = [raw_images]
            elif raw_images.ndim == 4:
                images = [np.asarray(frame) for frame in raw_images]
            else:
                raise ValueError(
                    f"`{field_name}` must have shape [H,W,3] or [N,H,W,3], got {tuple(raw_images.shape)}."
                )
        elif isinstance(raw_images, (list, tuple)):
            images = [np.asarray(frame) for frame in raw_images]
        else:
            raise TypeError(
                f"`{field_name}` must be np.ndarray/list/tuple, got {type(raw_images).__name__}."
            )

        checked: list[np.ndarray] = []
        for idx, image in enumerate(images):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(
                    f"`{field_name}[{idx}]` must have shape [H,W,3], got {tuple(image.shape)}."
                )
            checked.append(np.asarray(image, dtype=np.uint8))
        return checked

    def _extract_images(self, payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        primary_images = self._coerce_image_list(payload.get("primary_image", payload.get("image")), field_name="primary_image")
        wrist_images = self._coerce_image_list(payload.get("wrist_image"), field_name="wrist_image")

        if not primary_images and not wrist_images:
            raise KeyError("Request must contain `primary_image` and preferably `wrist_image`.")
        if not primary_images:
            primary_images = [wrist_images[0]]
        if not wrist_images:
            wrist_images = [primary_images[0]]
        return primary_images[0], wrist_images[0]

    @staticmethod
    def _coerce_state(state: Any) -> np.ndarray:
        arr = np.asarray(state, dtype=np.float32).reshape(-1)
        if arr.shape != (7,):
            raise ValueError(f"`state` must have shape [7], got {tuple(arr.shape)}.")
        return arr

    def _build_observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction = self._extract_instruction_from_payload(payload)
        primary_image, wrist_image = self._extract_images(payload)
        state = self._coerce_state(payload.get("state"))

        observation = {
            "video": {
                self.video_keys[0]: primary_image[None, None, ...],
                self.video_keys[1]: wrist_image[None, None, ...],
            },
            "state": {
                "eef_position": state[:3][None, None, :],
                "eef_orientation": state[3:6][None, None, :],
                "gripper": state[6:7][None, None, :],
            },
            "language": {
                self.language_key: [[instruction]],
            },
        }
        return observation

    def prepare_only(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._build_observation(payload)
        return {
            "status": "ok",
            "prepared": {
                "instruction": observation["language"][self.language_key][0][0],
                "video_keys": {
                    key: list(np.asarray(value).shape) for key, value in observation["video"].items()
                },
                "state_keys": {
                    key: list(np.asarray(value).shape) for key, value in observation["state"].items()
                },
                "language_key": self.language_key,
            },
            "metadata": self.metadata,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        observation = self._build_observation(payload)
        execution_horizon = payload.get("execution_horizon", self.args.execution_horizon)
        if execution_horizon is not None:
            execution_horizon = int(execution_horizon)
            if execution_horizon <= 0:
                raise ValueError(
                    f"`execution_horizon` must be > 0, got {execution_horizon}."
                )

        with self._policy_lock:
            action_dict, info = self.policy.get_action(observation)

        raw_actions = np.concatenate(
            [
                np.asarray(action_dict["eef_position"], dtype=np.float32)[0],
                np.asarray(action_dict["eef_orientation"], dtype=np.float32)[0],
                np.asarray(action_dict["gripper"], dtype=np.float32)[0],
            ],
            axis=-1,
        )
        if execution_horizon is not None:
            raw_actions = raw_actions[:execution_horizon]

        return {
            "status": "ok",
            "actions": raw_actions,
            "action_chunk_len": int(raw_actions.shape[0]),
            "action_dim": int(raw_actions.shape[1]),
            "raw_action_dict": action_dict,
            "info": info,
            "metadata": self.metadata,
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
                self._send_json(
                    {"status": "error", "message": f"Unknown path: {path}"},
                    status=HTTPStatus.NOT_FOUND,
                )

            def do_POST(self) -> None:
                path = urlparse(self.path).path
                if path not in {"/act", "/prepare"}:
                    self._send_json(
                        {"status": "error", "message": f"Unknown path: {path}"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                    payload = _decode_special_arrays(json.loads(raw_body.decode("utf-8")))
                    if not isinstance(payload, dict):
                        raise TypeError(
                            f"Request body must decode to a dict, got {type(payload).__name__}."
                        )
                    if path == "/prepare":
                        self._send_json(server.prepare_only(payload))
                    else:
                        self._send_json(server.predict(payload))
                except Exception as exc:
                    LOGGER.exception("HTTP %s failed", path)
                    self._send_json(
                        {"status": "error", "message": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )

            def _send_json(
                self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
            ) -> None:
                data = json.dumps(_to_jsonable(payload)).encode("utf-8")
                self.send_response(int(status))
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt: str, *args) -> None:
                LOGGER.info("%s - %s", self.address_string(), fmt % args)

        httpd = ThreadingHTTPServer((self.args.host, self.args.port), Handler)
        LOGGER.info(
            "Isaac-GR00T HTTP policy server listening on http://%s:%d",
            self.args.host,
            self.args.port,
        )
        LOGGER.info("Endpoints: POST /act | POST /prepare | GET /health | GET /metadata")
        httpd.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HTTP bridge for Franka real-world deployment using Isaac-GR00T."
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9886)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--embodiment_tag", type=str, default="new_embodiment")
    parser.add_argument("--execution_horizon", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    IsaacGR00THTTPPolicyServer(args).serve_forever()


if __name__ == "__main__":
    main()
