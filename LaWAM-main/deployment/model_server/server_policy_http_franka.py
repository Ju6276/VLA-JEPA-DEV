#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy.spatial.transform import Rotation as R

from deployment.model_server.server_policy import (
    build_policy_server_metadata,
    load_policy_from_checkpoint,
)
from starVLA.model.tools import read_mode_config

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
    if "__ndarray__" in keys and "dtype" in keys:
        data = decoded["__ndarray__"]
        dtype = np.dtype(decoded["dtype"])
        shape = tuple(decoded.get("shape", ()))
        arr = np.frombuffer(base64.b64decode(data), dtype=dtype) if isinstance(data, str) else np.asarray(data, dtype=dtype)
        return arr.reshape(shape) if shape else arr
    if {"data", "dtype", "shape"}.issubset(keys):
        data = decoded["data"]
        dtype = np.dtype(decoded["dtype"])
        shape = tuple(decoded["shape"])
        arr = np.frombuffer(base64.b64decode(data), dtype=dtype) if isinstance(data, str) else np.asarray(data, dtype=dtype)
        return arr.reshape(shape) if shape else arr
    return decoded


class FrankaHTTPPolicyServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.ckpt_path = Path(args.ckpt_path).expanduser().resolve()
        self.model_config, self.norm_stats = read_mode_config(self.ckpt_path)
        flow_cfg = self.model_config.get("framework", {}).get("action_model", {}).get("flow_cfg", {})
        self.model_uses_state = bool(flow_cfg.get("use_state", True))
        self.unnorm_key = self._check_unnorm_key(self.norm_stats, args.unnorm_key)
        self.action_norm_stats = self.norm_stats[self.unnorm_key]["action"]
        self.state_norm_stats = self.norm_stats[self.unnorm_key].get("state", {})
        self.policy = load_policy_from_checkpoint(
            args.ckpt_path,
            use_bf16=bool(args.use_bf16),
            device=args.device,
        )
        self._policy_lock = threading.Lock()
        self._recording_feature_lock = threading.Lock()
        self._recording_feature_buffers: dict[str, dict[str, Any]] = {}
        self.recording_dir = Path(args.recording_dir).expanduser().resolve()
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = build_policy_server_metadata(
            self.policy,
            ckpt_path=self.ckpt_path,
            server_type="starvla_http_franka",
            env="franka_http",
            supported_eval_envs=["franka"],
            extra_metadata={
                "dataset_contract": "pp_filtered_real_merged_v3.0",
                "unnorm_key": str(self.unnorm_key),
                "default_action_hz": float(self.args.action_hz),
                "default_embodiment_id": int(self.args.embodiment_id),
                "default_num_inference_steps": None if self.args.num_inference_steps is None else int(self.args.num_inference_steps),
                "image_layout": str(self.args.image_layout),
                "model_uses_state": bool(self.model_uses_state),
                "state_stats_dim": int(len(self.state_norm_stats.get("max", self.state_norm_stats.get("q99", [])))),
            },
        )

        if json_numpy is None:
            LOGGER.warning(
                "`json_numpy` is not installed in the server environment. "
                "Standard JSON requests still work; for full numpy-aware request decoding, install `json_numpy`."
            )

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint contains multiple dataset statistics; please pass `--unnorm-key`. "
                    f"Available: {list(norm_stats.keys())}"
                )
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(
                f"Invalid `--unnorm-key`={unnorm_key}. Available: {list(norm_stats.keys())}"
            )
        return unnorm_key

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
            if len(raw_images) == 0:
                return []
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

    def _extract_images(self, payload: dict[str, Any]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        raw_primary = payload.get("primary_image", payload.get("image", payload.get("images")))
        raw_wrist = payload.get("wrist_image", payload.get("wrist_images"))

        if isinstance(raw_primary, dict):
            raw_wrist = raw_primary.get("wrist_image", raw_primary.get("wrist_images", raw_wrist))
            raw_primary = raw_primary.get("primary_image", raw_primary.get("image", raw_primary.get("images")))

        primary_images = self._coerce_image_list(raw_primary, field_name="primary_image")
        wrist_images = self._coerce_image_list(raw_wrist, field_name="wrist_image")

        if not primary_images and not wrist_images:
            raise KeyError(
                "Request must contain images via `images`, `image`, `primary_image`, or explicit `wrist_image`."
            )

        if not primary_images and wrist_images:
            primary_images = [wrist_images.pop(0)]

        if wrist_images:
            return primary_images, wrist_images
        if self.args.image_layout == "all_primary" or len(primary_images) == 1:
            return primary_images, []
        return [primary_images[0]], primary_images[1:]

    @staticmethod
    def _first_present(payload: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in payload:
                return payload[key]
        return None

    @staticmethod
    def _coerce_vector(value: Any, *, field_name: str, expected_dim: int | None = None) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"`{field_name}` must be 1D after squeeze, got shape {tuple(arr.shape)}.")
        if expected_dim is not None and int(arr.shape[0]) != expected_dim:
            raise ValueError(
                f"`{field_name}` must have dim={expected_dim}, got shape {tuple(arr.shape)}."
            )
        return arr

    @classmethod
    def _pose_matrix_to_state_components(cls, pose_matrix: Any) -> tuple[np.ndarray, np.ndarray]:
        pose = np.asarray(pose_matrix, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"`ee_pose_T` must have shape [4,4], got {tuple(pose.shape)}.")
        xyz = pose[:3, 3].astype(np.float32, copy=True)
        rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False).astype(np.float32, copy=False)
        return xyz, rpy

    def _build_raw_state_from_payload(self, payload: dict[str, Any]) -> np.ndarray | None:
        explicit_state = payload.get("state")
        if explicit_state is not None:
            return np.asarray(explicit_state, dtype=np.float32)

        nested_state = None
        for state_key in ("robot_state", "observation", "obs"):
            candidate = payload.get(state_key)
            if isinstance(candidate, dict):
                nested_state = candidate
                break
        state_source = nested_state or payload

        ee_pose_t = self._first_present(state_source, "ee_pose_T", "pose", "pose_matrix")
        ee_pos = self._first_present(state_source, "ee_position", "position", "eef_position")
        ee_rpy = self._first_present(state_source, "ee_euler", "eef_euler", "rpy", "eef_orientation")
        ee_quat_xyzw = self._first_present(state_source, "ee_quat_xyzw", "eef_quat_xyzw")
        ee_quat_wxyz = self._first_present(state_source, "ee_quat_wxyz", "eef_quat_wxyz", "quaternion")
        gripper_width = self._first_present(
            state_source,
            "gripper_width",
            "eef_gripper_width",
            "gripper",
        )

        if ee_pose_t is not None:
            ee_pos_arr, ee_rpy_arr = self._pose_matrix_to_state_components(ee_pose_t)
        elif ee_pos is not None and (ee_rpy is not None or ee_quat_xyzw is not None or ee_quat_wxyz is not None):
            ee_pos_arr = self._coerce_vector(ee_pos, field_name="ee_position", expected_dim=3)
            if ee_rpy is not None:
                ee_rpy_arr = self._coerce_vector(ee_rpy, field_name="ee_euler", expected_dim=3)
            elif ee_quat_xyzw is not None:
                quat_xyzw = self._coerce_vector(ee_quat_xyzw, field_name="ee_quat_xyzw", expected_dim=4)
                ee_rpy_arr = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32, copy=False)
            else:
                quat_wxyz = self._coerce_vector(ee_quat_wxyz, field_name="ee_quat_wxyz", expected_dim=4)
                quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
                ee_rpy_arr = R.from_quat(quat_xyzw).as_euler("xyz", degrees=False).astype(np.float32, copy=False)
        else:
            return None

        if gripper_width is None:
            raise KeyError(
                "Request provides end-effector pose but no `gripper_width`. "
                "Provide `state` directly or include `gripper_width`."
            )
        gripper_arr = self._coerce_vector(gripper_width, field_name="gripper_width")
        if int(gripper_arr.shape[0]) < 1:
            raise ValueError("`gripper_width` must contain at least one scalar value.")
        return np.concatenate([ee_pos_arr, ee_rpy_arr, gripper_arr[:1]], axis=0).astype(np.float32, copy=False)

    def _normalize_state(self, state: Any) -> np.ndarray:
        if not self.state_norm_stats:
            return np.asarray(state, dtype=np.float32)

        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        if state.ndim != 2:
            raise ValueError(f"`state` must have shape [D] or [T,D], got {state.shape}.")

        high_key = "max" if "max" in self.state_norm_stats else "q99"
        low_key = "min" if "min" in self.state_norm_stats else "q01"
        state_high = np.asarray(self.state_norm_stats[high_key], dtype=np.float32)
        state_low = np.asarray(self.state_norm_stats[low_key], dtype=np.float32)

        stats_dim = int(state_high.shape[0])
        if state.shape[-1] != stats_dim:
            raise ValueError(
                f"`state` must match dataset contract dim={stats_dim}, got shape {tuple(state.shape)}."
            )

        normalized = state.copy()
        denom = state_high - state_low
        valid = np.abs(denom) > 1e-12
        if np.any(valid):
            normalized[..., valid] = (
                (normalized[..., valid] - state_low[valid]) / denom[valid] * 2.0 - 1.0
            )
        if np.any(~valid):
            normalized[..., ~valid] = 0.0
        return np.clip(normalized, -1.0, 1.0)

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        action_norm_stats = self.action_norm_stats
        high_key = "max" if "max" in action_norm_stats else "q99"
        low_key = "min" if "min" in action_norm_stats else "q01"

        action_high = np.asarray(action_norm_stats[high_key], dtype=np.float32)
        action_low = np.asarray(action_norm_stats[low_key], dtype=np.float32)
        mask = action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool))

        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]

        stats_dim = int(action_high.shape[0])
        if normalized_actions.shape[-1] < stats_dim:
            raise ValueError(
                "Policy action dimension is smaller than action statistics dimension: "
                f"action_dim={normalized_actions.shape[-1]}, stats_dim={stats_dim}."
            )
        if normalized_actions.shape[-1] > stats_dim:
            normalized_actions = normalized_actions[..., :stats_dim]

        if not np.all(np.isfinite(normalized_actions)):
            raise ValueError(
                "Policy returned non-finite normalized actions: "
                f"min={np.nanmin(normalized_actions)}, max={np.nanmax(normalized_actions)}"
            )

        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)

        denorm = 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low
        raw_actions = np.where(mask, denorm, normalized_actions)
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError(
                "Action denormalization produced non-finite actions: "
                f"min={np.nanmin(raw_actions)}, max={np.nanmax(raw_actions)}"
            )
        return raw_actions
    @staticmethod
    def _sanitize_recording_run_id(value: str | None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
        cleaned = cleaned.strip("._-")
        if not cleaned:
            cleaned = time.strftime("recording_%Y%m%d_%H%M%S")
        return cleaned[:128]

    def save_uploaded_recording(self, *, run_id: str | None, body: bytes) -> dict[str, Any]:
        if not body:
            raise ValueError("Empty recording upload body.")
        safe_run_id = self._sanitize_recording_run_id(run_id)
        out_dir = self.recording_dir / safe_run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "client_recording.npz"
        out_path.write_bytes(body)
        metadata = {
            "run_id": safe_run_id,
            "received_at": time.strftime("%Y%m%d_%H%M%S"),
            "bytes": int(len(body)),
            "path": str(out_path),
        }
        (out_dir / "upload_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Saved uploaded Franka recording to %s (%d bytes)", out_path, len(body))
        features_path = self._flush_recording_feature_buffer(safe_run_id, out_dir=out_dir)
        return {
            "status": "ok",
            "path": str(out_path),
            "bytes": int(len(body)),
            "run_id": safe_run_id,
            "features_path": None if features_path is None else str(features_path),
        }

    def _append_recording_intermediates(
        self,
        payload: dict[str, Any],
        intermediates: dict[str, Any] | None,
    ) -> None:
        if not bool(payload.get("record_intermediates", False)):
            return
        if intermediates is None:
            LOGGER.warning("Recording requested intermediates, but policy did not return them.")
            return

        safe_run_id = self._sanitize_recording_run_id(payload.get("run_id"))
        h_t = np.asarray(intermediates["h_t"], dtype=np.float32)
        h_t1_pred = np.asarray(intermediates["h_t1_pred"], dtype=np.float32)
        frame_h_t = h_t[0] if h_t.ndim == 3 else h_t
        frame_h_t1_pred = h_t1_pred[0] if h_t1_pred.ndim == 3 else h_t1_pred
        vision_tokens_hw_raw = intermediates["vision_tokens_hw"]
        vision_tokens_hw = (int(vision_tokens_hw_raw[0]), int(vision_tokens_hw_raw[1]))

        client_step_idx_raw = payload.get("client_step_idx", -1)
        client_timestamp_raw = payload.get("client_timestamp", np.nan)
        client_step_idx = int(client_step_idx_raw) if client_step_idx_raw is not None else -1
        client_timestamp = float(client_timestamp_raw) if client_timestamp_raw is not None else np.nan

        with self._recording_feature_lock:
            buffer = self._recording_feature_buffers.setdefault(
                safe_run_id,
                {
                    "h_t1_pred": [],
                    "h_t_first": None,
                    "vision_tokens_hw": vision_tokens_hw,
                    "client_step_idx": [],
                    "client_timestamps": [],
                    "request_idx": [],
                },
            )
            if tuple(buffer["vision_tokens_hw"]) != vision_tokens_hw:
                raise ValueError(
                    f"Recording vision_tokens_hw changed for run_id={safe_run_id}: "
                    f"{buffer['vision_tokens_hw']} -> {vision_tokens_hw}."
                )
            if buffer["h_t_first"] is None:
                buffer["h_t_first"] = frame_h_t.astype(np.float32, copy=True)
            buffer["h_t1_pred"].append(frame_h_t1_pred.astype(np.float32, copy=True))
            buffer["client_step_idx"].append(client_step_idx)
            buffer["client_timestamps"].append(client_timestamp)
            buffer["request_idx"].append(len(buffer["request_idx"]))

    def _flush_recording_feature_buffer(self, run_id: str, *, out_dir: Path) -> Path | None:
        safe_run_id = self._sanitize_recording_run_id(run_id)
        with self._recording_feature_lock:
            buffer = self._recording_feature_buffers.pop(safe_run_id, None)
        if not buffer or not buffer["h_t1_pred"]:
            return None

        out_path = out_dir / "features_from_act_requests.npz"
        metadata = {
            "run_id": safe_run_id,
            "num_requests": int(len(buffer["h_t1_pred"])),
            "source": "online_act_request_intermediates",
            "note": "These features correspond to /act requests, not every recorded client control step when action chunks are reused.",
        }
        np.savez_compressed(
            out_path,
            h_t1_pred=np.stack(buffer["h_t1_pred"], axis=0).astype(np.float32, copy=False),
            h_t_first=np.asarray(buffer["h_t_first"], dtype=np.float32),
            vision_tokens_hw=np.asarray(buffer["vision_tokens_hw"], dtype=np.int32),
            client_step_idx=np.asarray(buffer["client_step_idx"], dtype=np.int32),
            client_timestamps=np.asarray(buffer["client_timestamps"], dtype=np.float64),
            request_idx=np.asarray(buffer["request_idx"], dtype=np.int32),
            metadata_json=np.asarray(json.dumps(metadata, indent=2), dtype=np.str_),
        )
        LOGGER.info("Saved recording feature intermediates to %s", out_path)
        return out_path

    def _prepare_example(self, payload: dict[str, Any]) -> dict[str, Any]:
        primary_images, wrist_images = self._extract_images(payload)
        instruction = str(self.args.instruction) if str(self.args.instruction).strip() else self._extract_instruction_from_payload(payload)
        example: dict[str, Any] = {
            "primary_image": primary_images,
            "lang": instruction,
            "embodiment_id": int(self.args.embodiment_id),
            "action_hz": float(self.args.action_hz),
        }
        if wrist_images:
            example["wrist_image"] = wrist_images

        raw_state = self._build_raw_state_from_payload(payload)
        if raw_state is not None:
            example["state"] = self._normalize_state(raw_state)
        return example

    def prepare_only(self, payload: dict[str, Any]) -> dict[str, Any]:
        example = self._prepare_example(payload)
        raw_state = self._build_raw_state_from_payload(payload)
        return {
            "status": "ok",
            "prepared": {
                "lang": example["lang"],
                "embodiment_id": int(example["embodiment_id"]),
                "action_hz": float(example["action_hz"]),
                "primary_image_count": int(len(example["primary_image"])),
                "primary_image_shapes": [list(np.asarray(img).shape) for img in example["primary_image"]],
                "wrist_image_count": int(len(example.get("wrist_image", []))),
                "wrist_image_shapes": [
                    list(np.asarray(img).shape) for img in example.get("wrist_image", [])
                ],
                "raw_state": None if raw_state is None else np.asarray(raw_state, dtype=np.float32),
                "normalized_state": example.get("state"),
                "model_uses_state": bool(self.model_uses_state),
            },
            "metadata": self.metadata,
        }

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        example = self._prepare_example(payload)
        request_num_inference_steps = payload.get("num_inference_steps", None)
        if request_num_inference_steps is None:
            request_num_inference_steps = self.args.num_inference_steps
        if request_num_inference_steps is not None:
            request_num_inference_steps = int(request_num_inference_steps)
            if request_num_inference_steps <= 0:
                raise ValueError(
                    f"`num_inference_steps` must be > 0, got {request_num_inference_steps}."
                )
        predict_kwargs: dict[str, Any] = {
            "examples": [example],
            "num_inference_steps": request_num_inference_steps,
        }
        if bool(payload.get("record_intermediates", False)):
            predict_kwargs["return_intermediates"] = True
        with self._policy_lock:
            infer_output = self.policy.predict_action(**predict_kwargs)

        intermediates = infer_output.get("intermediates") if isinstance(infer_output, dict) else None
        normalized_actions = np.asarray(infer_output["normalized_actions"], dtype=np.float32)
        if normalized_actions.ndim == 3:
            normalized_actions = normalized_actions[0]
        elif normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        elif normalized_actions.ndim != 2:
            raise ValueError(
                f"Unexpected `normalized_actions` shape from policy: {tuple(normalized_actions.shape)}."
            )

        raw_actions = self._unnormalize_actions(normalized_actions)
        self._append_recording_intermediates(payload, intermediates)
        return {
            "status": "ok",
            "actions": raw_actions,
            "normalized_actions": normalized_actions,
            "action_chunk_len": int(raw_actions.shape[0]),
            "action_dim": int(raw_actions.shape[1]),
            "num_inference_steps": request_num_inference_steps,
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
                self._send_json({"status": "error", "message": f"Unknown path: {path}"}, status=HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                path = parsed.path
                if path not in {"/act", "/prepare", "/upload_recording"}:
                    self._send_json(
                        {"status": "error", "message": f"Unknown path: {path}"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    raw_body = self.rfile.read(content_length) if content_length > 0 else (b"" if path == "/upload_recording" else b"{}")
                    if path == "/upload_recording":
                        query = parse_qs(parsed.query)
                        run_id = query.get("run_id", [None])[0]
                        self._send_json(server.save_uploaded_recording(run_id=run_id, body=raw_body))
                        return
                    payload = _decode_special_arrays(json.loads(raw_body.decode("utf-8")))
                    if not isinstance(payload, dict):
                        raise TypeError(f"Request body must decode to a dict, got {type(payload).__name__}.")
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
        LOGGER.info("Franka HTTP policy server listening on http://%s:%d", self.args.host, self.args.port)
        LOGGER.info("Endpoints: POST /act | POST /prepare | POST /upload_recording | GET /health | GET /metadata")
        httpd.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HTTP bridge for examples/Franka/eval_files/franka_client.py, aligned with the LIBERO-style prepare/normalize/infer flow."
    )
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9876)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--embodiment_id", type=int, default=25)
    parser.add_argument("--action_hz", type=float, default=5.0)
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Override diffusion denoising steps at inference time. Default: use checkpoint config.",
    )
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument(
        "--recording_dir",
        "--recording-dir",
        type=str,
        default="logs/franka_client_recordings",
        help="Directory for uploaded Franka client observation recordings.",
    )
    parser.add_argument(
        "--image_layout",
        type=str,
        default="split_primary_wrist",
        choices=["split_primary_wrist", "all_primary"],
        help="`split_primary_wrist`: first image is primary and the rest are wrist views.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    FrankaHTTPPolicyServer(args).serve_forever()


if __name__ == "__main__":
    main()
