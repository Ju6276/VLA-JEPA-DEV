#!/usr/bin/env python3
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from deployment.model_server.server_policy import (
    build_policy_server_metadata,
    load_policy_from_checkpoint,
)
from starVLA.model.tools import read_mode_config


LOGGER = logging.getLogger(__name__)

FOLD_TOWEL_ACTION_ORDER = (
    "left_x",
    "left_y",
    "left_z",
    "left_roll",
    "left_pitch",
    "left_yaw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_roll",
    "right_pitch",
    "right_yaw",
    "right_gripper",
)


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


def _recv_exact(conn: socket.socket, count: int) -> bytes | None:
    chunks: list[bytes] = []
    remaining = int(count)
    while remaining > 0:
        chunk = conn.recv(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_json(conn: socket.socket) -> dict[str, Any] | None:
    header = _recv_exact(conn, 4)
    if not header:
        return None
    size = struct.unpack("<L", header)[0]
    body = _recv_exact(conn, size)
    if body is None:
        return None
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object, got {type(payload).__name__}.")
    return payload


def _send_json(conn: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(_to_jsonable(payload)).encode("utf-8")
    conn.sendall(struct.pack("<L", len(body)))
    conn.sendall(body)


def _recv_image(conn: socket.socket) -> np.ndarray:
    header = _recv_exact(conn, 4)
    if not header:
        raise ConnectionError("Connection closed while reading image length.")
    size = struct.unpack("<L", header)[0]
    image_bytes = _recv_exact(conn, size)
    if image_bytes is None:
        raise ConnectionError("Connection closed while reading image bytes.")
    encoded = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image from client.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class FoldTowelTCPPolicyServer:
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
        self._master_queue_by_client: dict[str, list[np.ndarray]] = {}
        self._ensemble_tail_by_client: dict[str, np.ndarray] = {}
        self.recording_dir = Path(args.recording_dir).expanduser().resolve()
        self.recording_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = build_policy_server_metadata(
            self.policy,
            ckpt_path=self.ckpt_path,
            server_type="starvla_tcp_fold_towel",
            env="fold_towel_tcp",
            supported_eval_envs=["fold_towel"],
            extra_metadata={
                "dataset_contract": "fold_towel_lerobot_v3.0",
                "unnorm_key": str(self.unnorm_key),
                "default_action_hz": float(args.action_hz),
                "default_embodiment_id": int(args.embodiment_id),
                "default_num_inference_steps": None if args.num_inference_steps is None else int(args.num_inference_steps),
                "move_steps": int(args.move_steps),
                "latency_step": int(args.latency_step),
                "command_dim": int(args.command_dim),
                "action_order": list(FOLD_TOWEL_ACTION_ORDER),
                "action_ensemble": bool(args.action_ensemble),
                "adaptive_ensemble_alpha": float(args.adaptive_ensemble_alpha),
                "model_uses_state": bool(self.model_uses_state),
                "state_stats_dim": int(len(self.state_norm_stats.get("max", self.state_norm_stats.get("q99", [])))),
            },
        )

    @staticmethod
    def _check_unnorm_key(norm_stats: dict[str, Any], unnorm_key: str | None) -> str:
        if unnorm_key is None:
            if len(norm_stats) != 1:
                raise ValueError(
                    "Checkpoint contains multiple dataset statistics; pass `--unnorm-key`. "
                    f"Available: {list(norm_stats.keys())}"
                )
            return next(iter(norm_stats.keys()))
        if unnorm_key not in norm_stats:
            raise ValueError(f"Invalid `--unnorm-key`={unnorm_key}. Available: {list(norm_stats.keys())}")
        return unnorm_key

    @staticmethod
    def _sanitize_recording_run_id(value: str | None) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
        cleaned = cleaned.strip("._-")
        if not cleaned:
            cleaned = time.strftime("recording_%Y%m%d_%H%M%S")
        return cleaned[:128]

    @staticmethod
    def _state14_from_payload(payload: dict[str, Any]) -> np.ndarray:
        left = np.asarray(payload["follow1_pos"], dtype=np.float32)
        right = np.asarray(payload["follow2_pos"], dtype=np.float32)
        if left.ndim == 1:
            left = left[None, :]
        if right.ndim == 1:
            right = right[None, :]
        if left.shape[-1] != 7 or right.shape[-1] != 7:
            raise ValueError(
                "`follow1_pos` and `follow2_pos` must end in dim 7, "
                f"got {left.shape} and {right.shape}."
            )
        if left.shape[0] != right.shape[0]:
            raise ValueError(f"State histories must have same length, got {left.shape[0]} and {right.shape[0]}.")
        return np.concatenate([left, right], axis=-1).astype(np.float32, copy=False)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        if not self.state_norm_stats:
            return np.asarray(state, dtype=np.float32)

        state = np.asarray(state, dtype=np.float32)
        if state.ndim == 1:
            state = state[None, :]
        high_key = "max" if "max" in self.state_norm_stats else "q99"
        low_key = "min" if "min" in self.state_norm_stats else "q01"
        state_high = np.asarray(self.state_norm_stats[high_key], dtype=np.float32)
        state_low = np.asarray(self.state_norm_stats[low_key], dtype=np.float32)
        stats_dim = int(state_high.shape[0])
        if state.shape[-1] > stats_dim:
            state = state[..., :stats_dim]
        normalized = state.copy()
        dims = min(int(normalized.shape[-1]), stats_dim)
        denom = state_high[:dims] - state_low[:dims]
        valid = np.abs(denom) > 1e-12
        normalized_dims = normalized[..., :dims]
        if np.any(valid):
            normalized_dims[..., valid] = (
                (normalized_dims[..., valid] - state_low[:dims][valid]) / denom[valid] * 2.0 - 1.0
            )
        if np.any(~valid):
            normalized_dims[..., ~valid] = 0.0
        return np.clip(normalized, -1.0, 1.0)

    def _unnormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        action_norm_stats = self.action_norm_stats
        if "min" not in action_norm_stats or "max" not in action_norm_stats:
            raise KeyError(
                "FoldTowelDataConfig trains actions with `min_max`; checkpoint action stats must contain `min` and `max`."
            )
        action_high = np.asarray(action_norm_stats["max"], dtype=np.float32)
        action_low = np.asarray(action_norm_stats["min"], dtype=np.float32)
        mask = np.asarray(action_norm_stats.get("mask", np.ones_like(action_low, dtype=bool)), dtype=bool)

        normalized_actions = np.asarray(normalized_actions, dtype=np.float32)
        if normalized_actions.ndim == 1:
            normalized_actions = normalized_actions[None, :]
        stats_dim = int(action_high.shape[0])
        if stats_dim != len(FOLD_TOWEL_ACTION_ORDER):
            raise ValueError(
                f"Fold-towel action stats must have dim={len(FOLD_TOWEL_ACTION_ORDER)} "
                f"with order {FOLD_TOWEL_ACTION_ORDER}, got stats_dim={stats_dim}."
            )
        if normalized_actions.shape[-1] < stats_dim:
            raise ValueError(
                "Policy action dimension is smaller than action statistics dimension: "
                f"action_dim={normalized_actions.shape[-1]}, stats_dim={stats_dim}."
            )
        if normalized_actions.shape[-1] > stats_dim:
            normalized_actions = normalized_actions[..., :stats_dim]
        if not np.all(np.isfinite(normalized_actions)):
            raise ValueError("Policy returned non-finite normalized actions.")
        normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        denorm = 0.5 * (normalized_actions + 1.0) * (action_high - action_low) + action_low
        raw_actions = np.where(mask, denorm, normalized_actions)
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("Action denormalization produced non-finite actions.")
        return raw_actions.astype(np.float32, copy=False)

    @staticmethod
    def _adaptive_ensemble_actions(previous_tail: np.ndarray, current_actions: np.ndarray, alpha: float) -> np.ndarray:
        previous_tail = np.asarray(previous_tail, dtype=np.float32)
        current_actions = np.asarray(current_actions, dtype=np.float32)
        if previous_tail.ndim != 2 or current_actions.ndim != 2:
            raise ValueError(
                "Adaptive ensemble expects [T, D] chunks, got "
                f"previous_tail={previous_tail.shape}, current_actions={current_actions.shape}."
            )
        out = current_actions.copy()
        overlap = min(int(previous_tail.shape[0]), int(current_actions.shape[0]))
        if overlap <= 0:
            return out

        for idx in range(overlap):
            preds = np.stack([previous_tail[idx], current_actions[idx]], axis=0).astype(np.float32, copy=False)
            ref = preds[-1]
            dot = np.sum(preds * ref[None, :], axis=1)
            norms = np.linalg.norm(preds, axis=1) * np.linalg.norm(ref)
            cosine = dot / (norms + 1e-7)
            weights = np.exp(float(alpha) * cosine)
            weights = weights / np.sum(weights)
            out[idx] = np.sum(weights[:, None] * preds, axis=0)
        return out

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
            "source": "fold_towel_tcp_act_request_intermediates",
            "note": "These features correspond to TCP act requests, not every interpolated robot-control tick.",
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

    def save_uploaded_recording(self, payload: dict[str, Any], conn: socket.socket) -> dict[str, Any]:
        byte_len = int(payload.get("byte_len", 0))
        if byte_len <= 0:
            raise ValueError("Upload request must contain positive `byte_len`.")
        body = _recv_exact(conn, byte_len)
        if body is None:
            raise ConnectionError("Connection closed while reading uploaded recording.")
        safe_run_id = self._sanitize_recording_run_id(payload.get("run_id"))
        out_dir = self.recording_dir / safe_run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "client_recording.npz"
        out_path.write_bytes(body)
        metadata = {
            "run_id": safe_run_id,
            "received_at": time.strftime("%Y%m%d_%H%M%S"),
            "bytes": int(len(body)),
            "path": str(out_path),
            "client_metadata": payload.get("metadata", {}),
        }
        (out_dir / "upload_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        features_path = self._flush_recording_feature_buffer(safe_run_id, out_dir=out_dir)
        LOGGER.info("Saved fold-towel client recording to %s (%d bytes)", out_path, len(body))
        return {
            "status": "ok",
            "path": str(out_path),
            "bytes": int(len(body)),
            "run_id": safe_run_id,
            "features_path": None if features_path is None else str(features_path),
        }

    def _prepare_example(self, payload: dict[str, Any], images: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
        camera_left, camera_front, camera_right = images
        instruction = str(self.args.instruction).strip() or str(payload.get("instruction") or payload.get("prompt") or "Fold the towel.")
        example = {
            "primary_image": [camera_front],
            "wrist_image": [camera_left, camera_right],
            "lang": instruction,
            "embodiment_id": int(self.args.embodiment_id),
            "action_hz": float(payload.get("action_hz", self.args.action_hz)),
        }
        state = self._state14_from_payload(payload)
        example["state"] = self._normalize_state(state)
        return example

    def prepare_only(self, payload: dict[str, Any], images: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
        example = self._prepare_example(payload, images)
        raw_state = self._state14_from_payload(payload)
        return {
            "status": "ok",
            "prepared": {
                "lang": example["lang"],
                "embodiment_id": int(example["embodiment_id"]),
                "action_hz": float(example["action_hz"]),
                "primary_image_shapes": [list(np.asarray(img).shape) for img in example["primary_image"]],
                "wrist_image_shapes": [list(np.asarray(img).shape) for img in example["wrist_image"]],
                "raw_state_shape": list(raw_state.shape),
                "normalized_state_shape": None if "state" not in example else list(np.asarray(example["state"]).shape),
                "model_uses_state": bool(self.model_uses_state),
            },
            "metadata": self.metadata,
        }

    def predict(self, payload: dict[str, Any], images: tuple[np.ndarray, np.ndarray, np.ndarray], client_key: str) -> dict[str, Any]:
        example = self._prepare_example(payload, images)
        request_num_inference_steps = payload.get("num_inference_steps", self.args.num_inference_steps)
        if request_num_inference_steps is not None:
            request_num_inference_steps = int(request_num_inference_steps)
            if request_num_inference_steps <= 0:
                raise ValueError(f"`num_inference_steps` must be > 0, got {request_num_inference_steps}.")

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
            raise ValueError(f"Unexpected `normalized_actions` shape from policy: {tuple(normalized_actions.shape)}.")

        raw_actions = self._unnormalize_actions(normalized_actions)
        command_dim = int(self.args.command_dim)
        if raw_actions.shape[-1] < command_dim:
            raise ValueError(f"Expected at least {command_dim} action dims, got {raw_actions.shape[-1]}.")
        raw_actions = raw_actions[:, :command_dim]

        start = max(0, int(self.args.latency_step))
        move_steps = int(payload.get("move_steps", self.args.move_steps))
        predicted = raw_actions[start : start + move_steps]
        if predicted.shape[0] == 0:
            raise ValueError(f"No actions left after latency_step={start}; model returned {raw_actions.shape[0]} actions.")

        previous_tail = self._ensemble_tail_by_client.get(client_key)
        ensemble_used = False
        if bool(self.args.action_ensemble) and previous_tail is not None:
            predicted = self._adaptive_ensemble_actions(
                previous_tail=previous_tail,
                current_actions=predicted,
                alpha=float(self.args.adaptive_ensemble_alpha),
            )
            ensemble_used = True
        self._ensemble_tail_by_client[client_key] = raw_actions[start + move_steps : start + 2 * move_steps].copy()

        LOGGER.info(
            "Fold-towel action chunk: norm[min=%.3f max=%.3f] raw[min=%.3f max=%.3f] "
            "action_hz=%.1f start=%d move_steps=%d ensemble=%s tail_len=%d",
            float(np.min(normalized_actions)),
            float(np.max(normalized_actions)),
            float(np.min(raw_actions)),
            float(np.max(raw_actions)),
            float(example["action_hz"]),
            int(start),
            int(move_steps),
            bool(ensemble_used),
            int(self._ensemble_tail_by_client[client_key].shape[0]),
        )

        state14 = self._state14_from_payload(payload)
        current_command = state14[-1, :command_dim]
        previous_commands = self._master_queue_by_client.get(client_key)
        if not previous_commands:
            previous_commands = [current_command.astype(np.float32, copy=True)]

        response_actions = np.concatenate([previous_commands[-1][None, :], predicted], axis=0)
        self._master_queue_by_client[client_key] = [
            *previous_commands[-99:],
            *[row.astype(np.float32, copy=True) for row in predicted],
        ]
        self._append_recording_intermediates(payload, intermediates)
        follow1_pos = response_actions[:, :7].tolist()
        follow2_pos = response_actions[:, 7:14].tolist()
        return {
            "status": "ok",
            "follow1_pos": follow1_pos,
            "follow2_pos": follow2_pos,
            "action_chunk_len": int(response_actions.shape[0]),
            "model_action_chunk_len": int(raw_actions.shape[0]),
            "action_dim": int(response_actions.shape[1]),
            "num_inference_steps": request_num_inference_steps,
        }

    def handle_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        client_key = f"{addr[0]}:{addr[1]}"
        LOGGER.info("Connection from %s", client_key)
        with conn:
            conn.settimeout(None)
            while True:
                payload = _recv_json(conn)
                if payload is None:
                    LOGGER.info("Client %s disconnected", client_key)
                    return
                request_type = str(payload.get("__request_type__", payload.get("request_type", "act"))).lower()
                try:
                    if request_type == "health":
                        _send_json(conn, {"status": "ok", "metadata": self.metadata})
                    elif request_type == "metadata":
                        _send_json(conn, {"status": "ok", "metadata": self.metadata})
                    elif request_type == "reset":
                        self._master_queue_by_client.pop(client_key, None)
                        self._ensemble_tail_by_client.pop(client_key, None)
                        _send_json(conn, {"status": "ok"})
                    elif request_type == "upload_recording":
                        _send_json(conn, self.save_uploaded_recording(payload, conn))
                    elif request_type in {"act", "prepare"}:
                        images = (_recv_image(conn), _recv_image(conn), _recv_image(conn))
                        if request_type == "prepare":
                            _send_json(conn, self.prepare_only(payload, images))
                        else:
                            _send_json(conn, self.predict(payload, images, client_key))
                    else:
                        raise ValueError(f"Unknown request_type: {request_type}")
                except Exception as exc:
                    LOGGER.exception("Request from %s failed", client_key)
                    _send_json(conn, {"status": "error", "message": str(exc)})

    def serve_forever(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self.args.host, self.args.port))
        sock.listen(int(self.args.backlog))
        LOGGER.info("Fold-towel TCP policy server listening on %s:%d", self.args.host, self.args.port)
        LOGGER.info("Protocol: length-prefixed JSON + three JPEG images; upload_recording uses same JSON framing.")
        while True:
            conn, addr = sock.accept()
            if bool(self.args.single_client):
                self.handle_client(conn, addr)
            else:
                threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True).start()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StarVLA TCP policy server for the fold-towel dual-arm client.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=57770)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--embodiment_id", type=int, default=31)
    parser.add_argument("--action_hz", type=float, default=20.0)
    parser.add_argument("--move_steps", type=int, default=20)
    parser.add_argument("--latency_step", type=int, default=0)
    parser.add_argument("--command_dim", type=int, default=14)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--instruction", type=str, default="Fold the towel.")
    parser.add_argument("--recording_dir", type=str, default="logs/fold_towel_client_recordings")
    parser.add_argument("--action_ensemble", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive_ensemble_alpha", type=float, default=0.05)
    parser.add_argument("--backlog", type=int, default=1)
    parser.add_argument("--single_client", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    FoldTowelTCPPolicyServer(args).serve_forever()


if __name__ == "__main__":
    main()
