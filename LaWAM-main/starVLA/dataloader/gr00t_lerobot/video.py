# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import subprocess

import av
import cv2
import numpy as np


def _require_pyav_backend(video_backend: str) -> None:
    if video_backend != "pyav":
        raise NotImplementedError(
            f"Video backend {video_backend} not implemented. Only `pyav` is supported."
        )


def _to_int64_1d(indices: list[int] | np.ndarray) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        raise ValueError("indices must be non-empty")
    if np.any(idx < 0):
        raise ValueError("indices must be non-negative")
    return idx


def _to_float64_1d(timestamps: list[float] | np.ndarray) -> np.ndarray:
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise ValueError("timestamps must be non-empty")
    return ts


def _configure_pyav_stream(stream) -> None:
    """Limit decoder threading to reduce host-memory spikes under many DataLoader workers."""
    raw_thread_count = os.environ.get("LAM_PYAV_THREAD_COUNT", "1")
    try:
        thread_count = max(1, int(raw_thread_count))
    except ValueError:
        thread_count = 1

    thread_type = os.environ.get("LAM_PYAV_THREAD_TYPE", "NONE").upper()
    if thread_type not in {"AUTO", "FRAME", "SLICE", "NONE"}:
        thread_type = "NONE"

    try:
        stream.thread_type = thread_type
    except Exception:
        pass

    try:
        stream.codec_context.thread_count = thread_count
    except Exception:
        pass


def _resize_frame_short_side(frame: np.ndarray, short_side: int) -> np.ndarray:
    if short_side <= 0:
        raise ValueError(f"short_side must be > 0, got {short_side}")
    height, width = frame.shape[:2]
    if min(height, width) == short_side:
        return frame
    scale = float(short_side) / float(min(height, width))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def _resize_frames_preserve_aspect(
    frames: np.ndarray,
    resize_size: tuple[int, int] | int | None,
) -> np.ndarray:
    if resize_size is None:
        return frames
    if isinstance(resize_size, int):
        return np.asarray([_resize_frame_short_side(frame, resize_size) for frame in frames])
    return np.asarray([cv2.resize(frame, resize_size, interpolation=cv2.INTER_LINEAR) for frame in frames])


def get_frames_by_indices(
    video_path: str,
    indices: list[int] | np.ndarray,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    del video_backend_kwargs
    _require_pyav_backend(video_backend)
    idx = _to_int64_1d(indices)

    frame_dict: dict[int, np.ndarray] = {}
    needed = set(int(i) for i in np.unique(idx).tolist())
    max_idx = int(idx.max())
    last_frame = None
    last_idx = -1

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        _configure_pyav_stream(stream)
        for i, frame in enumerate(container.decode(video=0)):
            frame_rgb = frame.to_ndarray(format="rgb24")
            last_frame = frame_rgb
            last_idx = i
            if i in needed:
                frame_dict[i] = frame_rgb
                if len(frame_dict) == len(needed) and i >= max_idx:
                    break
            if i >= max_idx:
                break

    if last_frame is None:
        raise ValueError(f"Video has no frames: {video_path}")

    out_frames = []
    for i in idx.tolist():
        j = int(i)
        if j in frame_dict:
            out_frames.append(frame_dict[j])
        elif j > last_idx:
            # Keep previous behavior: clip over-range indices to last decoded frame.
            out_frames.append(last_frame)
        else:
            raise ValueError(f"Unable to load frame index {j} from {video_path}")
    return np.asarray(out_frames)


def get_frames_by_timestamps(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    """Get frames from a video at specified timestamps."""
    _require_pyav_backend(video_backend)
    ts = _to_float64_1d(timestamps)
    del video_backend_kwargs

    unique_ts, inverse = np.unique(ts, return_inverse=True)
    selected_unique: list[np.ndarray] = []
    try:
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            _configure_pyav_stream(stream)
            time_base = float(stream.time_base)
            if time_base <= 0:
                raise RuntimeError(f"Invalid time_base for video {video_path}: {time_base}")

            # Two-seek strategy: seek+decode independently for each target timestamp.
            for target_ts in unique_ts.tolist():
                seek_ts = max(float(target_ts), 0.0)
                seek_pts = int(seek_ts / time_base)
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

                prev_frame = None
                prev_ts = None
                chosen = None

                for frame in container.decode(video=0):
                    if frame.pts is None:
                        continue
                    current_ts = float(frame.pts * time_base)
                    frame_rgb = frame.to_ndarray(format="rgb24")

                    if current_ts < target_ts:
                        prev_frame = frame_rgb
                        prev_ts = current_ts
                        continue

                    if prev_frame is None:
                        chosen = frame_rgb
                    else:
                        if abs(prev_ts - target_ts) <= abs(current_ts - target_ts):
                            chosen = prev_frame
                        else:
                            chosen = frame_rgb
                    break

                if chosen is None:
                    # If the stream ended before target_ts, fallback to the last decoded frame.
                    if prev_frame is not None:
                        chosen = prev_frame
                    else:
                        raise RuntimeError(
                            f"No frames loaded from {video_path} for timestamp={target_ts}."
                        )
                selected_unique.append(chosen)
    except av.error.InvalidDataError:
        selected_unique = [_extract_frame_with_ffmpeg(video_path, float(target_ts)) for target_ts in unique_ts.tolist()]

    selected = [selected_unique[int(i)] for i in inverse.tolist()]
    return np.asarray(selected)


def _extract_frame_with_ffmpeg(video_path: str, timestamp: float) -> np.ndarray:
    """Fallback path for samples that PyAV fails to decode at a target timestamp."""
    seek_ts = max(float(timestamp), 0.0)
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{seek_ts:.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    frame_bgr = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError(f"ffmpeg fallback returned undecodable frame for {video_path} at {seek_ts:.6f}s")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def get_frames_by_timestamps_single_seek(
    video_path: str,
    timestamps: list[float] | np.ndarray,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
) -> np.ndarray:
    """Get frames from a video via one initial seek followed by sequential scan."""
    _require_pyav_backend(video_backend)
    ts = _to_float64_1d(timestamps)
    del video_backend_kwargs

    unique_ts, inverse = np.unique(ts, return_inverse=True)
    selected_unique: list[np.ndarray | None] = [None] * len(unique_ts)

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        _configure_pyav_stream(stream)
        time_base = float(stream.time_base)
        if time_base <= 0:
            raise RuntimeError(f"Invalid time_base for video {video_path}: {time_base}")

        seek_ts = max(float(unique_ts[0]), 0.0)
        seek_pts = int(seek_ts / time_base)
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

        prev_frame = None
        prev_ts = None
        last_frame = None
        target_idx = 0

        for frame in container.decode(video=0):
            if frame.pts is None:
                continue
            current_ts = float(frame.pts * time_base)
            frame_rgb = frame.to_ndarray(format="rgb24")
            last_frame = frame_rgb

            while target_idx < len(unique_ts) and current_ts >= float(unique_ts[target_idx]):
                target_ts = float(unique_ts[target_idx])
                if prev_frame is None or prev_ts is None:
                    selected_unique[target_idx] = frame_rgb
                elif abs(prev_ts - target_ts) <= abs(current_ts - target_ts):
                    selected_unique[target_idx] = prev_frame
                else:
                    selected_unique[target_idx] = frame_rgb
                target_idx += 1

            if target_idx == len(unique_ts):
                break

            prev_frame = frame_rgb
            prev_ts = current_ts

    if last_frame is None:
        raise RuntimeError(f"No frames loaded from {video_path}.")

    fallback_frame = prev_frame if prev_frame is not None else last_frame
    for idx, frame in enumerate(selected_unique):
        if frame is None:
            selected_unique[idx] = fallback_frame

    selected = np.asarray([selected_unique[int(i)] for i in inverse.tolist()])
    return selected


def get_all_frames(
    video_path: str,
    video_backend: str = "pyav",
    video_backend_kwargs: dict | None = None,
    resize_size: tuple[int, int] | int | None = None,
) -> np.ndarray:
    """Get all frames from a video."""
    del video_backend_kwargs
    _require_pyav_backend(video_backend)

    frames = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        _configure_pyav_stream(stream)
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    frames = np.asarray(frames)
    if frames.shape[0] == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    return _resize_frames_preserve_aspect(frames, resize_size)
