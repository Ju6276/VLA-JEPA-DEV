#!/usr/bin/env python3
"""Validate the VLA-JEPA HumanoidArena SONIC40 data contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_MODALITY = {
    "state": {
        "root_rot6d": {"start": 0, "end": 6},
        "joint_pos": {"start": 6, "end": 35},
        "joint_vel": {"start": 35, "end": 64},
    },
    "action": {
        "root_xy_delta": {"start": 0, "end": 2},
        "root_z": {"start": 2, "end": 3},
        "root_rot6d": {"start": 3, "end": 9},
        "joint_pos": {"start": 9, "end": 38},
        "hand_binary": {"start": 38, "end": 40},
    },
    "video": {"front": {"original_key": "observation.images.front"}},
    "annotation": {
        "human.task_description": {"original_key": "task_index"}
    },
}


def validate_dataset(root: Path) -> dict:
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    modality = json.loads((root / "meta/modality.json").read_text(encoding="utf-8"))
    stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))

    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Expected LeRobot v2.1, got {info.get('codebase_version')}")
    features = info.get("features", {})
    expected_shapes = {
        "observation.images.front": [480, 640, 3],
        "observation.state": [64],
        "action": [40],
    }
    for key, expected in expected_shapes.items():
        actual = features.get(key, {}).get("shape")
        if actual != expected:
            raise ValueError(f"{key} shape must be {expected}, got {actual}")
    protocol = info.get("vla_protocol", {})
    if protocol.get("schema") != "unitree_g1_gmt_refpose_v3_1":
        raise ValueError("Unexpected HumanoidArena VLA schema")
    if protocol.get("backend_source") != "sonic":
        raise ValueError("VLA-JEPA benchmark requires SONIC-only data")
    if protocol.get("action_semantics") != "reference_pose_not_robot_current_residual":
        raise ValueError("Actions must be canonical reference-pose targets")
    if modality != EXPECTED_MODALITY:
        raise ValueError("meta/modality.json order does not match semantic_v3")
    if len(stats.get("observation.state", {}).get("min", [])) != 64:
        raise ValueError("State statistics must be 64D")
    if len(stats.get("action", {}).get("min", [])) != 40:
        raise ValueError("Action statistics must be 40D")
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    info = validate_dataset(args.dataset)
    print(
        f"validated {args.dataset}: LeRobot=v2.1 episodes={info['total_episodes']} "
        f"frames={info['total_frames']} image=480x640x3 state=64D "
        "action=40D semantic_v3=row SONIC-only"
    )


if __name__ == "__main__":
    main()
