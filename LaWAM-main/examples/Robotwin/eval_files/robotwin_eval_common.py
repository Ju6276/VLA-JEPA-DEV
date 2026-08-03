from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

DEFAULT_TASKS: tuple[str, ...] = (
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle_horizontally",
    "shake_bottle",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
)


def sanitize_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip().replace("+", "_"))
    sanitized = sanitized.strip("._-")
    return sanitized or "unknown"


def derive_ckpt_alias(ckpt_path: str | os.PathLike[str]) -> str:
    path = Path(ckpt_path).expanduser()
    candidates = [path.parent.parent.name, path.parent.name, path.stem]
    for candidate in candidates:
        alias = sanitize_component(candidate)
        if alias not in {"final_model", "checkpoint", "checkpoints", "pytorch_model"}:
            return alias
    return sanitize_component(path.stem)


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    override_dict: dict[str, Any] = {}
    if not pairs:
        return override_dict
    if len(pairs) % 2 != 0:
        raise ValueError(f"`overrides` must contain key/value pairs, got {pairs}")
    for idx in range(0, len(pairs), 2):
        key = pairs[idx].lstrip("-")
        value = pairs[idx + 1]
        try:
            value = eval(value)
        except Exception:
            pass
        override_dict[key] = value
    return override_dict


def load_config_with_overrides(config_path: str | os.PathLike[str], overrides: list[str] | None) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.update(parse_overrides(overrides))
    return config


def write_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def normalize_bool_env(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def detect_gpu_ids() -> list[str]:
    if os.environ.get("GPU_IDS"):
        return [item for item in re.split(r"[,\s]+", os.environ["GPU_IDS"].strip()) if item]
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return [item for item in re.split(r"[,\s]+", os.environ["CUDA_VISIBLE_DEVICES"].strip()) if item]
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ["0"]
    gpu_ids = []
    for line in output.splitlines():
        match = re.match(r"^GPU\s+(\d+):", line.strip())
        if match:
            gpu_ids.append(match.group(1))
    return gpu_ids or ["0"]


def resolve_tasks_from_env() -> list[str]:
    task_string = os.environ.get("ROBOTWIN_TASKS", "").strip()
    if not task_string:
        return list(DEFAULT_TASKS)
    return [item for item in re.split(r"[,\s]+", task_string) if item]


def split_tasks_for_worker(tasks: list[str], worker_index: int, num_workers: int) -> list[str]:
    return [task for idx, task in enumerate(tasks) if idx % num_workers == worker_index]


def wait_for_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, int(port)))
            return
        except OSError:
            time.sleep(1.0)
        finally:
            try:
                sock.close()
            except OSError:
                pass
    raise TimeoutError(f"Timed out waiting for {host}:{port} after {timeout_sec:.0f}s")


def _resolve_bind_probe_host(host: str) -> str:
    normalized = str(host).strip().lower()
    if normalized in {"127.0.0.1", "localhost", "::1"}:
        return "0.0.0.0"
    return host


def can_bind_port(host: str, port: int) -> bool:
    probe_host = _resolve_bind_probe_host(host)
    family = socket.AF_INET6 if ":" in str(probe_host) else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((probe_host, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            sock.close()
        except OSError:
            pass


def find_available_port(host: str, preferred_port: int, *, max_tries: int = 128) -> int:
    preferred_port = int(preferred_port)
    max_tries = max(1, int(max_tries))
    for offset in range(max_tries):
        candidate = preferred_port + offset
        if can_bind_port(host, candidate):
            return candidate
    raise RuntimeError(
        f"Could not find a free port for host={host} starting at {preferred_port} "
        f"within {max_tries} attempts"
    )


def ensure_runtime_paths(robotwin_path: str | os.PathLike[str], starvla_root: str | os.PathLike[str]) -> None:
    robotwin_root = str(Path(robotwin_path).expanduser().resolve())
    starvla_root = str(Path(starvla_root).expanduser().resolve())
    eval_files = str(Path(__file__).resolve().parent)
    for path in (robotwin_root, starvla_root, eval_files):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.chdir(robotwin_root)


def build_run_group(ckpt_alias: str, task_config: str) -> str:
    return sanitize_component(os.environ.get("ROBOTWIN_RUN_GROUP", f"{ckpt_alias}__{task_config}"))


def build_run_tag(default: str) -> str:
    env_value = os.environ.get("ROBOTWIN_RUN_TAG", "").strip()
    return sanitize_component(env_value or default)


@dataclass(frozen=True)
class TaskCompletionState:
    task_name: str
    task_dir: Path
    status: str
    reason: str
    summary_path: Path
    result_path: Path


def get_task_output_dir(run_dir: str | os.PathLike[str], task_name: str) -> Path:
    return Path(run_dir).expanduser() / "tasks" / sanitize_component(task_name)


def _load_task_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {summary_path}")
    return payload


def _parse_result_file(result_path: Path) -> float:
    lines = [line.strip() for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"empty result file: {result_path}")
    return float(lines[-1])


def inspect_task_completion(
    run_dir: str | os.PathLike[str],
    task_name: str,
    *,
    expected_num_episodes: int | None = None,
) -> TaskCompletionState:
    task_dir = get_task_output_dir(run_dir, task_name)
    summary_path = task_dir / "summary.json"
    result_path = task_dir / "_result.txt"

    if not task_dir.is_dir():
        return TaskCompletionState(
            task_name=task_name,
            task_dir=task_dir,
            status="missing",
            reason="missing_task_dir",
            summary_path=summary_path,
            result_path=result_path,
        )

    if not summary_path.is_file():
        return TaskCompletionState(
            task_name=task_name,
            task_dir=task_dir,
            status="incomplete",
            reason="missing_summary",
            summary_path=summary_path,
            result_path=result_path,
        )

    if not result_path.is_file():
        return TaskCompletionState(
            task_name=task_name,
            task_dir=task_dir,
            status="incomplete",
            reason="missing_result",
            summary_path=summary_path,
            result_path=result_path,
        )

    try:
        summary = _load_task_summary(summary_path)
    except Exception as exc:
        return TaskCompletionState(
            task_name=task_name,
            task_dir=task_dir,
            status="corrupted",
            reason=f"invalid_summary:{type(exc).__name__}",
            summary_path=summary_path,
            result_path=result_path,
        )

    try:
        _parse_result_file(result_path)
    except Exception as exc:
        return TaskCompletionState(
            task_name=task_name,
            task_dir=task_dir,
            status="corrupted",
            reason=f"invalid_result:{type(exc).__name__}",
            summary_path=summary_path,
            result_path=result_path,
        )

    if expected_num_episodes is not None:
        expected_num_episodes = int(expected_num_episodes)
        summary_num_episodes = int(summary.get("n_episodes", -1))
        if summary_num_episodes != expected_num_episodes:
            return TaskCompletionState(
                task_name=task_name,
                task_dir=task_dir,
                status="incomplete",
                reason="unexpected_n_episodes",
                summary_path=summary_path,
                result_path=result_path,
            )
        episodes = summary.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != expected_num_episodes:
            return TaskCompletionState(
                task_name=task_name,
                task_dir=task_dir,
                status="incomplete",
                reason="unexpected_episode_records",
                summary_path=summary_path,
                result_path=result_path,
            )

    return TaskCompletionState(
        task_name=task_name,
        task_dir=task_dir,
        status="completed",
        reason="complete",
        summary_path=summary_path,
        result_path=result_path,
    )


def collect_task_completion(
    run_dir: str | os.PathLike[str],
    task_names: list[str],
    *,
    expected_num_episodes: int | None = None,
) -> dict[str, TaskCompletionState]:
    return {
        task_name: inspect_task_completion(
            run_dir,
            task_name,
            expected_num_episodes=expected_num_episodes,
        )
        for task_name in task_names
    }


def select_incomplete_tasks(
    run_dir: str | os.PathLike[str],
    task_names: list[str],
    *,
    expected_num_episodes: int | None = None,
) -> list[str]:
    states = collect_task_completion(
        run_dir,
        task_names,
        expected_num_episodes=expected_num_episodes,
    )
    return [task_name for task_name, state in states.items() if state.status != "completed"]


def count_completed_tasks(
    run_dir: str | os.PathLike[str],
    task_names: list[str],
    *,
    expected_num_episodes: int | None = None,
) -> int:
    states = collect_task_completion(
        run_dir,
        task_names,
        expected_num_episodes=expected_num_episodes,
    )
    return sum(1 for state in states.values() if state.status == "completed")
