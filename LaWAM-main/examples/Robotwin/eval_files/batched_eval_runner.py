#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
STARVLA_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(STARVLA_ROOT) not in sys.path:
    sys.path.insert(0, str(STARVLA_ROOT))

from examples.Robotwin.eval_files.robotwin_eval_common import (  # noqa: E402
    build_run_group,
    build_run_tag,
    count_completed_tasks,
    derive_ckpt_alias,
    detect_gpu_ids,
    find_available_port,
    inspect_task_completion,
    normalize_bool_env,
    resolve_tasks_from_env,
    select_incomplete_tasks,
    write_json,
)


def install_interrupt_handlers() -> None:
    def _raise_keyboard_interrupt(signum, frame):
        del signum, frame
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch Robotwin benchmark orchestrator.")
    parser.add_argument("--mode", choices=("master", "worker", "prepare", "monitor"), default="master")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--task_config", type=str, default=os.getenv("TASK_CONFIG", "demo_clean"))
    parser.add_argument("--run_tag", type=str, default=os.getenv("RUN_TAG", datetime.now().strftime("%Y%m%d_%H%M%S")))
    parser.add_argument(
        "--resume_run_dir",
        type=str,
        default=os.getenv("ROBOTWIN_RESUME_RUN_DIR", os.getenv("RESUME_RUN_DIR", "")),
    )
    parser.add_argument("--worker_index", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


def resolve_starvla_python() -> str:
    env_value = os.getenv("STAR_VLA_PYTHON")
    if env_value:
        return env_value
    for candidate in (
        "/root/miniconda3/envs/starvla/bin/python",
    ):
        if Path(candidate).is_file():
            return candidate
    return sys.executable


def resolve_robotwin_python() -> str:
    env_value = os.getenv("ROBOTWIN_PYTHON")
    if env_value:
        return env_value
    return "python"


def resolve_robotwin_path() -> str:
    return os.getenv("ROBOTWIN_PATH", "../RoboTwin")


def log_line(handle, text: str) -> None:
    timestamp = datetime.now().strftime("%F %T")
    handle.write(f"[{timestamp}] {text}\n")
    handle.flush()


def terminate_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def build_bridge_command(
    *,
    ckpt_path: str,
    task_name: str,
    task_config: str,
    ckpt_setting: str,
    host: str,
    port: int,
    seed: int,
) -> list[str]:
    optional_env_args = (
        ("UNNORM_KEY", "--unnorm_key"),
        ("ROBOTWIN_REPLAN_STEPS", "--replan_steps"),
        ("ROBOTWIN_ACTION_ENSEMBLE", "--action_ensemble"),
        ("ROBOTWIN_ACTION_ENSEMBLE_ALPHA", "--action_ensemble_alpha"),
        ("ACTION_REORDER", "--action_reorder"),
    )
    command = [
        resolve_robotwin_python(),
        str(SCRIPT_DIR / "robotwin_batch_bridge.py"),
        "--config",
        str(SCRIPT_DIR / "deploy_policy.yml"),
        "--overrides",
        "--task_name",
        task_name,
        "--task_config",
        task_config,
        "--ckpt_setting",
        ckpt_setting,
        "--seed",
        str(seed),
        "--policy_name",
        "model2robotwin_interface",
        "--host",
        host,
        "--port",
        str(port),
        "--policy_ckpt_path",
        ckpt_path,
    ]
    for env_name, arg_name in optional_env_args:
        env_value = os.getenv(env_name)
        if env_value:
            command.extend([arg_name, env_value])
    return command


def tail_text(path: Path, max_lines: int = 40) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return ""
    if max_lines <= 0:
        return ""
    return "\n".join(lines[-max_lines:])


def wait_for_server_ready(
    *,
    host: str,
    port: int,
    timeout_sec: float,
    server_proc: subprocess.Popen,
    server_log_path: Path,
) -> None:
    deadline = time.time() + float(timeout_sec)
    while time.time() < deadline:
        return_code = server_proc.poll()
        if return_code is not None:
            server_tail = tail_text(server_log_path)
            message = f"Policy server exited before binding {host}:{port} (exit_code={return_code})"
            if server_tail:
                message = f"{message}\nLast server log lines:\n{server_tail}"
            raise RuntimeError(message)

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

    server_tail = tail_text(server_log_path)
    message = f"Timed out waiting for {host}:{port} after {timeout_sec:.0f}s"
    if server_tail:
        message = f"{message}\nLast server log lines:\n{server_tail}"
    raise TimeoutError(message)


def resolve_task_scheduler_heartbeat_sec() -> float:
    return max(5.0, float(os.getenv("ROBOTWIN_TASK_HEARTBEAT_SEC", "60")))


def resolve_task_scheduler_lease_timeout_sec(heartbeat_sec: float) -> float:
    default_timeout = max(heartbeat_sec * 4.0, 600.0)
    return max(heartbeat_sec * 2.0, float(os.getenv("ROBOTWIN_TASK_LEASE_TIMEOUT_SEC", str(int(default_timeout)))))


def task_scheduler_state_path(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir).expanduser() / ".task_scheduler.json"


def _normalize_scheduler_state(payload: Any) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    pending = payload.get("pending", [])
    in_progress = payload.get("in_progress", {})
    completed = payload.get("completed", {})
    failed = payload.get("failed", {})
    return {
        "pending": [str(task) for task in pending if str(task).strip()],
        "in_progress": {str(task): dict(info) for task, info in in_progress.items() if str(task).strip()},
        "completed": {str(task): dict(info) for task, info in completed.items() if str(task).strip()},
        "failed": {str(task): dict(info) for task, info in failed.items() if str(task).strip()},
        "updated_at": str(payload.get("updated_at", "")),
    }


@contextmanager
def locked_task_scheduler_state(run_dir: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    run_dir = Path(run_dir).expanduser()
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".task_scheduler.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        state_path = task_scheduler_state_path(run_dir)
        if state_path.is_file():
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        else:
            payload = {}
        state = _normalize_scheduler_state(payload)
        try:
            yield state
        finally:
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            state_path.write_text(
                json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def initialize_task_scheduler(run_dir: str | os.PathLike[str], scheduled_tasks: list[str]) -> None:
    unique_tasks = list(dict.fromkeys(str(task) for task in scheduled_tasks if str(task).strip()))
    with locked_task_scheduler_state(run_dir) as state:
        state.clear()
        state.update(
            {
                "pending": unique_tasks,
                "in_progress": {},
                "completed": {},
                "failed": {},
            }
        )


def _prepend_unique_tasks(tasks: list[str], prefix_tasks: list[str]) -> list[str]:
    seen = set()
    ordered: list[str] = []
    for task_name in [*prefix_tasks, *tasks]:
        if task_name in seen:
            continue
        seen.add(task_name)
        ordered.append(task_name)
    return ordered


def _is_task_completed(
    run_dir: str | os.PathLike[str],
    task_name: str,
    *,
    expected_num_episodes: int | None = None,
) -> bool:
    return inspect_task_completion(
        run_dir,
        task_name,
        expected_num_episodes=expected_num_episodes,
    ).status == "completed"


def claim_next_task(
    run_dir: str | os.PathLike[str],
    *,
    worker_index: int,
    expected_num_episodes: int | None = None,
    lease_timeout_sec: float,
) -> str | None:
    now_ts = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    with locked_task_scheduler_state(run_dir) as state:
        reclaimed_tasks: list[str] = []
        for task_name, claim_info in list(state["in_progress"].items()):
            heartbeat_ts = float(claim_info.get("heartbeat_at", claim_info.get("claimed_at", 0.0)))
            if now_ts - heartbeat_ts <= lease_timeout_sec:
                continue
            state["in_progress"].pop(task_name, None)
            if _is_task_completed(run_dir, task_name, expected_num_episodes=expected_num_episodes):
                state["completed"][task_name] = {
                    **claim_info,
                    "status": "completed",
                    "exit_code": int(claim_info.get("exit_code", 0)),
                    "finished_at": now_ts,
                    "finished_at_iso": now_iso,
                    "reason": "completed_before_reclaim",
                }
                continue
            reclaimed_tasks.append(task_name)
        if reclaimed_tasks:
            state["pending"] = _prepend_unique_tasks(state["pending"], reclaimed_tasks)

        while state["pending"]:
            task_name = state["pending"].pop(0)
            if _is_task_completed(run_dir, task_name, expected_num_episodes=expected_num_episodes):
                state["completed"][task_name] = {
                    "worker_index": worker_index,
                    "status": "skipped",
                    "exit_code": 0,
                    "finished_at": now_ts,
                    "finished_at_iso": now_iso,
                    "reason": "already_completed",
                }
                continue
            state["in_progress"][task_name] = {
                "worker_index": int(worker_index),
                "claimed_at": now_ts,
                "claimed_at_iso": now_iso,
                "heartbeat_at": now_ts,
                "heartbeat_at_iso": now_iso,
            }
            return task_name
    return None


def heartbeat_claimed_task(run_dir: str | os.PathLike[str], *, task_name: str, worker_index: int) -> bool:
    now_ts = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    with locked_task_scheduler_state(run_dir) as state:
        claim_info = state["in_progress"].get(task_name)
        if not isinstance(claim_info, dict) or int(claim_info.get("worker_index", -1)) != int(worker_index):
            return False
        claim_info["heartbeat_at"] = now_ts
        claim_info["heartbeat_at_iso"] = now_iso
        return True


def release_claimed_task(
    run_dir: str | os.PathLike[str],
    *,
    task_name: str,
    worker_index: int,
    expected_num_episodes: int | None = None,
    reason: str,
) -> bool:
    now_ts = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    with locked_task_scheduler_state(run_dir) as state:
        claim_info = state["in_progress"].get(task_name)
        if not isinstance(claim_info, dict) or int(claim_info.get("worker_index", -1)) != int(worker_index):
            return False
        state["in_progress"].pop(task_name, None)
        if _is_task_completed(run_dir, task_name, expected_num_episodes=expected_num_episodes):
            state["completed"][task_name] = {
                **claim_info,
                "status": "completed",
                "exit_code": 0,
                "finished_at": now_ts,
                "finished_at_iso": now_iso,
                "reason": reason,
            }
            return True
        state["pending"] = _prepend_unique_tasks(state["pending"], [task_name])
        return True


def finish_claimed_task(
    run_dir: str | os.PathLike[str],
    *,
    task_name: str,
    worker_index: int,
    return_code: int,
) -> None:
    now_ts = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")
    with locked_task_scheduler_state(run_dir) as state:
        claim_info = state["in_progress"].pop(task_name, {})
        if not isinstance(claim_info, dict):
            claim_info = {}
        bucket = "completed" if int(return_code) == 0 else "failed"
        state[bucket][task_name] = {
            **claim_info,
            "worker_index": int(worker_index),
            "status": "ok" if int(return_code) == 0 else "failed",
            "exit_code": int(return_code),
            "finished_at": now_ts,
            "finished_at_iso": now_iso,
        }


def wait_for_task_completion(
    task_proc: subprocess.Popen,
    *,
    run_dir: str | os.PathLike[str],
    task_name: str,
    worker_index: int,
    heartbeat_sec: float,
) -> int:
    while True:
        try:
            return task_proc.wait(timeout=heartbeat_sec)
        except subprocess.TimeoutExpired:
            heartbeat_claimed_task(run_dir, task_name=task_name, worker_index=worker_index)


def worker_main(args: argparse.Namespace) -> int:
    if args.worker_index is None or args.num_workers is None:
        raise ValueError("worker mode requires --worker_index and --num_workers")

    task_config = str(args.task_config)
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    worker_index = int(args.worker_index)
    num_workers = int(args.num_workers)

    gpu_ids = detect_gpu_ids()
    gpu_id = gpu_ids[worker_index % len(gpu_ids)]
    host = os.getenv("HOST", "127.0.0.1")
    port_base = int(os.getenv("PORT_BASE", "6694"))
    requested_port = port_base + worker_index
    port_search_limit = int(os.getenv("PORT_SEARCH_LIMIT", "128"))
    port = find_available_port(host, requested_port, max_tries=port_search_limit)
    server_timeout = float(os.getenv("SERVER_STARTUP_TIMEOUT_SEC", "600"))
    seed = int(os.getenv("SEED", "0"))
    expected_test_num = int(os.getenv("ROBOTWIN_TEST_NUM", "100"))
    resume_incomplete = normalize_bool_env(os.getenv("ROBOTWIN_RESUME_INCOMPLETE"), default=False)
    output_root = Path(
        os.getenv(
            "OUTPUT_ROOT",
            os.getenv("ROBOTWIN_EVAL_ROOT", str(STARVLA_ROOT / "results/eval_runs/robotwin")),
        )
    ).expanduser()
    ckpt_alias = os.getenv("ROBOTWIN_CKPT_ALIAS", derive_ckpt_alias(ckpt_path))
    run_group = build_run_group(ckpt_alias, task_config)
    run_tag = build_run_tag(str(args.run_tag))
    run_dir = output_root / run_group / run_tag
    if normalize_bool_env(os.getenv("ROBOTWIN_WAIT_FOR_SCHEDULER"), default=False):
        timeout_sec = float(os.getenv("ROBOTWIN_SCHEDULER_READY_TIMEOUT_SEC", "1800"))
        deadline = time.time() + timeout_sec
        state_path = task_scheduler_state_path(run_dir)
        while not state_path.is_file():
            if time.time() >= deadline:
                raise TimeoutError(f"Timed out waiting for Robotwin scheduler state at {state_path}")
            time.sleep(2.0)

    worker_dir = run_dir / "workers" / f"worker_{worker_index}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / ".task_status"
    progress_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_sec = resolve_task_scheduler_heartbeat_sec()
    lease_timeout_sec = resolve_task_scheduler_lease_timeout_sec(heartbeat_sec)

    tasks_file = worker_dir / "tasks.txt"
    tasks_file.write_text("", encoding="utf-8")

    worker_log_path = worker_dir / "worker.log"
    server_log_path = worker_dir / "server.log"
    worker_log = open(worker_log_path, "a", encoding="utf-8", buffering=1)
    server_log = open(server_log_path, "a", encoding="utf-8", buffering=1)
    log_line(
        worker_log,
        " ".join(
            [
                f"worker={worker_index}",
                f"num_workers={num_workers}",
                f"hostname={socket.gethostname()}",
                f"pid={os.getpid()}",
                f"gpu_ids={','.join(gpu_ids)}",
                f"selected_gpu={gpu_id}",
                f"cuda_visible_devices={os.getenv('CUDA_VISIBLE_DEVICES', '')}",
            ]
        ),
    )
    bridge_env = os.environ.copy()
    bridge_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    bridge_env["ROBOTWIN_PATH"] = resolve_robotwin_path()
    bridge_env["ROBOTWIN_PYTHON"] = resolve_robotwin_python()
    bridge_env["POLICY_CKPT_PATH"] = ckpt_path
    bridge_env["ROBOTWIN_CKPT_ALIAS"] = ckpt_alias
    bridge_env["ROBOTWIN_RUN_GROUP"] = run_group
    bridge_env["ROBOTWIN_RUN_TAG"] = run_tag
    bridge_env["ROBOTWIN_EVAL_ROOT"] = str(output_root)
    bridge_env["PYTHONPATH"] = f"{bridge_env['ROBOTWIN_PATH']}:{STARVLA_ROOT}:{SCRIPT_DIR}:{bridge_env.get('PYTHONPATH', '')}"

    server_proc: subprocess.Popen | None = None
    current_task_proc: subprocess.Popen | None = None
    current_task_name: str | None = None
    status = 0
    try:
        install_interrupt_handlers()
        if port != requested_port:
            log_line(
                worker_log,
                f"worker={worker_index} requested_port={requested_port} unavailable; using fallback_port={port}",
            )

        ckpt_setting_prefix = os.getenv("CKPT_SETTING_PREFIX", "benchmark")
        ckpt_setting = f"{ckpt_setting_prefix}_{task_config}"
        claimed_tasks: list[str] = []

        while True:
            current_task_name = claim_next_task(
                run_dir,
                worker_index=worker_index,
                expected_num_episodes=expected_test_num if resume_incomplete else None,
                lease_timeout_sec=lease_timeout_sec,
            )
            if current_task_name is None:
                if not claimed_tasks:
                    log_line(worker_log, f"worker={worker_index} found no pending tasks")
                else:
                    log_line(worker_log, f"worker={worker_index} finished claimed_tasks={len(claimed_tasks)}")
                return status

            claimed_tasks.append(current_task_name)
            tasks_file.write_text("\n".join(claimed_tasks) + "\n", encoding="utf-8")

            if server_proc is None:
                server_cmd = [
                    resolve_starvla_python(),
                    "-m",
                    "deployment.model_server.server_policy",
                    "--ckpt_path",
                    ckpt_path,
                    "--port",
                    str(port),
                ]
                if os.getenv("USE_BF16", "1") == "1":
                    server_cmd.append("--use_bf16")
                try:
                    server_proc = subprocess.Popen(
                        server_cmd,
                        cwd=str(STARVLA_ROOT),
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                        stdout=server_log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    wait_for_server_ready(
                        host=host,
                        port=port,
                        timeout_sec=server_timeout,
                        server_proc=server_proc,
                        server_log_path=server_log_path,
                    )
                    log_line(worker_log, f"worker={worker_index} gpu={gpu_id} server_ready port={port}")
                except Exception:
                    release_claimed_task(
                        run_dir,
                        task_name=current_task_name,
                        worker_index=worker_index,
                        expected_num_episodes=expected_test_num if resume_incomplete else None,
                        reason="server_start_failed",
                    )
                    current_task_name = None
                    raise

            task_log_path = worker_dir / f"{current_task_name}.log"
            log_line(worker_log, f"worker={worker_index} start task={current_task_name}")
            with open(task_log_path, "w", encoding="utf-8") as task_log:
                current_task_proc = subprocess.Popen(
                    build_bridge_command(
                        ckpt_path=ckpt_path,
                        task_name=current_task_name,
                        task_config=task_config,
                        ckpt_setting=ckpt_setting,
                        host=host,
                        port=port,
                        seed=seed,
                    ),
                    cwd=str(STARVLA_ROOT),
                    env=bridge_env,
                    stdout=task_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                return_code = wait_for_task_completion(
                    current_task_proc,
                    run_dir=run_dir,
                    task_name=current_task_name,
                    worker_index=worker_index,
                    heartbeat_sec=heartbeat_sec,
                )
            current_task_proc = None
            write_json(
                progress_dir / f"{current_task_name}.json",
                {
                    "task_name": current_task_name,
                    "worker_index": worker_index,
                    "status": "ok" if return_code == 0 else "failed",
                    "exit_code": int(return_code),
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            finish_claimed_task(
                run_dir,
                task_name=current_task_name,
                worker_index=worker_index,
                return_code=return_code,
            )
            if return_code != 0:
                status = 1
                log_line(worker_log, f"worker={worker_index} task={current_task_name} failed exit_code={return_code}")
            else:
                log_line(worker_log, f"worker={worker_index} done task={current_task_name}")
            current_task_name = None
        return status
    finally:
        terminate_process(current_task_proc)
        if current_task_name is not None:
            release_claimed_task(
                run_dir,
                task_name=current_task_name,
                worker_index=worker_index,
                expected_num_episodes=expected_test_num if resume_incomplete else None,
                reason="worker_shutdown",
            )
        terminate_process(server_proc)
        server_log.close()
        worker_log.close()


def master_main(args: argparse.Namespace) -> int:
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.task_config not in {"demo_clean", "demo_randomized"}:
        raise ValueError(f"task_config must be demo_clean or demo_randomized, got {args.task_config}")

    all_tasks = resolve_tasks_from_env()
    expected_test_num = int(os.getenv("ROBOTWIN_TEST_NUM", "100"))
    gpu_ids = detect_gpu_ids()
    num_workers = int(args.num_workers or os.getenv("NUM_WORKERS", len(gpu_ids) * 3))
    if num_workers < 1:
        raise ValueError(f"NUM_WORKERS must be >= 1, got {num_workers}")

    ckpt_alias = os.getenv("ROBOTWIN_CKPT_ALIAS", derive_ckpt_alias(ckpt_path))
    resume_run_dir = Path(str(args.resume_run_dir).strip()).expanduser().resolve() if str(args.resume_run_dir).strip() else None
    if resume_run_dir is not None:
        if not resume_run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {resume_run_dir}")
        output_root = resume_run_dir.parent.parent
        run_group = resume_run_dir.parent.name
        run_tag = resume_run_dir.name
        run_dir = resume_run_dir
        scheduled_tasks = select_incomplete_tasks(
            run_dir,
            all_tasks,
            expected_num_episodes=expected_test_num,
        )
        resume_mode = True
    else:
        output_root = Path(
            os.getenv(
                "OUTPUT_ROOT",
                os.getenv("ROBOTWIN_EVAL_ROOT", str(STARVLA_ROOT / "results/eval_runs/robotwin")),
            )
        ).expanduser()
        run_group = build_run_group(ckpt_alias, args.task_config)
        run_tag = build_run_tag(str(args.run_tag))
        run_dir = output_root / run_group / run_tag
        scheduled_tasks = list(all_tasks)
        resume_mode = False

    (run_dir / "workers").mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / ".task_status"
    progress_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = len(scheduled_tasks)

    run_meta_path = run_dir / "run_meta.json"
    existing_meta: dict[str, object] = {}
    if run_meta_path.is_file():
        try:
            payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing_meta = payload
        except Exception:
            existing_meta = {}

    write_json(
        run_meta_path,
        {
            **existing_meta,
            "run_group": run_group,
            "run_tag": run_tag,
            "run_dir": str(run_dir),
            "checkpoint_path": ckpt_path,
            "checkpoint_alias": ckpt_alias,
            "task_config": args.task_config,
            "gpu_ids": gpu_ids,
            "num_workers": num_workers,
            "total_tasks": len(all_tasks),
            "scheduled_tasks": total_tasks,
            "requested_tasks": all_tasks,
            "resume_incomplete": resume_mode,
            "resume_run_dir": str(run_dir) if resume_mode else None,
            "expected_test_num": expected_test_num,
            "task_assignment_strategy": "dynamic_shared_queue",
            "robotwin_num_slots": int(os.getenv("ROBOTWIN_NUM_SLOTS", os.getenv("ROBOTWIN_BATCH_SIZE", "1"))),
            "save_video": str(os.getenv("ROBOTWIN_SAVE_VIDEO", "0")).strip().lower() in {"1", "true", "yes", "on"},
            "task_heartbeat_sec": resolve_task_scheduler_heartbeat_sec(),
            "task_lease_timeout_sec": resolve_task_scheduler_lease_timeout_sec(resolve_task_scheduler_heartbeat_sec()),
        },
    )

    if total_tasks == 0:
        print(f"No incomplete Robotwin tasks found under {run_dir}")
        return 0

    initialize_task_scheduler(run_dir, scheduled_tasks)

    worker_procs: list[subprocess.Popen] = []
    status = 0
    finished_pids: set[int] = set()
    progress_bar = tqdm(total=total_tasks, desc="Robotwin Tasks", unit="task", dynamic_ncols=True)
    try:
        install_interrupt_handlers()
        worker_env = {
            **os.environ,
            "GPU_IDS": " ".join(gpu_ids),
            "ROBOTWIN_TASKS": " ".join(scheduled_tasks),
            "ROBOTWIN_EVAL_ROOT": str(output_root),
            "OUTPUT_ROOT": str(output_root),
            "ROBOTWIN_RUN_GROUP": run_group,
            "ROBOTWIN_RUN_TAG": run_tag,
        }
        if resume_mode:
            worker_env["ROBOTWIN_RESUME_INCOMPLETE"] = "1"
            worker_env["ROBOTWIN_RESUME_RUN_DIR"] = str(run_dir)
        for worker_index in range(num_workers):
            cmd = [
                resolve_starvla_python(),
                str(Path(__file__).resolve()),
                "--mode",
                "worker",
                "--ckpt_path",
                ckpt_path,
                "--task_config",
                str(args.task_config),
                "--run_tag",
                run_tag,
                "--worker_index",
                str(worker_index),
                "--num_workers",
                str(num_workers),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=str(STARVLA_ROOT),
                env=worker_env,
                start_new_session=True,
            )
            worker_procs.append(proc)
            launch_delay = float(os.getenv("LAUNCH_DELAY_SEC", "1"))
            if worker_index < num_workers - 1 and launch_delay > 0:
                time.sleep(launch_delay)

        while True:
            completed_tasks = count_completed_tasks(
                run_dir,
                scheduled_tasks,
                expected_num_episodes=expected_test_num,
            )
            if completed_tasks > progress_bar.n:
                progress_bar.update(completed_tasks - progress_bar.n)

            running = False
            for proc in worker_procs:
                return_code = proc.poll()
                if return_code is None:
                    running = True
                    continue
                if proc.pid not in finished_pids:
                    finished_pids.add(proc.pid)
                    if return_code != 0:
                        status = 1

            if not running:
                break
            time.sleep(1.0)

        completed_tasks = count_completed_tasks(
            run_dir,
            scheduled_tasks,
            expected_num_episodes=expected_test_num,
        )
        if completed_tasks > progress_bar.n:
            progress_bar.update(completed_tasks - progress_bar.n)
        return status
    finally:
        progress_bar.close()
        for proc in worker_procs:
            if proc.poll() is None:
                terminate_process(proc)


def _prepare_run(args: argparse.Namespace) -> tuple[Path, list[str], int, bool, dict[str, object]]:
    ckpt_path = str(Path(args.ckpt_path).expanduser().resolve())
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if args.task_config not in {"demo_clean", "demo_randomized"}:
        raise ValueError(f"task_config must be demo_clean or demo_randomized, got {args.task_config}")

    all_tasks = resolve_tasks_from_env()
    expected_test_num = int(os.getenv("ROBOTWIN_TEST_NUM", "100"))
    gpu_ids = detect_gpu_ids()
    num_workers = int(args.num_workers or os.getenv("NUM_WORKERS", len(gpu_ids) * 3))
    if num_workers < 1:
        raise ValueError(f"NUM_WORKERS must be >= 1, got {num_workers}")

    ckpt_alias = os.getenv("ROBOTWIN_CKPT_ALIAS", derive_ckpt_alias(ckpt_path))
    resume_run_dir = Path(str(args.resume_run_dir).strip()).expanduser().resolve() if str(args.resume_run_dir).strip() else None
    if resume_run_dir is not None:
        if not resume_run_dir.is_dir():
            raise FileNotFoundError(f"Resume run directory not found: {resume_run_dir}")
        output_root = resume_run_dir.parent.parent
        run_group = resume_run_dir.parent.name
        run_tag = resume_run_dir.name
        run_dir = resume_run_dir
        scheduled_tasks = select_incomplete_tasks(
            run_dir,
            all_tasks,
            expected_num_episodes=expected_test_num,
        )
        resume_mode = True
    else:
        output_root = Path(
            os.getenv(
                "OUTPUT_ROOT",
                os.getenv("ROBOTWIN_EVAL_ROOT", str(STARVLA_ROOT / "results/eval_runs/robotwin")),
            )
        ).expanduser()
        run_group = build_run_group(ckpt_alias, args.task_config)
        run_tag = build_run_tag(str(args.run_tag))
        run_dir = output_root / run_group / run_tag
        scheduled_tasks = list(all_tasks)
        resume_mode = False

    (run_dir / "workers").mkdir(parents=True, exist_ok=True)
    (run_dir / "tasks").mkdir(parents=True, exist_ok=True)
    progress_dir = run_dir / ".task_status"
    progress_dir.mkdir(parents=True, exist_ok=True)

    run_meta_path = run_dir / "run_meta.json"
    existing_meta: dict[str, object] = {}
    if run_meta_path.is_file():
        try:
            payload = json.loads(run_meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                existing_meta = payload
        except Exception:
            existing_meta = {}

    meta = {
        **existing_meta,
        "run_group": run_group,
        "run_tag": run_tag,
        "run_dir": str(run_dir),
        "checkpoint_path": ckpt_path,
        "checkpoint_alias": ckpt_alias,
        "task_config": args.task_config,
        "gpu_ids": gpu_ids,
        "num_workers": num_workers,
        "total_tasks": len(all_tasks),
        "scheduled_tasks": len(scheduled_tasks),
        "requested_tasks": all_tasks,
        "resume_incomplete": resume_mode,
        "resume_run_dir": str(run_dir) if resume_mode else None,
        "expected_test_num": expected_test_num,
        "task_assignment_strategy": "dynamic_shared_queue",
        "robotwin_num_slots": int(os.getenv("ROBOTWIN_NUM_SLOTS", os.getenv("ROBOTWIN_BATCH_SIZE", "1"))),
        "save_video": str(os.getenv("ROBOTWIN_SAVE_VIDEO", "0")).strip().lower() in {"1", "true", "yes", "on"},
        "task_heartbeat_sec": resolve_task_scheduler_heartbeat_sec(),
        "task_lease_timeout_sec": resolve_task_scheduler_lease_timeout_sec(resolve_task_scheduler_heartbeat_sec()),
    }
    write_json(run_meta_path, meta)
    return run_dir, scheduled_tasks, expected_test_num, resume_mode, meta


def prepare_main(args: argparse.Namespace) -> int:
    run_dir, scheduled_tasks, _, _, _ = _prepare_run(args)
    initialize_task_scheduler(run_dir, scheduled_tasks)
    print(f"Robotwin scheduler prepared: {run_dir}")
    print(f"Scheduled tasks: {len(scheduled_tasks)}")
    return 0


def _scheduler_terminal_counts(run_dir: Path) -> tuple[int, int, int, int]:
    state_path = task_scheduler_state_path(run_dir)
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    state = _normalize_scheduler_state(payload)
    return (
        len(state["pending"]),
        len(state["in_progress"]),
        len(state["completed"]),
        len(state["failed"]),
    )


def monitor_main(args: argparse.Namespace) -> int:
    run_dir, scheduled_tasks, expected_test_num, _, _ = _prepare_run(args)
    total_tasks = len(scheduled_tasks)
    if total_tasks == 0:
        print(f"No incomplete Robotwin tasks found under {run_dir}")
        return 0

    status = 0
    last_status_log_at = 0.0
    status_log_interval_sec = max(5.0, float(os.getenv("ROBOTWIN_MONITOR_LOG_INTERVAL_SEC", "600")))
    progress_bar = tqdm(total=total_tasks, desc="Robotwin Tasks", unit="task", dynamic_ncols=True)
    try:
        while True:
            completed_tasks = count_completed_tasks(
                run_dir,
                scheduled_tasks,
                expected_num_episodes=expected_test_num,
            )
            if completed_tasks > progress_bar.n:
                progress_bar.update(completed_tasks - progress_bar.n)

            pending, in_progress, _, failed = _scheduler_terminal_counts(run_dir)
            if failed > 0:
                status = 1
            now_ts = time.time()
            if now_ts - last_status_log_at >= status_log_interval_sec:
                print(
                    "[Robotwin monitor] "
                    f"completed={completed_tasks}/{total_tasks} "
                    f"failed={failed} pending={pending} in_progress={in_progress} "
                    f"run_dir={run_dir}",
                    flush=True,
                )
                last_status_log_at = now_ts
            if pending == 0 and in_progress == 0 and completed_tasks + failed >= total_tasks:
                break
            time.sleep(2.0)
        print(
            "[Robotwin monitor] "
            f"finished completed={completed_tasks}/{total_tasks} failed={failed} status={status} "
            f"run_dir={run_dir}",
            flush=True,
        )
        return status
    finally:
        progress_bar.close()


def main() -> int:
    args = parse_args()
    if args.mode == "worker":
        return worker_main(args)
    if args.mode == "prepare":
        return prepare_main(args)
    if args.mode == "monitor":
        return monitor_main(args)
    return master_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
