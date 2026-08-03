#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_ROOT}/../../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CKPT="${DEFAULT_CKPT:-results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt}"
your_ckpt="${1:-${CKPT_PATH:-${DEFAULT_CKPT}}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
# task_config="${2:-${TASK_CONFIG:-demo_randomized}}"
default_run_tag=""
if [[ -n "${RUN_TAG:-}" ]]; then
  default_run_tag="${RUN_TAG}"
elif [[ -n "${ROBOTWIN_RUN_TAG:-}" ]]; then
  default_run_tag="${ROBOTWIN_RUN_TAG}"
elif [[ -n "${LAWAM_RUN_TIMESTAMP:-}" ]]; then
  default_run_tag="${LAWAM_RUN_TIMESTAMP}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  default_run_tag="slurm_${SLURM_JOB_ID}"
else
  default_run_tag="$(date +"%Y%m%d_%H%M%S")"
fi
run_tag="${3:-${default_run_tag}}"
resume_run_dir="${4:-${ROBOTWIN_RESUME_RUN_DIR:-${RESUME_RUN_DIR:-}}}"

export NUM_WORKERS="${NUM_WORKERS:-32}" 
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-7}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-50}"
export ROBOTWIN_SAVE_VIDEO="${ROBOTWIN_SAVE_VIDEO:-0}"
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN="${ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN:-1}"
export ROBOTWIN_REPLAN_STEPS="${ROBOTWIN_REPLAN_STEPS:-36}"
export ROBOTWIN_ACTION_ENSEMBLE="${ROBOTWIN_ACTION_ENSEMBLE:-0}"
export ROBOTWIN_ACTION_ENSEMBLE_ALPHA="${ROBOTWIN_ACTION_ENSEMBLE_ALPHA:-0}"
export ROBOTWIN_WAIT_FOR_SCHEDULER="${ROBOTWIN_WAIT_FOR_SCHEDULER:-1}"

if [[ -n "${STAR_VLA_PYTHON:-}" ]]; then
  :
else
  STAR_VLA_PYTHON="python"
fi

if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || [[ "${NUM_WORKERS}" -lt 1 ]]; then
  echo "NUM_WORKERS must be a positive integer, got: ${NUM_WORKERS}" >&2
  exit 1
fi

LOCAL_RANK_VALUE="${LOCAL_RANK:-${SLURM_LOCALID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${MV2_COMM_WORLD_LOCAL_RANK:-}}}}"
if [[ -n "${LOCAL_RANK_VALUE}" && "${LOCAL_RANK_VALUE}" != "0" ]]; then
  echo "LOCAL_RANK=${LOCAL_RANK_VALUE}; only local rank 0 launches Robotwin workers on each node."
  exit 0
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"
else
  NUM_GPUS="$("${STAR_VLA_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
fi

if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "Failed to detect visible GPU count, got: ${NUM_GPUS}" >&2
  exit 1
fi

RAW_WORLD_SIZE="${WORLD_SIZE:-${SLURM_NTASKS:-${OMPI_COMM_WORLD_SIZE:-${MV2_COMM_WORLD_SIZE:-}}}}"
RAW_RANK="${RANK:-${SLURM_PROCID:-${OMPI_COMM_WORLD_RANK:-${MV2_COMM_WORLD_RANK:-}}}}"
LOCAL_WORLD_SIZE="${LOCAL_WORLD_SIZE:-}"
if ! [[ "${LOCAL_WORLD_SIZE}" =~ ^[0-9]+$ ]] || [[ "${LOCAL_WORLD_SIZE}" -lt 1 ]]; then
  LOCAL_WORLD_SIZE="${NUM_GPUS}"
fi

if [[ -n "${GPU_IDS:-}" ]]; then
  read -r -a visible_gpu_ids <<<"$(echo "${GPU_IDS}" | tr ',' ' ')"
elif [[ -n "${LOCAL_RANK_VALUE}" ]]; then
  visible_gpu_ids=()
  for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
    visible_gpu_ids+=("${gpu_idx}")
  done
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  read -r -a visible_gpu_ids <<<"$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' ' ')"
else
  visible_gpu_ids=()
  for ((gpu_idx = 0; gpu_idx < NUM_GPUS; gpu_idx++)); do
    visible_gpu_ids+=("${gpu_idx}")
  done
fi
LOCAL_GPU_COUNT="${#visible_gpu_ids[@]}"
if [[ "${LOCAL_GPU_COUNT}" -lt 1 ]]; then
  echo "No visible GPU IDs resolved." >&2
  exit 1
fi

NNODES="${NNODES:-${NUM_MACHINES:-${SLURM_NNODES:-}}}"
NODE_RANK="${NODE_RANK:-${MACHINE_RANK:-${SLURM_NODEID:-${GROUP_RANK:-}}}}"

if [[ -z "${NNODES}" || -z "${NODE_RANK}" ]]; then
  registry_info="$(
    "${STAR_VLA_PYTHON}" - \
      "${your_ckpt}" \
      "${task_config}" \
      "${run_tag}" \
      "${resume_run_dir}" \
      "${NUM_WORKERS}" \
      "${LOCAL_GPU_COUNT}" \
      "${NNODES:-}" <<'PY'
import fcntl
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

from examples.Robotwin.eval_files.robotwin_eval_common import build_run_group, build_run_tag, derive_ckpt_alias


def _resolve_run_dir() -> Path:
    ckpt_path = str(Path(sys.argv[1]).expanduser().resolve())
    task_config = str(sys.argv[2])
    run_tag = build_run_tag(str(sys.argv[3]))
    resume_run_dir = str(sys.argv[4]).strip()
    if resume_run_dir:
        return Path(resume_run_dir).expanduser().resolve()

    output_root = Path(
        os.getenv(
            "OUTPUT_ROOT",
            os.getenv(
                "ROBOTWIN_EVAL_ROOT",
                str(Path.cwd() / "results" / "eval_runs" / "robotwin"),
            ),
        )
    ).expanduser()
    ckpt_alias = os.getenv("ROBOTWIN_CKPT_ALIAS", derive_ckpt_alias(ckpt_path))
    run_group = build_run_group(ckpt_alias, task_config)
    return output_root / run_group / run_tag


run_dir = _resolve_run_dir()
run_dir.mkdir(parents=True, exist_ok=True)
registry_path = run_dir / ".node_registry.json"
lock_path = run_dir / ".node_registry.lock"

num_workers = max(1, int(sys.argv[5]))
local_gpu_count = max(1, int(sys.argv[6]))
explicit_nnodes = str(sys.argv[7]).strip()
expected_nnodes = int(explicit_nnodes) if explicit_nnodes else math.ceil(num_workers / local_gpu_count)
expected_nnodes = max(1, expected_nnodes)

node_id = os.getenv("ROBOTWIN_NODE_ID", "").strip() or socket.gethostname()
hostname = socket.gethostname()
timeout_sec = max(30.0, float(os.getenv("ROBOTWIN_NODE_REGISTRATION_TIMEOUT_SEC", "1800")))
poll_sec = max(0.2, float(os.getenv("ROBOTWIN_NODE_REGISTRATION_POLL_SEC", "1")))
deadline = time.time() + timeout_sec

while True:
    with open(lock_path, "a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if registry_path.is_file():
                payload = json.loads(registry_path.read_text(encoding="utf-8"))
            else:
                payload = {}
        except Exception:
            payload = {}

        nodes = payload.get("nodes", [])
        if not isinstance(nodes, list):
            nodes = []

        existing_rank = None
        for idx, entry in enumerate(nodes):
            if isinstance(entry, dict) and entry.get("node_id") == node_id:
                existing_rank = idx
                break

        if existing_rank is None:
            if len(nodes) >= expected_nnodes:
                raise RuntimeError(
                    f"Robotwin node registry at {registry_path} already has {len(nodes)} entries; "
                    f"cannot register extra node_id={node_id}."
                )
            nodes.append(
                {
                    "node_id": node_id,
                    "hostname": hostname,
                    "registered_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            )
            existing_rank = len(nodes) - 1

        payload = {
            "expected_nnodes": expected_nnodes,
            "nodes": nodes,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        current_count = len(nodes)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    if current_count >= expected_nnodes:
        print(f"{existing_rank}\t{expected_nnodes}\t{node_id}\t{run_dir}")
        break
    if time.time() >= deadline:
        raise TimeoutError(
            f"Timed out waiting for {expected_nnodes} nodes to register in {registry_path}; "
            f"currently have {current_count}."
        )
    time.sleep(poll_sec)
PY
  )"
  IFS=$'\t' read -r NODE_RANK NNODES NODE_ID_VALUE SHARED_RUN_DIR <<<"${registry_info}"
else
  NODE_ID_VALUE="${ROBOTWIN_NODE_ID:-$(hostname)}"
  SHARED_RUN_DIR="${resume_run_dir:-}"
fi

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [[ "${NNODES}" -lt 1 ]]; then
  echo "Invalid NNODES value: ${NNODES}" >&2
  exit 1
fi

if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [[ "${NODE_RANK}" -lt 0 ]]; then
  echo "Invalid NODE_RANK value: ${NODE_RANK}" >&2
  exit 1
fi

if [[ "${NODE_RANK}" -ge "${NNODES}" ]]; then
  echo "NODE_RANK ${NODE_RANK} must be in [0, ${NNODES})" >&2
  exit 1
fi

base_workers=$(( NUM_WORKERS / NNODES ))
extra_workers=$(( NUM_WORKERS % NNODES ))
if (( NODE_RANK < extra_workers )); then
  local_worker_count=$(( base_workers + 1 ))
  local_worker_start=$(( NODE_RANK * local_worker_count ))
else
  local_worker_count="${base_workers}"
  local_worker_start=$(( extra_workers * (base_workers + 1) + (NODE_RANK - extra_workers) * base_workers ))
fi

if (( local_worker_count > LOCAL_GPU_COUNT )) && [[ "${ALLOW_GPU_OVERSUBSCRIBE:-0}" != "1" ]]; then
  echo "Refusing to launch ${local_worker_count} Robotwin workers on ${LOCAL_GPU_COUNT} visible GPUs." >&2
  echo "Set NUM_WORKERS to the cluster-wide GPU count, or set ALLOW_GPU_OVERSUBSCRIBE=1 if you really want multiple workers per GPU." >&2
  exit 1
fi

runner_args=(
  --ckpt_path "${your_ckpt}"
  --task_config "${task_config}"
  --run_tag "${run_tag}"
  --num_workers "${NUM_WORKERS}"
)

if [[ -n "${resume_run_dir}" ]]; then
  runner_args+=(--resume_run_dir "${resume_run_dir}")
fi

worker_pids=()
monitor_pid=""

collect_descendant_pids() {
  local parent_pid="$1"
  local child_pid
  while read -r child_pid; do
    [[ -n "${child_pid}" ]] || continue
    collect_descendant_pids "${child_pid}"
    printf '%s\n' "${child_pid}"
  done < <(pgrep -P "${parent_pid}" || true)
}

terminate_pid_tree() {
  local root_pid="$1"
  local signal_name="${2:-TERM}"
  local pid
  local -a targets=()

  while read -r pid; do
    [[ -n "${pid}" ]] || continue
    targets+=("${pid}")
  done < <(collect_descendant_pids "${root_pid}")

  if kill -0 "${root_pid}" 2>/dev/null; then
    targets+=("${root_pid}")
  fi

  [[ ${#targets[@]} -gt 0 ]] || return 0
  kill -"${signal_name}" "${targets[@]}" 2>/dev/null || true
}

cleanup() {
  local pid
  for pid in "${worker_pids[@]}"; do
    terminate_pid_tree "${pid}" TERM
  done
  if [[ -n "${monitor_pid}" ]]; then
    terminate_pid_tree "${monitor_pid}" TERM
  fi
  sleep 2
  for pid in "${worker_pids[@]}"; do
    terminate_pid_tree "${pid}" KILL
    wait "${pid}" 2>/dev/null || true
  done
  if [[ -n "${monitor_pid}" ]]; then
    terminate_pid_tree "${monitor_pid}" KILL
    wait "${monitor_pid}" 2>/dev/null || true
  fi
}

on_exit() {
  local exit_code="$1"
  trap - EXIT INT TERM
  cleanup
  exit "${exit_code}"
}

on_interrupt() {
  trap - EXIT INT TERM
  cleanup
  exit 130
}

on_terminate() {
  trap - EXIT INT TERM
  cleanup
  exit 143
}

trap 'on_exit $?' EXIT
trap on_interrupt INT
trap on_terminate TERM

echo "Starting distributed Robotwin eval"
echo "Checkpoint: ${your_ckpt}"
echo "Task config: ${task_config}"
echo "Run tag: ${run_tag}"
echo "Local GPU count: ${LOCAL_GPU_COUNT}"
echo "Raw rank/world/local_rank/local_world: ${RAW_RANK:-}/${RAW_WORLD_SIZE:-}/${LOCAL_RANK_VALUE:-}/${LOCAL_WORLD_SIZE}"
echo "Node ID: ${NODE_ID_VALUE}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
if [[ -n "${SHARED_RUN_DIR}" ]]; then
  echo "Shared run dir: ${SHARED_RUN_DIR}"
fi
echo "NUM_WORKERS(total): ${NUM_WORKERS}"
echo "Local worker start/count: ${local_worker_start}/${local_worker_count}"
echo "Visible GPU IDs: ${visible_gpu_ids[*]}"
echo "ROBOTWIN_NUM_SLOTS: ${ROBOTWIN_NUM_SLOTS}"
echo "ROBOTWIN_TEST_NUM: ${ROBOTWIN_TEST_NUM}"
echo "ROBOTWIN_REPLAN_STEPS: ${ROBOTWIN_REPLAN_STEPS}"
echo "ROBOTWIN_ACTION_ENSEMBLE: ${ROBOTWIN_ACTION_ENSEMBLE}"
echo "ROBOTWIN_ACTION_ENSEMBLE_ALPHA: ${ROBOTWIN_ACTION_ENSEMBLE_ALPHA}"

if [[ "${NODE_RANK}" == "0" ]]; then
  "${STAR_VLA_PYTHON}" "${EVAL_ROOT}/batched_eval_runner.py" --mode prepare "${runner_args[@]}"
  "${STAR_VLA_PYTHON}" "${EVAL_ROOT}/batched_eval_runner.py" --mode monitor "${runner_args[@]}" &
  monitor_pid="$!"
fi

for ((local_idx = 0; local_idx < local_worker_count; local_idx++)); do
  worker_index=$(( local_worker_start + local_idx ))
  worker_gpu_id="${visible_gpu_ids[$(( local_idx % LOCAL_GPU_COUNT ))]}"
  echo "Launching worker_index=${worker_index} on GPU ${worker_gpu_id}"
  GPU_IDS="${worker_gpu_id}" "${STAR_VLA_PYTHON}" "${EVAL_ROOT}/batched_eval_runner.py" \
    --mode worker \
    "${runner_args[@]}" \
    --worker_index "${worker_index}" &
  worker_pids+=("$!")
  launch_delay="${LAUNCH_DELAY_SEC:-1}"
  if (( local_idx < local_worker_count - 1 )) && [[ "${launch_delay}" != "0" ]]; then
    sleep "${launch_delay}"
  fi
done

status=0
for pid in "${worker_pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ -n "${monitor_pid}" ]]; then
  if ! wait "${monitor_pid}"; then
    status=1
  fi
fi

trap - EXIT INT TERM
exit "${status}"
