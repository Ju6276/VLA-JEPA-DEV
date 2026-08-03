#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(libero_repo_root)"
cd "${REPO_ROOT}"

your_ckpt="${1:-${CKPT_PATH:-}}"
task_suite_name="${2:-${TASK_SUITE_NAME:-}}"
run_index="${3:-${RUN_INDEX:-}}"

if [ -z "${your_ckpt}" ] || [ -z "${task_suite_name}" ] || [ -z "${run_index}" ]; then
  echo "Usage: $0 <ckpt_path> <task_suite_name> <run_index>" >&2
  exit 1
fi
if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi

export LIBERO_HOME="${LIBERO_HOME:-../LIBERO}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${LIBERO_python:-python}}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
export PYTHONPATH="${LIBERO_HOME}:${REPO_ROOT}:${PYTHONPATH:-}"

if [ ! -x "${LIBERO_PYTHON}" ]; then
  echo "[ERROR] LIBERO_PYTHON is not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [ ! -x "${STAR_VLA_PYTHON}" ]; then
  echo "[ERROR] STAR_VLA_PYTHON is not executable: ${STAR_VLA_PYTHON}" >&2
  exit 1
fi
if [ ! -d "${LIBERO_HOME}" ]; then
  echo "[ERROR] LIBERO_HOME does not exist: ${LIBERO_HOME}" >&2
  exit 1
fi

ensure_libero_eval_config "${LIBERO_HOME}" "${LIBERO_CONFIG_PATH}"

num_gpus="$(detect_num_gpus "${STAR_VLA_PYTHON}")"
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 1
fi

gpu_id="${GPU_ID:-}"
if [ -z "${gpu_id}" ]; then
  gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${GPU_IDS:-}")"
fi
eval_gpu_id="${EVAL_GPU_ID:-}"
if [ -z "${eval_gpu_id}" ] && [ -n "${EVAL_GPU_IDS:-}" ]; then
  eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${EVAL_GPU_IDS}")"
fi
if [ -z "${eval_gpu_id}" ]; then
  eval_gpu_id="${gpu_id}"
fi
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
num_workers="${NUM_WORKERS:-1}"
if [ -z "${MUJOCO_GL:-}" ]; then
  if [ "${num_workers}" -gt 1 ]; then
    export MUJOCO_GL="osmesa"
  else
    export MUJOCO_GL="egl"
  fi
fi
if [ -z "${PYOPENGL_PLATFORM:-}" ] && { [ "${MUJOCO_GL}" = "egl" ] || [ "${MUJOCO_GL}" = "osmesa" ]; }; then
  export PYOPENGL_PLATFORM="${MUJOCO_GL}"
fi
max_tasks="${MAX_TASKS:-}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
save_similarity_video="${SAVE_SIMILARITY_VIDEO:-False}"
sim_src_row="${SIM_SRC_ROW:-3}"
sim_src_col="${SIM_SRC_COL:-7}"
sim_vmin="${SIM_VMIN:-0.4}"
sim_vmax="${SIM_VMAX:-1.0}"
sim_alpha="${SIM_ALPHA:-0.5}"
sim_cmap="${SIM_CMAP:-jet}"
host="${HOST:-127.0.0.1}"
port_base="${PORT_BASE:-5694}"
preferred_port=$((port_base + run_index))
port_search_limit="${PORT_SEARCH_LIMIT:-200}"
reserve_port "${STAR_VLA_PYTHON}" "${preferred_port}" "${port_search_limit}"
base_port="${RESERVED_PORT}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-600}"
benchmark_variant="${BENCHMARK_VARIANT:-libero}"
enable_category_aggregation="${ENABLE_CATEGORY_AGGREGATION:-False}"
unnorm_key="${UNNORM_KEY:-}"
log_path="${LOG_PATH:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
worker_sync_timeout_sec="${WORKER_SYNC_TIMEOUT_SEC:-1.0}"
eval_action_chunk_len="${EVAL_ACTION_CHUNK_LEN:-}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
export MUJOCO_EGL_DEVICE_ID="${gpu_id}"

output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero}"
run_group="${LIBERO_RUN_GROUP:-${LIBERO_CKPT_ALIAS:-$(derive_ckpt_alias "${your_ckpt}")}}"
run_tag="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
suite_dir="${output_root}/${run_group}/${run_tag}/suites/${task_suite_name}"
mkdir -p "${suite_dir}"

server_log="${suite_dir}/server.log"
eval_log="${suite_dir}/eval.log"
server_pid=""

eval_cmd=(
  "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
  --args.pretrained-path "${your_ckpt}"
  --args.host "${host}"
  --args.port "${base_port}"
  --args.task-suite-name "${task_suite_name}"
  --args.num-trials-per-task "${num_trials_per_task}"
  --args.num-workers "${num_workers}"
  --args.worker-sync-timeout-sec "${worker_sync_timeout_sec}"
  --args.video-out-path "${suite_dir}"
  --args.benchmark-variant "${benchmark_variant}"
  --args.enable-category-aggregation "${enable_category_aggregation}"
  --args.worker-result-timeout-sec "${worker_result_timeout_sec}"
)

if [ "${save_videos}" = "True" ]; then
  eval_cmd+=(--args.save-videos)
else
  eval_cmd+=(--args.no-save-videos)
fi
if [ "${save_only_failure_videos}" = "True" ]; then
  eval_cmd+=(--args.save-only-failure-videos)
else
  eval_cmd+=(--args.no-save-only-failure-videos)
fi
if [ "${save_similarity_video}" = "True" ]; then
  eval_cmd+=(--args.save-similarity-video)
else
  eval_cmd+=(--args.no-save-similarity-video)
fi
eval_cmd+=(
  --args.sim-src-row "${sim_src_row}"
  --args.sim-src-col "${sim_src_col}"
  --args.sim-vmin "${sim_vmin}"
  --args.sim-vmax "${sim_vmax}"
  --args.sim-alpha "${sim_alpha}"
  --args.sim-cmap "${sim_cmap}"
)

if [ -n "${unnorm_key}" ]; then
  eval_cmd+=(--args.unnorm-key "${unnorm_key}")
fi
if [ -n "${max_tasks}" ]; then
  eval_cmd+=(--args.max-tasks "${max_tasks}")
fi
if [ -n "${log_path}" ]; then
  eval_cmd+=(--args.log-path "${log_path}")
fi
if [ -n "${eval_action_chunk_len}" ]; then
  eval_cmd+=(--args.eval-action-chunk-len "${eval_action_chunk_len}")
fi

cleanup() {
  local exit_code=$?
  if [ -n "${server_pid}" ] && kill -0 "${server_pid}" 2>/dev/null; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
  release_reserved_port
  exit "${exit_code}"
}
trap cleanup EXIT INT TERM

echo "Starting LIBERO suite benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  task_suite: ${task_suite_name}"
echo "  run_index: ${run_index}"
echo "  server_gpu_id: ${gpu_id}"
echo "  eval_gpu_id: ${eval_gpu_id}"
echo "  num_workers: ${num_workers}"
echo "  mujoco_gl: ${MUJOCO_GL}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  save_similarity_video: ${save_similarity_video}"
echo "  sim_src_row: ${sim_src_row}"
echo "  sim_src_col: ${sim_src_col}"
echo "  sim_vmin: ${sim_vmin}"
echo "  sim_vmax: ${sim_vmax}"
echo "  sim_alpha: ${sim_alpha}"
echo "  sim_cmap: ${sim_cmap}"
echo "  worker_sync_timeout_sec: ${worker_sync_timeout_sec}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  server_startup_timeout_sec: ${server_startup_timeout_sec}"
echo "  eval_action_chunk_len: ${eval_action_chunk_len:-full_chunk}"
echo "  port: ${base_port}"
if [ "${save_only_failure_videos}" = "True" ] && [ "${save_videos}" != "True" ]; then
  echo "[WARN] SAVE_ONLY_FAILURE_VIDEOS=True is ignored because SAVE_VIDEOS=False." >&2
  echo "[WARN] Set SAVE_VIDEOS=True SAVE_ONLY_FAILURE_VIDEOS=True to save only failed episodes." >&2
fi
if [ "${base_port}" != "${preferred_port}" ]; then
  echo "  preferred_port: ${preferred_port} (occupied, auto-switched)"
fi
echo "  output: ${suite_dir}"

CUDA_VISIBLE_DEVICES="${gpu_id}" "${STAR_VLA_PYTHON}" deployment/model_server/server_policy.py \
  --ckpt_path "${your_ckpt}" \
  --port "${base_port}" \
  --use_bf16 \
  > "${server_log}" 2>&1 &
server_pid=$!

if ! wait_for_port "${STAR_VLA_PYTHON}" "${host}" "${base_port}" "${server_startup_timeout_sec}"; then
  echo "[ERROR] Policy server failed to become ready on ${host}:${base_port}" >&2
  echo "[ERROR] server_log: ${server_log}" >&2
  echo "[ERROR] Try increasing SERVER_STARTUP_TIMEOUT_SEC if the checkpoint is still loading." >&2
  tail -n 40 "${server_log}" >&2 || true
  exit 1
fi

set +e
PYTHONFAULTHANDLER=1 \
CUDA_VISIBLE_DEVICES="${eval_gpu_id}" MUJOCO_EGL_DEVICE_ID="${eval_gpu_id}" \
  "${eval_cmd[@]}" 2>&1 | tee "${eval_log}"
eval_status=${PIPESTATUS[0]}
set -e

if [ "${eval_status}" -ne 0 ]; then
  echo "[ERROR] Evaluation command failed with exit code ${eval_status}" >&2
  tail -n 40 "${server_log}" >&2 || true
  exit "${eval_status}"
fi

echo "LIBERO suite benchmark completed: ${suite_dir}"
