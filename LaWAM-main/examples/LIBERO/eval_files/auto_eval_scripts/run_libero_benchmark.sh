#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

REPO_ROOT="$(libero_repo_root)"
cd "${REPO_ROOT}"

SCRIPT_PATH="${SCRIPT_DIR}/run_libero_suite_benchmark.sh"
DEFAULT_CKPT="results/Checkpoints/libero/lawam_libero_sft_release/final_model/pytorch_model.pt"

your_ckpt="${1:-${CKPT_PATH:-${DEFAULT_CKPT}}}"
run_index_base="${2:-${RUN_INDEX_BASE:-0}}"
suites="${SUITES:-libero_10 libero_object libero_goal libero_spatial}"
# suites="${SUITES:-libero_10}"
launch_delay_sec="${LAUNCH_DELAY_SEC:-0.1}"
legacy_max_concurrent_suites="${MAX_CONCURRENT_SUITES:-}"
num_trials_per_task="${NUM_TRIALS_PER_TASK:-50}"
num_workers="${NUM_WORKERS:-25}"
max_tasks="${MAX_TASKS:-}"
gpu_ids="${GPU_IDS:-}"
eval_gpu_ids="${EVAL_GPU_IDS:-}"
worker_result_timeout_sec="${WORKER_RESULT_TIMEOUT_SEC:-600}"
server_startup_timeout_sec="${SERVER_STARTUP_TIMEOUT_SEC:-600}"
eval_action_chunk_len="${EVAL_ACTION_CHUNK_LEN:-}"
save_videos="${SAVE_VIDEOS:-False}"
save_only_failure_videos="${SAVE_ONLY_FAILURE_VIDEOS:-False}"
save_similarity_video="${SAVE_SIMILARITY_VIDEO:-False}"
sim_src_row="${SIM_SRC_ROW:-3}"
sim_src_col="${SIM_SRC_COL:-7}"
sim_vmin="${SIM_VMIN:-0.4}"
sim_vmax="${SIM_VMAX:-1.0}"
sim_alpha="${SIM_ALPHA:-0.5}"
sim_cmap="${SIM_CMAP:-jet}"
json_save_videos="$(printf '%s' "${save_videos}" | tr '[:upper:]' '[:lower:]')"
json_save_only_failure_videos="$(printf '%s' "${save_only_failure_videos}" | tr '[:upper:]' '[:lower:]')"
json_save_similarity_video="$(printf '%s' "${save_similarity_video}" | tr '[:upper:]' '[:lower:]')"
if [ -n "${eval_action_chunk_len}" ]; then
  json_eval_action_chunk_len="${eval_action_chunk_len}"
else
  json_eval_action_chunk_len="null"
fi

if [ ! -f "${your_ckpt}" ]; then
  echo "[ERROR] Checkpoint not found: ${your_ckpt}" >&2
  exit 1
fi
if [ -n "${eval_action_chunk_len}" ]; then
  case "${eval_action_chunk_len}" in
    ''|*[!0-9]*)
      echo "[ERROR] EVAL_ACTION_CHUNK_LEN must be a positive integer when set, got: ${eval_action_chunk_len}" >&2
      exit 1
      ;;
  esac
  if [ "${eval_action_chunk_len}" -lt 1 ]; then
    echo "[ERROR] EVAL_ACTION_CHUNK_LEN must be >= 1, got: ${eval_action_chunk_len}" >&2
    exit 1
  fi
fi
IFS=' ,' read -r -a SUITE_ARRAY <<< "${suites}"
if [ "${#SUITE_ARRAY[@]}" -eq 0 ]; then
  echo "[ERROR] No task suites were provided. Set SUITES." >&2
  exit 1
fi
if [ -n "${legacy_max_concurrent_suites}" ] && [ "${legacy_max_concurrent_suites}" != "1" ]; then
  echo "[ERROR] MAX_CONCURRENT_SUITES has been removed. Suites always run serially; unset it or set it to 1." >&2
  exit 1
fi

output_root="${OUTPUT_ROOT:-${REPO_ROOT}/results/eval_runs/libero}"
ckpt_alias="${LIBERO_CKPT_ALIAS:-$(derive_ckpt_alias "${your_ckpt}")}"
run_group="${LIBERO_RUN_GROUP:-${ckpt_alias}}"
run_tag="${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}"
run_dir="${output_root}/${run_group}/${run_tag}"
mkdir -p "${run_dir}/suites"

export LIBERO_HOME="${LIBERO_HOME:-../LIBERO}"
export STAR_VLA_PYTHON="${STAR_VLA_PYTHON:-${starVLA_python:-/usr/local/miniconda3/bin/python}}"
num_gpus="$(detect_num_gpus "${STAR_VLA_PYTHON}")"
if ! [[ "${num_gpus}" =~ ^[0-9]+$ ]] || [ "${num_gpus}" -lt 1 ]; then
  echo "[ERROR] NUM_GPUS must be a positive integer, got: ${num_gpus}" >&2
  exit 1
fi

cat > "${run_dir}/run_meta.json" <<EOF
{
  "run_group": "${run_group}",
  "run_tag": "${run_tag}",
  "run_dir": "${run_dir}",
  "checkpoint_path": "${your_ckpt}",
  "checkpoint_alias": "${ckpt_alias}",
  "suites": "${suites}",
  "run_index_base": ${run_index_base},
  "suite_execution_mode": "serial",
  "num_trials_per_task": ${num_trials_per_task},
  "num_workers": ${num_workers},
  "max_tasks": ${max_tasks:-null},
  "worker_result_timeout_sec": ${worker_result_timeout_sec},
  "server_startup_timeout_sec": ${server_startup_timeout_sec},
  "save_videos": ${json_save_videos},
  "save_only_failure_videos": ${json_save_only_failure_videos},
  "save_similarity_video": ${json_save_similarity_video},
  "sim_src_row": ${sim_src_row},
  "sim_src_col": ${sim_src_col},
  "sim_vmin": ${sim_vmin},
  "sim_vmax": ${sim_vmax},
  "sim_alpha": ${sim_alpha},
  "sim_cmap": "${sim_cmap}",
  "eval_action_chunk_len": ${json_eval_action_chunk_len},
  "gpu_ids": "${gpu_ids}",
  "eval_gpu_ids": "${eval_gpu_ids}"
}
EOF

echo "Launching LIBERO benchmark"
echo "  checkpoint: ${your_ckpt}"
echo "  suites: ${suites}"
echo "  run_index_base: ${run_index_base}"
echo "  num_trials_per_task: ${num_trials_per_task}"
echo "  suite_execution_mode: serial"
echo "  per_suite_workers: ${num_workers}"
echo "  worker_result_timeout_sec: ${worker_result_timeout_sec}"
echo "  server_startup_timeout_sec: ${server_startup_timeout_sec}"
echo "  save_videos: ${save_videos}"
echo "  save_only_failure_videos: ${save_only_failure_videos}"
echo "  save_similarity_video: ${save_similarity_video}"
echo "  sim_src_row: ${sim_src_row}"
echo "  sim_src_col: ${sim_src_col}"
echo "  sim_vmin: ${sim_vmin}"
echo "  sim_vmax: ${sim_vmax}"
echo "  sim_alpha: ${sim_alpha}"
echo "  sim_cmap: ${sim_cmap}"
echo "  gpu_ids: ${gpu_ids:-auto}"
echo "  eval_gpu_ids: ${eval_gpu_ids:-same_as_gpu_ids}"
echo "  eval_action_chunk_len: ${eval_action_chunk_len:-full_chunk}"
echo "  run_dir: ${run_dir}"

if [ "${save_only_failure_videos}" = "True" ] && [ "${save_videos}" != "True" ]; then
  echo "[WARN] SAVE_ONLY_FAILURE_VIDEOS=True is ignored because SAVE_VIDEOS=False." >&2
  echo "[WARN] Set SAVE_VIDEOS=True SAVE_ONLY_FAILURE_VIDEOS=True to save only failed episodes." >&2
fi

failed_suites=0
for idx in "${!SUITE_ARRAY[@]}"; do
  task_suite_name="${SUITE_ARRAY[$idx]}"
  [ -z "${task_suite_name}" ] && continue

  run_index=$((run_index_base + idx))
  suite_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${gpu_ids}")"
  suite_eval_gpu_id="${suite_gpu_id}"
  if [ -n "${eval_gpu_ids}" ]; then
    suite_eval_gpu_id="$(resolve_gpu_id "${run_index}" "${num_gpus}" "${eval_gpu_ids}")"
  fi

  suite_launch_log="${run_dir}/suites/${task_suite_name}/launcher.log"
  mkdir -p "$(dirname "${suite_launch_log}")"

  echo "Running suite serially"
  echo "  suite: ${task_suite_name}"
  echo "  run_index: ${run_index}"
  echo "  server_gpu_id: ${suite_gpu_id}"
  echo "  eval_gpu_id: ${suite_eval_gpu_id}"
  echo "  launcher_log: ${suite_launch_log}"

  set +e
  OUTPUT_ROOT="${output_root}" \
  LIBERO_CKPT_ALIAS="${ckpt_alias}" \
  LIBERO_RUN_GROUP="${run_group}" \
  RUN_TAG="${run_tag}" \
  GPU_ID="${suite_gpu_id}" \
  EVAL_GPU_ID="${suite_eval_gpu_id}" \
  NUM_TRIALS_PER_TASK="${num_trials_per_task}" \
  NUM_WORKERS="${num_workers}" \
  MAX_TASKS="${max_tasks}" \
  WORKER_RESULT_TIMEOUT_SEC="${worker_result_timeout_sec}" \
  SERVER_STARTUP_TIMEOUT_SEC="${server_startup_timeout_sec}" \
  SAVE_VIDEOS="${save_videos}" \
  SAVE_ONLY_FAILURE_VIDEOS="${save_only_failure_videos}" \
  SAVE_SIMILARITY_VIDEO="${save_similarity_video}" \
  SIM_SRC_ROW="${sim_src_row}" \
  SIM_SRC_COL="${sim_src_col}" \
  SIM_VMIN="${sim_vmin}" \
  SIM_VMAX="${sim_vmax}" \
  SIM_ALPHA="${sim_alpha}" \
  SIM_CMAP="${sim_cmap}" \
  EVAL_ACTION_CHUNK_LEN="${eval_action_chunk_len}" \
  bash "${SCRIPT_PATH}" "${your_ckpt}" "${task_suite_name}" "${run_index}" \
    2>&1 | tee "${suite_launch_log}"
  suite_status=${PIPESTATUS[0]}
  set -e

  if [ "${suite_status}" -eq 0 ]; then
    echo "[OK] Suite completed: ${task_suite_name}"
  else
    echo "[ERROR] Suite failed: ${task_suite_name} (exit_code=${suite_status})" >&2
    echo "  launcher_log: ${suite_launch_log}" >&2
    failed_suites=$((failed_suites + 1))
  fi

  if [ "${idx}" -lt "$(( ${#SUITE_ARRAY[@]} - 1 ))" ] && [ "${launch_delay_sec}" -gt 0 ]; then
    sleep "${launch_delay_sec}"
  fi
done

if [ "${failed_suites}" -ne 0 ]; then
  echo "[ERROR] ${failed_suites} suite(s) failed. Check logs under: ${run_dir}/suites" >&2
  exit 1
fi

echo "LIBERO benchmark completed: ${run_dir}"
