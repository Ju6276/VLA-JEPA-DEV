#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EVAL_ROOT}/../../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_CKPT="results/Checkpoints/robotwin/lawam_robotwin_sft_release/final_model/pytorch_model.pt"
your_ckpt="${1:-${CKPT_PATH:-${DEFAULT_CKPT}}}"
task_config="${2:-${TASK_CONFIG:-demo_clean}}"
# task_config="${2:-${TASK_CONFIG:-demo_randomized}}"
run_tag="${3:-${RUN_TAG:-$(date +"%Y%m%d_%H%M%S")}}"
resume_run_dir="${4:-${ROBOTWIN_RESUME_RUN_DIR:-${RESUME_RUN_DIR:-}}}"

export NUM_WORKERS="${NUM_WORKERS:-8}"
export ROBOTWIN_NUM_SLOTS="${ROBOTWIN_NUM_SLOTS:-7}"
export ROBOTWIN_TEST_NUM="${ROBOTWIN_TEST_NUM:-14}"
export ROBOTWIN_SAVE_VIDEO="${ROBOTWIN_SAVE_VIDEO:-0}"
export ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN="${ROBOTWIN_SKIP_GET_OBS_WITHIN_REPLAN:-1}"
export ROBOTWIN_REPLAN_STEPS="${ROBOTWIN_REPLAN_STEPS:-36}"
export ROBOTWIN_ACTION_ENSEMBLE="${ROBOTWIN_ACTION_ENSEMBLE:-0}"
export ROBOTWIN_ACTION_ENSEMBLE_ALPHA="${ROBOTWIN_ACTION_ENSEMBLE_ALPHA:-0}"

echo "ROBOTWIN_TEST_NUM=${ROBOTWIN_TEST_NUM}"
echo "ROBOTWIN_NUM_SLOTS=${ROBOTWIN_NUM_SLOTS}"
echo "ROBOTWIN_REPLAN_STEPS=${ROBOTWIN_REPLAN_STEPS}"
echo "ROBOTWIN_ACTION_ENSEMBLE=${ROBOTWIN_ACTION_ENSEMBLE}"
echo "ROBOTWIN_ACTION_ENSEMBLE_ALPHA=${ROBOTWIN_ACTION_ENSEMBLE_ALPHA}"

if [[ -n "${STAR_VLA_PYTHON:-}" ]]; then
  :
else
  STAR_VLA_PYTHON="python"
fi

runner_args=(
  --mode master
  --ckpt_path "${your_ckpt}"
  --task_config "${task_config}"
  --run_tag "${run_tag}"
)

if [[ -n "${resume_run_dir}" ]]; then
  runner_args+=(--resume_run_dir "${resume_run_dir}")
fi

runner_pid=""

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

cleanup_runner() {
  [[ -n "${runner_pid}" ]] || return 0
  terminate_pid_tree "${runner_pid}" TERM
  sleep 2
  terminate_pid_tree "${runner_pid}" KILL
  wait "${runner_pid}" 2>/dev/null || true
}

on_exit() {
  local exit_code="$1"
  trap - EXIT INT TERM
  cleanup_runner
  exit "${exit_code}"
}

on_interrupt() {
  trap - EXIT INT TERM
  cleanup_runner
  exit 130
}

on_terminate() {
  trap - EXIT INT TERM
  cleanup_runner
  exit 143
}

trap 'on_exit $?' EXIT
trap on_interrupt INT
trap on_terminate TERM

"${STAR_VLA_PYTHON}" "${EVAL_ROOT}/batched_eval_runner.py" "${runner_args[@]}" &
runner_pid="$!"
wait "${runner_pid}"
