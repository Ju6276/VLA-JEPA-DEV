#!/usr/bin/env bash
set -euo pipefail
export NO_ALBUMENTATIONS_UPDATE=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Keep runtime caches under dataset root by default (avoid system paths).
DEFAULT_DATASET_ROOT="$(cd "${REPO_ROOT}/../.." && pwd)/datasets"
DATASET_ROOT_DIR="${DATASET_ROOT_DIR:-${DEFAULT_DATASET_ROOT}}"

# ---------------------- Default configuration ----------------------
DEFAULT_CONFIG_YAML="starVLA/config/training/train_libero.yaml"
DEFAULT_ACCELERATE_CONFIG="starVLA/config/accelerate/ddp_bf16.yaml"

CONFIG_YAML="${CONFIG_YAML:-${DEFAULT_CONFIG_YAML}}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-${DEFAULT_ACCELERATE_CONFIG}}"

# Accept config overrides either as the first positional argument or via flags.
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_yaml=*)
      CONFIG_YAML="${1#*=}"
      shift
      ;;
    --config_yaml)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --config_yaml" >&2
        exit 1
      fi
      CONFIG_YAML="$2"
      shift 2
      ;;
    --accelerate_config=*)
      ACCELERATE_CONFIG="${1#*=}"
      shift
      ;;
    --accelerate_config)
      if [[ $# -lt 2 ]]; then
        echo "Missing value for --accelerate_config" >&2
        exit 1
      fi
      ACCELERATE_CONFIG="$2"
      shift 2
      ;;
    *)
      if [[ "${1}" != --* && "${CONFIG_YAML}" == "${DEFAULT_CONFIG_YAML}" ]]; then
        CONFIG_YAML="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

# ---------------------- Runtime environment -----------------------
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
# Prefer the non-deprecated PyTorch NCCL env var, but accept legacy input.
if [[ -n "${NCCL_ASYNC_ERROR_HANDLING:-}" && -z "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}" ]]; then
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING}"
else
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
fi
unset NCCL_ASYNC_ERROR_HANDLING
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10000}"
export NCCL_SOCKET_TIMEOUT_MS="${NCCL_SOCKET_TIMEOUT_MS:-360000}"
export DEEPSPEED_LOG_LEVEL="${DEEPSPEED_LOG_LEVEL:-error}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache}"

# Accept legacy TRANSFORMERS_CACHE input once, then switch to HF_HOME layout.
if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$(dirname "${TRANSFORMERS_CACHE}")"
fi
export HF_HOME="${HF_HOME:-${DATASET_ROOT_DIR}/.hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
unset TRANSFORMERS_CACHE
mkdir -p "${TRITON_CACHE_DIR}/autotune"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${HF_DATASETS_CACHE}"

# Auto-select NCCL network interface if user does not provide one.
if [[ -z "${NCCL_SOCKET_IFNAME:-}" ]]; then
  if ip -o link show | awk -F': ' '{print $2}' | grep -qx "bond0" >/dev/null 2>&1; then
    export NCCL_SOCKET_IFNAME="bond0"
  else
    default_if="$(ip route | awk '/default/ {print $5; exit}')"
    if [[ -n "${default_if}" ]]; then
      export NCCL_SOCKET_IFNAME="${default_if}"
    fi
  fi
fi

# Optional IB HCA setting; keep auto when not provided.
if [[ -z "${NCCL_IB_HCA:-}" && -d "/sys/class/infiniband" ]]; then
  hca_list="$(ls /sys/class/infiniband 2>/dev/null | paste -sd, - || true)"
  if [[ -n "${hca_list}" ]]; then
    export NCCL_IB_HCA="${hca_list}"
  fi
fi

# ---------------------- Argument and file checks ------------------
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config file not found: ${CONFIG_YAML}" >&2
  exit 1
fi

if [[ ! -f "${ACCELERATE_CONFIG}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG}" >&2
  exit 1
fi

readarray -t run_path_info < <(
  python - "${CONFIG_YAML}" "${EXTRA_ARGS[@]}" <<'PY'
import sys
import yaml

config_yaml = sys.argv[1]
extra_args = sys.argv[2:]

with open(config_yaml, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

run_root_dir = str(cfg.get("run_root_dir", "results/Checkpoints"))
run_id = str(cfg.get("run_id", "starvla_run"))

i = 0
while i < len(extra_args):
    arg = extra_args[i]
    if arg.startswith("--run_root_dir="):
        run_root_dir = arg.split("=", 1)[1]
    elif arg == "--run_root_dir" and i + 1 < len(extra_args):
        run_root_dir = extra_args[i + 1]
        i += 1
    elif arg.startswith("--run_id="):
        run_id = arg.split("=", 1)[1]
    elif arg == "--run_id" and i + 1 < len(extra_args):
        run_id = extra_args[i + 1]
        i += 1
    i += 1

print(run_root_dir)
print(run_id)
PY
)

RUN_ROOT_DIR="${run_path_info[0]}"
RUN_ID="${run_path_info[1]}"

NUM_PROCESSES="${NUM_PROCESSES:-$(python - <<'PY'
import torch
print(max(torch.cuda.device_count(), 1))
PY
)}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)}"

if [[ "${NUM_MACHINES}" != "1" || "${MACHINE_RANK}" != "0" || -n "${MAIN_PROCESS_IP}" ]]; then
  echo "train_lawam.sh only supports single-node training." >&2
  echo "Use train_lawam_distributed.sh for distributed launch." >&2
  exit 1
fi

# ---------------------- Logging -----------------------------------
TIMESTAMP="${LAWAM_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export LAWAM_RUN_TIMESTAMP="${TIMESTAMP}"
RUN_OUTPUT_DIR="${RUN_ROOT_DIR}/${TIMESTAMP}+${RUN_ID}"
LOG_DIR="${RUN_OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train.log"

echo "Starting LaWAM training"
echo "Launch mode: single-node"
echo "Repo root: ${REPO_ROOT}"
echo "Config YAML: ${CONFIG_YAML}"
echo "Accelerate config: ${ACCELERATE_CONFIG}"
echo "Run root dir: ${RUN_ROOT_DIR}"
echo "Run ID: ${RUN_ID}"
echo "Run output dir: ${RUN_OUTPUT_DIR}"
echo "Num processes: ${NUM_PROCESSES}"
echo "Main process port: ${MAIN_PROCESS_PORT}"
echo "TORCH_NCCL_ASYNC_ERROR_HANDLING: ${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
echo "HF cache root: ${HF_HOME}"
echo "HF hub cache: ${HF_HUB_CACHE}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "Log file: ${LOG_FILE}"

cp "${CONFIG_YAML}" "${LOG_DIR}/$(basename "${CONFIG_YAML}")"

accelerate_cmd=(
  accelerate launch
  --config_file "${ACCELERATE_CONFIG}"
  --num_processes "${NUM_PROCESSES}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
)

# Forward any extra CLI overrides, e.g.:
#   bash train_lawam.sh --trainer.max_train_steps 1000
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  accelerate_cmd+=("${EXTRA_ARGS[@]}")
fi

"${accelerate_cmd[@]}" 2>&1 | tee -a "${LOG_FILE}"
