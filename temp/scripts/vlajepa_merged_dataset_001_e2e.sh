#!/bin/bash
# End-to-end privileged VLA-JEPA training on merged_dataset_001.
# Usage:
#   bash scripts/vlajepa_merged_dataset_001_e2e.sh
#   NUM_PROCESSES=4 BATCH_SIZE=1 WANDB_MODE=offline \
#     bash scripts/vlajepa_merged_dataset_001_e2e.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-VLA_JEPA}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
elif [ -f "/home/d024/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/home/d024/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "conda was not found; activate ${CONDA_ENV_NAME} before running this script." >&2
  exit 1
fi

CONFIG_PATH="./scripts/config/vlajepa_merged_dataset_001_e2e.yaml"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/dataset}"
BASE_VLM="${BASE_VLM:-${PROJECT_ROOT}/Qwen3-VL-2B-Instruct}"
VJEPA_ENCODER="${VJEPA_ENCODER:-/cpfs_infra/shared/xiaoxinyu/VLA-JEPA/VLA-JEPA-DEV/VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt}"
# One process per visible GPU; more processes than GPUs makes NCCL abort with
# "Duplicate GPU detected".
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  DETECTED_GPUS="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  DETECTED_GPUS="$(nvidia-smi --list-gpus | wc -l)"
fi
NUM_PROCESSES="${NUM_PROCESSES:-${DETECTED_GPUS}}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_ID="${RUN_ID:-merged_dataset_001_e2e_vjepa21}"

for required_path in \
  "${CONFIG_PATH}" \
  "${DATA_ROOT}/merged_dataset_001" \
  "${BASE_VLM}" \
  "${VJEPA_ENCODER}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [ "${NUM_PROCESSES}" -gt "${DETECTED_GPUS}" ]; then
  echo "NUM_PROCESSES=${NUM_PROCESSES} exceeds visible GPUs (${DETECTED_GPUS})." >&2
  exit 1
fi

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export TMPDIR="${TMPDIR:-/tmp}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-xinyu-xiao-kinetix-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLA_JEPA_merged_dataset_001_e2e}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_API_KEY is not set. Set it, or use WANDB_MODE=offline." >&2
fi

echo "W&B: mode=${WANDB_MODE}, entity=${WANDB_ENTITY}, project=${WANDB_PROJECT}"
echo "Processes: ${NUM_PROCESSES}; per-device batch: ${BATCH_SIZE}"
echo "Dataset: ${DATA_ROOT}/merged_dataset_001"
echo "Output: checkpoints/${RUN_ID}"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_PATH}" \
  --run_id "${RUN_ID}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.vj2_model.base_encoder "${VJEPA_ENCODER}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
