#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" != 1 ]]; then
  echo "Usage: $0 <opendoor|double_desk|football|pp_box|boxing|sit_sofa|vision_navi>" >&2
  exit 2
fi

TASK="$1"
case "${TASK}" in
  opendoor|double_desk|football|pp_box|boxing|sit_sofa|vision_navi) ;;
  *) echo "Unsupported HumanoidArena task: ${TASK}" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${REPO_ROOT}/scripts/config/vlajepa_humanoidarena_sonic40.yaml"
TRAIN_PYTHON="${TRAIN_PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
DATASET_PATH="${DATA_ROOT}/humanoidarena_sonic_v31_${TASK}"
BASE_VLM="${BASE_VLM:-Qwen/Qwen3-VL-2B-Instruct}"
VJEPA2_ENCODER="${VJEPA2_ENCODER:-facebook/vjepa2-vitl-fpc64-256}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/checkpoints/humanoidarena}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_IDS <<<"${CUDA_VISIBLE_DEVICES}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_IDS[@]}}"
BATCH_DENOMINATOR=$((NUM_PROCESSES * GRADIENT_ACCUMULATION_STEPS))
if (( GLOBAL_BATCH_SIZE % BATCH_DENOMINATOR != 0 )); then
  echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by processes=${NUM_PROCESSES} x accumulation=${GRADIENT_ACCUMULATION_STEPS}" >&2
  exit 2
fi
PER_DEVICE_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / BATCH_DENOMINATOR))

export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1000}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-ju-dong6276-technical-university-of-munich}"
export WANDB_PROJECT="${WANDB_PROJECT:-HumanoidArena}"

cd "${REPO_ROOT}"
"${TRAIN_PYTHON}" examples/HumanoidArena/validate_dataset.py "${DATASET_PATH}"

RUN_ID="humanoidarena-sonic-${TASK}-vlajepa-canonical40-h30-100k"
echo "Training VLA-JEPA task=${TASK} processes=${NUM_PROCESSES} global_batch=${GLOBAL_BATCH_SIZE} per_device_batch=${PER_DEVICE_BATCH_SIZE} accumulation=${GRADIENT_ACCUMULATION_STEPS} steps=${MAX_STEPS}"

exec "${TRAIN_PYTHON}" -m accelerate.commands.launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_PATH}" \
  --run_id "${RUN_ID}" \
  --run_root_dir "${OUTPUT_ROOT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  --wandb_project "${WANDB_PROJECT}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.vj2_model.base_encoder "${VJEPA2_ENCODER}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "humanoidarena_sonic40_${TASK}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --trainer.max_train_steps "${MAX_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
