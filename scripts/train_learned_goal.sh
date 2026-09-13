#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

: "${DATA_ROOT:?请设置 DATA_ROOT（所选 data_mix 的数据根目录）}"
: "${VJEPA21_CKPT:?请设置 VJEPA21_CKPT（V-JEPA 2.1 384px 的 .pt 权重）}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints}"
RUN_ID="${RUN_ID:-learned_goal}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "DATA_ROOT 不存在: ${DATA_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${VJEPA21_CKPT}" ]]; then
  echo "VJEPA21_CKPT 不存在或不是文件: ${VJEPA21_CKPT}" >&2
  exit 1
fi
if ! command -v accelerate >/dev/null 2>&1; then
  echo "找不到 accelerate；请先激活训练环境并安装依赖。" >&2
  exit 1
fi

export WANDB_MODE
export TOKENIZERS_PARALLELISM=false
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-3600}"
export FFMPEG_THREADS="${FFMPEG_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

: "${CONFIG_YAML:?请通过仿真或 SONIC 启动脚本选择训练配置}"
CONFIG="${CONFIG_YAML}"
if [[ ! -f "${CONFIG}" ]]; then
  echo "CONFIG_YAML 不存在: ${CONFIG}" >&2
  exit 1
fi

echo "启动自动目标训练: ${CONFIG}"
echo "  GPU 数: ${NUM_PROCESSES}"
echo "  单卡 batch: ${PER_DEVICE_BATCH_SIZE}"
echo "  梯度累积与全局 batch: 以 trainer 配置和训练启动日志为准"
echo "  输出目录: ${OUTPUT_ROOT}/${RUN_ID}"
echo "  目标来源: 当前 ego 图像 + 指令 + 状态（无需 SUBGOALS_PATH）"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}" \
  --run_id "${RUN_ID}" \
  --run_root_dir "${OUTPUT_ROOT}" \
  --framework.qwenvl.base_vlm "${QWEN_MODEL}" \
  --framework.vj2_model.base_encoder "${VJEPA21_CKPT}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  "$@"
