#!/bin/bash
# Continue all_merged fine-tune from steps_70000 for another 20k steps.
# World-model encoder: V-JEPA 2.1 ViT-L/384 (weights under VJEPA21/).
# Usage:
#   bash scripts/vlajepa_all_merged_ft_from70k.sh
#   NUM_PROCESSES=4 bash scripts/vlajepa_all_merged_ft_from70k.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

CONDA_SH="/cpfs_infra/shared/xiaoxinyu/opt/miniconda3/etc/profile.d/conda.sh"
if [ -f "${CONDA_SH}" ]; then
  source "${CONDA_SH}"
  conda activate VLA_JEPA
else
  export PATH="/cpfs_infra/shared/xiaoxinyu/opt/miniconda3/envs/VLA_JEPA/bin:${PATH}"
fi

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export TMPDIR=/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=8

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_ENTITY="${WANDB_ENTITY:-xinyu-xiao-kinetix-ai}"
export WANDB_PROJECT="${WANDB_PROJECT:-VLA_JEPA_garbage_vjepa21}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "Warning: WANDB_API_KEY is not set. Set WANDB_MODE=offline to train without W&B."
fi

NUM_PROCESSES="${NUM_PROCESSES:-8}"
echo "Using NUM_PROCESSES=${NUM_PROCESSES}"
echo "Config: scripts/config/vlajepa_all_merged_ft_from70k.yaml (V-JEPA 2.1, 384px)"
echo "Encoder: VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt"
echo "Init ckpt: checkpoints/all_merged_vjepa21_ft/checkpoints/steps_70000_pytorch_model.pt"
echo "Dataset: dataset (data_mix=garbage)"
echo "Output: checkpoints/all_merged_vjepa21_ft_from70k"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_all_merged_ft_from70k.yaml
