#!/bin/bash
# G1 pick-hug-container fine-tuning with V-JEPA 2.1 ViT-L/384 (weights under VJEPA21/).
# NOTE: DLC 默认使用 /bin/sh，不支持 source。请用 bash 运行本脚本：
#   bash scripts/vlajepa_g1_pick_hug_container_ft.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

# 激活 conda 环境（在 bash 脚本内部使用 source 是安全的）
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
export OMP_NUM_THREADS=1

# W&B: login once before training if needed
# wandb login

# Default 8 GPUs; override with NUM_PROCESSES=N if needed
NUM_PROCESSES="${NUM_PROCESSES:-8}"
echo "Using NUM_PROCESSES=${NUM_PROCESSES}"
echo "Config: scripts/config/vlajepa_g1_pick_hug_container_ft.yaml (V-JEPA 2.1, 384px)"
echo "Encoder: VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt"
echo "Output: checkpoints/g1_pick_hug_container_vjepa21_ft"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_g1_pick_hug_container_ft.yaml
