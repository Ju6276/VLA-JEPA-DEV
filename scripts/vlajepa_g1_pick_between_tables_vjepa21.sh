#!/bin/bash
# G1 pick-between-tables fine-tuning with V-JEPA 2.1 + Delta-JEPA
#   bash scripts/vlajepa_g1_pick_between_tables_vjepa21.sh

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
export OMP_NUM_THREADS=1

NUM_PROCESSES="${NUM_PROCESSES:-8}"
echo "Using NUM_PROCESSES=${NUM_PROCESSES}"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_g1_pick_between_tables_vjepa21.yaml
