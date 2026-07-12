export NCCL_IB_DISABLE=1
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export CUDA_VISIBLE_DEVICES=0
export VLA_USE_DEEPSPEED=0
export TMPDIR=/home/dataset-local/tmp
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=1

export WANDB_MODE=${WANDB_MODE:-online}

accelerate launch \
  --num_processes 1 \
  --main_process_port 0 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/sonic_latent_ft.yaml
