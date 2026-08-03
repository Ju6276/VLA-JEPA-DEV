export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)
#export NCCL_DEBUG=INFO
#export NCCL_DEBUG_SUBSYS=ALL
export FFMPEG_THREADS=1
export OMP_NUM_THREADS=8

export WANDB_MODE=online
export WANDB_ENTITY="${WANDB_ENTITY:-ju-dong6276-technical-university-of-munich}"
export WANDB_PROJECT="${WANDB_PROJECT:-vlajepa_sonic_latent}"

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "Error: WANDB_API_KEY is not set. Export it before training, e.g.:"
  echo "  export WANDB_API_KEY=\"your_wandb_api_key\""
  exit 1
fi

# Number of GPUs to use for training. Adjust to your machine.
NUM_PROCESSES="${NUM_PROCESSES:-8}"

accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_sonic_latent.yaml
