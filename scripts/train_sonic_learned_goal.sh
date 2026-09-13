#!/usr/bin/env bash
# SONIC real-robot data: state 46, action 78, horizon 40.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/scripts/config/vlajepa_sonic_latent_learned_goal.yaml}"
export RUN_ID="${RUN_ID:-sonic_learned_goal_core}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
exec bash "${PROJECT_ROOT}/scripts/train_learned_goal.sh" "$@"
