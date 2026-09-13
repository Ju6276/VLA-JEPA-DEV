#!/usr/bin/env bash
# Non-SONIC simulation: state 32, action 36, horizon 30.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/scripts/config/vlajepa_g1_pick_between_tables_vjepa21_8xa100.yaml}"
export RUN_ID="${RUN_ID:-simple_learned_goal_core_8xa100}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
exec bash "${PROJECT_ROOT}/scripts/train_learned_goal.sh" "$@"
