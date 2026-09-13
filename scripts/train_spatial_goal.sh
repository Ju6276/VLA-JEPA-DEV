#!/usr/bin/env bash
# Annotation-free spatial goals with interface-specific control dimensions.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/train_spatial_goal.sh simple|sonic [training overrides...]" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$1" in
  simple)
    export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/scripts/config/vlajepa_simple_spatial_goal.yaml}"
    export RUN_ID="${RUN_ID:-simple_spatial_goal_8xa100}"
    SPATIAL_ACCUMULATION_STEPS=32
    ;;
  sonic)
    export CONFIG_YAML="${CONFIG_YAML:-${PROJECT_ROOT}/scripts/config/vlajepa_sonic_spatial_goal.yaml}"
    export RUN_ID="${RUN_ID:-sonic_spatial_goal_8xa100}"
    SPATIAL_ACCUMULATION_STEPS=4
    ;;
  *)
    echo "Unknown control interface: $1 (expected simple or sonic)" >&2
    exit 2
    ;;
esac
shift
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
exec bash "${PROJECT_ROOT}/scripts/train_learned_goal.sh" \
  --trainer.gradient_accumulation_steps "${SPATIAL_ACCUMULATION_STEPS}" "$@"
