#!/usr/bin/env bash
# Delta/inverse-dynamics ablations with the remaining learned-goal setup shared.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/train_learned_goal_ablation.sh simple|sonic core|delta|ctrl|full [training overrides...]

Variants (lambda_delta / lambda_ctrl):
  core   0 / 0
  delta  0.05 / 0
  ctrl   0 / 0.02
  full   0.05 / 0.02

Set DATA_ROOT and VJEPA21_CKPT as for the standard training launchers.
SEED defaults to 42; RUN_ID defaults to <interface>_learned_goal_<variant>_seed<seed>.
Seed and auxiliary-loss weights are applied after other training overrides.
EOF
}

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
  usage
  exit 0
fi
if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

INTERFACE="$1"
VARIANT="$2"
shift 2

case "${INTERFACE}" in
  simple) LAUNCHER=train_g1_delta_jepa_8xa100.sh ;;
  sonic) LAUNCHER=train_sonic_learned_goal.sh ;;
  *) usage >&2; exit 2 ;;
esac

case "${VARIANT}" in
  core) LAMBDA_DELTA=0; LAMBDA_CTRL=0 ;;
  delta) LAMBDA_DELTA=0.05; LAMBDA_CTRL=0 ;;
  ctrl) LAMBDA_DELTA=0; LAMBDA_CTRL=0.02 ;;
  full) LAMBDA_DELTA=0.05; LAMBDA_CTRL=0.02 ;;
  *) usage >&2; exit 2 ;;
esac

ABLATION_SEED="${SEED:-42}"
if [[ ! "${ABLATION_SEED}" =~ ^[0-9]+$ || ${#ABLATION_SEED} -gt 10 ]] ||
   (( 10#${ABLATION_SEED} > 4294967295 )); then
  echo "SEED must be an integer between 0 and 4294967295." >&2
  usage >&2
  exit 2
fi
ABLATION_SEED=$((10#${ABLATION_SEED}))

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUN_ID="${RUN_ID:-${INTERFACE}_learned_goal_${VARIANT}_seed${ABLATION_SEED}}"

exec bash "${PROJECT_ROOT}/scripts/${LAUNCHER}" "$@" \
  --seed "${ABLATION_SEED}" \
  --framework.delta_jepa.lambda_delta "${LAMBDA_DELTA}" \
  --framework.delta_jepa.lambda_ctrl "${LAMBDA_CTRL}"
