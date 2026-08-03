#!/bin/bash

set -e
export NO_ALBUMENTATIONS_UPDATE=1
export LAM_VIDEO_CACHE_BUILD_WORKERS=16
export LAM_PYAV_THREAD_COUNT=5

DATA_ROOT_DIR="/mnt/xx/xx/datasets"
DATA_MIX=""
CONFIG_FILE=""
VIDEO_BACKEND="pyav"
DATASET_NAME=""
ROBOT_TYPE=""
ENABLE_VIDEO_FRAME_CACHE=0

show_help() {
    cat << EOF
Usage: $0 [options]

Precompute the current StarVLA dataset cache on CPU nodes:
  - meta/stats_gr00t.json
  - optional on-disk video frame cache

Options:
    -h, --help              Show this help message
    -c, --config FILE       Use a YAML config file
    -d, --data-root DIR     Dataset root directory
    -m, --mix NAME          Dataset mixture name
    --dataset-name NAME     Single dataset name
    --robot-type NAME       robot_type for the single dataset
    -b, --backend BACKEND   Optional video backend override (pyav only)
    --video-cache           Also prebuild the on-disk video frame cache

Examples:
    $0 --config starVLA/config/training/starvla_train_oxe.yaml

    $0 --data-root /data/lerobot --mix bridge_rt_1

    $0 --data-root /data/lerobot --mix libero --backend pyav --video-cache

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--data-root)
            DATA_ROOT_DIR="$2"
            shift 2
            ;;
        -m|--mix)
            DATA_MIX="$2"
            shift 2
            ;;
        -b|--backend)
            VIDEO_BACKEND="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --robot-type)
            ROBOT_TYPE="$2"
            shift 2
            ;;
        --video-cache)
            ENABLE_VIDEO_FRAME_CACHE=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# source /path/to/venv/bin/activate

echo "=========================================="
echo "Starting dataset cache precomputation"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "CPU info: $(nproc) cores"
echo "=========================================="

PYTHON_CMD="python ${SCRIPT_DIR}/precompute_dataset_cache.py"

if [ -n "$CONFIG_FILE" ]; then
    echo "Using config file: $CONFIG_FILE"
    PYTHON_CMD="$PYTHON_CMD --config $CONFIG_FILE"
fi

if [ -n "$DATA_ROOT_DIR" ]; then
    echo "Dataset root directory: $DATA_ROOT_DIR"
    PYTHON_CMD="$PYTHON_CMD --data_root_dir $DATA_ROOT_DIR"
fi

if [ -n "$DATA_MIX" ]; then
    echo "Dataset mixture: $DATA_MIX"
    PYTHON_CMD="$PYTHON_CMD --data_mix $DATA_MIX"
fi

if [ -n "$DATASET_NAME" ]; then
    echo "Single dataset: $DATASET_NAME"
    PYTHON_CMD="$PYTHON_CMD --dataset_name $DATASET_NAME"
fi

if [ -n "$ROBOT_TYPE" ]; then
    echo "Robot type: $ROBOT_TYPE"
    PYTHON_CMD="$PYTHON_CMD --robot_type $ROBOT_TYPE"
fi

if [ -n "$VIDEO_BACKEND" ]; then
    if [ "$VIDEO_BACKEND" != "pyav" ]; then
        echo "Error: only --backend pyav is currently supported, got: $VIDEO_BACKEND"
        exit 1
    fi
    PYTHON_CMD="$PYTHON_CMD --video_backend $VIDEO_BACKEND"
fi

if [ "$ENABLE_VIDEO_FRAME_CACHE" -eq 1 ]; then
    echo "Video frame disk cache: enabled"
    PYTHON_CMD="$PYTHON_CMD --enable-video-frame-cache"
fi

echo ""
echo "Command:"
echo "$PYTHON_CMD"
echo "=========================================="
echo ""

START_TIME=$(date +%s)
$PYTHON_CMD
EXIT_CODE=$?
END_TIME=$(date +%s)

ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Precomputation completed!"
else
    echo "✗ Precomputation failed (exit code: $EXIT_CODE)"
fi
echo "Total elapsed time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

exit $EXIT_CODE
