#!/bin/bash

export PYTHONDONTWRITEBYTECODE=1
export LIBERO_HOME=/home/d013/桌面/LIBERO # your LIBERO code path
export LIBERO_CONFIG_PATH="$(pwd)/.libero"
export main_python="${main_python:-python}" # your VLA-JEPA python path

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export sim_python=/home/d013/anaconda3/envs/libero/bin/python # your LIBERO conda path

your_ckpt=/home/d013/桌面/VLA-JEPA/CKPT/LIBERO/checkpoints/VLA-JEPA-LIBERO.pt
folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
benchmark_root="${LIBERO_HOME}/libero/libero"

mkdir -p "${LIBERO_CONFIG_PATH}" "${LIBERO_CONFIG_PATH}/datasets"

if [ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]; then
    printf '%s\n' \
        "benchmark_root: ${benchmark_root}" \
        "bddl_files: ${benchmark_root}/bddl_files" \
        "init_states: ${benchmark_root}/init_files" \
        "datasets: ${LIBERO_CONFIG_PATH}/datasets" \
        "assets: ${benchmark_root}/assets" \
        > "${LIBERO_CONFIG_PATH}/config.yaml"
fi

if ! command -v "${main_python}" >/dev/null 2>&1; then
    echo "Error: main_python is not executable from PATH: ${main_python}" >&2
    echo "Hint: activate the VLA_JEPA environment or export main_python=/path/to/python" >&2
    exit 1
fi

if ! "${sim_python}" - <<'PY' >/dev/null 2>&1
import tyro
import websockets.sync.client
import msgpack
import rich
import accelerate
PY
then
    echo "Error: the LIBERO environment is missing required packages." >&2
    echo "Install them with:" >&2
    echo "  ${sim_python} -m pip install tyro websockets msgpack rich accelerate" >&2
    exit 1
fi

gpu_count=$("${main_python}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)

if [ "${gpu_count}" -lt 1 ]; then
    echo "Error: no CUDA devices are visible to ${main_python}." >&2
    echo "Hint: verify the VLA_JEPA environment can access your GPU, or set CUDA_VISIBLE_DEVICES before running." >&2
    exit 1
fi

#items=("libero_10" "libero_goal" "libero_object" "libero_spatial")
items=("libero_goal")
if [ "${gpu_count}" -lt "${#items[@]}" ]; then
    items=("${items[@]:0:${gpu_count}}")
fi
host="127.0.0.1"
base_port=15083
unnorm_key="franka"
index=0
num_trials_per_task=50
with_state="true"

cleanup() {
    jobs -pr | xargs -r kill
}
trap cleanup EXIT

echo "Starting LIBERO evaluation for: ${items[*]}"

# start each task suite on specific GPU.
for task_suite_name in "${items[@]}"
do
    gpu_id=${index}
    port=$((base_port+index+1))

    fuser -k "${port}/tcp" >/dev/null 2>&1 || true

    "${main_python}" ./deployment/model_server/server_policy.py \
        --ckpt_path ${your_ckpt} \
        --port ${port} \
        --use_bf16 \
        --cuda ${gpu_id} &

    video_out_path="results/${task_suite_name}/${folder_name}"

    LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
    mkdir -p ${LOG_DIR}
    mkdir -p ${video_out_path}


    # export DEBUG=true

    ${sim_python} ./examples/LIBERO/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port ${port}\
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" > "${video_out_path}/eval.log" \
        --args.with_state "$with_state" &
    index=$((index+1))
done

wait
