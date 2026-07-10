

#cd /mnt/petrelfs/yejinhui/Projects/starVLA
#export PYTHONPATH=$(pwd):${PYTHONPATH}

port=6680
gpu_id=0
# export DEBUG=true
#export star_vla_python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/starVLA/bin/python

your_ckpt=/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt
base_vlm_path=/home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct
base_encoder_path=/home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} python deployment/model_server/server_policy_simple_g1.py \
    --ckpt_path ${your_ckpt} \
    --base_vlm_path ${base_vlm_path} \
    --base_encoder_path ${base_encoder_path} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
