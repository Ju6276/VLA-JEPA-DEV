# G1 Pick Between Tables：8×A100 训练步骤

## 1. 进入仓库并激活环境

```bash
cd /path/to/VLA-JEPA-DEV
conda activate VLA_JEPA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

## 2. 准备文件

需要以下三个路径：

```text
/path/to/data_root/
└── dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/

/path/to/vjepa2_1_vitl_dist_vitG_384.pt

/path/to/subgoals/
```

`/path/to/subgoals/` 必须是 `--method jepa_change` 生成的 subgoal 目录或 pkl。

如果尚未抽取 subgoal：

```bash
export LEROBOT_DATASET="${DATA_ROOT}/dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
export SUBGOALS_PATH=/path/to/subgoals

python -m starVLA.tools.extract_subgoals \
  --lerobot_dataset "${LEROBOT_DATASET}" \
  --episode_index 0 \
  --output_dir "${SUBGOALS_PATH}" \
  --encoder_path "${VJEPA21_CKPT}" \
  --method jepa_change \
  --num_subgoals 5 \
  --image_key observation.images.rs_view \
  --frame_stride 5 \
  --device cuda
```

## 3. 设置训练路径

```bash
export DATA_ROOT=/path/to/data_root
export VJEPA21_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export SUBGOALS_PATH=/path/to/subgoals
export QWEN_MODEL=Qwen/Qwen3-VL-2B-Instruct
export OUTPUT_ROOT=/path/to/checkpoints
```

默认使用离线 W&B。需要在线记录时执行：

```bash
export WANDB_MODE=online
export WANDB_API_KEY=你的密钥
```

## 4. 启动 8 卡训练

```bash
bash scripts/train_g1_delta_jepa_8xa100.sh
```

默认配置：

```text
GPU：8×A100
精度：BF16
并行：DeepSpeed ZeRO-2
单卡 batch size：32
梯度累积：1
全局 batch size：256
V-JEPA：V-JEPA 2.1 ViT-L 384px
训练 loss：action + wm + delta + ctrl
Delta-JEPA：开启
subgoal：开启
verifier：默认开启
候选动作数：8
```

显存不足时：

```bash
PER_DEVICE_BATCH_SIZE=2 bash scripts/train_g1_delta_jepa_8xa100.sh
```

修改训练步数或保存间隔：

```bash
bash scripts/train_g1_delta_jepa_8xa100.sh \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 5000
```

## 5. 检查训练输出

```bash
ls "${OUTPUT_ROOT}/g1_pick_between_tables_delta_jepa_8xa100"
tail -f "${OUTPUT_ROOT}/g1_pick_between_tables_delta_jepa_8xa100/summary.jsonl"
```

日志中每个训练 step 应包含：

```text
action_loss
wm_loss
delta_loss
ctrl_loss
```

## 6. 启动默认 verifier 推理

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path "${OUTPUT_ROOT}/g1_pick_between_tables_delta_jepa_8xa100/checkpoints/steps_40000_pytorch_model.pt" \
  --cuda 0 \
  --use_bf16 \
  --port 10093
```

checkpoint 保存的配置已经包含：

```text
delta_jepa.enabled=true
delta_jepa.use_verifier=true
delta_jepa.subgoals_path=<训练时的 SUBGOALS_PATH>
```

因此不需要额外添加 `--use_verifier` 或 `--subgoals_path`。
