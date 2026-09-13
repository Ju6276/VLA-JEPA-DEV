# 8×A100 训练指南

[主 README](README.md)

## 1. 环境与数据

按主 README 安装环境，准备 Qwen3-VL-2B-Instruct、V-JEPA 2.1 ViT-L 384px 和对应的 LeRobot 数据集。

```bash
conda activate VLA_JEPA
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export VJEPA21_CKPT=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
export QWEN_MODEL=Qwen/Qwen3-VL-2B-Instruct
export OUTPUT_ROOT=/path/to/checkpoints
export WANDB_MODE=offline
```

在线 W&B 记录可通过 `wandb login` 配置凭据，再设置 `WANDB_MODE=online`。

## 2. 选择控制接口

| 参数 | SIMPLE | SONIC |
|---|---|---|
| state / action | 32 / 36 | 46 / 78 |
| action horizon | 30 | 40 |
| 每卡 batch | 32 | 4 |
| 全局 batch，梯度累积为 1 | 256 | 32 |
| 目标偏移 | t+30 | t+40 |

SIMPLE 的 `DATA_ROOT` 下包含 `dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/`：

```bash
export DATA_ROOT=/path/to/sim_data_root
bash scripts/train_g1_delta_jepa_8xa100.sh
```

SONIC 的 `DATA_ROOT` 下包含 `merged_dataset_001/`：

```bash
export DATA_ROOT=/path/to/sonic_data_root
bash scripts/train_sonic_learned_goal.sh
```

两个启动脚本共用 BF16、DeepSpeed ZeRO-2 和自动目标预测训练流程，各自选择对应的模型维度、数据配置及输出 run_id。

## 3. 调整训练参数

```bash
PER_DEVICE_BATCH_SIZE=2 bash scripts/train_sonic_learned_goal.sh \
  --trainer.max_train_steps 40000 \
  --trainer.save_interval 5000
```

启动器支持 `NUM_PROCESSES`、`PER_DEVICE_BATCH_SIZE`、`NUM_WORKERS`、`OUTPUT_ROOT`、`RUN_ID` 和 `QWEN_MODEL`。命令末尾的配置覆盖参数直接传给训练入口。

## 4. 输出与日志

默认输出目录：

```text
checkpoints/
├── g1_pick_between_tables_delta_jepa_8xa100/
└── sonic_latent_learned_goal/
```

每个 run 保存配置、`dataset_statistics.json`、`summary.jsonl` 和 `checkpoints/`。日志包含：

```text
action_loss
wm_loss
delta_loss
ctrl_loss
action_prior_loss
goal_proposal_loss
goal_prediction_loss
```

查看 SONIC 日志：

```bash
tail -f "${OUTPUT_ROOT}/sonic_latent_learned_goal/summary.jsonl"
```

## 5. 启动推理服务

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path "${OUTPUT_ROOT}/sonic_latent_learned_goal/checkpoints/steps_40000_pytorch_model.pt" \
  --cuda 0 --use_bf16 --port 10093
```

SIMPLE 使用对应 run 目录中的 checkpoint。模型配置默认开启自动目标和 verifier，每次请求输入当前 ego 图像、指令与归一化 state。客户端按该 run 的数据统计恢复动作，并通过对应控制器执行，详见 [部署文档](deployment/model_server/README.md)。
