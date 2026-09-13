# SpatialGoal-JEPA：SIMPLE 单机 8×A100 训练

本指南提供 SIMPLE 移动抓取任务的完整策略训练、续训和部署命令。环境安装、模型下载和数据下载见 [主 README](README.md)。

## 1. 准备环境与路径

完成主 README 的环境安装后，在仓库根目录执行。将 `/path/to/...` 替换为实际路径。

```bash
conda activate VLA_JEPA
wandb login
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export QWEN_MODEL=/path/to/models/Qwen3-VL-2B-Instruct
export VJEPA21_CKPT=/path/to/models/vjepa2_1_vitl_dist_vitG_384.pt
export DATA_ROOT=/path/to/simple_data
export OUTPUT_ROOT=/path/to/checkpoints
export RUN_ID=simple_spatial_goal_8xa100
export WANDB_MODE=online
export WANDB_PROJECT=SPATIAL_JEPA
export NUM_WORKERS=4
export OMP_NUM_THREADS=4
export FFMPEG_THREADS=1
```

`wandb login` 用于首次登录。在线曲线记录到 `SPATIAL_JEPA` 项目；团队工作区可另设置 `export WANDB_ENTITY=你的团队名称`。凭据由 W&B CLI 管理。

数据根目录结构：

```text
${DATA_ROOT}/dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/
├── data/
├── videos/
└── meta/
```

训练使用该数据集的全部 99 个 episode。数据加载器按示范读取当前图像、指令、32 维 state、30×36 动作 chunk、未来帧及过去真实观测；未来帧在训练时编码为目标监督。首次启动会计算并缓存归一化统计，数据目录需要可写。

## 2. 启动完整训练

```bash
bash scripts/train_spatial_goal.sh simple
```

入口选择 [SIMPLE 空间目标配置](scripts/config/vlajepa_simple_spatial_goal.yaml)，联合训练 Qwen、Action Expert、世界模型预测器、GRU action prior、全局与空间目标预测器、目标条件动作模块。V-JEPA 2.1 编码器保持冻结。五类主损失为动作学习、世界模型预测、action prior、目标条件动作重建、目标预测；目标预测包含全局和空间两个分量。`delta_loss` 与 `ctrl_loss` 默认权重为零，可用于消融。

| 参数 | 默认值 |
|---|---:|
| GPU 数 | 8 |
| 精度 / 优化器状态分片 | BF16 / DeepSpeed ZeRO-2 |
| 每卡 batch | 1 |
| 梯度累积步数 | 32 |
| 有效 batch | 256 |
| 优化器更新次数 | 40,000 |
| warmup 更新次数 | 2,000 |
| checkpoint 间隔 | 10,000 次更新 |
| 训练指标记录间隔 | 10 次更新 |
| 动作误差诊断间隔 | 500 次更新 |
| Qwen / Action Expert 学习率 | `1e-5` / `1e-4` |
| 其余可训练模块学习率 | `3e-5` |

有效 batch = GPU 数 × 每卡 batch × 梯度累积步数。训练步数按优化器更新计数；数据读完后继续采样，直到达到 40,000 次更新。数据采用随机采样，默认保留并补齐示范尾段。动作误差诊断计算当前训练 batch 的 MAE / MSE；任务成功率通过仿真执行单独评估。

可在保持有效 batch 为 256 的情况下调整显存与吞吐配置：

```bash
RUN_ID=simple_spatial_batch2 PER_DEVICE_BATCH_SIZE=2 \
  bash scripts/train_spatial_goal.sh simple \
    --trainer.gradient_accumulation_steps 16
```

启动新实验时使用新的 `RUN_ID`。命令末尾的参数覆盖配置文件默认值，例如 `--trainer.save_interval 5000` 或 `--seed 123`。

## 3. 日志与输出

```text
${OUTPUT_ROOT}/${RUN_ID}/
├── config.yaml
├── config.json
├── dataset_statistics.json
├── wandb_run.json                 # W&B run ID 与工作区
├── metrics.jsonl                  # loss、学习率等训练指标
├── summary.jsonl                  # checkpoint 保存步数
├── tensorboard/
├── wandb/
├── checkpoints/
│   ├── steps_10000/               # 完整训练状态，可续训
│   ├── steps_10000_pytorch_model.pt
│   └── ...
└── final_model/
    └── pytorch_model.pt
```

本地查看训练曲线和最近指标：

```bash
tensorboard --logdir "${OUTPUT_ROOT}/${RUN_ID}/tensorboard" --port 6006
```

```bash
tail -f "${OUTPUT_ROOT}/${RUN_ID}/metrics.jsonl"
```

W&B 上传实际训练配置、loss、学习率、数据读取与计算耗时，以及定期动作 MAE / MSE；曲线横轴使用 `optimizer_step`。8 个训练进程共用一个云端 run，由主进程记录。loss 为主进程最近一个 microbatch 的值，动作误差诊断在各训练进程间归约。需要离线记录时设置 `WANDB_MODE=offline`，本地 TensorBoard 与 `metrics.jsonl` 保持可用。

## 4. 从 checkpoint 续训

保持原实验的模型、数据路径、GPU 数、每卡 batch 与累积步数。以下命令读取原 run 保存的配置，并从第 10,000 次更新继续到第 40,000 次更新：

```bash
export RUN_ID=simple_spatial_goal_8xa100
CONFIG_YAML="${OUTPUT_ROOT}/${RUN_ID}/config.yaml" \
PER_DEVICE_BATCH_SIZE=1 \
  bash scripts/train_spatial_goal.sh simple \
    --trainer.gradient_accumulation_steps 32 \
    --trainer.max_train_steps 40000 \
    --trainer.resume_from_checkpoint \
    "${OUTPUT_ROOT}/${RUN_ID}/checkpoints/steps_10000"
```

`steps_10000/` 恢复模型、优化器、学习率调度、随机状态、训练步数与数据位置。独立 `.pt` 文件用于部署，或通过 `--trainer.pretrained_checkpoint /path/to/model.pt` 初始化新训练的模型权重。

在线续训自动读取 run 目录的 `wandb_run.json`，继续使用原 W&B run；`WANDB_PROJECT`、`WANDB_ENTITY` 保持与原实验一致。迁移训练时一同保留该文件。恢复较旧 checkpoint 会保留已有曲线，并记录恢复后的实际训练步数。旧版实验可设置 `WANDB_RUN_ID` 为原云端 run 的 ID；缺少身份记录时创建新的 W&B run。

## 5. 启动 SIMPLE 推理服务

```bash
CUDA_VISIBLE_DEVICES=0 python -m deployment.model_server.server_policy \
  --ckpt_path "${OUTPUT_ROOT}/${RUN_ID}/checkpoints/steps_40000_pytorch_model.pt" \
  --cuda 0 --use_bf16 --port 10093
```

保留该 run 的 `config.yaml`、`dataset_statistics.json` 与所选权重，确保配置中的预训练模型路径可访问。客户端发送当前 ego RGB、任务指令、归一化的 32 维 state、观测时间和 episode 标识。服务自动预测 subgoal latent，返回 `[30,36]` 归一化动作；客户端用该 run 的统计量与 SIMPLE 控制规则恢复动作后执行。

请求格式见 [主 README 的部署示例](README.md#部署)，动作字段和转换见 [控制接口](docs/control_interfaces.md)。
