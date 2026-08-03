# merged_dataset_001 端到端训练指南

本文记录在本仓库（`temp`）上，对 `merged_dataset_001` 做 **privileged VLA-JEPA 端到端训练**（V-JEPA 2.1）的启动方式。对应产出目录：

```text
checkpoints/merged_dataset_001_e2e_vjepa21/
```

该 run 已跑满 `max_train_steps=90000`，断点保存在 `checkpoints/merged_dataset_001_e2e_vjepa21/checkpoints/steps_*_pytorch_model.pt`。

---

## 1. 训练内容概览

| 项 | 值 |
|---|---|
| Framework | `VLA_JEPA`（privileged latent 端到端） |
| 数据集 | `dataset/merged_dataset_001`（mix: `sonic_merged_dataset_001`） |
| Action / State | 78-dim `motion_token` / 46-dim state |
| VLM | `Qwen3-VL-2B-Instruct`（本仓库根目录） |
| World model | V-JEPA 2.1 ViT-L/384（冻结 teacher encoder） |
| 启动脚本 | `scripts/vlajepa_merged_dataset_001_e2e.sh` |
| 配置 | `scripts/config/vlajepa_merged_dataset_001_e2e.yaml` |
| 训练入口 | `starVLA/training/train_starvla.py` + DeepSpeed ZeRO-2 |
| Run ID | `merged_dataset_001_e2e_vjepa21` |

除 V-JEPA teacher encoder 外，其余模块默认可训（`freeze_modules: ''`）。

---

## 2. 环境与依赖路径

```bash
cd /cpfs_infra/shared/xiaoxinyu/myjepa/temp
conda activate VLA_JEPA   # 脚本也会自动尝试激活该环境
```

启动前需保证下列路径存在（脚本会检查）：

| 依赖 | 默认路径 |
|---|---|
| 配置 | `./scripts/config/vlajepa_merged_dataset_001_e2e.yaml` |
| 数据 | `./dataset/merged_dataset_001` |
| Qwen3-VL-2B | `./Qwen3-VL-2B-Instruct` |
| V-JEPA 2.1 权重 | `/cpfs_infra/shared/xiaoxinyu/VLA-JEPA/VLA-JEPA-DEV/VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt` |

可用环境变量覆盖：

```bash
export DATA_ROOT=/path/to/dataset          # 其下需有 merged_dataset_001/
export BASE_VLM=/path/to/Qwen3-VL-2B-Instruct
export VJEPA_ENCODER=/path/to/vjepa2_1_vitl_dist_vitG_384.pt
```

---

## 3. 启动指令

在仓库根目录执行：

```bash
cd /cpfs_infra/shared/xiaoxinyu/myjepa/temp

# 默认：可见 GPU 全开，每卡 batch=32，W&B online
bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

脚本实际调用：

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_merged_dataset_001_e2e.yaml \
  --run_id merged_dataset_001_e2e_vjepa21 \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.vj2_model.base_encoder "${VJEPA_ENCODER}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.per_device_batch_size "${BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}"
```

### 常用覆盖示例

```bash
# 指定 GPU / 进程数 / batch（进程数不能超过可见 GPU 数）
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 BATCH_SIZE=8 \
  bash scripts/vlajepa_merged_dataset_001_e2e.sh

# 离线 W&B
WANDB_MODE=offline bash scripts/vlajepa_merged_dataset_001_e2e.sh

# 在线 W&B（需提前 export API key）
export WANDB_API_KEY=xxxx
WANDB_MODE=online bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

脚本默认环境变量：

| 变量 | 默认 |
|---|---|
| `NUM_PROCESSES` | 可见 GPU 数 |
| `BATCH_SIZE` | `32` |
| `NUM_WORKERS` | `8` |
| `RUN_ID` | `merged_dataset_001_e2e_vjepa21` |
| `WANDB_MODE` | `online` |
| `WANDB_ENTITY` | `xinyu-xiao-kinetix-ai` |
| `WANDB_PROJECT` | `VLA_JEPA_merged_dataset_001_e2e` |

---

## 4. 关键训练超参（YAML）

来自 `scripts/config/vlajepa_merged_dataset_001_e2e.yaml`：

- `max_train_steps`: 90000
- `num_warmup_steps`: 1000
- `save_interval`: 10000
- `per_device_batch_size`: 32
- `learning_rate`: base `3e-5` / Qwen `1e-5` / action `1e-4`
- `lr_scheduler_type`: `cosine_with_min_lr`（`min_lr=1e-6`）
- privileged loss：`lambda_act=1.0`, `lambda_wm=0.5`, `lambda_teacher_wm=0.1`, `lambda_distill=0.1`, `lambda_latent=0.1`

当前 YAML 里还配置了从 step 20k 权重继续训练的字段（用于续跑时对齐 LR schedule）：

```yaml
trainer:
  pretrained_checkpoint: .../steps_20000_pytorch_model.pt
  is_resume: false
  resume_step: 20000
```

若要从零开训，请清空或注释 `pretrained_checkpoint`，并将 `resume_step` 设为 `null`。

---

## 5. 产出位置

```text
checkpoints/merged_dataset_001_e2e_vjepa21/
├── config.yaml / config.json
├── dataset_statistics.json
├── summary.jsonl
├── checkpoints/
│   ├── steps_10000_pytorch_model.pt
│   ├── ...
│   └── steps_90000_pytorch_model.pt
├── final_model/
└── wandb/
```

已完成的本次训练在 `summary.jsonl` 中记录到 `steps: 90000`。
