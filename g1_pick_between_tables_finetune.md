# G1WholebodyLocomotionPickBetweenTablesTeleop-v0 微调指南

本文档说明如何在 [VLA-JEPA-DEV](https://github.com/Ju6276/VLA-JEPA-DEV/tree/vlajepa-xxy) 仓库中，使用 **LeRobot v2.1** 格式的 G1 人形机器人数据集 `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` 完成 VLA-JEPA 微调（fine-tuning）的全流程，包括环境配置、数据集放置、YAML 参数说明与启动命令。

---

## 目录

- [1. 任务与数据集概览](#1-任务与数据集概览)
- [2. 环境准备](#2-环境准备)
- [3. 预训练模型下载](#3-预训练模型下载)
- [4. 数据集准备](#4-数据集准备)
- [5. 代码注册关系（已内置，通常无需修改）](#5-代码注册关系已内置通常无需修改)
- [6. 训练配置文件详解](#6-训练配置文件详解)
- [7. 他人下载代码后的路径修改清单](#7-他人下载代码后的路径修改清单)
- [8. 启动微调](#8-启动微调)
- [9. 训练产出与断点续训](#9-训练产出与断点续训)
- [10. 从已有 VLA-JEPA checkpoint 继续微调](#10-从已有-vla-jepa-checkpoint-继续微调)
- [11. 部署推理（简要）](#11-部署推理简要)
- [12. 常见问题排查](#12-常见问题排查)
- [13. 扩展到其他 G1 数据集](#13-扩展到其他-g1-数据集)

---

## 1. 任务与数据集概览

| 项 | 说明 |
|---|---|
| 数据集名称 | `G1WholebodyLocomotionPickBetweenTablesTeleop-v0` |
| 机器人类型 | Unitree G1 全身遥操作（wholebody teleop） |
| 数据格式 | LeRobot v2.1 |
| 任务描述 | 从 table1 拿起 cracker box，行走至 table2，并放置到 table2 上 |
| Episode 数 | 99 |
| 总帧数 | 62,764 |
| 帧率 | 50 FPS |
| 视角 | 单目 egocentric 视频（`observation.images.egocentric`，原始 360×640） |
| 状态维度 | **32D** = 左手(7) + 右手(7) + 左臂(7) + 右臂(7) + rpy(3) + height(1) |
| 动作维度 | **36D** = 上述 32D + torso_vx(1) + torso_vy(1) + torso_vyaw(1) + target_yaw(1) |
| 动作类型 | `joint`（关节空间，非 Sonic latent-action 的 `motion_token`） |

本仓库已内置该数据集对应的 **data mix**、**data config** 和 **训练脚本**，无需从零编写 dataloader。

---

## 2. 环境准备

```bash
# 1. 克隆仓库
git clone https://github.com/Ju6276/VLA-JEPA-DEV.git
cd VLA-JEPA-DEV
git checkout vlajepa-xxy

# 2. 创建 conda 环境
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA

# 3. 安装依赖
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

> 训练脚本默认使用 **8 GPU + DeepSpeed ZeRO-2**。GPU 数量不足时可通过环境变量 `NUM_PROCESSES` 调整（见 [§8](#8-启动微调)）。

---

## 3. 预训练模型下载

微调需要两个 backbone，请在仓库根目录（或任意路径，后续在 YAML 中指向即可）下载：

| 模型 | 用途 | 下载方式 |
|---|---|---|
| **Qwen3-VL-2B-Instruct** | 视觉-语言 backbone | [HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |
| **V-JEPA 2 ViT-L** | 世界模型 encoder（默认） | [facebook/vjepa2-vitl-fpc64-256](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) |

```bash
# 示例：下载到仓库根目录（推荐命名与 .gitignore 一致）
cd /path/to/VLA-JEPA-DEV

# Qwen3-VL-2B（需 huggingface-cli 或 git lfs）
huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir Qwen3-VL-2B-Instruct

# V-JEPA 2
huggingface-cli download facebook/vjepa2-vitl-fpc64-256 --local-dir vjepa2-vitl-fpc64-256
```

> **可选**：若使用 V-JEPA 2.1（384px），需额外下载 `.pt` 权重并修改 YAML 中 `framework.vj2_model.base_encoder` 为 `.pt` 文件路径。本任务默认配置使用 V-JEPA 2（256px）。

---

## 4. 数据集准备

### 4.1 目录结构

将完整数据集放到仓库根目录下的 `dataset/` 子目录：

```
/path/to/VLA-JEPA-DEV/
└── dataset/
    └── G1WholebodyLocomotionPickBetweenTablesTeleop-v0/
        ├── data/
        │   └── chunk-000/
        │       └── episode_*.parquet
        ├── videos/
        │   └── chunk-000/
        │       └── egocentric/
        │           └── episode_*.mp4
        └── meta/
            ├── info.json
            ├── modality.json      # 必须存在，定义 state/action/video 切片
            ├── episodes.jsonl
            ├── tasks.jsonl
            └── stats.json           # 归一化统计量
```

### 4.2 关键 meta 文件

`meta/modality.json` 已将原始字段映射为训练所需 key，例如：

- 视频：`video.rs_view` ← `observation.images.egocentric`
- 语言：`annotation.human.task_description` ← `task_index`
- 状态/动作：从 `states` / `action` 向量按维度切片

`meta/tasks.jsonl` 中的任务文本示例：

```json
{"task_index": 0, "task": " pick up the cracker box from table1,locomotion to table2,and place  on table2."}
```

### 4.3 路径解析规则

训练配置中：

- `datasets.vla_data.data_root_dir` = 仓库根目录（如 `/path/to/VLA-JEPA-DEV`）
- `datasets.vla_data.data_mix` = `g1_pick_between_tables`

代码在 `starVLA/dataloader/gr00t_lerobot/mixtures.py` 中将 mix 解析为：

```
{data_root_dir}/dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0
```

**最终数据集绝对路径** = `data_root_dir` + `/` + mixture 中的相对路径。

---

## 5. 代码注册关系（已内置，通常无需修改）

本数据集已在以下三处注册，他人直接使用即可：

### 5.1 Data Config — `starVLA/dataloader/gr00t_lerobot/data_config.py`

`G1HandoverDataConfig` 定义了 G1 wholebody 数据的 video/state/action/language key 及归一化方式：

- 状态/手部/臂部/rpy/height → `min_max` 归一化
- torso 速度类动作 → `mean_std` 归一化

```python
"g1_pick_between_tables": G1HandoverDataConfig,
```

### 5.2 Data Mixture — `starVLA/dataloader/gr00t_lerobot/mixtures.py`

```python
"g1_pick_between_tables": [
    ("dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0", 1.0, "g1_pick_between_tables"),
],
```

三元组含义：`(数据集相对路径, 采样权重, robot_type)`。

### 5.3 Embodiment Tag — `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`

```python
"g1_pick_between_tables": EmbodimentTag.G1_PICK_BETWEEN_TABLES,  # index = 29
```

---

## 6. 训练配置文件详解

官方配置文件：

- **YAML**：`scripts/config/vlajepa_g1_pick_between_tables_ft.yaml`
- **启动脚本**：`scripts/vlajepa_g1_pick_between_tables_ft.sh`

### 6.1 全局与日志

| 参数 | 默认值 | 说明 |
|---|---|---|
| `run_id` | `g1_pick_between_tables_ft` | 实验名，决定输出目录 |
| `run_root_dir` | `checkpoints` | checkpoint 根目录 |
| `seed` | `42` | 随机种子 |
| `trackers` | `[json, wandb]` | 训练日志后端 |
| `wandb_entity` | `xinyu-xiao-kinetix-ai` | W&B 团队/用户名，**需按实际情况修改** |
| `wandb_project` | `VLA_JEPA21_SIMPLE` | W&B 项目名 |

输出目录：`checkpoints/g1_pick_between_tables_ft/`

### 6.2 模型框架 `framework`

#### Qwen-VL backbone

```yaml
framework:
  qwenvl:
    base_vlm: /path/to/VLA-JEPA-DEV/Qwen3-VL-2B-Instruct
    attn_implementation: sdpa
    vl_hidden_dim: 2048
```

#### Action Model（DiT 扩散头）

```yaml
  action_model:
    action_dim: 36          # 必须与数据集 action 维度一致
    state_dim: 32           # 必须与数据集 state 维度一致
    action_horizon: 30      # 一次预测的未来动作步数
    future_action_window_size: 29
    past_action_window_size: 0
```

> `action_horizon` 与 `framework.vj2_model.num_frames` 分别传入 dataloader 作为动作窗口和视频帧窗口长度。

#### V-JEPA 2 世界模型 encoder

```yaml
  vj2_model:
    base_encoder: /path/to/VLA-JEPA-DEV/vjepa2-vitl-fpc64-256
    num_video_views: 1
    num_frames: 8           # 输入视频帧数（video_horizon）
    num_action_tokens_per_timestep: 8
    num_embodied_action_tokens_per_instruction: 32
```

### 6.3 数据集 `datasets.vla_data`

```yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets
    data_root_dir: /path/to/VLA-JEPA-DEV
    data_mix: g1_pick_between_tables
    action_type: joint
    CoT_prompt: "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."
    resolution_size: 224
    image_size: [224, 224]
    per_device_batch_size: 32
    video_resolution_size: 256
    load_all_data_for_training: true
    with_state: true
    delete_pause_frame: false
    duplicate_single_view: false
    num_workers: 16
    prefetch_factor: 4
```

| 参数 | 说明 |
|---|---|
| `dataset_py` | 固定为 `lerobot_datasets`（LeRobot v2.1 格式） |
| `data_mix` | 引用 `mixtures.py` 中注册的 mix 名称 |
| `action_type` | G1 joint 数据使用 `joint`（Sonic latent 数据才用 `motion_token`） |
| `per_device_batch_size` | 每 GPU batch size；全局 batch = 该值 × GPU 数 × `gradient_accumulation_steps` |
| `video_resolution_size` | V-JEPA encoder 输入分辨率（256，对应 V-JEPA 2） |
| `resolution_size` | VLM 输入分辨率（224） |
| `with_state` | 是否使用 proprio state |

默认 **全局 batch size** = 32 × 8 GPU × 1 = **256**。

### 6.4 训练器 `trainer`

```yaml
trainer:
  epochs: 100
  max_train_steps: 40000
  num_warmup_steps: 2000
  save_interval: 10000       # 每 10000 step 保存一次 checkpoint
  eval_interval: 500
  pretrained_checkpoint: null  # 从 Qwen+V-JEPA 初始化；若填路径则从该 ckpt 加载
  learning_rate:
    base: 3.0e-05
    qwen_vl_interface: 1.0e-05
    action_model: 1.0e-04
  lr_scheduler_type: cosine_with_min_lr
  scheduler_specific_kwargs:
    min_lr: 1.0e-06
  freeze_modules: ''           # 留空表示不冻结；可填模块名冻结部分 backbone
  loss_scale:
    vla: 1.0
    vlm: 0.1
  gradient_accumulation_steps: 1
  enable_mixed_precision_training: true
```

---

## 7. 他人下载代码后的路径修改清单

从 GitHub 克隆后，**至少需要修改** `scripts/config/vlajepa_g1_pick_between_tables_ft.yaml` 中以下 3 处绝对路径：

```yaml
framework:
  qwenvl:
    base_vlm: /你的路径/VLA-JEPA-DEV/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: /你的路径/VLA-JEPA-DEV/vjepa2-vitl-fpc64-256

datasets:
  vla_data:
    data_root_dir: /你的路径/VLA-JEPA-DEV
```

可选修改：

| 参数 | 何时修改 |
|---|---|
| `wandb_entity` / `wandb_project` | 使用自己的 W&B 账号 |
| `per_device_batch_size` | GPU 显存不足时减小（如 16 或 8） |
| `max_train_steps` | 数据量较小时可减少 |
| `num_workers` | 按 CPU 核数调整 |
| `run_id` | 区分不同实验 |

启动脚本 `scripts/vlajepa_g1_pick_between_tables_ft.sh` 中的 conda 路径也需按本机环境修改：

```bash
CONDA_SH="/你的路径/miniconda3/etc/profile.d/conda.sh"
```

---

## 8. 启动微调

### 8.1 标准启动（8 GPU）

```bash
cd /path/to/VLA-JEPA-DEV
conda activate VLA_JEPA

# 可选：W&B 在线日志
export WANDB_MODE=online
export WANDB_API_KEY=your_wandb_api_key
# wandb login

bash scripts/vlajepa_g1_pick_between_tables_ft.sh
```

### 8.2 指定 GPU 数量

```bash
# 例如 4 卡训练
NUM_PROCESSES=4 bash scripts/vlajepa_g1_pick_between_tables_ft.sh
```

### 8.3 等价的手动启动命令

脚本内部实际执行：

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_g1_pick_between_tables_ft.yaml
```

### 8.4 通过命令行覆盖 YAML 参数（无需改文件）

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_g1_pick_between_tables_ft.yaml \
  trainer.max_train_steps=20000 \
  datasets.vla_data.per_device_batch_size=16
```

### 8.5 快速验证 dataloader（可选）

确认数据集路径与 modality 配置正确：

```bash
python -m starVLA.dataloader.lerobot_datasets \
  --config_yaml ./scripts/config/vlajepa_g1_pick_between_tables_ft.yaml
```

> 默认会开启 debugpy 监听端口 10092；仅用于本地 debug，正式训练请直接跑 `train_starvla.py`。

---

## 9. 训练产出与断点续训

### 9.1 输出目录结构

```
checkpoints/g1_pick_between_tables_ft/
├── config.yaml                  # 训练配置快照
├── config.json
├── dataset_statistics.json      # 数据集归一化统计
├── summary.jsonl                # 每步 loss 等指标
├── wandb/                       # W&B 本地缓存
├── tensorboard/
└── checkpoints/
    ├── steps_10000_pytorch_model.pt
    ├── steps_20000_pytorch_model.pt
    ├── steps_30000_pytorch_model.pt
    └── steps_40000_pytorch_model.pt
```

### 9.2 断点续训

在 YAML 中设置：

```yaml
trainer:
  pretrained_checkpoint: checkpoints/g1_pick_between_tables_ft/checkpoints/steps_10000_pytorch_model.pt
  is_resume: true
  resume_from_checkpoint: checkpoints/g1_pick_between_tables_ft/checkpoints/steps_10000
```

> `pretrained_checkpoint` 指向 `.pt` 权重文件；`resume_from_checkpoint` 指向 DeepSpeed 状态目录（不含 `_pytorch_model.pt` 后缀），用于恢复 optimizer / scheduler 状态。

---

## 10. 从已有 VLA-JEPA checkpoint 继续微调

默认 `pretrained_checkpoint: null` 表示：**Qwen3-VL 和 V-JEPA 从各自预训练权重初始化，action head 随机初始化**，在本 G1 数据集上从头微调。

若已有同维度（action_dim=36, state_dim=32）的 G1 checkpoint，可设置：

```yaml
trainer:
  pretrained_checkpoint: /path/to/steps_XXXXX_pytorch_model.pt
```

若要从 **不同 action 维度** 的 checkpoint（如 LIBERO 7-dim）迁移，不能直接全量加载，需使用部分加载：

```yaml
trainer:
  pretrained_checkpoint: /path/to/libero_ckpt/steps_100000_pytorch_model.pt
  reload_modules: "qwenvl_interface,vj2_model"   # 仅加载 VLM 和世界模型，跳过 action head
```

`reload_modules` 为逗号分隔的模块路径，对应 `model` 下的子模块名。

---

## 11. 部署推理（简要）

训练完成后，可通过 WebSocket policy server 部署：

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path checkpoints/g1_pick_between_tables_ft/checkpoints/steps_40000_pytorch_model.pt \
  --port 10093 \
  --cuda 0
```

冒烟测试：

```bash
python deployment/model_server/debug_server_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --test infer
```

> 部署时会从 checkpoint 同目录的上级 `config.yaml` 重建模型，因此 `base_vlm` 和 `base_encoder` 在部署机器上也必须可访问。

---

## 12. 常见问题排查

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| `KeyError: g1_pick_between_tables` | 代码版本过旧 | 确保使用 `vlajepa-xxy` 分支 |
| 找不到 parquet / mp4 | 数据集路径不对 | 检查 `data_root_dir` + mixture 拼接后的绝对路径 |
| `action_dim` 不匹配 | YAML 与 modality 维度不一致 | 保持 `action_dim: 36`, `state_dim: 32` |
| CUDA OOM | batch size 过大 | 降低 `per_device_batch_size` 或 `NUM_PROCESSES` |
| W&B 报错 | 未设置 API Key | `export WANDB_MODE=offline` 或配置 `WANDB_API_KEY` |
| NCCL 超时 | 多机/网络配置 | 启动脚本已设 `NCCL_IB_DISABLE=1`；检查 `NCCL_SOCKET_IFNAME` |
| 视频解码慢 | ffmpeg/decord 线程过多 | 脚本已设 `FFMPEG_THREADS=1`, `OMP_NUM_THREADS=1` |

---

## 13. 扩展到其他 G1 数据集

本仓库中同系列的 G1 wholebody 数据集（共享 `G1HandoverDataConfig`，36D action / 32D state）：

| data_mix | 数据集目录 | 配置文件 | 启动脚本 |
|---|---|---|---|
| `g1_handover` | `dataset/G1WholebodyHandoverTeleop-v0` | `vlajepa_g1_handover_ft.yaml` | `vlajepa_g1_handover_ft.sh` |
| `g1_pick_between_tables` | `dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0` | `vlajepa_g1_pick_between_tables_ft.yaml` | `vlajepa_g1_pick_between_tables_ft.sh` |
| `g1_pick_hug_container` | `dataset/G1WholebodyPickAndPlaceAndHugContainerTeleop-v0` | `vlajepa_g1_pick_hug_container_ft.yaml` | `vlajepa_g1_pick_hug_container_ft.sh` |
| `g1_open_oven` | `dataset/G1WholebodyOpenOvenTeleop-v0` | `vlajepa_g1_open_oven_ft.yaml` | `vlajepa_g1_open_oven_ft.sh` |

若需接入**全新的自定义 G1 数据集**（LeRobot v2.1 格式），参考 [README.md § Custom Dataset Training](./README.md#custom-dataset-training)：

1. 编写/复用 `data_config.py` 中的 DataConfig 类（确保 key 与 `modality.json` 一致）；
2. 在 `ROBOT_TYPE_CONFIG_MAP` 注册 robot_type；
3. 在 `mixtures.py` 添加 data_mix；
4. 在 `embodiment_tags.py` 添加 EmbodimentTag；
5. 复制本任务的 YAML / shell，修改 `data_mix` 和 `run_id`。

---

## 快速参考（一键流程）

```bash
# 0. 环境
conda activate VLA_JEPA

# 1. 确认数据集已放置
ls dataset/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/meta/modality.json

# 2. 修改 YAML 中的 base_vlm / base_encoder / data_root_dir

# 3. 启动
bash scripts/vlajepa_g1_pick_between_tables_ft.sh

# 4. 产出
ls checkpoints/g1_pick_between_tables_ft/checkpoints/
```
