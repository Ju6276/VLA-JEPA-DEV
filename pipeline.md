# SonicStar (Unitree G1) Latent-Action 训练 Pipeline

本文件记录本次为接入 **SonicStar / Unitree G1 人形机器人 latent-action 数据集**（`merged_dataset_001`）所做的全部改动，以及完整的训练流程，方便复现与回溯。

---

## 1. 总览

本次目标：在已有的 `VLA_JEPA` 框架上，新增对 Unitree G1 人形机器人数据集的支持，并跑通训练（含 W&B 监控）。

涉及的改动分两类：

- **已跟踪文件的修改**（`git diff`）：4 个文件
- **新增的未跟踪文件**：训练脚本 + 训练配置

| 文件 | 类型 | 作用 |
| --- | --- | --- |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | 修改 | 新增 `SonicLatentDataConfig` 数据配置 |
| `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py` | 修改 | 注册新 embodiment tag |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 修改 | 注册数据集 mixture |
| `starVLA/training/train_starvla.py` | 修改 | 重新启用 W&B 记录 |
| `scripts/config/vlajepa_sonic_latent.yaml` | 新增 | 训练超参 / 模型 / 数据配置 |
| `scripts/vlajepa_sonic_latent.sh` | 新增 | 启动脚本（accelerate + 环境变量） |

---

## 2. 数据集说明（`merged_dataset_001`）

LeRobot v2.1 格式，Unitree G1 人形机器人遥操作数据。

- 总 episode：303，总帧数：234,973，单任务，`fps = 50`
- 视频：`observation.images.ego_view`（480×640，h264，第一视角）

### 输入 / 输出 modality

根据 `meta/modality.json`，本次训练选用如下字段：

**State（输入，46 维）**

| key | 维度 |
| --- | --- |
| `state.left_leg` | 6 |
| `state.right_leg` | 6 |
| `state.waist` | 3 |
| `state.left_arm` | 7 |
| `state.left_hand` | 7 |
| `state.right_arm` | 7 |
| `state.right_hand` | 7 |
| `state.projected_gravity` | 3 |
| **合计** | **46** |

**Action（输出，78 维）**

| key | 维度 | 来源 |
| --- | --- | --- |
| `action.motion_token` | 64 | latent motion token |
| `action.left_hand_joints` | 7 | `teleop.left_hand_joints` |
| `action.right_hand_joints` | 7 | `teleop.right_hand_joints` |
| **合计** | **78** | |

**Video**：`video.ego_view`（第一视角，交给 world model / V-JEPA2 处理）
**Language**：`annotation.human.task_description`

> 关键点：action 不是常规的关节角，而是 **latent motion token（64 维）+ 双手关节（14 维）**，这也是 "latent" 命名的由来。

---

## 3. 代码改动详情

### 3.1 `data_config.py` —— 新增 `SonicLatentDataConfig`

新增一个 `@dataclass`，定义 G1 数据集的 modality 与 transform：

- `video_keys` / `state_keys` / `action_keys` / `language_keys`：按上表配置。
- `__init__(observation_indices, action_indices)`：
  - `observation_indices` 用于 video（对齐 world model 的 `num_frames`）；
  - `action_indices` 用于 action chunk；
  - `state_indices = [0]`，即只取当前帧 state。
- `modality_config()`：返回 video / state / action / language 四个 `ModalityConfig`。
- `transform()`：对所有 state、action 字段做 `StateActionToTensor` + `StateActionTransform`，归一化方式统一为 `min_max`。

并在 `ROBOT_TYPE_CONFIG_MAP` 中注册：

```python
"sonic_latent_humanoid": SonicLatentDataConfig,
```

### 3.2 `embodiment_tags.py` —— 注册 embodiment tag

由于 G1 不在已有 embodiment 列表中，复用 `NEW_EMBODIMENT`：

```python
"sonic_latent_humanoid": EmbodimentTag.NEW_EMBODIMENT,
```

### 3.3 `mixtures.py` —— 注册数据 mixture

```python
"sonic_merged_dataset_001": [
    ("merged_dataset_001", 1.0, "sonic_latent_humanoid"),
],
```

将数据集名、采样权重、robot_type 三元组绑定。配置文件中通过 `data_mix: sonic_merged_dataset_001` 引用。

### 3.4 `train_starvla.py` —— 重新启用 W&B

把原先注释掉的 W&B 三处逻辑恢复：

- `self._init_wandb()`（初始化）
- `wandb.log(metrics, step=self.completed_steps)`（每步记录指标）
- 训练结束时 `wandb.finish()`（仅主进程）

---

## 4. 训练配置（`scripts/config/vlajepa_sonic_latent.yaml`）

### 4.1 框架 `VLA_JEPA` 三大模块

- **qwenvl**：`Qwen3-VL-2B-Instruct`，`flash_attention_2`，`vl_hidden_dim = 2048`
- **action_model**：`DiT-B`，flow-matching diffusion head
  - `action_dim = 78`、`state_dim = 46`（对齐本数据集）
  - `action_horizon = 40`、`future_action_window_size = 39`、`past_action_window_size = 0`
  - `num_inference_timesteps = 4`，`repeated_diffusion_steps = 8`
- **vj2_model**：`vjepa2-vitl-fpc64-256`（V-JEPA2 world model）
  - `num_frames = 8`、`num_action_tokens_per_timestep = 8`
  - `num_embodied_action_tokens_per_instruction = 32`

> 相对 `vlajepa_robot_ft.yaml`（LIBERO Franka，`action_dim=7 / state_dim=8 / action_horizon=7`）的主要差异，就是把动作/状态维度和 horizon 调整到了 G1 latent-action 的尺寸，并切换了模型与数据路径。

### 4.2 数据

```yaml
data_root_dir: /mnt/workspace/VLA-JEPA/data/merged_dataset_001
data_mix: sonic_merged_dataset_001
resolution_size: 224          # VLA 图像输入
video_resolution_size: 256    # world model 视频输入
per_device_batch_size: 4
with_state: true
```

### 4.3 训练超参

- `max_train_steps = 90000`，`num_warmup_steps = 100`
- `save_interval = 30000`，`eval_interval = 100`
- 分层学习率：`base 3e-5` / `qwen_vl_interface 1e-5` / `action_model 1e-4`
- 调度器：`cosine_with_min_lr`（`min_lr = 1e-6`）
- 优化器：`AdamW`，`betas=[0.9, 0.95]`
- loss 权重：`vla 1.0` / `vlm 0.1`
- 开启梯度检查点与混合精度训练

---

## 5. 启动脚本（`scripts/vlajepa_sonic_latent.sh`）

主要内容：

- **NCCL / 通信环境变量**：禁用 IB、指定 `eth0`、阻塞式 wait、超时 1 小时，保证多卡通信稳定与 checkpoint 落盘安全。
- **W&B 环境变量**：`WANDB_MODE=online`，`WANDB_PROJECT=vlajepa_sonic_latent`，并在 online 模式下强制校验 `WANDB_API_KEY` 是否已设置。
- **启动命令**：

```bash
accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_PROCESSES:-8}" \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_sonic_latent.yaml
```

即：8 卡 + DeepSpeed ZeRO-2。

---

## 6. 完整训练流程

```
merged_dataset_001 (LeRobot v2.1)
        │
        ▼
data_mix: sonic_merged_dataset_001  ──►  robot_type: sonic_latent_humanoid
        │                                        │
        ▼                                        ▼
SonicLatentDataConfig                    EmbodimentTag.NEW_EMBODIMENT
  · modality_config()  (video/state/action/language)
  · transform()        (min_max 归一化)
        │
        ▼
   DataLoader  ──►  VLA_JEPA framework
                      ├── Qwen3-VL-2B   (VLM)
                      ├── V-JEPA2 ViT-L (world model, 8 frames ego_view)
                      └── DiT-B flow-matching head (78-dim action)
        │
        ▼
  accelerate + DeepSpeed ZeRO-2 (8 GPU)
        │
        ▼
  checkpoints/sonic_latent/  +  W&B (project: vlajepa_sonic_latent)
```

### 运行命令

```bash
export WANDB_API_KEY="<your_key>"
bash scripts/vlajepa_sonic_latent.sh
```

---

## 7. 与基线（LIBERO `robot_ft`）的对比小结

| 项 | `vlajepa_robot_ft`（基线） | `vlajepa_sonic_latent`（本次） |
| --- | --- | --- |
| 数据集 | LIBERO Franka | Unitree G1 `merged_dataset_001` |
| robot_type | `libero_franka` | `sonic_latent_humanoid`（新增） |
| action_dim | 7 | 78（latent token + 双手） |
| state_dim | 8 | 46 |
| action_horizon | 7 | 40 |
| per_device_batch_size | 32 | 4 |
| max_train_steps | 30000 | 90000 |
| W&B | 关闭（仅 json） | 开启（json + wandb） |

---

## 8. 变更记录（Changelog）

### 2026-06-23 —— 切换 world model 视觉编码器

**改动**：把 `vlajepa_sonic_latent.yaml` 的 `vj2_model.base_encoder` 从
`vjepa2-vitl-fpc64-256` 换成 `vjepa2-vitl-fpc16-256-ssv2`。

```yaml
# scripts/config/vlajepa_sonic_latent.yaml
vj2_model:
  base_encoder: /dev/shm/models/vjepa2-vitl-fpc16-256-ssv2
```

**背景 / 决策过程**：

- 原计划升级到 **V-JEPA 2.1 ViT-L/16（300M, 384）**，但遇到两个硬阻塞：
  1. 本机无外网（默认 HF / GitHub 均不可达），V-JEPA 2.1 权重无法下载（本地也没有）；
  2. 当前 `transformers 5.8.1` 的 `VJEPA2Config` 不含 2.1 架构字段（HF 对 2.1 的支持仍在 PR `huggingface/transformers#45497`，且 Meta 尚未把 2.1 权重传到 HF Hub）。
- 因此改用现成可下载的 `facebook/vjepa2-vitl-fpc16-256-ssv2`（V-JEPA2 在 Something-Something-V2 上的分类微调版）。

**下载方式**：默认 HF 端点不可达，改用镜像 `HF_ENDPOINT=https://hf-mirror.com` 下载到
`/mnt/workspace/models/vjepa2-vitl-fpc16-256-ssv2`（与原编码器同在 `models/` 下）。

**磁盘清理**：下载时 `/mnt/workspace`（30G）已 100% 满。删除冗余文件
`models/vjepa2-vitl-fpc64-256/original/model.pth`（4.8G，HF `AutoModel` 加载用不到，
随时可重下）腾出空间后下载完成。

**兼容性验证**：

- `AutoModel.from_pretrained(...)` 加载得到 `VJEPA2Model`（纯 backbone），分类头
  （`pooler.*` / `classifier.weight`）作为 UNEXPECTED 被自动忽略。
- `get_vision_features(...)` 正常，输出特征 `(1, 2048, 1024)`。
- backbone 维度与原编码器一致（`image_size=256`、`tubelet_size=2`、`hidden_size=1024`、
  `num_hidden_layers=24`），框架其余部分无需改动；分辨率仍为 256。

**注意事项 / 风险**：

- 该模型是**分类任务微调过的 backbone**（非纯 SSL 编码器，且 `frames_per_clip=16`），
  特征已偏向 SSv2 动作识别，作为通用 world model 编码器可能不如原始 SSL 版本通用；
  效果需通过训练/评测对比确认。
- 若后续要用真正的 V-JEPA 2.1，需要：拿到 2.1 权重 + 升级 transformers（带 #45497）
  或改用 Meta `facebookresearch/vjepa2` 的 torch.hub 加载路径，并把分辨率调到 384。

### 2026-06-23 —— 模型权重迁移到内存盘（tmpfs）

**起因**：`/mnt/workspace`（30G 磁盘）已满；而内存极充裕（1.6Ti）。

**改动**：把整个模型目录从磁盘移到内存盘 `/dev/shm`（tmpfs，RAM 支撑，1.6T 可用），
并把配置里的路径直接改掉（不使用软链接）。

```bash
mv /mnt/workspace/models /dev/shm/models
```

`scripts/config/vlajepa_sonic_latent.yaml` 路径更新：

```yaml
qwenvl:
  base_vlm: /dev/shm/models/Qwen3-VL-2B-Instruct
vj2_model:
  base_encoder: /dev/shm/models/vjepa2-vitl-fpc16-256-ssv2
```

**效果**：磁盘占用从 100% 降到约 66%（释放 ~11G，可用 ~9.8G）；模型读取走内存，更快。

**重要提醒（易失性）**：`/dev/shm` 是 tmpfs，**容器/Pod 重启后内容会丢失**，需要重新
下载模型（用镜像 `HF_ENDPOINT=https://hf-mirror.com`）。因此：

- 模型权重放 `/dev/shm`（可重下，OK）；
- 训练 checkpoints 仍写到真磁盘 `checkpoints/`（需长期保留，**不要**放 tmpfs）。
- 注：`checkpoints/sonic_latent/config.{json,yaml}` 是历史运行快照，仍保留旧的
  `/mnt/workspace/models/...` 路径，未改动（作为记录）。

### 2026-06-23 —— 为 V-JEPA 2.1 新建独立 config（不动现有 config）

**改动**：新增两个文件，当前在用的 `vlajepa_sonic_latent.yaml`（ssv2 编码器）保持不变。

- `scripts/config/vlajepa_sonic_latent_vjepa21.yaml`
- `scripts/vlajepa_sonic_latent_vjepa21.sh`

**与现有 config 的差异**：

| 项 | `vlajepa_sonic_latent`（在用） | `vlajepa_sonic_latent_vjepa21`（新建，待权重） |
| --- | --- | --- |
| `run_id` | `sonic_latent` | `sonic_latent_vjepa21` |
| `wandb_project` | `vlajepa_sonic_latent` | `vlajepa_sonic_latent_vjepa21` |
| `base_encoder` | `/dev/shm/models/vjepa2-vitl-fpc16-256-ssv2` | `/dev/shm/models/vjepa2.1-vitl-384`（占位，待放权重） |
| `video_resolution_size` | 256 | 384 |

其余（action_model / 数据集 / trainer 超参）完全一致；backbone 维度不变
（hidden 1024 / tubelet 2 / patch 16），动作模型与预测器无需改。

**运行前置条件（尚未满足，已写在该 yaml 头部注释里）**：

1. **权重**：下载 V-JEPA 2.1 ViT-L/384 放到 `base_encoder` 路径。当前 Meta 经 PyTorch
   Hub 发布（`torch.hub.load('facebookresearch/vjepa2','vjepa2_1_vit_large_384')`），
   HF Hub 上传仍 pending（`facebookresearch/vjepa2#137`）。
2. **加载支持**：现装的 `transformers 5.8.1` 不支持 2.1 架构；需升级到含
   `huggingface/transformers#45497` 的版本，或改 `VLA_JEPA.py` 走 torch.hub 加载。
3. **分辨率**：2.1 ViT-L 终版是 384px，已在 yaml 设 `video_resolution_size: 384`。

> 即：config 已就绪，等"权重 + 加载支持"两件事齐了即可 `bash scripts/vlajepa_sonic_latent_vjepa21.sh` 启动。

### 2026-06-23 —— 删除 ssv2、改用真正的 V-JEPA 2.1 权重

**纠正**：之前下载的 `vjepa2-vitl-fpc16-256-ssv2` 其实是 **V-JEPA 2.0** 在 SSv2 上的分类
微调版，**不是 2.1**。已删除它，改下载真正的 2.1。

**操作**：

- 删除 `/dev/shm/models/vjepa2-vitl-fpc16-256-ssv2`。
- 把在用的 `vlajepa_sonic_latent.yaml` 的 `base_encoder` 退回原版
  `/dev/shm/models/vjepa2-vitl-fpc64-256`（V-JEPA 2.0 纯 SSL，`model.safetensors` 仍在，
  可正常加载），不再指向已删的 ssv2。
- 下载真正的 2.1 ViT-L/384 原始权重（`dl.fbaipublicfiles.com` 可达）：

```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -O /dev/shm/models/vjepa2_1_vitl_dist_vitG_384.pt   # ~4.8GB
```

**网络可达性（已测）**：`dl.fbaipublicfiles.com`（权重）✅、`raw.githubusercontent.com`
（取代码）✅、`hf-mirror.com` ✅；`github.com` / `huggingface.co` ❌（DNS 不通）。
→ 因此 `torch.hub.load(...)`（走 GitHub）不可用，改用 CDN 直链下原始 `.pt`。

**待办（让框架真正加载 2.1）**：该 `.pt` 是 Meta 原始格式，且 `transformers 5.8.1` 不支持
2.1 架构。两条路：
- A：转 HF 格式 + 升级 transformers（含 `huggingface/transformers#45497`）；
- B（更稳，倾向）：用 Meta 建模代码（经 `raw.githubusercontent.com` 或 pypi）+ 在
  `VLA_JEPA.py` 写适配器，暴露 `.config` 与 `get_vision_features`，对齐现有接口。
  随后把 `vlajepa_sonic_latent_vjepa21.yaml` 的 `base_encoder` 指到该 `.pt`。

### 2026-06-23 —— V-JEPA 2.1 集成完成并跑通训练（路线 B）

**结果**：`sonic_latent_vjepa21` 训练已用真正的 V-JEPA 2.1 ViT-L/384 成功启动并稳定迭代
（step 50/90000，`action_loss≈1.10`、`wm_loss≈0.12`，~1.7s/it）。

**1) 下载真 2.1 权重**

```bash
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt \
  -O /dev/shm/models/vjepa2_1_vitl_dist_vitG_384.pt   # 5,151,198,524 bytes
```

**2) Vendor Meta 官方 2.1 encoder 代码**（经 `raw.githubusercontent.com`，因 `github.com` 不通）
到 `starVLA/model/modules/world_model/vjepa21_vendor/`：

```
app/vjepa_2_1/models/vision_transformer.py   # vit_large 等
app/vjepa_2_1/models/utils/modules.py        # Block (RoPE/attn)
app/vjepa_2_1/models/utils/patch_embed.py
src/masks/utils.py
src/utils/tensors.py
（各级 __init__.py）
```

依赖仅 `torch / timm / einops`（均已具备）。

**3) 适配器**：新增 `starVLA/model/modules/world_model/vjepa21_encoder.py`
- `load_vjepa21_encoder(ckpt, img_size=384)`：按 `src/hub/backbones.py::vjepa2_1_vit_large_384`
  的配方建 encoder（`use_rope / interpolate_rope / img_temporal_dim_size=1`），加载 `.pt` 的
  `ema_encoder` 键（`pos_embed` 因走 RoPE 而缺失，属正常）。
- `VJEPA21VisionEncoder`：暴露 `.config.{tubelet_size,image_size,hidden_size}` 与
  `get_vision_features(pixel_values_videos=[B,T,C,H,W]) -> [B,N,1024]`，与 HF `VJEPA2Model` 同接口。
- `build_vjepa21_processor(384)`：ImageNet 归一化、384×384 的 `VJEPA2VideoProcessor`。
- 验证：304.7M 参数（=ViT-L/384），8 帧@384 输出 `[B, 2304, 1024]`。

**4) 框架接入**：`starVLA/model/framework/VLA_JEPA.py` 的 `__init__` 加分支——
`base_encoder` 以 `.pt` 结尾时走 2.1 适配器，否则保持原 `AutoModel` 路径（向后兼容）。

**5) 配置**：`vlajepa_sonic_latent_vjepa21.yaml` 的 `base_encoder` 指向该 `.pt`，并设
`arch: vit_large`、`image_size: 384`、`video_resolution_size: 384`。

**关键踩坑记录**：

- **真正的运行环境是 `.venv`（Python 3.10）**，不是系统 Python 3.12。`.venv` 里依赖齐全
  （pytorch3d 0.7.6 / numpydantic / albumentations / **transformers 4.57.0** / deepspeed…）。
  → 启动训练必须用 `.venv`，例如 `.venv/bin/accelerate launch ...`。
  （注：`.venv` 的 transformers 是 4.57.0，同样不支持 2.1，故路线 B 的适配器是必需的。）
- **wandb**：当前 shell 无 `WANDB_API_KEY`，而 `.sh` 脚本强制 `WANDB_MODE=online` 会退出。
  本次用 `WANDB_MODE=offline` 启动（本地记录）。要 online，请 `export WANDB_API_KEY=...`
  后用脚本启动。

**启动命令（本次实际使用）**：

```bash
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0 NCCL_BLOCKING_WAIT=1 \
       NCCL_ASYNC_ERROR_HANDLING=1 NCCL_TIMEOUT=1000 FFMPEG_THREADS=1 \
       OMP_NUM_THREADS=1 WANDB_MODE=offline
.venv/bin/accelerate launch \
  --config_file ./starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  ./starVLA/training/train_starvla.py \
  --config_yaml ./scripts/config/vlajepa_sonic_latent_vjepa21.yaml
```
