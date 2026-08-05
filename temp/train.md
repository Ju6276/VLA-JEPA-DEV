# merged_dataset_001 一阶段训练与 Smoke Test

本文对应 `temp/` 中的 One-Stage Action-Grounded Privileged VLA-JEPA。训练配置为：

- 冻结 V-JEPA 2.1 target encoder；
- 可训练 privileged TeacherEncoder、deployable Student、shared world decoder、Delta action decoder 和 Flow Action Head；
- 同一个 `forward()` 联合计算 teacher/student/world/action losses；
- 部署时只保留 Student 与 Action Head。

详细结构见 `privileged_e2e_scheme.md`，简版故事见 `scheme_brief.md`。

## 1. 依赖路径

训练脚本接受以下环境变量：

| 变量 | 含义 |
|---|---|
| `DATA_ROOT` | 数据集根目录，其下应有 `merged_dataset_001/` |
| `BASE_VLM` | 本地 Qwen3-VL-2B-Instruct 目录 |
| `VJEPA_ENCODER` | V-JEPA 2.1 `.pt` 权重 |
| `CONDA_ENV_NAME` | Conda 环境名，默认 `VLA_JEPA` |
| `BATCH_SIZE` | 每卡 batch size |
| `NUM_WORKERS` | 每进程 dataloader workers |
| `NUM_PROCESSES` | Accelerate 进程数 |
| `WANDB_MODE` | `online`、`offline` 或 `disabled` |

配置入口：`scripts/config/vlajepa_merged_dataset_001_e2e.yaml`。

## 2. 正式训练

```bash
cd temp

DATA_ROOT=/path/to/Datasets \
BASE_VLM=/path/to/Qwen3-VL-2B-Instruct \
VJEPA_ENCODER=/path/to/vjepa2_1_vitl_dist_vitG_384.pt \
NUM_PROCESSES=1 BATCH_SIZE=1 NUM_WORKERS=0 WANDB_MODE=offline \
bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

默认配置现在从零开始：`pretrained_checkpoint: null`、`resume_step: null`。如需加载旧权重，请同时确认旧 checkpoint 与当前新增模块的兼容性；loader 会报告 missing/unexpected keys。

## 3. 单卡真实数据 Smoke Test

Smoke 脚本读取一个真实 mini-batch，加载 Qwen 与 V-JEPA，执行完整前向和反向，并检查主要模块的 loss 与梯度是否为有限非零值。它不会保存 checkpoint。

```bash
cd temp

PYTHONPATH=. python scripts/smoke_test_action_grounded.py \
  --data-root /path/to/Datasets \
  --base-vlm /path/to/Qwen3-VL-2B-Instruct \
  --vjepa-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --batch-size 1
```

显存紧张时可先做轻量排错：

```bash
PYTHONPATH=. python scripts/smoke_test_action_grounded.py \
  --data-root /path/to/Datasets \
  --base-vlm /path/to/Qwen3-VL-2B-Instruct \
  --vjepa-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --batch-size 1 --action-layers 2 --freeze-vlm
```

`--action-layers` 只在内存中的 smoke 配置上临时覆盖 Action Head 深度，不修改 YAML。

## 4. 核心单元测试

不加载大模型权重即可验证 Teacher 未来条件、完整动作 chunk、Knowledge Insulation 与联合梯度路径：

```bash
cd temp
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

## 5. 默认损失权重

| 配置 | 默认值 |
|---|---:|
| `lambda_act` | 1.0 |
| `lambda_current_align` | 0.1 |
| `lambda_student_wm` | 0.5 |
| `lambda_teacher_wm` | 0.1 |
| `lambda_distill` | 0.1 |
| `lambda_ldad_gt` | 0.05 |
| `lambda_ldad_pred` | 0.05 |
| `lambda_latent` | 0.0 |
| `lambda_state` | 0.0 |

第一版默认启用 `knowledge_insulation` 与 `delta_action_grounding`，暂不加入 Music-JEPA temporal prior 和 Causal-JEPA object-level masking。
