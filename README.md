# One-Stage Action-Grounded Privileged VLA-JEPA

本分支基于 `origin/xxy_LaWAM_JEPA` 中的 `temp/` codebase，提供一阶段的 privileged teacher/student 训练、LaWAM-style Knowledge Insulation，以及 Delta-JEPA-style multi-step action grounding。

工作目录是 `temp/`；`LaWAM-main/` 提供本分支复用的 `LAMEncoder`、`LAMDecoder_v2` 等模块。

## 1. 怎么启动

### 1.1 准备环境

```bash
git clone https://github.com/Ju6276/VLA-JEPA-DEV.git
cd VLA-JEPA-DEV
git switch feat/one-stage-action-grounded-lawam
cd temp

conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA
pip install -r requirements.txt
pip install -e .
```

下列启动和测试命令均假定当前目录是仓库中的 `temp/`。

训练前准备三个路径：

| 环境变量 | 要求 |
|---|---|
| `DATA_ROOT` | 数据根目录；其下必须存在 `merged_dataset_001/` |
| `BASE_VLM` | 本地 `Qwen3-VL-2B-Instruct` 模型目录 |
| `VJEPA_ENCODER` | 本地 `vjepa2_1_vitl_dist_vitG_384.pt` 权重 |

### 1.2 先跑真实数据 smoke test

建议先执行单卡、`batch_size=1` 的完整 forward/backward。它会加载真实数据、Qwen 和 V-JEPA，检查各项 loss 与关键模块梯度，但不会保存 checkpoint。

```bash
conda activate VLA_JEPA

PYTHONPATH=. python scripts/smoke_test_action_grounded.py \
  --data-root /path/to/Datasets \
  --base-vlm /path/to/Qwen3-VL-2B-Instruct \
  --vjepa-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --batch-size 1
```

显存紧张时，可先缩小仅用于 smoke test 的 Action Head，并冻结 Qwen：

```bash
PYTHONPATH=. python scripts/smoke_test_action_grounded.py \
  --data-root /path/to/Datasets \
  --base-vlm /path/to/Qwen3-VL-2B-Instruct \
  --vjepa-encoder /path/to/vjepa2_1_vitl_dist_vitG_384.pt \
  --batch-size 1 \
  --action-layers 2 \
  --freeze-vlm
```

`--action-layers` 只覆盖当前 smoke 进程中的配置，不会修改训练 YAML。

### 1.3 启动正式训练

下面是推荐的单卡小 batch 启动方式：

```bash
DATA_ROOT=/path/to/Datasets \
BASE_VLM=/path/to/Qwen3-VL-2B-Instruct \
VJEPA_ENCODER=/path/to/vjepa2_1_vitl_dist_vitG_384.pt \
CUDA_VISIBLE_DEVICES=0 \
NUM_PROCESSES=1 \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
WANDB_MODE=offline \
bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

多卡训练示例：

```bash
DATA_ROOT=/path/to/Datasets \
BASE_VLM=/path/to/Qwen3-VL-2B-Instruct \
VJEPA_ENCODER=/path/to/vjepa2_1_vitl_dist_vitG_384.pt \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NUM_PROCESSES=4 \
BATCH_SIZE=8 \
NUM_WORKERS=8 \
WANDB_MODE=offline \
bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

主要入口：

- 启动脚本：`temp/scripts/vlajepa_merged_dataset_001_e2e.sh`
- 训练配置：`temp/scripts/config/vlajepa_merged_dataset_001_e2e.yaml`
- Trainer：`temp/starVLA/training/train_starvla.py`
- 输出目录：`temp/checkpoints/<RUN_ID>/`

当前 YAML 默认从零训练：

```yaml
pretrained_checkpoint: null
resume_step: null
```

如果需要加载旧 checkpoint，需要检查新增 Teacher/Delta 模块的 missing keys，并明确设置续训步数。

### 1.4 运行核心单元测试

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```

测试覆盖：Teacher 是否使用未来视觉、Delta decoder 的完整 action chunk 与梯度、Knowledge Insulation 的梯度边界，以及一阶段联合 loss 的梯度路由。

## 2. 本分支改了什么

### 2.1 修复 privileged Teacher 没有读取未来视觉的问题

原始 `temp/` 的 Teacher 实际只读取当前视觉 latent：

```text
u_0 + concat(state_0, state_T) -> TeacherEncoder -> z_teacher
```

本分支改为两个对齐的视觉/状态端点：

```text
(u_0_target, state_0)
(u_T_target, state_T)
        -> TeacherEncoder
        -> z_teacher
```

两个视觉 target 都来自冻结的 V-JEPA；Teacher 只在训练时存在。

### 2.2 对齐 Student 当前 latent

新增：

```text
L_current_align = SmoothL1(u_student, u_0_target)
```

原始版本只监督预测的未来 latent，没有直接保证 `u_student` 与 V-JEPA 当前 latent 位于同一个空间。现在当前端和未来端都有监督：

```text
u_student       <-> u_0_target
u_student_hat_T <-> u_T_target
```

### 2.3 加入 LaWAM-style Knowledge Insulation

Action Head 读取 dynamics tokens，但 flow-matching loss 不通过这两个 token 接口直接回写 StudentPredictor/SharedWorldDecoder：

```text
embodied tokens
+ projection(stopgrad(u_student_hat_T))
+ projection(stopgrad(z_student))
        -> Flow Action Head
```

Action Head、投影层以及 embodied-token 对应的 Qwen 路径仍然可训练。Knowledge Insulation 只截断 future latent 和 latent-action token 两条直接梯度路径。

### 2.4 加入 Delta-JEPA-style multi-step action grounding

本分支新增 `MultiStepDeltaActionDecoder`，从长时域 latent displacement 恢复完整动作序列：

```text
delta_gt   = u_T_target - u_0_target
delta_pred = u_student_hat_T - u_student

delta_gt   -> DeltaActionDecoder -> complete action chunk
delta_pred -> DeltaActionDecoder -> complete action chunk
```

对应两个辅助目标：

```text
L_ldad_gt
L_ldad_pred
```

默认不向 Delta decoder 输入 state，避免通过 proprioception 绕过视觉 displacement。

### 2.5 一阶段联合损失

```text
L_total =
    lambda_act        * L_act
  + lambda_current    * L_current_align
  + lambda_student_wm * L_student_wm
  + lambda_teacher_wm * L_teacher_wm
  + lambda_distill    * L_distill
  + lambda_ldad_gt    * L_ldad_gt
  + lambda_ldad_pred  * L_ldad_pred
  + lambda_latent     * L_latent
  + lambda_state      * L_state
```

默认权重：

| Loss | 权重 |
|---|---:|
| `L_act` | 1.0 |
| `L_current_align` | 0.1 |
| `L_student_wm` | 0.5 |
| `L_teacher_wm` | 0.1 |
| `L_distill` | 0.1 |
| `L_ldad_gt` | 0.05 |
| `L_ldad_pred` | 0.05 |
| `L_latent` | 0.0 |
| `L_state` | 0.0 |

本版本仍然只有一个训练 stage，没有拆成 world-model pretraining 和 policy training 两个阶段。

### 2.6 训练和部署路径

训练时：

```text
Student + privileged Teacher + frozen V-JEPA
+ SharedWorldDecoder + DeltaActionDecoder + ActionHead
```

部署时：

```text
image + language + state_0
        -> Student
        -> z_student, u_student_hat_T
        -> Action Head
        -> action chunk
```

部署不需要未来视频或 `state_T`，也不会构建 V-JEPA、TeacherEncoder 和 DeltaActionDecoder。

`temp/deployment/model_server/simple_g1_adapter.py` 是原始 codebase 已有的 SIMPLE/G1 handover 部署适配器；本分支只增加了训练模块过滤。它当前包含 36 维 G1 handover 的反归一化规则，不是 78 维 Sonic action 的通用部署接口。

### 2.7 当前没有加入什么

第一版暂未加入：

- Music-JEPA temporal prior；
- Causal-JEPA object-level masking；
- verifier scoring；
- 显式外部 subgoal proposal；
- 第二训练 stage。

当前统一故事是：LaWAM 提供 privileged transition teacher 和 Knowledge Insulation，VLA-JEPA 提供冻结视觉 target 与部署侧 VLM policy，Delta-JEPA 提供 latent displacement 到完整动作序列的 grounding。

## 3. 主要代码位置

| 内容 | 路径 |
|---|---|
| 一阶段总框架 | `temp/starVLA/model/framework/VLA_JEPA.py` |
| Teacher / shared decoder / KI | `temp/starVLA/model/modules/world_model/privileged_latent.py` |
| Multi-step Delta decoder | `temp/starVLA/model/modules/world_model/delta_action_decoder.py` |
| 训练配置 | `temp/scripts/config/vlajepa_merged_dataset_001_e2e.yaml` |
| Smoke test | `temp/scripts/smoke_test_action_grounded.py` |
| 单元测试 | `temp/tests/test_action_grounded_privileged_latent.py` |
| 完整设计说明 | `temp/privileged_e2e_scheme.md` |
| 简版故事 | `temp/scheme_brief.md` |

## 4. 本机验证结果

在 NVIDIA GeForce RTX 4090 D 上，使用真实 `merged_dataset_001`、Qwen3-VL-2B、V-JEPA 2.1、生产配置 16 层 Action Head 和 `batch_size=1` 完成 forward/backward：

```text
SMOKE_TEST_OK
loss_total=1.694084
peak_cuda_gib=11.805
```

核心单元测试共 4 项，全部通过。

## 5. 更多说明

- 原始 VLA-JEPA 使用说明：`temp/README.md`
- 详细训练说明：`temp/train.md`
- LaWAM 原始项目快照：`LaWAM-main/README.md`
