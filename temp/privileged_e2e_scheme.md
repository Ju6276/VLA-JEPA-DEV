# Privileged 端到端 VLA-JEPA 方案

本文描述当前仓库已实现的训练方案：把两阶段 latent distillation 折叠进**一条端到端训练流程**。

对应代码：

- `starVLA/model/framework/VLA_JEPA.py`
- `starVLA/model/modules/world_model/privileged_latent.py`
- 配置示例：`scripts/config/vlajepa_merged_dataset_001_e2e.yaml`

一句话概括：

- 冻结的 `VJEPA21` 只在训练时提供视觉 endpoint latent
- `teacher` 用未来信息构造 `z_teacher`
- `student` 在部署条件下预测 `z_pred`，再解码 `u_hat_T` 并输出动作

---

## 1. 设计目标

| 目标 | 做法 |
|---|---|
| 保留 latent world model 结构 | student 显式预测 `z_pred`，再解码未来 latent `u_hat_T` |
| 不拆 Stage1 / Stage2 工程 | 同一 `forward()` 内联合训练 |
| 部署路径尽量简单 | 推理只跑 student，不跑 V-JEPA / teacher |
| latent 有外部监督 | teacher 提供 `z_teacher`，真实视频提供 `u_T` |

---

## 2. 前提与可训练模块

- `VJEPA21` 全程冻结（训练 / 验证 / 推理都不更新）
- 推理时不加载、不调用 V-JEPA
- 真正学习的模块：
  - `Qwen3-VL`（student backbone）
  - `StudentCurrentAdapter`
  - `StudentPredictor`
  - `TeacherEncoder`（仅训练）
  - `SharedWorldDecoder`
  - `Action Head`（FlowMatching / DiT）
  - 可选 `StateDeltaPredictor`（当前默认关闭，`lambda_state=0`）

默认 V-JEPA 权重路径由 YAML 中 `framework.vj2_model.base_encoder` 指定，例如：

```text
/cpfs_infra/shared/xiaoxinyu/VLA-JEPA/VLA-JEPA-DEV/VJEPA21/vjepa2_1_vitl_dist_vitG_384.pt
```

### 模块职责

| 模块 | 作用 |
|---|---|
| `Qwen3-VL` | 编码当前图像 + 语言，得到 `current_context` 与 embodied tokens |
| `StudentCurrentAdapter` | 把 Qwen context 映射到 V-JEPA latent 空间，得到 `u` |
| `StudentPredictor` | 由 `current_context + state_0` 预测部署侧 latent action `z_pred` |
| `TeacherEncoder` | LaWAM `LAMEncoder`：`u_teacher + [state_0; state_T] -> z_teacher` |
| `SharedWorldDecoder` | LaWAM `LAMDecoder_v2`：`(u, z) -> u_hat_T`，两分支共用 |
| `future_latent_to_qwen` / `latent_action_to_qwen` | 把 `u_hat_T`、`z_pred` 投到 Qwen 隐空间后拼进 action condition |
| `Action Head` | FlowMatching 预测动作 chunk |

空间对齐约定：

- student 当前表征来自 Qwen，不经过 V-JEPA
- `StudentCurrentAdapter` 把 `current_context` 对齐到 decoder / V-JEPA 空间
- `L_wm(u_hat_T, u_T)` 给该 adapter 与 decoder 提供训练信号

---

## 3. 训练过程

入口：`VLA_JEPA.forward(examples)`。

### 3.1 训练输入

每个 sample 至少包含：

| 字段 | 形状 / 语义 | 进入哪条分支 |
|---|---|---|
| `image` | 当前观测图像（student 可见） | student |
| `lang` | 任务指令 | student |
| `state` / `state_0` | 当前状态，`[D_state]`（如 46 维） | student + teacher |
| `state_T` | 未来终态，`[D_state]` | teacher |
| `video` | 当前→未来视频片段，用于切 endpoint latent | teacher（经冻结 VJEPA21） |
| `action` | 动作标签 chunk，用于 `L_act` | action head |

可选 mask：

- `action_mask`
- `video_mask` / `video_endpoint_mask`（无效 endpoint 时不计入 `L_wm` / `L_teacher_wm`）
- `state_mask`（仅在开启 `L_state` 时使用）

集合形式：

```text
训练输入 = {image, lang, state_0, state_T, video, action}
```

### 3.2 训练中间过程

训练时同时跑 **student 主路径** 和 **teacher 辅助分支**。

#### A. Student 主路径

```text
1) image + lang
      -> Qwen3-VL
      -> last_hidden / embodied_action_tokens
      -> current_context = mean(embodied_action_tokens)

2) current_context
      -> StudentCurrentAdapter
      -> u                          # 对齐到 V-JEPA 空间的当前 latent

3) current_context + state_0
      -> StudentPredictor
      -> z_pred                     # 部署侧 latent action

4) u + z_pred
      -> SharedWorldDecoder
      -> u_hat_T                    # 预测的未来 latent / latent subgoal

5) embodied_action_tokens
   + future_latent_to_qwen(u_hat_T)
   + latent_action_to_qwen(z_pred)
      -> action_condition
      -> ActionHead(+ state_0)
      -> a_hat / action_loss
```

含义：

- `current_context`：只由当前图像和语言得到，推理时也能构造
- `u`：给 shared decoder 用的当前条件
- `z_pred`：student 要学会预测的 latent action
- `u_hat_T`：由 `u + z_pred` 解码出的未来视觉 latent
- `a_hat`：最终动作

#### B. Teacher 辅助分支（仅训练）

```text
1) video
      -> frozen VJEPA21.get_vision_features
      -> 按 temporal tubelet 切分
      -> u_teacher = 第一个时间步 spatial mean
      -> u_T       = 最后一个时间步 spatial mean，并 detach

2) u_teacher + state_0 + state_T
      -> TeacherEncoder
      -> z_teacher                  # privileged latent action

3) u_teacher + z_teacher
      -> SharedWorldDecoder（与 student 共用）
      -> u_teacher_hat_T            # teacher 侧未来 latent 预测
```

含义：

- `u_teacher`：teacher 看到的“当前端”视觉 latent（来自真实视频）
- `u_T`：真实未来视觉 latent，作为 world-model 监督目标
- `z_teacher`：结合视觉转移与状态转移得到的 teacher latent
- `u_teacher_hat_T`：用 `z_teacher` 解码未来，用来约束 teacher latent 真的被 `u_T` 锚定

Teacher 不直接输出动作；它只提供 `z_teacher` 和 `u_T` 给 student 侧 loss 使用。

#### C. 两条分支如何交互

```text
z_teacher  --L_distill-->  z_pred  --decoder-->  u_hat_T  --action-->  a_hat
                                             ^
                                          L_wm ← u_T

u_teacher + z_teacher --decoder--> u_teacher_hat_T --L_teacher_wm--> u_T
```

### 3.3 训练输出

`forward()` 返回的是 loss / 监控量字典（不是直接返回动作张量）：

| 字段 | 含义 |
|---|---|
| `loss_total` | 加权总损失，用于反传 |
| `action_loss` | `L_act` |
| `wm_loss` | `L_wm` |
| `teacher_wm_loss` | `L_teacher_wm` |
| `distill_loss` | `L_distill` |
| `latent_loss` | `L_latent` |
| `state_loss` | `L_state`（默认 0） |
| `z_teacher_norm` / `z_pred_norm` / `z_cosine` | 仅监控，不进总损失 |

训练阶段内部还会产生这些中间量（用于算 loss，不一定全部写入返回字典）：

- student：`current_context`, `u`, `z_pred`, `u_hat_T`, `a_hat`
- teacher：`u_teacher`, `u_T`, `z_teacher`, `u_teacher_hat_T`

### 3.4 训练 Loss

总损失（与代码一致）：

```text
L_total =
    λ_act         * L_act
  + λ_wm          * L_wm
  + λ_teacher_wm  * L_teacher_wm
  + λ_distill     * L_distill
  + λ_latent      * L_latent
  + λ_state       * L_state
```

| Loss | 定义 | 作用 | 默认 λ（e2e yaml） |
|---|---|---|---|
| `L_act` | ActionHead 对 `action` 的 flow-matching loss | 保证最终控制能力 | 1.0 |
| `L_wm` | `SmoothL1(u_hat_T, u_T)` | student 真正预测未来 latent | 0.5 |
| `L_teacher_wm` | `SmoothL1(u_teacher_hat_T, u_T)` | 锚定 `z_teacher`，避免 teacher 崩塌 | 0.1 |
| `L_distill` | `SmoothL1(z_pred, z_teacher.detach())` | teacher → student latent 蒸馏 | 0.1 |
| `L_latent` | `mean(z_pred²)` | 约束 latent 幅度，抑制捷径 | 0.1 |
| `L_state` | 可选 `SmoothL1(Δ̂, state_T - state_0)` | 让 latent 贴近状态变化 | 0.0（关闭） |

注意：

- `z_teacher` **detach** 后再监督 `z_pred`，蒸馏梯度主要流向 student
- `u_T` 来自冻结 VJEPA21，并 `detach`，不回传到 encoder
- `z_teacher` 不直接监督动作；它先约束 `z_pred`，再通过 decoder / action condition 间接影响策略

---

## 4. 推理过程

入口：`VLA_JEPA.predict_action(batch_images, instructions, state)`。

### 4.1 推理输入

```text
推理输入 = {image, lang, state_0}
```

| 字段 | 含义 |
|---|---|
| `image` | 当前观测 |
| `lang` | 任务指令 |
| `state` / `state_0` | 当前状态（必需） |

推理时**没有**：

- `video`
- `state_T`
- `action`
- teacher 分支
- V-JEPA 前向

### 4.2 推理中间过程

只保留 student 主路径：

```text
1) image + lang
      -> Qwen3-VL
      -> embodied_action_tokens / current_context

2) current_context
      -> StudentCurrentAdapter
      -> u

3) current_context + state_0
      -> StudentPredictor
      -> z_pred

4) u + z_pred
      -> SharedWorldDecoder
      -> u_hat_T

5) embodied_action_tokens
   + future_latent_to_qwen(u_hat_T)
   + latent_action_to_qwen(z_pred)
      -> ActionHead.predict_action(+ state_0)
      -> a_hat
```

与训练的差别：

- 不跑 `video -> VJEPA21`
- 不跑 `TeacherEncoder`
- 不算任何 loss
- `ActionHead` 走 `predict_action`（采样），而不是训练时的 denoising loss

### 4.3 推理输出

`predict_action()` 返回字典：

| 字段 | 含义 |
|---|---|
| `normalized_actions` | 主输出：归一化动作 chunk `a_hat`，真正用于控制 |
| `z_pred` | 可选中间量：预测的 latent action |
| `u_hat_T` | 可选中间量：预测的未来 latent |

部署时通常只用 `normalized_actions`；`z_pred` / `u_hat_T` 用于调试或可视化。

---

## 5. 与 VLA-JEPA / LaWAM 的关系

**保留自 VLA-JEPA**

- 冻结世界模型 encoder（本实现为 V-JEPA 2.1）
- Qwen-VL 作为 student 视觉语言主干
- future latent alignment（`u_hat_T ↔ u_T`）
- Action Head 作为最终控制出口

**借鉴自 LaWAM**

- 显式 latent action（`z_teacher` / `z_pred`）
- teacher → student distillation
- `LAMEncoder` / `LAMDecoder_v2` 作为 teacher / shared decoder
- 可选 state-delta 辅助头

---

## 6. 启动入口

```bash
# 端到端训练（merged_dataset_001 + VJEPA21）
bash scripts/vlajepa_merged_dataset_001_e2e.sh
```

更完整的启动说明见 `train.md`。
