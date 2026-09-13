# 自动目标与动作验证

[返回主 README](../README.md)

## 表示与条件

模型输入为当前 ego 图像 I_t、任务指令 l 和机器人状态 s_t。V-JEPA 2.1 提取 patch 特征，池化并归一化得到视觉状态 z_t；Qwen3-VL 提取图像／指令条件 tokens。

默认 ViT-L 视觉维度为 1024，Qwen 条件维度为 2048。SIMPLE 使用 32 维 state 和 36 维 action；SONIC 使用 46 维 state 和 78 维 action，见 [控制接口](control_interfaces.md)。

## 目标预测器

`LatentGoalPredictor` 将 z_t、平均后的 Qwen embodied-action 条件 tokens 和 s_t 拼接，经以下网络输出归一化的目标向量 g_t：

```text
Linear(1024 + 2048 + state_dim, 512)
GELU
Linear(512, 512)
GELU
LayerNorm(512)
Linear(512, 1024)
L2 normalization
```

Qwen 条件 tokens 由当前图像与指令计算。目标预测器输出的是未来视觉表征。训练标签为同一示范在 t+H 时刻的视觉特征，其中 H 是动作 chunk 长度。

训练和部署都使用单张当前图像重复构成 8 帧编码器输入。训练目标图像同样重复后独立编码，当前图像与未来图像分别调用同一份冻结编码器。这样训练输入与部署输入一致，未来帧仅参与监督目标的计算。

## 时间对齐

| 配置 | H | 视频采样偏移 |
|---|---:|---|
| SIMPLE | 30 | `[0, 4, 8, 12, 16, 20, 24, 30]` |
| SONIC | 40 | `[0, 6, 12, 18, 24, 30, 35, 40]` |

完整片段的动作标签覆盖 `[t, t+H)`，目标图像取 t+H。偏移由 `datasets.vla_data.video_frame_offsets` 配置；序列从 0 开始严格递增，长度与 `framework.vj2_model.num_frames` 一致。

默认 `datasets.vla_data.require_full_horizon: false` 保留尾段补齐规则：越界动作使用原始零值后归一化，越界图像重复末帧，补齐动作参与训练损失。此时临近末端的目标图像时刻为 `min(t+H, episode_length-1)`。设置为 `true` 时，仅采样所有动作／视频偏移均有效的片段；这会减少尾段样本。两种选择均使用版本化索引缓存，根据采样策略、暂停帧过滤、动作／视频偏移和轨迹编号／长度区分并校验。配置或轨迹元数据变化时自动重建索引；缺少这些信息的旧缓存不参与读取，旧文件保持不变。数据目录只读时直接在内存中生成索引。

索引重建导致每轮 batch 数变化时，完整续训会拒绝恢复旧数据位置；可使用 `trainer.pretrained_checkpoint` 加载模型权重开始新的训练。

目标预测器和 proposal 的当前 latent 均来自独立当前观察编码的最后时间块；learned-goal 的逆动力学基准与部署评分使用同一个定义。

## 动作候选

`GoalConditionedActionProposal` 输入 `[z_t, g_t, g_t-z_t, s_t]`，经两层 MLP 与可学习的时间位置嵌入，输出 H×A 的动作 chunk。

默认候选集合包含一个目标条件 proposal 与七个 Action Expert 采样结果。所有候选共享同一个目标 g_t。训练 proposal 时，真实未来目标与预测目标各占 0.5；预测目标在该分支中使用 stop-gradient。

## 世界模型与控制监督

`CandidateActionEncoder` 将动作 chunk 转换为条件 tokens，与 Qwen 条件 tokens 相加，输入动作条件视觉预测器。世界模型由重复当前图像的特征预测终点 patch tokens，并与独立编码的未来图像特征对齐。

`LatentInverseDynamics` 从视觉特征位移和当前 state 恢复动作标签，默认使用 chunk 第一步动作；真实位移和预测位移均参与监督。

| 损失 | 定义 | 默认权重 |
|---|---|---:|
| `action_loss` | Action Expert 的动作学习目标 | 1.0 |
| `wm_loss` | 预测终点与目标 patch tokens 的 L1 距离 | 0.1 |
| `delta_loss` | 预测位移与目标位移的 MSE | 0.05 |
| `ctrl_loss` | 从真实／预测位移恢复动作的 MSE 之和 | 0.02 |
| `action_prior_loss` | GRU 下一步动作预测 MSE | 0.01 |
| `goal_proposal_loss` | 目标条件动作 chunk 重建 MSE | 0.01 |
| `goal_prediction_loss` | 预测目标与真实目标的 cosine 距离 | 0.05 |

框架返回的损失值已乘以上述权重。V-JEPA 编码器显式冻结、保持 eval 模式并排除在优化器之外；目标预测器、动作模块、世界模型与 Qwen 按训练配置优化。

动作学习的噪声重复次数由 `trainer.repeated_diffusion_steps` 控制，默认4。训练期间的 MAE/MSE 是当前训练 batch 的动作误差诊断：暂时关闭 dropout、跨进程聚合后恢复训练模式。MSE 按逐元素平方误差的均值计算。

## GRU 动作先验

`ActionDynamicsPrior` 的结构为 `Linear(A+S,512) → GELU → GRU(512,2 layers) → LayerNorm → Linear(512,A)`。

它学习在当前 state 条件下，从候选序列中的前序动作预测下一步动作。训练使用示范动作的 teacher forcing，推理为每个候选计算：

```text
E_prior(A, s_t) = mean over time and action dimensions of (a_next - a_next_predicted)^2
```

SONIC 中 A=78，包括 64 维 motion token 和 14 维双手控制。该评分度量动作序列的统计一致性，使用归一化动作空间中的等权 MSE。GRU 在每个候选开始时初始化隐藏状态，当前 state 在 chunk 内重复使用。

## 候选选择

令 d 为 cosine 距离，z_future 为候选动作对应的预测终点：

```text
score(A) = d(z_t, g_t) - d(z_future(A), g_t) - beta * E_prior(A, s_t)
```

默认 beta=0.1。最高分候选作为输出，服务返回 `normalized_actions`、`all_candidates` 和选中候选的 `verification_scores`。

目标来源按以下顺序选择：

1. 请求中的 `subgoal_images`。
2. 配置加载的 `SubgoalTracker`。
3. `LatentGoalPredictor`。

对应的 `goal_source` 为 `images`、`tracker` 或 `predicted`。默认配置选择自动目标预测。

## 实验配置

| 实验 | 配置方式 |
|---|---|
| 普通 Action Expert 推理 | 请求设置 `use_verifier=False` |
| 自动目标与人工目标对比 | 同一模型分别使用默认目标或传入 `subgoal_images` |
| GRU prior 评分对比 | 对同一候选集合比较 beta=0 与 beta=0.1 |
| 目标条件 proposal 对比 | 分别训练 `goal_action_proposal_enabled: false / true` 的模型 |
| 候选数量对比 | 请求设置 `num_candidates` |

对比 prior 时固定输入、候选集合与随机种子；分别记录任务成功率、动作变化量、手部控制误差和推理延迟。自动目标与外部目标可在同一 checkpoint 上比较。

关闭 `learned_goal_enabled` 会选择原有序列 teacher-forcing 世界模型监督；比较该配置时需要同时考虑世界模型训练输入和目标定义的变化。

## checkpoint 初始化

训练输出包含 `goal_predictor.*` 参数。部署通过完整 checkpoint 严格恢复模型。

从相同控制接口的旧 goal-proposal checkpoint 继续训练时，可以选择加载已有模块：

```bash
bash scripts/train_g1_delta_jepa_8xa100.sh \
  --trainer.pretrained_checkpoint /path/to/checkpoints/steps_40000_pytorch_model.pt \
  --trainer.reload_modules qwen_vl_interface,action_model,vj_encoder,vj_predictor,candidate_action_encoder,inv_dyn_decoder,action_dynamics_prior,goal_action_proposal
```

目标预测器随新一轮训练学习。跨 SIMPLE / SONIC 接口初始化时，选择共享骨干并重新训练控制模块：

```bash
bash scripts/train_sonic_learned_goal.sh \
  --trainer.pretrained_checkpoint /path/to/sim/checkpoints/steps_40000_pytorch_model.pt \
  --trainer.reload_modules qwen_vl_interface,vj_encoder
```

两个命令均使用主 README 中的 `DATA_ROOT`、`VJEPA21_CKPT` 与 `QWEN_MODEL` 环境变量。

## 恢复训练

`trainer.pretrained_checkpoint` 用于从权重开始新训练；`trainer.resume_from_checkpoint` 指定训练状态目录，用于继续同一次训练：

```bash
bash scripts/train_sonic_learned_goal.sh \
  --trainer.resume_from_checkpoint /path/to/run/checkpoints/steps_10000
```

保存点 `steps_N/` 包含 Accelerate/DeepSpeed 模型与优化器状态、学习率调度器、各进程随机状态、训练步数和 epoch／batch 位置；同级 `steps_N_pytorch_model.pt` 用于部署。`trainer_state.json` 在所有状态保存完成后写入。

恢复时保持相同的数据、每进程 batch、进程数与梯度累积设置。数据集 epoch 会传递到持续运行的 worker，恢复后从对应 batch 位置继续。worker 内的数据增强和预取随机状态不序列化，因此随机预处理不保证逐位重放。旧的单独 `.pt` 文件继续用于权重初始化和部署。
