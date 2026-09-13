# 消融实验

[主 README](../README.md) · [模型与训练目标](learned_goals.md) · [推理协议](../deployment/model_server/README.md)

实验分别回答三个问题：辅助损失是否必要、候选评分各分量是否有贡献、自动目标与目标条件 proposal 是否改善策略。SIMPLE 与 SONIC 各自使用固定的数据划分、控制接口和评估任务。

## 辅助损失：四组训练

主配置 `core` 使用五项有效损失：Action Expert 动作学习、世界模型预测、动作先验、目标条件动作重建和目标预测。其余变体只改变两个辅助损失的权重：

| 变体 | `lambda_delta` | `lambda_ctrl` | 非零损失数 |
|---|---:|---:|---:|
| `core` | 0 | 0 | 5 |
| `delta` | 0.05 | 0 | 6 |
| `ctrl` | 0 | 0.02 | 6 |
| `full` | 0.05 | 0.02 | 7 |

四组保留相同的网络结构、候选生成流程和评分公式。关闭辅助权重用于测量训练约束的作用，模型参数数量保持一致。日志仍包含七个加权损失字段，关闭项为零。

在主 README 的环境、模型、数据路径基础上运行。以下示例在单机 8 张 GPU 上依次启动四组 SIMPLE 训练，每卡 batch 为 1，累积 32 次，有效 batch 为 256：

```bash
export NUM_PROCESSES=8
export DATA_ROOT="${SIMPLE_DATA_ROOT}"
export PER_DEVICE_BATCH_SIZE=1
unset RUN_ID

for variant in core delta ctrl full; do
  SEED=42 bash scripts/train_learned_goal_ablation.sh simple "${variant}" \
    --trainer.gradient_accumulation_steps 32 \
    --trainer.max_train_steps 40000 \
    --trainer.save_interval 10000 || exit
done
```

SONIC 的单组示例使用累积 4 次，有效 batch 为 32：

```bash
DATA_ROOT="${SONIC_DATA_ROOT}" PER_DEVICE_BATCH_SIZE=1 SEED=42 \
  bash scripts/train_learned_goal_ablation.sh sonic core \
    --trainer.gradient_accumulation_steps 4
```

`SEED` 指定训练随机种子。默认输出名称为 `<interface>_learned_goal_<variant>_seed<seed>`，例如 `simple_learned_goal_full_seed42`。可以用 `RUN_ID` 显式设置输出名称。脚本将变体权重与 `SEED` 作为最终覆盖项；其他训练选项可追加在命令末尾。

使用相同的一组训练种子（例如 42、43、44）重复四个变体。每组从相同的预训练骨干初始化，固定训练步数、有效 batch、数据划分、尾段采样策略和评估 checkpoint 规则。对完整模型关闭一项 loss 后继续训练属于微调实验，需要与从相同起点训练的消融分别报告。

对照解释：

- `delta` 与 `core`：全局未来 latent 对齐的增益。
- `ctrl` 与 `core`：单步逆动力学辅助监督的增益。
- `full` 与 `delta`：已有全局对齐时，逆动力学的额外作用。
- `full` 与 `ctrl`：已有逆动力学时，全局对齐的额外作用。
- `full` 与 `core`：两项辅助监督的联合增益。

历史 prior 基线由其原始 commit、训练配置和 checkpoint 定义。这里的 `full` 是 learned-goal 架构的七项损失配置；自动目标、proposal、数据处理和训练输入均应按各自保存的配置记录。

续训沿用该 run 的辅助权重与训练配置。例如继续上面 seed 42 的 SIMPLE `full` 运行：

```bash
DATA_ROOT="${SIMPLE_DATA_ROOT}" PER_DEVICE_BATCH_SIZE=1 SEED=42 \
  bash scripts/train_learned_goal_ablation.sh simple full \
    --trainer.gradient_accumulation_steps 32 \
    --trainer.resume_from_checkpoint \
    "${OUTPUT_ROOT}/simple_learned_goal_full_seed42/checkpoints/steps_10000"
```

既有七项损失 run 使用 `RUN_ID` 指定原名称，并选择 `full` 继续。配置中使用自定义辅助权重的运行，通过普通训练入口传入原权重。

## 候选评分：同一模型的四组推理

固定训练好的 checkpoint、当前观测、目标与候选动作集合。令 `G` 为每个候选的视觉目标进展，`E` 为 GRU 的下一动作预测误差：

| 评分方式 | 选择规则 | 对照作用 |
|---|---|---|
| 无评分 | 从候选集合均匀随机选择 | 候选生成本身的基线 |
| 仅视觉 | 最大化 `G` | 世界模型目标评分的作用 |
| 仅 prior | 最小化 `E` | 动作序列一致性的作用 |
| 视觉＋prior | 最大化 `G - beta * E` | 两种评分是否互补 |

默认 `beta=0.1`。比较仅视觉与联合评分时，保持 prior 的训练权重不变，只改变选择规则。无评分也使用相同的候选集合。

启用 verifier 的推理响应提供以下数组：

```text
all_candidates             [B, N, H, A]
candidate_goal_progress    [B, N]
candidate_prior_error      [B, N]
candidate_scores           [B, N]
```

例如，对一份已经取得的响应进行配对选择：

```python
import numpy as np

# result 为 client.infer(payload)["data"]，请求启用 verifier。
actions = result["all_candidates"]
progress = result["candidate_goal_progress"]
prior_error = result["candidate_prior_error"]
scores = result["candidate_scores"]
batch, count = progress.shape
rng = np.random.default_rng(42)

indices = {
    "uniform": rng.integers(count, size=batch),
    "visual": progress.argmax(axis=1),
    "prior": prior_error.argmin(axis=1),
    "combined": scores.argmax(axis=1),
}
selected_chunks = {
    name: actions[np.arange(batch), index]
    for name, index in indices.items()
}
```

这些数组可以用于比较同一观测下的候选选择、选择分歧和分数分布。闭环成功率需要将四种选择规则分别用于完整任务，在相同任务初始条件和评估种子上重复执行；轨迹分叉后，各策略自然会看到不同观测。

`use_verifier=False` 切换到普通 Action Expert 推理，也改变了候选生成过程。该对照测量完整选择流程的增益。上表四组固定候选来源，专门测量评分机制。

## 目标与 proposal：分别控制条件和候选来源

| 问题 | 对照设计 |
|---|---|
| 自动目标能否替代外部目标？ | 同一 checkpoint 使用预测目标或通过 `subgoal_images` 提供示范目标，固定其他选项；预先固定示范目标的选取协议 |
| proposal 是否提供更好的候选？ | 两组均为 N 个候选，共享 N−1 个 Action Expert 候选；最后一个分别为额外 Expert 候选或目标条件 proposal，使用相同评分 |
| 移除 proposal 训练模块的系统影响是什么？ | 单独训练 `goal_action_proposal_enabled=false` 的模型，与开启组比较；这是训练模块消融，与上一行的推理候选消融分别报告 |
| 更多采样是否足以解释增益？ | 固定候选来源与评分规则，比较相同候选预算下的方法，并报告候选数量对应的延迟 |

proposal 训练中的视觉目标使用 stop-gradient，其重建损失直接训练 proposal 头。关闭 `learned_goal_enabled` 会同时切换世界模型的训练输入和目标构造。仅比较目标来源时，通过推理请求切换目标，保持 learned-goal 架构与训练方式一致。

示范目标是额外信息条件，应记录来源与选取方法。自动目标评估只向策略提供当前图像、任务指令和机器人状态。

## 结果记录

主指标为闭环任务成功率，同时记录完成时间、动作变化量、控制接口相关的手部误差和推理延迟。需要解释身体控制的实验另外记录控制器跟踪误差、失稳与碰撞事件，使用环境或机器人测量值。

每个结果对应 commit、配置、数据划分、训练 seed、checkpoint、候选数量、目标来源、评分规则和评估种子。报告各训练种子的结果与汇总统计。训练损失、候选误差与任务成功率分别记录，以区分监督拟合、候选选择和闭环执行三种作用。
