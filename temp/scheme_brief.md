# 简化方案总览

## 1. 方案定位

这份文档描述的是一个比 `privileged_e2e_scheme.md` 更短、但比一句话摘要更完整的方案说明。

目标不是给出所有实现细节，而是先明确下面几件事：

- 输入到底有哪些
- 输出到底有哪些
- `teacher` 和 `student` 两个分支各自做什么
- 训练时哪些量互相监督
- 当前方案保留了哪些 `VLA-JEPA` 的东西
- 当前方案借鉴了哪些 `LaWAM` 的东西

如果把这套方案压缩成一句话，可以写成：

- 用冻结的 `VJEPA21` 提供视觉 latent
- 用 `teacher` 分支在训练时借助未来信息构造 `z_teacher`
- 用 `student` 分支在部署条件下预测 `z_pred`
- 再通过 `world decoder` 和 `action head` 完成 future latent 预测与动作预测

## 2. 前提

- `JEPA encoder` 固定为 `/home/d013/桌面/project/VLA-JEPA-DEV/VJEPA21`
- `JEPA encoder` 在训练、验证、推理中始终冻结
- 整体采用 `student 主路径 + teacher 辅助分支` 的端到端训练
- `teacher` 只在训练时存在，推理时完全删除
- `student` 是真正的部署路径

这个前提意味着：

- `VJEPA21` 只负责提供稳定视觉表征
- 它不负责适配任务
- 它也不负责直接学习控制
- 真正需要学习的是：
  - `TeacherEncoder`
  - `Student latent predictor`
  - `World Decoder`
  - `Action Head`
  - 可选 `State Head`

## 3. 输入

### 3.1 训练输入

训练时，一个 batch 建议至少包含：

- `obs`
  - 当前观测图像
  - 用于 student 主路径
- `video`
  - 从当前到未来的一段视频片段
  - 用于 teacher 分支，也用于构造真实 future latent target
- `lang`
  - 任务指令
  - 只进入 student 主路径
- `state_0`
  - 当前状态
  - 进入 teacher 和 student 两个分支
- `state_T`
  - 未来终态
  - 只进入 teacher 分支
- `action`
  - 动作标签
  - 用于动作监督

可选字段：

- `state_mask`
- `delta_proprio = state_T - state_0`
- `embodiment_id`

如果写成集合形式，可以记成：

- 训练输入：
  - `{obs, video, lang, state_0, state_T, action}`

### 3.2 推理输入

推理时只能使用部署时拿得到的量，所以输入变成：

- `obs`
- `lang`
- `state_0`

这意味着推理时**没有**：

- `video`
- `state_T`
- `action`
- teacher 分支

如果写成集合形式，就是：

- 推理输入：
  - `{obs, lang, state_0}`

## 4. 输出

### 4.1 训练输出

训练时的核心输出分成两组。

teacher 分支输出：

- `z_teacher`
  - teacher 生成的 privileged latent
- `u_T`
  - 从真实未来视频中切出来的 future latent target

student 分支输出：

- `z_pred`
  - student 在当前观测条件下预测出的 latent
- `u_hat_T`
  - student 预测的 future latent / latent subgoal
- `a_hat`
  - student 最终预测的动作

如果加状态辅助头，还可以再有：

- `delta_hat`

### 4.2 推理输出

推理时只保留 student 主路径，所以输出很简单：

- 主输出：`a_hat`
- 可选中间量：
  - `z_pred`
  - `u_hat_T`

其中：

- `a_hat` 是真正用于控制的输出
- `z_pred` 和 `u_hat_T` 更多用于调试、可视化或分析

## 5. 中间过程

## 5.1 公共视觉编码阶段

训练时和推理时都要经过同一个视觉编码器：

- `obs -> VJEPA21 -> u`
- `video -> VJEPA21 -> video_latents`

其中：

- `u` 是 student 主路径可见的当前视觉 latent
- `video_latents` 是 teacher 分支用来切分 `u_teacher` 和 `u_T` 的整段视频 latent

这里默认：

- `VJEPA21` 全程冻结
- 分辨率按 `384 x 384`
- 时间维要和 `tubelet_size = 2` 对齐

## 5.2 Teacher 分支

teacher 分支的目标不是直接出动作，而是利用未来信息生成更好的 latent teacher。

数据流是：

- `video -> VJEPA21 -> video_latents`
- `video_latents -> (u_teacher, u_T)`
- `u_teacher + u_T + state_0 + state_T -> TeacherEncoder -> z_teacher`

这里每个量的语义是：

- `u_teacher`
  - teacher 看到的“当前部分”视觉上下文
- `u_T`
  - teacher 看到的真实未来视觉目标
- `state_0`
  - 当前状态
- `state_T`
  - 未来终态
- `z_teacher`
  - teacher 结合视觉转移和状态转移得到的 privileged latent

teacher 分支只在训练时存在。

它的主要作用不是做完整预测，而是给 student 提供：

- `z_teacher`
- `u_T`

## 5.3 Student 分支

student 分支是部署时真正保留下来的路径。

数据流是：

- `obs -> VJEPA21 -> u`
- `obs + lang + state_0 -> VLM / latent predictor -> z_pred`
- `u + z_pred -> World Decoder -> u_hat_T`
- `context + u_hat_T (+ z_pred) -> Action Head -> a_hat`

这里每个量的语义是：

- `u`
  - 当前视觉 latent
- `z_pred`
  - student 在当前可见信息下预测出来的 latent action
- `u_hat_T`
  - 由 `u + z_pred` 解码出的未来 latent / latent subgoal
- `a_hat`
  - 最终动作输出

student 分支既要学 latent，也要学 future latent prediction，还要学动作预测。

## 5.4 Teacher 和 Student 怎么交互

两个分支的交互不是每个量一一对应，而是主要发生在两处：

- `z_teacher` 监督 `z_pred`
- `u_T` 监督 `u_hat_T`

更具体地说：

- latent 层：
  - `L_distill(z_pred, z_teacher)`
- future latent 层：
  - `L_wm(u_hat_T, u_T)`
- action 层：
  - `L_act(a_hat, action)`

所以整条训练链可以压缩成：

- `z_teacher -> z_pred -> u_hat_T -> a_hat`

其中：

- `z_teacher` 不直接监督动作
- 它先约束 `z_pred`
- `z_pred` 再通过 decoder 影响 `u_hat_T`
- `u_hat_T` 再影响动作输出 `a_hat`

## 6. Loss

这套方案理论上可以包含 5 个子 loss，但它们不是同等优先级。

### 6.1 必需项

- `L_act(a_hat, action)`
  - 保证最终控制能力
- `L_wm(u_hat_T, u_T)`
  - 保证 world model 分支真的在预测未来

这两个 loss 是这套方案最核心的基础。

如果没有它们：

- 要么动作学不好
- 要么 latent world model 结构会失去意义

### 6.2 推荐项

- `L_distill(z_pred, z_teacher)`
  - 让 teacher 分支对 student latent 提供 privileged supervision
  - 这是这个方案区别于“纯端到端”的关键
- `L_latent = mean(||z_pred||^2)`
  - 让 latent 更稳定
  - 防止 `z_pred` 发散或变成 decoder 的高带宽捷径

这两个 loss 我建议第一版就保留。

### 6.3 可选项

- `L_state(delta_hat, state_T - state_0)`
  - 用真实状态变化去约束 latent
  - 能让 latent 更贴近真实机器人运动语义

但第一版里它最适合作为：

- 可选增强项
- 或 ablation 项

### 6.4 第一版推荐组合

如果现在想先做一个最小可行版本，我建议：

- 开启：
  - `L_act`
  - `L_wm`
  - `L_distill`
  - `L_latent`
- 关闭：
  - `L_state`

也就是说，第一版更建议的总目标是：

- `L_total = L_act + lambda_wm * L_wm + lambda_distill * L_distill + lambda_latent * L_latent`

## 7. 保留的 VLA-JEPA 部分

当前方案没有完全推翻原始 `VLA-JEPA`，主要保留了下面这些部分：

- `VJEPA21` 作为冻结视觉编码器
- `VLM` 仍放在 student 主路径中处理 `obs + lang`
- `future latent alignment` 的 world-model 思路
  - 即 `u_hat_T` 对齐 `u_T`
- `Action Head` 仍作为最终动作预测模块
- 整体仍然偏联合训练风格
  - 不是拆成两个完全分离的工程

换句话说，当前方案仍然保留了：

- `VLA-JEPA` 的视觉主干
- `VLA-JEPA` 的语言主路径
- `VLA-JEPA` 的 future latent prediction 思想
- `VLA-JEPA` 的 action prediction 终点

## 8. 引入的 LaWAM 部分

当前方案也显式借鉴了 `LaWAM` 的几个关键思想：

- 显式 `latent action` 中间变量
  - 即 `z_teacher / z_pred`
- `teacher -> student` 的 latent distillation
- `当前 latent + latent action -> 未来 latent` 的 world decoder 结构
- 把 `u_hat_T` 作为 latent subgoal 再送入 action head
- 可选的状态变化辅助监督 `L_state`

所以它不是纯 `VLA-JEPA`，也不是纯 `LaWAM`，而是：

- 保留 `VLA-JEPA` 的主干
- 引入 `LaWAM` 的 latent 组织方式

## 9. 一句话总结

这套方案本质上是：

- 用冻结的 `VJEPA21` 提供视觉 latent
- 用 `teacher` 分支借助未来视频和未来状态构造 `z_teacher`
- 用 `student` 分支在部署条件下预测 `z_pred`
- 用 `z_pred` 驱动 world decoder 预测 `u_hat_T`
- 再用 `u_hat_T` 作为 latent subgoal 预测最终动作 `a_hat`

从结构上看，它是端到端；
从监督方式看，它吸收了 `LaWAM` 的 latent distillation 思想；
从工程上看，它比完整两阶段更轻，但比纯端到端更有结构。
