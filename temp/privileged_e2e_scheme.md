# VLA-JEPA-DEV 端到端 + 未来信息辅助分支方案

这份文档描述的是一个**不显式拆成 Stage 1 / Stage 2 独立训练**的方案，而是把两阶段思想折叠进**一个端到端训练流程**里。

它的核心思想是：

- 部署主路径只使用当前可观测信息
- 训练时额外引入一个使用未来信息的 teacher 辅助分支
- 用 teacher 分支给 student 主路径提供 latent 监督
- `JEPA encoder` 固定使用：
  - `/home/d013/桌面/project/VLA-JEPA-DEV/VJEPA21`
- 并且在：
  - 训练
  - 验证
  - 推理
  中**始终冻结**

这个方案可以理解成：

- 结构上是端到端
- 监督方式上吸收了“两阶段 latent distillation”的思想

---

## 1. 方案目标

这个方案想解决的问题是：

- 我们希望保留 latent world model 的结构意义
- 但不想真的拆成两个独立训练工程
- 我们希望 latent 不只是 action head 的附属中间变量
- 同时又希望部署路径足够简单清楚

所以这里采用：

- `student 主路径`
  - 面向部署
  - 只看当前可用信息
- `teacher 辅助分支`
  - 只在训练时存在
  - 使用训练时额外可见的未来信息生成更强的 latent teacher

---

## 2. 全局前提

### 2.1 JEPA encoder 固定

这里我们明确约束：

- 视觉编码器固定使用 `VJEPA21`
- 路径为：
  - `/home/d013/桌面/project/VLA-JEPA-DEV/VJEPA21`

在实现上，对应的是当前仓库里的 `VJEPA21` 适配器路径：

- `starVLA/model/modules/world_model/vjepa21_encoder.py`

### 2.2 JEPA encoder 全程冻结

这个前提是本方案的基础：

- 训练时冻结
- 验证时冻结
- 推理时冻结

也就是说，`VJEPA21` 在本方案中的职责仅仅是：

- 提供稳定的视觉 latent

它不承担：

- 任务适配
- 控制策略学习
- latent action 学习

真正需要学习的是：

- student latent predictor
- teacher latent encoder
- latent world decoder
- action head
- 可选 state head

---

## 3. 整体结构

整个训练图可以拆成两条分支，但这两条分支的职责不同：

- `student 主路径`
  - 是真正的部署路径
  - 训练时参与 `L_act / L_wm / L_distill / L_latent`
  - 推理时完整保留
- `teacher 辅助分支`
  - 是训练时的未来信息监督路径
  - 训练时提供 `z_teacher`
  - 同时通过 teacher 侧 world-model 约束让 `z_teacher` 被 `u_T` 锚定
  - 推理时完全删除

### 3.1 Student 主路径总览

输入：

- 当前观测 `obs`
- 语言指令 `lang`
- 当前状态 `state_0`

中间过程：

- `obs + lang -> Qwen3-VL / student backbone -> current context`
- `current context -> pooling -> StudentCurrentAdapter -> 当前上下文 latent u`
- `current context + state_0 -> latent predictor -> z_pred`
- `u + z_pred -> SharedWorldDecoder -> u_hat_T`
- `current context + u_hat_T + z_pred + state_0 -> action head -> a_hat`

输出：

- `u`
- `z_pred`
- `u_hat_T`
- `a_hat`

这里：

- `u` 是从 student 当前上下文里抽取并经过 adapter 后得到的当前 latent，供 shared world decoder 使用
- `z_pred` 是 student 预测出的 latent action
- `u_hat_T` 是 student 预测的未来 latent / latent subgoal
- `a_hat` 是最终动作

这是推理时真正保留下来的主路径。

这里需要明确一个空间对齐约定：

- student 分支不使用 `VJEPA21`
- `u_raw` 来自 `Qwen3-VL` 的 `current context` pooling
- `u_raw` 需要经过 `StudentCurrentAdapter` 得到 `u`
- `u` 再进入 `SharedWorldDecoder`
- `SharedWorldDecoder` 的输出空间对齐冻结 `VJEPA21` 的未来 latent `u_T`

这样做的原因是：

- `Qwen3-VL` 表征和 `VJEPA21` 表征天然不在同一个 latent 空间
- 但第一版仍然希望 student 推理路径不依赖 `VJEPA21`
- 所以用可训练的 `StudentCurrentAdapter` 把 student 当前上下文映射到 shared world decoder 可使用的条件空间
- `L_wm(u_hat_T, u_T)` 会给这个 adapter 提供训练信号

### 3.2 Teacher 辅助分支总览

输入：

- 当前视频片段 `video`
- 当前状态 `state_0`
- 未来终态 `state_T`

中间过程：

- `video -> VJEPA21 -> video_latents`
- `video_latents -> 拆成当前部分与未来部分 -> (u_teacher, u_T)`
- `u_teacher + state_0 + state_T -> teacher latent encoder -> z_teacher`
- `u_teacher + z_teacher -> SharedWorldDecoder -> u_teacher_hat_T`

输出：

- `u_teacher`
- `u_T`
- `z_teacher`
- `u_teacher_hat_T`

这里：

- `u_teacher` 是 teacher 分支用于推断 latent 的当前视觉上下文
- `u_T` 是真实未来 latent target
- `z_teacher` 是 teacher 依据当前视觉上下文和状态信息生成的 latent teacher
- `u_teacher_hat_T` 是 teacher 侧用 `u_teacher + z_teacher` 通过共享 world decoder 预测出的未来 latent，用于计算 `L_teacher_wm`

这里需要明确：

- `u_T` 不作为 `TeacherEncoder` 的输入
- `u_T` 只作为 teacher 侧 `L_teacher_wm` 的未来 latent 监督目标
- student 侧仍然使用同一个 `u_T` 计算 `L_wm(u_hat_T, u_T)`

这条分支只在训练时存在，用于提供基于未来信息构造的 latent supervision。

---

## 4. 训练时输入是什么

训练时，整套系统建议看到的 batch 至少包括：

- `obs`
  - 当前观测
  - 一般可以是当前时刻多视角图像
- `video`
  - 用于 world modeling 和 teacher 分支的视频片段
  - 第一版要求覆盖当前 action chunk 对应的时间窗口 `[t0, t0 + chunk_horizon]`
  - `video_length` 跟随训练参数 `chunk_size` 构造
- `lang`
  - 任务语言指令
- `state_0`
  - 当前状态
- `state_T`
  - 未来目标终态
- `action`
  - 动作标签

可选字段：

- `state_mask`
- `delta_proprio = state_T - state_0`
- `embodiment_id`

这里把它们写成“可选”，不是说它们没有价值，而是说：

- 第一版最小可行系统不一定依赖这三个字段才能成立
- 它们更多属于增强训练鲁棒性、跨 embodiment 泛化、或状态监督的附加结构

更具体地说：

### `state_mask` 什么时候可省，什么时候建议保留

`state_mask` 的作用是：

- 标记哪些 state 维度有效
- 避免无效维度进入状态监督或 teacher 编码器

如果你当前的数据满足下面这些条件：

- 只有一种机器人
- state 维度固定
- 每一维都有效
- 没有缺失字段

那么第一版可以默认：

- 所有 state 维度都有效

这时 `state_mask` 可以先省略。

但如果你后面会遇到：

- 多机器人混合训练
- 不同数据源 state 维度语义不完全一致
- 某些维度缺失或只在部分样本中可用

那就建议保留 `state_mask`。

### `delta_proprio` 什么时候可省，什么时候建议保留

`delta_proprio` 的作用是：

- 显式表示 `state_T - state_0`
- 给 `L_state` 提供监督目标

如果第一版里你先不开：

- `L_state`

那么 `delta_proprio` 就不是核心必需字段，可以先不构造。

但如果你后面打开：

- `L_state`

那么 `delta_proprio` 基本就会变成必需项，因为这时模型需要一个明确的“状态变化量”监督目标。

### `embodiment_id` 什么时候可省，什么时候建议保留

`embodiment_id` 的作用是：

- 告诉 teacher / student 当前样本属于哪个机器人本体
- 帮助模型区分不同 embodiment 的状态转移模式

如果你当前只有：

- 单一机器人本体
- 单一状态空间

那么 `embodiment_id` 对第一版来说可以先省略。

但如果你后面会做：

- 多 embodiment 训练
- 多机器人联合训练
- 跨平台迁移

那么 `embodiment_id` 会变得更重要，建议尽早纳入 batch 定义。

如果写成一个更紧凑的集合形式，可以记成：

- 训练输入：
  - `{obs, video, lang, state_0, state_T, action}`

---

## 5. 推理时输入是什么

推理时不能再依赖未来信息，所以只能输入部署时能获取到的量：

- `obs`
- `lang`
- `state_0`

如果写成集合形式，就是：

- 推理输入：
  - `{obs, lang, state_0}`

注意，推理时**没有**：

- `state_T`
- `action`
- teacher 分支

部署配置必须显式关闭 VJEPA21 加载：

- `framework.privileged_latent.load_vjepa: false`

这样部署初始化时不加载或运行 VJEPA21；VJEPA21 只属于训练期 teacher / target encoder 路径。

---

## 6. 训练时输出是什么

训练时建议区分两类输出：

### 6.1 Student 主路径输出

- `z_pred`
  - student 预测出的 latent action
- `u_hat_T`
  - 由 latent world decoder 预测的未来 latent / latent subgoal
- `a_hat`
  - 最终动作预测

### 6.2 Teacher 辅助分支输出

- `z_teacher`
  - teacher latent
- `u_teacher_hat_T`
  - teacher 侧预测出的未来 latent，用于约束 `z_teacher`

所以训练时核心输出可以记成：

- `{z_pred, z_teacher, u_teacher_hat_T, u_hat_T, a_hat}`

如果后面加 state 辅助头，还可以再有：

- `delta_hat`

---

## 7. 推理时输出是什么

推理时只保留 student 主路径。

核心输出是：

- `a_hat`

可选中间输出：

- `z_pred`
- `u_hat_T`

所以推理时可以理解成：

- 最重要输出：
  - `a_hat`
- 可调试输出：
  - `{z_pred, u_hat_T}`

---

## 8. 中间过程怎么走

下面不用抽象箭头，而是直接按训练时和推理时各走一遍。

### 8.1 训练时的完整过程

训练时你有这些输入：

- `obs`
- `video`
- `lang`
- `state_0`
- `state_T`
- `action`

系统会同时跑 `teacher` 和 `student` 两条链，最后在 loss 处汇合。

#### 8.1.1 Teacher 分支先处理视频

teacher 分支先只处理：

- `video`

这里明确是：

- `video -> VJEPA21 -> video_latents`

注意：

- 只有 `video` 进入 `VJEPA21`
- `state_0` 和 `state_T` 不进入 `VJEPA21`
- `VJEPA21` 只做前向，不更新参数

第一版采用一个明确的 chunk 对齐约定：

- `chunk_size` 是训练参数
- `video_length` 跟随 `chunk_size` 构造
- `obs` 对应 chunk 起点 `t0`
- `state_0` 对应 chunk 起点 `t0`
- `action` 覆盖当前 chunk，即执行 `chunk_size` 个 action
- `state_T` 对应执行完当前 chunk 后的终态 `t0 + chunk_horizon`
- `video` 覆盖 `[t0, t0 + chunk_horizon]`

离散 step index 约定为：

- 若 `action_horizon = H`
- `action` 取 `base + [0, 1, ..., H-1]`
- `state_0` 取 `base + 0`
- `state_T` 取 `base + H`
- `video` 的首尾 endpoint 分别对齐 `base + 0` 和 `base + H`

也就是说，dataloader 需要保证：

- 当前样本里的 `obs / state_0 / action / state_T / video` 来自同一个时间窗口
- `video` 的第一帧对应当前观测附近
- `video` 的最后一帧对应执行完当前 action chunk 后的末端附近

如果原始视频长度是 `T_raw`，并且 `tubelet_size = 2`，那么经过 `VJEPA21` 后会得到：

- latent 时间长度 `T_latent = T_raw / 2`

第一版采用 endpoint latent 消费方案：

- `T_latent` 不固定写死，而是由视频超参数自动决定
- 期望关系为 `T_latent = num_frames // tubelet_size`
- 例如 `num_frames=8, tubelet_size=2` 时，`T_latent=4`
- 例如 `num_frames=4, tubelet_size=2` 时，`T_latent=2`
- 模型侧只消费首尾两个时间 latent
- `latent[0]` 表示当前 / chunk 起点
- `latent[-1]` 表示未来 / chunk 末端
- 中间时间 latent 暂不进入 loss 或 action head

这里不要求训练代码显式处理所有物理时间戳，但要求 dataloader 构造出的 `video` 已经和当前 action chunk 对齐，并且视频编码后能稳定得到当前和未来两个不同时间 latent。

#### 8.1.2 Teacher 从视频 latent 中切出当前部分和未来部分

接着把 `video_latents` 按时间切成两块：

- `u_teacher`
  - teacher 使用的当前部分视觉上下文
- `u_T`
  - 真实未来 latent target

第一版固定采用最简单的终态 latent 对齐：

- `u_teacher = video_latents` 的第一个时间 latent / 当前 latent
- `u_T = video_latents` 的最后一个时间 latent / chunk 末端 latent
- `T_latent` 根据 `num_frames // tubelet_size` 自动得到，只要求 `T_latent >= 2`

这里：

- `u_teacher` 不是输出给 student 的目标
- `u_teacher` 是 teacher 内部用来生成 `z_teacher` 的当前视觉条件
- `u_T` 是后面拿来监督 teacher future latent 和 student future latent 的目标
- `u_T` 的语义固定为“当前 action chunk 末端的未来 latent”

这样即使 `chunk_size` 作为训练参数变化，也不会破坏语义：

- `chunk_size` 变大，`video_length` 跟着变长
- `video` 仍然只抽取 chunk 起点和 chunk 末端两个 endpoint latent
- `u_T` 仍然取编码后视频的最后一个时间 latent
- 因此 `u_T` 始终代表当前训练样本对应 action chunk 的末端

#### 8.1.3 Teacher 结合未来信息生成 `z_teacher`

然后 teacher 把下面这些量融合起来：

- `u_teacher`
- `state_0`
- `state_T`
- 可选 `embodiment_id`

经过：

- `TeacherEncoder`

得到：

- `z_teacher`

这一步的含义是：

- teacher 只在当前视觉上下文和真实状态条件下生成 `z_teacher`
- teacher 不直接读取 `u_T`
- 真实未来 latent `u_T` 只在后续 `L_teacher_wm` 中作为监督目标

teacher 分支训练时最重要的产物是：

- `z_teacher`
- `u_T`
- `u_teacher_hat_T`

#### 8.1.3.1 Teacher latent 如何被真实监督锚定

需要特别说明的是：

- `z_teacher` 本身不是数据集里天然存在的 ground truth
- 数据里有真实未来视频 latent `u_T`
- 数据里有真实动作 `action`
- 数据里可以有真实状态变化 `state_T - state_0`
- 但数据里没有一个直接标注好的“正确 `z_teacher`”

所以第一版不能把 `z_teacher` 简单当成天然标签。

本方案采用的第一版解决方法是：

- 用真实未来视觉 latent `u_T` 锚定 teacher
- 让 teacher 生成的 `z_teacher` 必须能够和当前 teacher 视觉上下文 `u_teacher` 一起预测 / 重建 `u_T`

也就是增加 teacher 自身的 world-model 约束：

- `u_teacher + z_teacher -> SharedWorldDecoder -> u_teacher_hat_T`
- `L_teacher_wm = loss(u_teacher_hat_T, u_T)`

这里的含义是：

- `u_T` 来自冻结的 `VJEPA21` 对真实未来视频的编码
- 因此 `u_T` 是稳定的未来 latent target
- `z_teacher` 虽然没有直接 ground truth，但它必须服务于预测 `u_T`
- 这样 `z_teacher` 就被真实未来信息间接固定住，而不是一个悬空 latent

训练 student 时，再用这个已经被 `u_T` 锚定过的 teacher latent 去监督 `z_pred`：

- `L_distill = loss(z_pred, stopgrad(z_teacher))`

这里的 `stopgrad(z_teacher)` 表示：

- `L_distill` 只更新 student 侧
- 不通过 distill loss 反向更新 `TeacherEncoder`
- `TeacherEncoder` 主要通过 `L_teacher_wm` 获得训练信号

所以第一版 teacher / student 的关系可以理解成：

- `u_T` 锚定 teacher
- teacher latent `z_teacher` 锚定 student latent `z_pred`
- student 再通过 `z_pred -> u_hat_T -> a_hat` 服务最终动作预测

#### 8.1.4 Student 分支先处理当前观测和语言

student 分支不看未来，只处理：

- `obs`
- `lang`
- `state_0`

先做：

- `obs + lang -> Qwen3-VL / student backbone -> current context`

这里的 `current context` 可以理解成：

- 当前视觉
- 当前语言
- 当前任务语义

融合后的上下文表示。

#### 8.1.5 Student 从当前上下文中得到 `u` 和 `z_pred`

接下来 student 从 `current context` 中抽取两个不同用途的量。

第一条支路：

- `current context -> pooling -> StudentCurrentAdapter -> u`

这里的 `u` 表示：

- 给 world decoder 使用的当前上下文 latent

第二条支路：

- `current context + state_0 -> StudentPredictor -> z_pred`

这里的 `z_pred` 表示：

- student 在看不到未来的条件下，对 latent action 的预测

#### 8.1.6 Student 用 `u + z_pred` 预测未来 latent

有了：

- `u`
- `z_pred`

之后，student 经过共享 world decoder：

- `u + z_pred -> SharedWorldDecoder -> u_hat_T`

这里的 `u_hat_T` 表示：

- student 预测出来的 future latent
- 也可以理解成 latent subgoal

#### 8.1.7 Student 用 `u_hat_T` 预测动作

最后 action head 再结合：

- `current context`
- `u_hat_T`
- `z_pred`
- `state_0`

输出：

- `a_hat`

可以写成：

- `a_hat = ActionHead(current context, u_hat_T, z_pred, state_0)`

#### 8.1.8 训练时两条分支如何汇合

到这里，teacher 分支给出：

- `z_teacher`
- `u_T`

student 分支给出：

- `z_pred`
- `u_hat_T`
- `a_hat`

然后在三个地方汇合：

第一处，latent 层：

- `z_teacher` 监督 `z_pred`
- 对应 `L_distill(z_pred, stopgrad(z_teacher))`
- 这条 loss 只更新 student 侧，不通过 `z_teacher` 回传到 teacher

第二处，future latent 层：

- `u_T` 监督 `u_hat_T`
- 对应 `L_wm(u_hat_T, u_T)`

第三处，teacher 自身的 future latent 层：

- `u_T` 监督 teacher 侧预测出来的 `u_teacher_hat_T`
- 对应 `L_teacher_wm(u_teacher_hat_T, u_T)`
- 这条 loss 用来让 `z_teacher` 被真实未来 latent 锚定

第四处，动作层：

- `action` 监督 `a_hat`
- 对应 `L_act(a_hat, action)`

如果启用状态辅助头，还会再有：

- `delta_proprio = state_T - state_0`
- `delta_hat`
- 对应 `L_state(delta_hat, delta_proprio)`

另外 student latent 上还会有：

- `L_latent = mean(||z_pred||^2)`

### 8.2 推理时的完整过程

推理时 teacher 分支完全删除，只保留 student 主路径。

推理输入只有：

- `obs`
- `lang`
- `state_0`

推理流程是：

1. `obs + lang -> Qwen3-VL / student backbone -> current context`
2. `current context -> pooling -> StudentCurrentAdapter -> u`
3. `current context + state_0 -> StudentPredictor -> z_pred`
4. `u + z_pred -> SharedWorldDecoder -> u_hat_T`
5. `current context + u_hat_T + z_pred + state_0 -> ActionHead -> a_hat`

推理时完全没有：

- `video`
- `state_T`
- `z_teacher`
- teacher 分支

---

## 9. Loss 应该怎么放

这个方案可以包含 6 个子 loss：

- `L_act`
- `L_wm`
- `L_teacher_wm`
- `L_state`
- `L_distill`
- `L_latent`

但它们**不是同等优先级**。第一版实现时，建议分成：

- 必需项
- 推荐项
- 可选项

总损失可以统一写成：

- `L_total = L_act + lambda_wm * L_wm + lambda_teacher_wm * L_teacher_wm + lambda_state * L_state + lambda_distill * L_distill + lambda_latent * L_latent`

但在第一版中，并不要求所有项都以同样重要的方式开启。

### 9.1 必需项

#### 9.1.1 `L_act`

定义：

- `a_hat` 与真实动作 `action` 的监督

作用：

- 保证最终控制能力

这是部署目标最直接的监督。

这是整个系统最不可缺少的目标之一。

#### 9.1.2 `L_wm`

定义：

- `u_hat_T` 与真实未来 latent `u_T` 的监督

作用：

- 保证 `z_pred` 和 decoder 确实编码了未来转移信息

这个项约束的是 world modeling 能力。

这是本方案保留 latent world model 结构意义的关键项。

如果没有 `L_act + L_wm`，这个方案基本就不再成立。

#### 9.1.3 `L_teacher_wm`

定义：

- `u_teacher_hat_T` 与真实未来 latent `u_T` 的监督

其中：

- `u_teacher_hat_T = SharedWorldDecoder(u_teacher, z_teacher)`
- `z_teacher = TeacherEncoder(u_teacher, state_0, state_T)`

作用：

- 用真实未来 latent `u_T` 锚定 `z_teacher`
- 避免 `z_teacher` 变成没有真实语义约束的悬空 latent
- 给 `TeacherEncoder` 提供独立训练信号

这是第一版 teacher 训练方式的核心。

因为 `z_teacher` 没有天然 ground truth，所以不能只依赖：

- `L_distill(z_pred, z_teacher)`

否则 teacher 和 student 可能只是互相靠近，而不是靠近某个真实目标。

更合理的梯度关系是：

- `L_teacher_wm` 更新 `TeacherEncoder` 和 teacher 侧 world decoder
- `L_distill` 使用 `stopgrad(z_teacher)`，只更新 student 侧

这样 teacher latent 的语义来源于真实未来 latent `u_T`，而 student latent 再去模仿这个被锚定过的 teacher latent。

### 9.2 推荐项

#### 9.2.1 `L_distill`

定义：

- `z_pred` 对齐 `z_teacher`

例如：

- `SmoothL1(z_pred, stopgrad(z_teacher))`

作用：

- 用 teacher 分支提供的未来信息监督去约束 student latent
- 让 student 在不看未来信息时，仍然学会输出有意义的 latent

这是这个方案和“纯端到端”最大的区别。

所以它虽然不属于“数学上绝对不能删”的项，但在**这个方案的设计哲学上非常重要**。

如果去掉 `L_distill`，teacher 分支的存在价值会明显下降，这个方案会更像：

- 带 `wm loss` 的普通端到端 policy 训练

#### 9.2.2 `L_latent`

这里默认采用我们前面已经确定的方案：

- `continuous latent + latent norm penalty`

例如：

- `L_latent = mean(||z_pred||^2)`

如果你愿意，也可以对 `z_teacher` 加同类正则，但第一版建议优先约束 student latent。

作用：

- 控制 latent 幅值
- 防止 latent 发散
- 避免 decoder 把 latent 当成高带宽捷径

这个项通常不需要太大权重，它更像一个：

- latent 稳定化项

所以它是推荐保留，但通常用小权重即可。

### 9.3 可选项

#### 9.3.1 `L_state`

定义：

- 如果加 state 辅助头，则用 `delta_hat` 对齐 `delta_proprio = state_T - state_0`

作用：

- 让 latent 更贴近真实机器人状态变化

这个项可以防止 latent 只学到视觉表面变化。

但在第一版实现里，它最适合先作为：

- `optional switch`

也就是：

- 第一版可以先关闭
- 跑通后再打开做 ablation

原因是：

- 它会增加一个额外的 state 辅助头
- 也会增加一条监督链
- 虽然它通常有帮助，但不是第一版最先要保证的部分

### 9.4 第一版建议保留哪些 loss

如果你现在想先做一个最小可行的版本，我建议：

#### 第一版默认开启

- `L_act`
- `L_wm`
- `L_teacher_wm`
- `L_distill`
- `L_latent`

#### 第一版默认关闭

- `L_state`

也就是说，第一版更推荐：

- `L_total = L_act + lambda_wm * L_wm + lambda_teacher_wm * L_teacher_wm + lambda_distill * L_distill + lambda_latent * L_latent`

第一版默认权重建议为：

- `lambda_act = 1.0`
- `lambda_wm = 0.5`
- `lambda_teacher_wm = 0.1`
- `lambda_distill = 0.1`
- `lambda_latent = 0.1`
- `lambda_state = 0.0`

也就是：

- `L_total = 1.0 * L_act + 0.5 * L_wm + 0.1 * L_teacher_wm + 0.1 * L_distill + 0.1 * L_latent`

这个权重设定的倾向是：

- 明确突出最终动作预测
- 保留较强但不压过 action 的 student world-model 监督
- teacher 锚定、distill、latent 正则作为辅助约束
- `L_state` 第一版关闭

等主链跑通以后，再把 `L_state` 打开做进一步比较。

### 9.4.1 第一版模块训练 / 冻结表

第一版建议：

- `Qwen3-VL / student backbone` 全量训练
- `StudentCurrentAdapter` 训练
- `StudentPredictor` 训练
- `SharedWorldDecoder` 训练
- `TeacherEncoder` 训练
- `ActionHead` 训练
- `VJEPA21` 冻结
- `u_T` 作为 target 使用 `detach`

其中：

- `StudentPredictor` 必须训练，因为它负责生成 student latent action `z_pred`
- `SharedWorldDecoder` 必须训练，因为 teacher 和 student 都通过它预测未来 latent
- `TeacherEncoder` 必须训练，因为 `z_teacher` 没有天然 ground truth，需要通过 `L_teacher_wm` 学出来
- `u_T.detach()` 的作用是明确 `u_T` 是冻结 `VJEPA21` 产生的监督目标，不参与反向传播
- 如果 `VJEPA21` 已经在 `no_grad` 下运行，`u_T.detach()` 对数值结果基本没有额外影响，但能防止后续误开梯度导致 target 漂移

### 9.5 一个简单的优先级排序

如果一定要把这 5 个 loss 排序，我建议按下面理解：

1. `L_act`
2. `L_wm`
3. `L_teacher_wm`
4. `L_distill`
5. `L_latent`
6. `L_state`

这不是说后面的项一定没用，而是说：

- 越靠前，越接近这个方案的核心成立条件
- 越靠后，越适合作为增强项和 ablation 项

---

## 10. Teacher / Student 变量之间如何监督和交互

这个方案里，teacher 分支和 student 分支虽然都产生了一些中间量，但**不是所有量都一一对齐**。

真正发生监督和交互的关系主要有三类：

- latent 层监督
- future latent 层监督
- action 层监督

下面把每个变量的角色分别说清楚。

### 10.1 Teacher 分支输出的角色

teacher 分支输出：

- `u_teacher`
- `u_T`
- `z_teacher`

其中：

#### 10.1.1 `u_teacher`

`u_teacher` 是 teacher 分支内部使用的“当前视觉上下文”。

它的主要作用是：

- 和 `state_0 / state_T`

一起输入到：

- `TeacherEncoder(u_teacher, state_0, state_T)`

最终生成：

- `z_teacher`

所以：

- `u_teacher` 本身**不直接监督** student 分支的某个量
- 它主要是 teacher 生成高质量 latent 的内部条件

也就是说，在第一版里：

- 不建议额外加 `u_teacher` 和 `u` 的直接对齐 loss

#### 10.1.2 `u_T`

`u_T` 是从真实未来视频 latent 中切出来的：

- 真实未来 latent target

它的作用是：

- 监督 student 分支预测出来的 `u_hat_T`

也就是：

- `L_wm(u_hat_T, u_T)`

所以：

- `u_T` 是 world model 的监督目标
- 它不直接作用在 `z_pred` 上

#### 10.1.3 `z_teacher`

`z_teacher` 是 teacher 分支最核心的输出：

- 基于未来信息构造的 latent teacher

但需要注意：

- `z_teacher` 没有天然 ground truth
- 它需要先通过 `L_teacher_wm` 被真实未来 latent `u_T` 锚定

teacher 侧先做：

- `u_teacher + z_teacher -> SharedWorldDecoder -> u_teacher_hat_T`
- `L_teacher_wm(u_teacher_hat_T, u_T)`

之后它再用于：

- 监督 student 分支预测出来的 `z_pred`

也就是：

- `L_distill(z_pred, stopgrad(z_teacher))`

所以：

- `z_teacher` 是 latent 层的 teacher target
- `L_teacher_wm` 负责训练和锚定 teacher
- `L_distill` 只负责把 teacher latent 蒸馏给 student

### 10.2 Student 分支输出的角色

student 分支输出：

- `u`
- `z_pred`
- `u_hat_T`
- `a_hat`

其中：

#### 10.2.1 `u`

`u` 是 student 分支从当前观测中提取出来的：

- 当前视觉 latent

它的作用是：

- 和 `z_pred` 一起作为 world decoder 的输入

也就是：

- `u + z_pred -> SharedWorldDecoder -> u_hat_T`

所以：

- `u` 不直接被 teacher 分支监督
- 它是 student world model 的当前条件输入

#### 10.2.2 `z_pred`

`z_pred` 是 student 分支最关键的 latent 预测结果。

它有两层作用：

- 第一层：
  - 被 `z_teacher` 监督
  - 对应 `L_distill`
- 第二层：
  - 作为 decoder 的条件输入
  - 决定 `u_hat_T`

所以：

- `z_pred` 是 teacher/student 桥梁里最核心的量

#### 10.2.3 `u_hat_T`

`u_hat_T` 是 student 用：

- `u`
- `z_pred`

预测出来的未来 latent / latent subgoal。

它的作用是：

- 被真实未来 latent `u_T` 监督
  - 对应 `L_wm`
- 作为 action head 的输入之一
  - 影响最终动作 `a_hat`

所以：

- `u_hat_T` 既承担 future prediction 角色
- 又承担 subgoal 角色

#### 10.2.4 `a_hat`

`a_hat` 是 student 分支的最终动作输出。

它不直接和 teacher 分支某个量对齐。

它只受真实动作标签监督：

- `L_act(a_hat, action)`

但是 teacher 分支会通过下面这条链路间接影响它：

- `z_teacher -> z_pred -> u_hat_T -> a_hat`

### 10.3 四条真正的监督链

把整件事压缩后，可以得到四条最关键的监督关系。

#### 10.3.1 teacher 自身的 future latent 监督

- `u_T` 监督 teacher 侧的 `u_teacher_hat_T`

对应：

- `L_teacher_wm(u_teacher_hat_T, u_T)`

这条监督链负责把没有天然 ground truth 的 `z_teacher` 锚定到真实未来 latent 上。

#### 10.3.2 latent distillation 监督

- `z_teacher` 监督 `z_pred`

对应：

- `L_distill(z_pred, stopgrad(z_teacher))`

这是 teacher/student 结构最核心的一条监督链。

#### 10.3.3 student future latent 层监督

- `u_T` 监督 `u_hat_T`

对应：

- `L_wm(u_hat_T, u_T)`

这条监督链保证 student 预测出来的 latent 真的能对应正确未来。

#### 10.3.4 action 层监督

- `action` 监督 `a_hat`

对应：

- `L_act(a_hat, action)`

这条监督链保证最终控制任务仍然是主目标。

### 10.4 哪些量只做内部条件，不直接对齐

第一版里，下面这些量建议只做内部条件，不额外加直接对齐损失：

- `u_teacher`
- `u`

原因是：

- `u_teacher` 的职责是帮助 teacher 生成 `z_teacher`
- `u` 的职责是作为 student decoder 的当前条件
- 它们虽然都可以被理解成“当前视觉上下文”，但来源和语义位置并不完全一致

所以第一版更建议：

- 不额外引入 `L(u, u_teacher)`

否则会让结构复杂化，而且未必带来清晰收益。

### 10.5 一个最简对应表

- `u_teacher`
  - teacher 内部条件
  - 不直接监督 student
- `u_T`
  - 真实未来 latent target
  - 监督 `u_hat_T`
- `z_teacher`
  - 基于未来信息构造的 latent teacher
  - 先被 `L_teacher_wm` 通过 `u_T` 锚定
  - 再通过 `stopgrad(z_teacher)` 监督 `z_pred`
- `u`
  - student 当前视觉 latent
  - 和 `z_pred` 一起输入 decoder
- `z_pred`
  - student latent action
  - 被 `z_teacher` 监督
- `u_hat_T`
  - student 预测未来 latent / latent subgoal
  - 被 `u_T` 监督
- `a_hat`
  - 最终动作输出
  - 被真实动作监督

---

## 10. 这个方案和真正两阶段方案的区别

### 10.1 相同点

- 都有 latent teacher / student 的思想
- 都有 world model 结构
- 都希望 latent 不只是 action head 的副产物

### 10.2 不同点

真正两阶段方案是：

- 先单独训练 Stage 1
- 再单独训练 Stage 2

而这个端到端方案是：

- 只训练一个系统
- teacher 分支和 student 主路径联合优化

所以这个方案可以理解为：

- 两阶段思想的端到端化

---

## 11. 这个方案的优点

- 比纯端到端更稳
- 比完整两阶段更省工程量
- 部署路径清晰
- teacher 分支只在训练时存在，不污染推理流程
- 能利用训练时额外可见的未来信息，例如 `state_T` 和未来视频对应的 latent

---

## 12. 这个方案的缺点

- 训练逻辑比纯端到端更复杂
- 需要同时维护 student 和 teacher 两条分支
- `distill` 权重需要调
- 如果 teacher 太强、数据又小，student 可能更容易过拟合训练样本上的 teacher 输出

---

## 13. 我建议的第一版实现原则

为了让第一版尽可能稳，我建议：

- `VJEPA21` 全程冻结
- 先做单视角版本
- `video_length` 跟随 `chunk_size` 构造，并覆盖当前 action chunk
- `u_teacher` 取 `video_latents` 的第一个时间 latent
- `u_T` 取 `video_latents` 的最后一个时间 latent
- teacher 分支输出 `z_teacher` 和 `u_teacher_hat_T`
- 用 `L_teacher_wm(u_teacher_hat_T, u_T)` 锚定 `z_teacher`
- student 分支输出 `z_pred -> u_hat_T -> a_hat`
- `L_distill` 用 `SmoothL1(z_pred, stopgrad(z_teacher))`
- `L_latent` 用小权重 norm penalty
- `L_state` 先做成可开关项

---

## 14. 一个最简训练数据流总结

训练时：

1. `obs + lang + state_0 -> z_pred`
2. `video + state_0 + state_T -> z_teacher`
3. `SharedWorldDecoder(u_teacher, z_teacher) -> u_teacher_hat_T`
4. `SharedWorldDecoder(u, z_pred) -> u_hat_T`
5. `ActionHead(...) -> a_hat`
6. 用
   - `L_teacher_wm(u_teacher_hat_T, u_T)`
   - `L_distill(z_pred, stopgrad(z_teacher))`
   - `L_wm(u_hat_T, u_T)`
   - `L_act(a_hat, action)`
   - `L_latent(z_pred)`
   - 可选 `L_state`
   联合训练

推理时：

1. `obs + lang + state_0 -> z_pred`
2. `SharedWorldDecoder(u, z_pred) -> u_hat_T`
3. `ActionHead(...) -> a_hat`

teacher 分支在推理时完全删除。

---

## 15. 当前文档的定位

这份文档是一个**方案草案**，不是最终实现说明。

它的作用是帮助我们在真正改代码之前先统一这几件事：

- 输入到底有哪些
- 哪些输入只在训练时存在
- 哪些路径属于部署主链
- 哪些 loss 是核心项
- `VJEPA21` 在整个系统中扮演什么角色

如果你认可这个方向，下一步最自然的工作就是继续细化成：

- 模块级设计
- 代码落点
- 训练脚本修改方案
- 与当前 `VLA-JEPA-DEV` 现有模块的映射关系

---

## 16. 尽可能复用现有代码的实现约束

> **重点工程原则：本方案不是只基于 `VLA-JEPA-DEV` 单独设计，而是基于以下两个现有项目共同制定：**
>
> - `/home/d013/桌面/project/LaWAM-main`
> - `/home/d013/桌面/project/VLA-JEPA-DEV`
>
> 后续实现必须尽可能复用这两个项目已经存在的代码、数据结构、模块接口、配置方式和训练约定。新增代码应优先采用：
>
> - 直接复用
> - 薄 wrapper
> - adapter
> - 最小接口扩展
>
> 不应在 `VLA-JEPA-DEV` 中重新实现一套与 `LaWAM-main` 平行的 latent action / latent world-model 系统。

这条原则意味着：

- 在新增模块前，先检查两个项目是否已经存在等价实现
- 优先复用已有模块，而不是复制代码后形成两个分叉版本
- 如果两个项目的接口不一致，优先增加 adapter 或 wrapper
- 如果两个项目的配置、batch schema 或 checkpoint 约定不一致，优先保留已有约定，并在边界处做转换
- 尽量保持现有 checkpoint、训练脚本、部署接口和数据统计逻辑的兼容性
- 只有在两个项目都没有可复用实现时，才新增全新模块

### 16.0 两个项目的职责分工

当前方案的代码复用方向建议如下。

`LaWAM-main` 优先作为以下内容的参考和复用来源：

- latent action model 的 encoder / decoder 结构
- latent world-model 的模块组织方式
- future video latent 和 start / end state 的 batch contract
- `states`、`state_mask`、`delta_proprio` 等状态数据组织
- latent action model 的 dataloader、collator 和 cache 逻辑
- latent-world runtime、component、contract 和 output mapping
- 已有的 latent action / latent subgoal 训练约定

重点检查路径包括：

- `latent_action_model/core/`
- `latent_action_model/data_loader/`
- `starVLA/model/framework/latent_world/`
- `starVLA/model/framework/lawam_framework.py`
- `starVLA/model/framework/vlas/lawam.py`
- `starVLA/dataloader/latent_world_train_collator.py`

`VLA-JEPA-DEV` 优先作为以下内容的参考和复用来源：

- Qwen3-VL 输入构造和 hidden-state 提取
- VJEPA21 encoder adapter 和冻结逻辑
- 现有 `VisionTransformerPredictorAC`
- 现有 Flow Matching / action head
- VLA forward、训练循环和 checkpoint 逻辑
- 当前 LeRobot 数据读取、action normalization 和部署接口

重点检查路径包括：

- `starVLA/model/framework/VLA_JEPA.py`
- `starVLA/model/modules/world_model/vjepa21_encoder.py`
- `starVLA/model/modules/world_model/vj2_predictor.py`
- `starVLA/model/modules/action_model/`
- `starVLA/dataloader/gr00t_lerobot/`
- `starVLA/training/train_vlajepa_cotrain.py`

### 16.0.1 两个项目之间的推荐组合

第一版推荐采用下面的组合关系：

- `VLA-JEPA-DEV` 的 Qwen3-VL 作为 student 主干
- `VLA-JEPA-DEV` 的 VJEPA21 adapter 作为冻结 teacher-side visual target encoder
- `LaWAM-main` 的 latent action / latent world-model 设计作为 latent 分支的结构参考
- `VLA-JEPA-DEV` 的 action head 作为最终动作生成器
- `LaWAM-main` 的 start / end state、mask 和 delta state batch 约定优先复用
- 两个项目中重复功能优先选择一个作为唯一实现，另一侧通过 adapter 接入

特别需要避免：

- 同时保留两套互相独立的 latent decoder
- 同时维护两套不兼容的 latent action 定义
- 为了适配新方案而复制整个 dataloader
- 在 student 推理路径中引入 `VJEPA21`
- 让 `LaWAM-main` 和 `VLA-JEPA-DEV` 各自维护一套不同的 loss / checkpoint 语义

因此，本方案后续的模块级设计和代码落点必须先回答：

- 这个功能在 `LaWAM-main` 是否已经存在？
- 这个功能在 `VLA-JEPA-DEV` 是否已经存在？
- 两边哪个版本更适合作为唯一实现？
- 是否只需要一个 adapter 就能连接两边？

只有在这些问题确认后，才允许新增对应模块。

### 16.1 现有代码中可以直接复用的部分

优先复用：

- `starVLA/model/framework/VLA_JEPA.py`
  - Qwen3-VL 输入构造
  - Qwen hidden states 提取
  - 视频 batch 整理
  - VJEPA21 视频编码
  - action head 调用
- `starVLA/model/modules/world_model/vjepa21_encoder.py`
  - VJEPA21 加载
  - VJEPA21 processor
  - `get_vision_features`
- `starVLA/model/modules/world_model/vj2_predictor.py`
  - `VisionTransformerPredictorAC`
  - transformer predictor block
  - latent token 的输入输出投影逻辑
- `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
  - 现有 Flow Matching action head
  - action chunk 预测
  - state condition 编码
- `starVLA/dataloader/gr00t_lerobot/`
  - LeRobot episode 读取
  - video / action / state / language 的现有样本组织
  - action 和 state normalization

### 16.2 Student 不使用 VJEPA21

需要保持一个明确约束：

- student 分支不调用 `VJEPA21`
- student 推理时不加载或运行 `VJEPA21`
- `VJEPA21` 只用于训练时生成 teacher 侧的 `u_teacher` 和 `u_T`

Student 当前上下文优先复用现有 Qwen 路径：

- `Qwen3-VL -> last_hidden / embodied_action_tokens`
- `embodied_action_tokens -> pooling -> u_raw`
- `u_raw -> StudentCurrentAdapter -> u`

这里优先使用现有的 `embodied_action_tokens` 做 pooling，因为它已经是当前代码中传给 action head 的任务条件表示，不需要重新定义一套视觉语言特征抽取路径。

### 16.3 第一版 latent shape 和 pooling

VJEPA21 当前输出是展平的时间和空间 token，形式近似：

- `[B, T_latent * N_spatial, D_vj]`

第一版固定采用 endpoint latent 消费方案：

- `T_latent` 由 `num_frames // tubelet_size` 自动决定
- 第一版只要求 `T_latent >= 2`
- `latent[0]` 表示当前 / chunk 起点
- `latent[-1]` 表示未来 / chunk 末端
- 不使用中间时间 latent

如果超参数产生 `T_latent > 2`，第一版仍然只取首尾：

- `latent[0]`
- `latent[-1]`

中间 latent 可以由 VJEPA21 编码产生，但暂不参与 world-model loss，也不进入 action head。

第一版只增加 reshape 和空间 pooling：

- `video_latents -> [B, T_latent, N_spatial, D_vj]`
- `u_teacher = mean(video_latents[:, 0], dim=spatial)`
- `u_T = mean(video_latents[:, -1], dim=spatial).detach()`

建议第一版固定使用：

- `u_teacher`: `[B, D_vj]`
- `u_T`: `[B, D_vj]`
- `u_hat_T`: `[B, D_vj]`
- `u_raw`: `[B, D_qwen]`
- `u`: `[B, D_vj]`
- `z_pred`: `[B, D_z]`
- `z_teacher`: `[B, D_z]`

其中：

- `D_qwen` 是 Qwen3-VL hidden size
- `D_vj` 是 VJEPA21 hidden size
- `StudentCurrentAdapter` 负责从 `D_qwen` 映射到 `D_vj`

第一版优先使用单视角，避免当前代码中多视角沿 feature dimension 拼接后造成额外 shape 复杂度。

`T_latent >= 2` 的必要性不是为了增加模型复杂度，而是为了保证 world-model loss 真的在学习：

- 当前 latent -> 未来 latent

如果 `T_latent = 1`，那么：

- `u_teacher` 和 `u_T` 会来自同一个时间位置
- `L_wm` 会退化成当前重建
- `SharedWorldDecoder` 可能学成近似 identity mapping
- `z_teacher / z_pred` 不必表达真实动作变化

因此第一版不把 `T_latent` 固定成某个常数，而是让它跟随超参数自动匹配。只要 `T_latent >= 2`，模型侧就使用首尾两个 latent；如果少于 2，则拒绝该样本或调整采样配置。

这里区分“模块外部接口”和“decoder 内部接口”：

- `TeacherEncoder`、`StudentPredictor` 和 loss 使用 pooled vector
- `u_teacher`、`u_T`、`u`、`u_hat_T` 的公开接口先统一为 `[B, D_vj]`
- `SharedWorldDecoder` 内部为了复用 `LAMDecoder_v2`，把 `[B, D_vj]` 临时扩展为 `[B, 1, K, D_vj]`
- decoder 输出后再对空间 token pooling 回 `[B, D_vj]`

第一版不把空间 token 暴露给 action head，也不让 action head 依赖 VJEPA21 的 token 数量。这样可以保持当前 Qwen/action head 的 pooled-condition 习惯，同时保留后续升级到空间 token 预测的可能性。

### 16.3.1 `D_z` 的第一版取值

`D_z` 是 latent action `z_pred / z_teacher` 的维度，不需要等于 `D_vj` 或 `D_qwen`。它是一个单独的瓶颈维度：

- `D_vj`：冻结 VJEPA21 的视觉特征维度
- `D_qwen`：Qwen3-VL hidden dimension
- `D_z`：student/teacher latent action 的维度

第一版建议：

- `D_z = 32`
- 参考 `LaWAM-main` 现有 `code_dim=32` 的配置
- `TeacherEncoder` 和 `StudentPredictor` 都输出 `[B, 32]`
- `SharedWorldDecoder` 内部通过 `latent condition projection` 将 `32` 映射到 decoder hidden dimension
- action head 通过独立 projection 将 `z_pred` 映射到 `D_qwen`

选择 `32` 的原因是：

- 与现有 LaWAM latent code 约定一致
- 参数量小，适合作为第一版 latent bottleneck
- 不会把 VJEPA21 的高维视觉表征直接复制成控制 latent

后续如果实验表明容量不足，只修改配置中的 `D_z` 以及对应 projection 层，不改变 batch schema 和 loss 语义。

### 16.4 SharedWorldDecoder 的复用方式

teacher 和 student 必须共享同一个 world decoder：

- `SharedWorldDecoder(u_teacher, z_teacher) -> u_teacher_hat_T`
- `SharedWorldDecoder(u, z_pred) -> u_hat_T`

第一版最终建议优先复用：

- `LaWAM-main/latent_action_model/core/utils/lam_decoder.py`
- 其中的 `LAMDecoder_v2`

原因是 `LAMDecoder_v2` 的接口语义正好是：

- 当前视觉特征
- latent code 条件
- 预测未来视觉特征

它比 `VisionTransformerPredictorAC` 更接近当前方案的：

- `u + z -> u_hat_T`

`VisionTransformerPredictorAC` 暂不作为第一版 `SharedWorldDecoder` 的实现。它更适合：

- 保留完整 VJEPA 时空 patch token
- 使用按时间排列的 action tokens
- 进行视频 token 级、带时间因果结构的预测

而当前方案的 `z_teacher / z_pred` 是单个 pooled latent code，不是 `VisionTransformerPredictorAC` 所要求的 `[B, T * num_action_tokens, D]` 动作 token 序列。直接套用会引入不必要的时间 token 和 attention mask 复杂度。

因此第一版使用一个薄的 `SharedWorldDecoder` wrapper 适配 `LAMDecoder_v2`：

- 输入：
  - `u`: `[B, D_vj]`
  - `z`: `[B, D_z]`
- wrapper 将 `u` 投影并扩展为 decoder 所需的 `[B, 1, K, D_vj]`
- wrapper 将 `z` 投影为 `LAMDecoder_v2` 的 latent condition
- `LAMDecoder_v2` 输出 `[B, 1, K, D_vj]`
- wrapper 对空间 token 做 mean pooling，得到 `[B, D_vj]` 的 `u_hat_T`

teacher 和 student 必须调用同一个 decoder 实例和同一组参数。后续如果需要恢复完整空间 token 监督，只扩展 wrapper 的内部接口，不改变 teacher/student 的 loss 语义。

真正需要新增的 decoder 相关代码只有：

- `SharedWorldDecoder` wrapper
- `D_z -> decoder hidden dimension` 的 projection
- `D_vj -> decoder context dimension` 的 projection（如果配置维度不同）

### 16.5 Action head 的复用方式

现有 action head 已经接收 Qwen hidden-state condition 和 state condition，不建议重新写 action generator。

现有 `FlowmatchingActionHead` 的 `encoder_hidden_states` 最终需要匹配 action head 配置中的 cross-attention hidden dimension。当前配置中：

- Qwen hidden dimension：`D_qwen = 2048`
- action head cross-attention dimension：`2048`

因此新的 future latent condition 先通过独立 projection 映射到 `D_qwen`，再追加到现有 condition token 序列：

- `embodied_action_tokens`
- `u_hat_T` projection token
- `z_pred` projection token

然后继续复用：

- `ActionHead(condition_tokens, action, state_0)`

因此逻辑上等价于：

- `ActionHead(current context, u_hat_T, z_pred, state_0)`

拼接位置固定为：

- 保留现有 `embodied_action_tokens` 的顺序
- 在其末尾追加 `u_hat_T` token 和 `z_pred` token
- 不把 `state_0` 拼到 VLM condition token 中，继续使用现有 action head 的 state encoder

因此：

- `ActionHead(current context, u_hat_T, z_pred, state_0)`
- 在代码中等价于：
  - `FlowmatchingActionHead(vl_embs=concat(current_tokens, future_tokens), state=state_0)`

不改变现有 action chunk 输出格式，也不改变 action loss 的目标格式。

### 16.6 Dataloader 复用和必要扩展

优先继续使用当前 LeRobot dataloader，不新建独立 video dataset。

训练 batch 最终需要组织成：

- `image`
- `video`
- `lang`
- `action`
- `state_0`
- `state_T`

当前 dataloader 已经能够提供：

- 当前图像
- video window
- action chunk
- 当前 state
- language

本方案只考虑真实机器人数据，不引入人类数据 loader，也不保留当前 co-training 中面向人类视频数据的第二个 dataloader。

需要补充或明确：

- `state_T`
  - action chunk 时间窗口末端对应的机器人 state
- `video` 的时间范围
  - 覆盖当前 action chunk
- `video_length`
  - 由 `chunk_size` 和机器人 action 频率换算得到

时间定义固定为：

- 当前观测时刻为 `t0`
- action chunk 包含 `chunk_size` 个动作，覆盖区间 `[t0, t0 + chunk_size / action_hz)`
- `state_0` 取 `t0`
- `state_T` 取窗口末端 `tT = t0 + chunk_size / action_hz`
- video 覆盖 `[t0, tT]`
- `u_teacher` 取 video latent 的第一个时间 latent
- `u_T` 取 video latent 的最后一个时间 latent

对应到离散 step index：

- 若 `action_horizon = H`
- action 序列为 `base + [0, 1, ..., H-1]`
- `state_T` / video 末端应取 `base + H`
- 不应把 `state_T` 定义成 `base + H - 1`，否则它表示最后一个 action 所在时刻，而不是执行完 `H` 个 action 后的终态

这里的关键是使用时间戳换算，而不是假设“action 数量等于 video 帧数”。例如：

- action 频率为 `20 Hz`
- `chunk_size=40`
- action 窗口时长为 `2 秒`
- video 需要覆盖这 `2 秒`
- video 的原始帧数由相机 fps 决定，不要求等于 `40`

video 采样、state 读取和 action chunk 必须基于同一 episode 的时间戳。若 state 或 video 在 `tT` 没有精确采样点，优先使用最近的有效时间戳，不在模型 forward 中插值。

当前目标数据集 `/home/d013/桌面/project/VLA-JEPA-DEV/merged_dataset_001` 已检查过时间网格：

- dataset metadata `fps = 50`
- 所有 parquet 的 `timestamp` 单调递增
- 相邻 row 的 timestamp 间隔稳定为 `0.02s`
- `frame_index` 等于 `0..episode_length-1`
- video fps 全部为 `50`
- video frame count 全部等于 episode length
- 当前 `action_horizon = 40` 时，`timestamp[base + 40] - timestamp[base] = 0.8s`

因此在当前数据集上，可以采用离散 row index 对齐作为第一版实现：

- `action = base + [0, ..., H-1]`
- `state_T = base + H`
- `video_T = base + H`

这里的 row index 与真实物理时间网格等价。后续如果更换数据集，必须重新做同类 sanity check；不能默认所有 LeRobot 数据都满足这个条件。

episode 末尾不足一个完整 endpoint window 时，第一版不使用 padding 后的假未来 endpoint 参与训练。

本方案当前目标数据集 `/home/d013/桌面/project/VLA-JEPA-DEV/merged_dataset_001` 中，每个 episode 长度都大于当前 chunk：

- `action_horizon = 40`
- `min_episode_length = 439`
- 不存在整条 episode 因短于 chunk 而无法训练的情况

因此第一版采用更简单、更干净的策略：

- 只采样满足 `base_index + action_horizon < episode_length` 的起点
- `action` 仍然取 `base + [0, ..., H-1]`
- `state_T` / video 末端取 `base + H`
- episode 尾部无法提供真实 `state_T` / video endpoint 的起点直接跳过
- 不把复制出来的最后 state / 最后 frame 当成真实未来监督目标

这样做会丢掉每条 episode 尾部约 `H` 个起点，但不会污染：

- `L_wm`
- `L_teacher_wm`
- `L_distill`
- `z_teacher`
- `z_pred`
- `SharedWorldDecoder`

对于当前数据集，按 `H=40` 估算，尾部无效起点约占全部候选起点的 `5.16%`，可以接受。后续实现上建议在 dataloader 构建 `all_steps` 时直接预过滤这些起点，而不是运行时采到无效 endpoint 后再 retry。

对于 VJEPA21，dataloader 或 video adapter 必须保证：

- `T_latent = num_frames // tubelet_size`
- 第一版要求 `T_latent >= 2`
- `latent[0]` 来自 chunk 起点
- `latent[-1]` 来自 chunk 末端
- 第一个 latent 和最后一个 latent 不能来自同一时间位置

如果 tubelet 或采样频率导致 `T_latent < 2`，应增加 video 采样帧数或拒绝该样本，而不是在模型中复制同一个 latent 冒充未来。

如果超参数或底层 VJEPA21 / processor 产生 `T_latent > 2`，第一版仍然只使用：

- `video_latents[:, 0]`
- `video_latents[:, -1]`

中间 latent 暂不参与 loss，也不进入 action head。

#### 16.6.1 这个时间对齐问题为什么重要

这个问题对本方案是关键正确性问题，因为 `L_wm` 和 `L_teacher_wm` 都依赖 `u_T` 是真实 chunk 末端未来。

student world-model loss 是：

- `u + z_pred -> u_hat_T`
- `L_wm(u_hat_T, u_T)`

teacher world-model loss 是：

- `u_teacher + z_teacher -> u_teacher_hat_T`
- `L_teacher_wm(u_teacher_hat_T, u_T)`

如果 action chunk 覆盖未来 2 秒，但 `u_T` 实际来自未来 1.5 秒或 2.5 秒，loss 仍然可能下降，但模型学到的是错误时间关系。这会污染：

- `SharedWorldDecoder`
- `z_teacher`
- `z_pred`
- `L_distill`

对于只做 action supervised learning 的模型，这个问题可能没那么显眼；但本方案额外引入了 future latent supervision，因此必须保证 action、video 末端和 `state_T` 对齐。

#### 16.6.2 LaWAM-main 中可以复用的类似机制

`LaWAM-main` 中没有直接叫 `T_latent` 的变量，但已经有非常接近的 endpoint 思路。

可复用的 dataloader 机制包括：

- `LaWAM-main/starVLA/dataloader/lerobot_datasets.py`
- 通过 `sec_chunk + fps` 计算 action chunk 长度
- 从 action delta indices 中采样 video delta indices
- `_sample_video_delta_indices()` 强制保留第一个和最后一个时间点

这和本方案需要的语义一致：

- video 起点对应 action chunk 起点
- video 终点对应 action chunk 末端

可复用的 state batch contract 包括：

- `LaWAM-main/latent_action_model/data_loader/collate.py`
- `states: [B, 2, max_state_dim]`
- `states[:, 0]` 是 start state
- `states[:, 1]` 是 end state
- 同时提供 `state_mask`
- 同时提供 `delta_proprio = end - start`

可复用的模型侧 endpoint latent 习惯包括：

- `LaWAM-main/starVLA/model/framework/vlas/lawam.py`
- `h_t = features[:, 0, :, :]`
- `h_t1_gt = features[:, -1, :, :]`

以及：

- `LaWAM-main/latent_action_model/core/lam_model.py`
- V-JEPA split-feature path 使用 endpoint clip 的 `T=2` 思路可参考
- 但本方案不强制 `T_latent=2`，只强制首尾 endpoint 语义和 `T_latent >= 2`

因此第一版实现建议不是重新设计一套时间系统，而是复用 LaWAM 的 endpoint 思路，并在 VLA-JEPA-DEV 中补齐两点：

- `state_T` 明确定义为 action chunk 末端状态
- video adapter 明确保证 endpoint latent：`latent[0]` 当前，`latent[-1]` chunk 末端未来

当前代码中 `action_horizon` 和 `video_horizon` 是两个独立参数，后续需要在 dataloader 层明确它们之间的采样频率换算，而不是在模型 forward 中猜测时间对应关系。

第一版建议保留当前 LeRobot dataloader 的样本组织方式，但对 endpoint window 起点做显式过滤：

- `state_T` 的 endpoint 读取
- 覆盖 action chunk 的 video 时间窗口
- endpoint latent 检查，第一版要求 `T_latent >= 2`
- 有效位置 mask
- `all_steps` 中优先只加入满足 `base_index + action_horizon < episode_length` 的完整 endpoint 起点

### 16.7 复用原则总结

后续实现遵循：

- 不在 student 中新增 VJEPA21 调用
- 不重新实现 Qwen hidden-state 提取
- 不重新实现 action diffusion / flow-matching head
- 不重新设计 LeRobot 样本格式
- VJEPA21 latent 只增加 reshape 和 spatial pooling
- `SharedWorldDecoder` 优先包装 LaWAM-main 的 `LAMDecoder_v2`
- `VisionTransformerPredictorAC` 暂作为完整 VJEPA 时空 token predictor 的备选实现
- dataloader 只扩展 `state_T` 和 chunk 对齐逻辑

真正需要新增的模块限制为：

- `StudentCurrentAdapter`
- `TeacherEncoder`
- `StudentPredictor`
- `SharedWorldDecoder` wrapper
- `u_hat_T / z_pred -> Qwen hidden dimension` 的 action condition projection
- 新的 loss 汇总和训练输出

### 16.8 TeacherEncoder 和 StudentPredictor 的代码映射

这两个名字在两个项目中都没有一个可以直接原样替换的现成类，因此需要设计任务特化 wrapper，但不需要从零设计完整 latent action model。

#### `TeacherEncoder`

优先复用：

- `LaWAM-main/latent_action_model/core/utils/lam_encoder.py`
- 其中的 `LAMEncoder`
- `QFormer`
- `CategorySpecificMLP`
- state token projection

建议 wrapper 接口：

- `TeacherEncoder(u_teacher, state_0, state_T) -> z_teacher`

实现方式：

- 将 pooled `u_teacher` 恢复为单时间步视觉 token 输入
- 用 endpoint state adapter 编码 `state_0` 和 `state_T`
- 将 endpoint state 作为一个或两个 state token 注入 `LAMEncoder` 的 QFormer
- 通过 `out_proj` 输出 `[B, D_z]`

`u_T` 不进入 `TeacherEncoder`。它只在后续的：

- `L_teacher_wm(SharedWorldDecoder(u_teacher, z_teacher), u_T)`

中作为真实未来监督目标。

#### `StudentPredictor`

当前两个项目都没有一个完全匹配：

- Qwen current context
- `state_0`
- 输出单个 `z_pred`

的现成模块。

因此建议实现轻量 wrapper：

- `StudentPredictor(current_context_pool, state_0) -> z_pred`

内部优先复用：

- VLA-JEPA-DEV 现有 Qwen hidden-state pooling
- LaWAM 的 projection / `CategorySpecificMLP` 状态编码思路
- 必要时使用一到两层 MLP，而不是完整 LAMEncoder

student 不调用 VJEPA21，也不把未来视频或 `state_T` 暴露给 `StudentPredictor`。

### 16.9 联合训练代码落点

当前 VLA-JEPA-DEV 的训练流程存在两个 dataloader 和两次 optimizer step。新方案必须改为真机机器人单 batch 联合训练：

- 一个 robot dataloader
- 一个 batch 同时包含 `image / video / lang / action / state_0 / state_T`
- 一次 joint forward 同时得到 teacher 和 student 输出
- 计算一个 `L_total`
- 只执行一次 `backward()` 和一次 `optimizer.step()`

梯度路径固定为：

- `L_act`：更新 Qwen3-VL、StudentPredictor、SharedWorldDecoder、ActionHead
- `L_wm`：更新 StudentPredictor、StudentCurrentAdapter、SharedWorldDecoder
- `L_teacher_wm`：更新 TeacherEncoder、SharedWorldDecoder
- `L_distill`：只更新 StudentPredictor 主链，`z_teacher` 使用 `stopgrad`
- `L_latent`：只约束 `z_pred`，不把梯度传到 teacher
- `VJEPA21`：始终冻结并处于 `no_grad`

### 16.10 checkpoint 和部分加载策略

现有 VLA-JEPA checkpoint 不包含新增加的：

- `TeacherEncoder`
- `StudentPredictor`
- `SharedWorldDecoder`
- latent/action projection

因此加载旧 checkpoint 时采用：

- `strict=False`
- 旧有 Qwen、action head、VJEPA21 adapter 权重尽可能加载
- 新模块保持随机初始化
- 打印 missing keys 和 unexpected keys
- 第一版不允许静默忽略 shape mismatch

如果旧 checkpoint 中的 action head 输入 projection 维度与新配置不一致：

- 只跳过不兼容的 projection 参数
- 保留其余 action head 参数
- 记录明确的 partial-load 日志

`D_z=32` 会改变新 latent 分支的 projection 形状，但不会影响旧 Qwen 或 action head 的主体权重。后续修改 `D_z` 时，只需重新初始化对应 latent projection，不应尝试把不同 `D_z` 的权重强行 reshape。

---

## 17. 工程防坑清单：训练 / 仿真 / 真机部署

下面 5 个点是后续实现最容易踩坑、且会真实影响训练效果、仿真评测可信度和真机部署稳定性的地方。实现和 review 时需要重点检查。

### 17.1 时间对齐错了，但 loss 仍然可能下降

这是本方案最大的正确性风险。

如果：

- action chunk 覆盖未来 2 秒
- `u_T` 实际来自未来 1.5 秒或 2.5 秒
- `state_T` 又对应另一个时间点

那么 `L_wm` / `L_teacher_wm` 仍然可能下降，但模型学到的是错误时间关系。

影响：

- 训练时 latent world model 学偏
- 仿真评测中动作可能提前或滞后
- 真机上可能表现为反应慢半拍、过冲或到达错误 subgoal

必须检查：

- `obs` 和 `state_0` 对应 chunk 起点 `t0`
- `action` 覆盖 `[t0, tT)`
- `state_T` 对应 chunk 末端 `tT`
- video 覆盖 `[t0, tT]`
- endpoint latent 使用 `latent[0]` 和 `latent[-1]`
- `T_latent = num_frames // tubelet_size` 且必须 `>= 2`

对于当前 `merged_dataset_001`，row-index 对齐已经通过以下 sanity check：

- `timestamp[i+1] - timestamp[i] == 1 / fps`
- `timestamp[base + action_horizon] - timestamp[base] == action_horizon / fps`
- `frame_index == arange(episode_length)`
- video fps 与 metadata fps 一致
- video frame count 与 episode length 一致

后续训练启动时建议打印或断言这些统计，尤其是在更换数据集、修改 `action_horizon` 或修改 video backend 之后。

### 17.2 student 和 teacher latent 空间没有真正接上

teacher 侧：

- `u_teacher / u_T` 来自冻结 `VJEPA21`

student 侧：

- `u` 来自 Qwen3-VL current context pooling
- 再通过 `StudentCurrentAdapter` 映射到 shared world decoder 的输入空间

这两个空间天然不一致。如果 adapter 太弱、`L_wm` 训练不稳定，或者 shape/pooling 实现错了，student 分支可能学不到可被 `SharedWorldDecoder` 使用的 `u`。

影响：

- teacher 侧 `L_teacher_wm` 正常下降
- student 侧 `L_wm` 不稳定或长期不下降
- action head 可能绕开 latent 分支，只依赖 Qwen tokens
- 真机泛化时 latent subgoal 不可靠

必须监控：

- `L_teacher_wm`
- `L_wm`
- `norm(u)`
- `norm(u_T)`
- `norm(u_hat_T)`
- `cosine(u_hat_T, u_T)` 或同类相似度指标

### 17.3 auxiliary loss 抢主任务，导致 action 变差

本方案的主任务仍然是动作预测：

- `L_act`

下面这些都是辅助项：

- `L_wm`
- `L_teacher_wm`
- `L_distill`
- `L_latent`

如果辅助 loss 权重过大，模型可能优先学习预测 VJEPA latent，而不是学习执行正确动作。

影响：

- 总 loss 看起来更好
- action loss 下降变慢
- 仿真中视觉 latent prediction 合理，但动作质量一般
- 真机上动作可能更犹豫或更不精确

第一版默认权重保持：

- `1.0 * L_act`
- `0.5 * L_wm`
- `0.1 * L_teacher_wm`
- `0.1 * L_distill`
- `0.1 * L_latent`

训练时必须优先观察：

- `L_act` 是否被辅助 loss 压制
- 关闭辅助 loss 后 action baseline 是否更好
- 开启 latent 分支后仿真 success rate 是否真实提升

### 17.4 训练路径和推理路径不一致

teacher 分支只在训练时存在，真机部署时完全不存在。

训练时禁止把下面这些 teacher-only 或 future-only 信息直接喂给 student/action head：

- `state_T`
- `u_T`
- `z_teacher`
- `u_teacher`
- 未来 video
- teacher branch 输出

否则训练和评测结果可能虚高，但真机部署会掉性能。

部署路径只能依赖：

- `obs`
- `lang`
- `state_0`
- Qwen3-VL
- `StudentCurrentAdapter`
- `StudentPredictor`
- `SharedWorldDecoder`
- `ActionHead`

部署配置必须显式设置：

- `framework.privileged_latent.load_vjepa: false`

这表示部署时不加载、不初始化、不运行 VJEPA21；如果沿用训练配置导致部署进程仍加载 VJEPA21，应视为部署配置错误。

必须检查：

- `predict_action()` 不调用 `VJEPA21`
- `predict_action()` 不需要 `video`
- `predict_action()` 不需要 `state_T`
- action head 的部署输入和训练 student 路径一致
- 仿真评测代码不能偷偷使用 teacher 分支或 future target

### 17.5 action / state normalization 在训练、仿真、真机不一致

这是机器人部署中最容易出现“模型没错但执行全错”的坑。

必须确认训练、仿真和真机部署使用同一套语义：

- `action_dim`
- `state_dim`
- `action_horizon`
- `future_action_window_size`
- action normalization stats
- state normalization stats
- gripper 维度
- EEF / joint 顺序
- delta / absolute action 语义
- 坐标系定义

典型错误包括：

- 训练输出 normalized action，部署时没有反归一化
- 训练是 delta action，真机按 absolute action 执行
- gripper 维度位置不一致
- EEF rotation 表示不一致
- action horizon 和部署执行步数不一致

影响：

- 训练 loss 正常
- 仿真动作幅度异常
- 真机执行过大动作、方向错误或夹爪行为异常

实现时必须把 normalization / denormalization 明确放在同一套 policy interface 中，不允许训练、仿真和真机各自维护一套隐式转换逻辑。
