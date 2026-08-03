# LaWAM 各训练阶段与推理流程说明

这份文档从**代码实现**出发，梳理 `LaWAM-main` 的三个核心部分：

1. `Stage 1`：Latent Action Model / LaWM 预训练
2. `Stage 2`：LaWAM 策略训练
3. `Inference`：部署与推理

每一部分都重点回答三个问题：

1. 输入是什么？
2. 输出是什么？
3. 中间经过了什么过程？

注意：这里的描述以**仓库中的实际实现**为准，而不只是论文中的抽象公式。

## 总览

仓库可以粗分为两大块：

- `latent_action_model/`
  对应 `Stage 1`。训练 latent action model，并保留其 forward decoder 作为 latent world model。
- `starVLA/`
  对应 `Stage 2` 和推理。训练一个策略模型来预测 latent action、解码 latent visual subgoal，并最终生成 action chunk。

整体数据流可以先记成下面三句话：

- `Stage 1` 学的是：
  `当前视觉 latent + latent action -> 未来视觉 latent`
- `Stage 2` 学的是：
  `当前观测 + 指令 -> latent action -> latent subgoal -> action chunk`
- 推理时实际执行的是：
  `当前观测 + 指令 -> latent action -> latent subgoal -> action chunk`

## Stage 1：Latent Action Model / LaWM 预训练

### 代码入口

- 启动脚本：
  `latent_action_model/train.sh`
- Lightning CLI 入口：
  `latent_action_model/main.py`
- LightningModule：
  `latent_action_model/core/lam_lightinng.py`
- 主模型：
  `latent_action_model/core/lam_model.py`

### 这一阶段的目标

`Stage 1` 的目标是：从视觉状态转移中学习一个 latent action model。

训练结束后，会留下三类重要能力：

- latent inverse-dynamics 路径
  后面给 `Stage 2` 做 latent distillation teacher
- forward decoder
  后面直接作为 `LaWM` 使用
- frozen visual feature extractor
  后面继续用于构造当前 latent state 和未来 latent target

在代码里，最关键的可复用模块是：

- `latent_action_model.core.lam_model.LatentLAMModel.decoder`

也就是 `Stage 2` 里真正会复用的 latent world decoder。

### Stage 1 的训练输入

一个 batch 里主要包含：

- `videos`
  形状一般是 `[B, T, C, H, W]`
  表示当前 clip，给 latent inverse-dynamics encoder 使用
- `dec_videos`
  形状一般是 `[B, T, C, H, W]`
  表示 decoder 侧使用的 clip，用来构造当前 latent feature 和未来 latent target
- `states`
  形状是 `[B, 2, D]`
  只保留起始和终止 state / proprio
- `state_mask`
  形状是 `[B, 2, D]`
  标记哪些 state 维度有效
- `delta_proprio`
  形状是 `[B, D]`
  表示 `end - start`，给辅助状态损失使用
- `embodiment_ids`
  形状是 `[B]`
  表示 embodiment / robot id

相关 collate 逻辑在：

- `latent_action_model/data_loader/collate.py`

这里有两个实现细节很重要：

- batch 里保存的是**起点状态和终点状态**，不是整段完整状态序列
- `delta_proprio` 是显式构造出来的：
  `end - start`

### Stage 1 的中间过程

#### 1. 先提取视觉特征

模型会先调用冻结的视觉编码器，通常是 DINOv3，配置在：

- `latent_action_model/config/*.yaml` 里的 `vision_model_id`

在 `LatentLAMModel._run(...)` 中：

- `videos` 会被编码成 `enc_in`
- `dec_videos` 会被编码成：
  - `dec_in`
    当前 latent visual feature
  - `tgt`
    未来 latent visual feature target

从语义上理解：

- `enc_in`
  用来推断 latent action 的转移上下文
- `dec_in`
  当前 latent state，近似论文里的 `u`
- `tgt`
  未来 latent state，近似论文里的 `u_T`

#### 2. latent inverse dynamics 推断 latent action

主编码器会执行：

- `self.encoder(enc_in, states_for_model, embodiment_id=emb_tensor)`

得到：

- `nodes`
  形状一般是 `[B, num_queries, code_dim]`

然后这些 `nodes` 会经过 quantizer / VAE 风格 bottleneck，得到：

- `quantized`

这个 `quantized` 就是后面给 decoder 用的 latent action 表示。

从论文角度，可以把这一步近似看成：

- `z ~ q(z | u, u_T)`

#### 3. forward decoder 预测未来 latent state

如果开启未来预测：

- `recon = self.decoder(features=dec_in, actions=quantized)`

这里的含义是：

- 输入：
  当前 latent 特征 `dec_in` 和 latent action `quantized`
- 输出：
  预测的未来 latent 特征 `recon`

这一步就是代码里的 latent world prediction，对应论文里的：

- `LaWM(u, z) -> u_T`

#### 4. 辅助状态头预测 embodied motion

除了视觉未来预测外，模型还会计算：

- `s_pred = self.state_decoder(z_t=quantized, state_0=dec_states, embodiment_id=emb_tensor)`

不过在当前实现里，辅助损失并不是去拟合绝对的 `s_T`，而是和：

- `delta_proprio`

对齐。也就是说，这里的辅助目标更接近“状态变化量”而不是“绝对终态”。

### Stage 1 的训练输出

`Stage 1` 的主模型会返回：

- `recon`
  预测出来的未来 latent visual feature
- `tgt`
  真实未来 latent visual feature
- `quantized`
  latent action
- `s_pred`
  辅助状态预测输出
- `vq_loss`
  quantizer / latent bottleneck loss
- `perplexity`
- `entropy_loss`

然后 `lam_lightinng.py` 会把这些量组合成 loss，主要包括：

- `recon` 和 `tgt` 之间的 future latent reconstruction loss
- latent bottleneck / regularization loss
- 对 `delta_proprio` 的辅助状态损失

所以从代码实现角度，`Stage 1` 的训练目标可以概括成：

- 学会未来 latent visual feature 预测
- 约束 latent action 空间
- 用辅助状态损失让 latent action 更贴近真实 embodied motion

### Stage 1 最终留下什么给后续用

`Stage 2` 会复用 `Stage 1` 的三个部分：

- latent world decoder：
  `lam.decoder`
- latent teacher 路径：
  `lam.get_latent_action(...)`
- 视觉特征提取器：
  `lam.extract_vision_features(...)`

## Stage 2：LaWAM 策略训练

### 代码入口

- 启动脚本：
  `train_lawam.sh`
  `train_lawam_distributed.sh`
- 训练循环：
  `starVLA/training/train_starvla.py`
- framework 包装层：
  `starVLA/model/framework/lawam_framework.py`
- 核心策略后端：
  `starVLA/model/framework/vlas/lawam.py`
- 训练 batch collator：
  `starVLA/dataloader/latent_world_train_collator.py`

### 这一阶段的目标

`Stage 2` 的目标是训练一个真正面向测试时使用的 policy，它要完成三件事：

1. 从当前观测和语言指令中预测 latent action
2. 用 `Stage 1` 训练好的 decoder 把 latent action 解码成 latent visual subgoal
3. 基于语义上下文和这个未来 latent subgoal，生成 action chunk

### Stage 2 的训练输入

训练时，collator 会组装出一个 `LatentWorldPolicyTrainBatch`，其中主要包括：

- `pixel_values`
  Qwen3-VL 的图像输入
- `input_ids`
  文本和多模态 prompt 的 token ids
- `attention_mask`
  Qwen attention mask
- `act_placeholder_mask`
  latent-action query placeholder 所在位置
- `flow_placeholder_mask`
  action-flow query placeholder 所在位置
- `primary_video`
  形状一般是 `[B, T, C, H, W]`
  主视角视频，用来提取 latent-world 特征
- `state`
  形状一般是 `[B, D]`
  当前 state
- `state_mask`
  形状一般是 `[B, D]`
  哪些 state 维度有效
- `embodiment_id`
  形状是 `[B]`
- `action_hz`
  形状是 `[B]`
  控制频率
- `actions`
  形状一般是 `[B, action_horizon, action_dim]`
  目标 action chunk
- `actions_mask`
  形状一般是 `[B, action_horizon, action_dim]`
  action 的有效位置和维度掩码

从更高层看，`Stage 2` 的外部监督输入就是：

- 当前 observation video / images
- 语言 instruction
- 目标 action chunk
- 可选 state / embodiment / control frequency 元数据

### Stage 2 的中间过程

#### 1. Qwen3-VL 编码当前语义上下文

VLM 会吃进去：

- 图像
- 指令
- 插入到 prompt 中的 placeholder token

这里会注入两类 query：

- latent-action queries
- flow queries

Qwen forward 完成后，得到：

- `h_vlm`
  形状一般是 `[B, L, hidden_dim]`
  表示完整的 VLM hidden states

然后代码会从 `act_placeholder_mask` 对应的位置取出 hidden states，并送入：

- `VLMToLAMQFormer`

得到：

- `pred_action_emb`
  在正常配置下通常是 `[B, 1, lam_code_dim]`

这一步就是 `Stage 2` 里的 policy prior，对应：

- `p(z_hat | o, l)`

#### 2. 用 Stage 1 的 LAM teacher 做 latent distillation

训练时，代码还会调用预训练好的 `Stage 1` LAM：

- `self.lam.get_latent_action(...)`

得到 teacher latent action，然后计算：

- `loss_distill`

也就是让 `Stage 2` 预测出来的 latent action 去对齐 `Stage 1` teacher。

这一步的作用是：让 `Stage 2` 学会稳定地驱动 `Stage 1` 留下来的 decoder。

#### 3. 构造当前和未来的 latent visual feature

预训练好的 `Stage 1` 视觉特征提取器会被调用：

- `features = self.lam.extract_vision_features(primary_visual_input)`

然后从中取出：

- `h_t_original`
  当前 latent visual tokens
- `h_t`
  给 action head 用的当前 latent visual tokens
- `h_t1_gt`
  从训练 clip 中拿到的未来 latent visual tokens

接着，如果开启 future prediction：

- `h_t1_pred = self._decode_future_tokens_strict_single_query(h_t, pred_action_emb, ...)`

内部其实就是：

- `self.lam.decoder(h_t, pred_action_emb)`

这一步就是 `Stage 2` 真正复用 `Stage 1` decoder 来构造 latent visual subgoal。

#### 4. 对未来 latent subgoal 做监督

代码会计算：

- `loss_perceptual = mse(h_t1_pred, h_t1_gt)`

这一步对应的是：让 policy 驱动出来的 latent future 去贴近训练数据里的真实未来 latent。

#### 5. action head 学习生成 action chunk

最终动作专家是：

- `ConditionalFlowMatchingHead`

它接收的条件包括：

- `h_t`
  当前 latent visual tokens
- `h_t1_star`
  未来条件 latent tokens
  训练时这部分可以在 predicted / GT 之间按策略混合
- `h_vlm`
  来自 Qwen3-VL 的语义上下文
- `state`
- `actions`
- `action_hz`
- `embodiment_id`
- 各类 mask

它的训练输出是：

- `loss_flow`

这就是 action chunk generation 的主损失。

### Stage 2 的训练输出

`Stage 2` forward 最终返回：

- `loss_flow`
- `loss_perceptual`
- `loss_distill`
- `loss_total`

所以这一阶段的训练目标可以概括成三块：

- latent action distillation
- latent future / latent subgoal supervision
- action chunk generation loss

### Stage 2 最终学到的接口

训练完成后，实际可用的策略接口就是：

- `observation + language`
  -> `latent action`
  -> `latent visual subgoal`
  -> `action chunk`

如果后面你要改模型结构，这一层通常是最值得改的地方。

## Inference：推理与部署

### 代码入口

- framework 推理入口：
  `starVLA/model/framework/lawam_framework.py`
- inference batch builder：
  `starVLA/model/framework/latent_world/batch_builder.py`
- runtime runner：
  `starVLA/model/framework/latent_world/runtime/runner.py`
- policy backend：
  `starVLA/model/framework/vlas/lawam.py`
- 部署 server：
  `deployment/model_server/server_policy.py`

### 推理输入

公开推理接口使用的是 `LatentWorldPolicyInferExample`。
单个 example 主要包含：

- `primary_image`
  必填
  一个或多个当前 RGB 视角
- `lang`
  必填
  任务指令
- `action_hz`
  必填
  控制频率
- `embodiment_id`
  必填
- `state`
  可选
- `state_mask`
  可选
- `wrist_image`
  可选

组 batch 后，backend 实际收到的是：

- Qwen 输入：
  `pixel_values`, `input_ids`, `attention_mask`
- query placeholder mask：
  `act_placeholder_mask`, `flow_placeholder_mask`
- `primary_image`
  归一化后的当前图像张量
- `state`, `state_mask`
- `embodiment_id`
- `action_hz`

### 推理时的中间过程

#### 1. 用 Qwen3-VL 构造语义上下文

和 `Stage 2` 训练时一样，backend 会：

- 注入 latent-action query 和 flow query
- 跑一遍 Qwen3-VL
- 取出 latent-action query 对应的 hidden states

然后得到：

- `pred_action_emb = self.vlm_to_lam(...)`

这就是测试时预测出来的 latent action。

#### 2. 构造当前 latent visual feature

接着会调用 `Stage 1` 留下来的视觉提取器：

- `features = self.lam.extract_vision_features(primary_image)`

从而得到：

- `h_t_original`
- `h_t`

#### 3. 解码 latent visual subgoal

如果配置里开启 future prediction：

- `h_t1_pred = self.lam.decoder(h_t, pred_action_emb)`

这一步就是测试时真正使用的 latent visual subgoal。

注意：

- 推理时没有 ground-truth future feature
- 所以这里只有 `predicted future latent`，没有 `h_t1_gt`

#### 4. 生成 action chunk

最后 action head 会调用：

- `self.flow.sample_actions_cfg(...)`

条件输入包括：

- `h_t`
- `h_t1_pred`
- `h_vlm`
- `state`
- `action_hz`
- `embodiment_id`

最终返回归一化后的 action chunk。

### 推理输出

运行时主输出是：

- `normalized_actions`
  形状一般是 `[B, effective_action_len, action_dim]`

如果开启调试 / 中间结果返回，还可能附带：

- `h_t`
- `h_t1_pred`
- `vision_tokens_hw`

这些量对于后续观察 latent future 分支、替换 latent future interface 很有帮助。

## 端到端简版总结

### Stage 1

输入：

- 当前 clip
- 未来 clip
- 起点 / 终点 state
- embodiment id

中间过程：

- frozen visual encoder 提取 latent 特征
- inverse-dynamics encoder 推 latent action
- forward decoder 预测未来 latent feature
- auxiliary state head 学 embodied motion

输出：

- latent world decoder
- latent teacher 路径

### Stage 2

输入：

- 当前 observation clip / images
- 指令
- 目标 action chunk
- state / embodiment / control frequency 元数据

中间过程：

- Qwen3-VL 编码语义上下文
- latent-action queries 预测 latent action
- 用 Stage 1 decoder 预测 latent visual subgoal
- flow-matching action head 生成 action chunk
- 三类损失共同训练：latent action、latent future、action generation

输出：

- 训练好的 LaWAM policy backend

### Inference

输入：

- 当前图像
- 指令
- action frequency
- embodiment id
- 可选 state

中间过程：

- Qwen3-VL -> latent action
- Stage 1 decoder -> latent subgoal
- flow-matching head -> action chunk

输出：

- normalized action chunk

## 如果后面要改结构，建议重点看的边界

如果你的目标是改模型结构，可以优先把这些位置当作模块边界：

- 改 Stage 1 latent world predictor：
  `latent_action_model/core/lam_model.py`
- 改 Stage 2 如何从 VLM feature 预测 latent action：
  `starVLA/model/framework/vlas/lawam.py`
- 改 latent future 如何条件化 action expert：
  `starVLA/model/framework/vlas/lawam.py`
  和
  `starVLA/model/framework/vlas/flowmatching_expert.py`
- 改 train / infer batch 合约：
  `starVLA/dataloader/latent_world_train_collator.py`
  和
  `starVLA/model/framework/latent_world/batch_builder.py`

## 最核心的结构理解

这个仓库最重要的设计点可以总结成三句：

- `Stage 1` 不直接输出 action
- `Stage 1` 输出的是一个可复用的 latent world interface
- `Stage 2` 学的是如何预测进入这个 interface，再基于这个 interface 出动作
