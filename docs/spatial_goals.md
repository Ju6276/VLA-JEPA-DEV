# 空间目标与动作选择

本分支从当前单 ego 图像、任务指令、state 和真实过去观测预测未来视觉目标，并生成、比较动作 chunk。空间分支使用固定数据集已有的示范视频和动作，不需要边界框、分割图、物体 ID 或独立定位训练阶段。

## 表示与目标预测

冻结的 V-JEPA 2.1 对每张图像独立编码：将该图像重复为编码器需要的视频长度，取最后一个时间块的 patch 特征。当前图像、历史图像与示范未来图像分别编码，未来图像不会进入当前条件或历史。

设当前 patch 特征为 $Z_t$，示范动作跨度为 $H$，未来帧特征为 $Z_{t+H}$。全局分支使用归一化平均池化 $z_t$，空间分支使用固定平均汇聚：

$$
G_t=\operatorname{Pool}_{8\times8}(Z_t),\qquad
G^*_{t+H}=\operatorname{stopgrad}(\operatorname{Pool}_{8\times8}(Z_{t+H})).
$$

每个网格 token 保留冻结编码器的原始特征空间。网格位置对应各自图像的空间位置，不假设同一格在前后帧中对应同一个物体。

全局目标 MLP 根据 $z_t$、Qwen 图像与指令条件 $c_t$、state $s_t$ 预测归一化目标 $\hat z_g$。空间目标预测器同时预测：

$$
\hat G_g=f_\theta(G_t,c_t,s_t,\mathcal H_t).
$$

`SpatialGoalPredictor` 使用 64 个未来网格查询、256 维隐藏层和 4 头 cross-attention。当前网格、历史网格、任务 tokens 和 state 条件组成 attention 的输入；网格带固定二维位置编码，历史带实际观测间隔编码。输出以当前网格为残差参考，预测未来原始 JEPA 特征。该残差形式不构成物体对应或几何运动约束。

## 任务相关空间读取与动作解码

`TaskSpatialReader` 根据当前 Qwen 条件和 state 产生 4 个查询 $q_t$，分别读取当前、目标及候选未来网格：

$$
r_t=R(G_t;q_t),\quad r_g=R(\hat G_g;q_t),\quad
\hat r^{(i)}=R(\hat G^{(i)}_{t+H};q_t).
$$

查询在一次决策的所有候选中共享。attention 的位置可以随图像内容变化；value 直接使用网格中的原始 JEPA 特征，没有可训练的 value 投影。每个查询的输出经过 L2 归一化。

全局 proposal 解码完整动作 chunk，空间适配器从 $[r_t,r_g,r_g-r_t]$ 预测同形状的残差：

$$
A_{\mathrm{proposal}}=D(z_t,\hat z_g,s_t)+D_{\mathrm{spatial}}(r_t,r_g).
$$

空间适配器使用 MLP 与每个动作时间步的可训练嵌入。SIMPLE 输出 $30\times36$，SONIC 输出 $40\times78$；其余候选由 Action Expert 产生，默认总数为 8。

训练时，真实未来目标和预测目标各占默认 0.5 的 proposal 重建权重。预测目标在动作重建路径中停止梯度，目标预测器由其未来特征监督训练；空间读取器和动作适配器仍保留到动作重建损失的梯度。因此读取器学习对动作有用的关注位置，无需定位标签。

## 训练目标

默认加权总损失为：

$$
\mathcal L=\mathcal L_{\mathrm{action}}
+0.1\mathcal L_{\mathrm{wm}}
+0.01\mathcal L_{\mathrm{prior}}
+0.01\mathcal L_{\mathrm{proposal}}
+0.05\left(\mathcal L_{\mathrm{goal,global}}
+\alpha\mathcal L_{\mathrm{goal,spatial}}\right),\qquad\alpha=1.
$$

| 日志损失 | 监督内容 |
|---|---|
| `action_loss` | Action Expert 的示范动作 flow matching |
| `wm_loss` | 动作条件世界模型预测未来完整 patch 特征的 L1 |
| `action_prior_loss` | 状态条件 GRU 的下一步动作预测 MSE |
| `goal_proposal_loss` | 全局 proposal 加空间残差后的完整 chunk 重建 MSE |
| `goal_prediction_loss` | 全局目标 cosine 距离与固定空间网格 L1 之和 |

其中：

$$
\mathcal L_{\mathrm{goal,global}}=1-\cos(\hat z_g,z^*_{t+H}),\qquad
\mathcal L_{\mathrm{goal,spatial}}=\operatorname{mean}|\hat G_g-G^*_{t+H}|.
$$

全局和空间分量分别记录为 `goal_prediction_global_component`、`goal_prediction_spatial_component`，记录值包含各自权重。它们仅用于诊断，总损失中只通过 `goal_prediction_loss` 计入一次。五类损失中的目标预测项包含两个明确的监督分量。

`delta_loss` 和 `ctrl_loss` 是可选辅助约束，主配置的 `lambda_delta`、`lambda_ctrl` 为零。前者监督预测与真实未来的全局 latent 位移，后者从 latent 位移与 state 重建指定动作，默认监督 chunk 第一步。它们不代替部署时的完整 chunk proposal。

## 候选评分

动作条件世界模型分别预测每个候选动作的未来 patch 特征。全局目标进展为候选未来与目标的 cosine 相似度减去当前与目标的相似度；空间进展为 4 个共享查询读取结果的对应进展均值：

$$
p^{(i)}_{\mathrm{spatial}}=\frac14\sum_{k=1}^{4}
\left[\cos(\hat r_k^{(i)},r_{g,k})-\cos(r_{t,k},r_{g,k})\right].
$$

默认联合分数：

$$
s_i=p^{(i)}_{\mathrm{global}}+0.5p^{(i)}_{\mathrm{spatial}}
-0.1e^{(i)}_{\mathrm{prior}}.
$$

`framework.spatial_goal.score_weight` 控制空间进展权重，`framework.delta_jepa.verifier_action_prior_weight` 控制 prior 权重。全部候选使用同一个当前查询与同一个目标，取得分最大的 chunk。

这些分数是特征空间的进展与动作序列一致性指标，不是校准后的成功概率、稳定性保证或碰撞检测结果。attention 可以呈现任务相关位置，但没有物体 ID 监督，不应解释为已获得精确检测框或跨帧身份跟踪。

## 数据与历史

训练样本继续使用已有 LeRobot 数据。空间配置额外返回：

| 字段 | 内容 |
|---|---|
| `jepa_image` | 当前单 ego 图像，按 JEPA 的 384 方形分辨率处理 |
| `history_images` | 固定数量的过去图像，顺序与历史时间偏移一致 |
| `history_valid` | 每个过去观测槽位是否有效 |
| `history_ages` | 当前减去实际历史时间戳，单位秒 |
| `timestamp` | 当前 episode 内观测时间戳 |

Qwen 使用独立的 224 分辨率图像路径。JEPA 当前、未来与历史使用原始视频经方形 resize 的图像；无需额外离线文件。

默认 `history_offsets_seconds: [-0.8, -0.4]`。对每个偏移选择同一 episode 内不晚于请求时刻的最近观测，最多允许比请求时刻早 `history_tolerance_seconds: 0.15` 秒。缺失槽位使用当前图像占位并标记无效，不将占位当作真实过去，也不跨 episode 取帧。训练中每个有效历史以 `history_dropout: 0.25` 的概率独立屏蔽，让模型同时接触完整、部分缺失与无历史条件，无需新增损失。

空间目标的未来时刻仍与动作跨度对齐：SIMPLE 为 $t+30$ 个数据步，SONIC 为 $t+40$ 个数据步。历史偏移以秒定义，独立于这两个动作步数。尾段仍按原采样配置补齐；`require_full_horizon: true` 可仅保留完整起始位置。

历史偏移和容差应根据实际部署观测频率设置。例如，执行完整动作 chunk 后才获取下一张图像，会产生较稀疏的历史；若缓存中没有符合某个时间槽的观测，该槽会被 mask。缓存无法补出没有采集的图像。配置通过 OmegaConf 插值将训练与部署的历史采样参数绑定。

## 推理接口

在线每个 WebSocket 连接维护一条轨迹，`batch_images` 为 `[[ego_rgb]]`，`state` 为 `[1,1,S]`。推荐提供观测的 `timestamp` 秒数与 `episode_id`。若不提供时间，模型使用服务器处理观测时的单调时钟。

以下情况开始新的历史：新连接、`client.reset()`、指令变化、`episode_id` 变化、时钟来源改变、时间不递增，或两次请求相隔超过 `memory_max_gap_seconds: 2.0`。只在成功推理后提交当前真实观测特征，候选未来不写入缓存。在线缓存有 `memory_max_frames` 上限，默认 256。

元数据新增 `spatial_goal_enabled`、`spatial_memory_enabled`。成功的 verifier 响应包含：

| 字段 | 形状 | 含义 |
|---|---|---|
| `normalized_actions` | `[B,H,A]` | 选中的归一化动作 chunk |
| `candidate_goal_progress` | `[B,N]` | 全局目标进展 |
| `candidate_spatial_progress` | `[B,N]` | 空间目标进展 |
| `candidate_prior_error` | `[B,N]` | 动作 prior 误差 |
| `candidate_scores` | `[B,N]` | 实际用于选择的联合分数 |
| `spatial_attention` | `[B,4,64]` | 当前图像的空间读取权重 |
| `spatial_goal_attention` | `[B,4,64]` | 预测目标的空间读取权重 |
| `spatial_history_used` | `[B]` | 本次有效过去观测的数量 |

权重可 reshape 成 8×8 热图。动作采用与对应数据集匹配的归一化统计恢复，参见 [控制接口](control_interfaces.md)。混合精度下，复现服务选择应直接使用 `candidate_scores`，而非对导出的 float32 分量重新组合。

离线批量诊断可向模型 `predict_action` 显式提供 `history_images[B][M]`、`history_valid[B,M]`、`history_ages[B,M]`，并设置 `update_memory=False`。该路径使用显式历史且不修改在线缓存。没有显式历史时，`update_memory=False` 表示当前观测条件的无缓存推理。启用在线记忆的 WebSocket 服务保持单连接单轨迹的 batch size 1。

## 消融

| 对照 | 配置方式 | 比较内容 |
|---|---|---|
| 关闭空间评分 | 同一 checkpoint 将 `framework.spatial_goal.score_weight` 设为 `0.0` | 相同候选上的空间进展项 |
| 无历史目标预测 | 训练时设 `framework.spatial_goal.memory_enabled=false` | 真实过去观测的作用 |
| 仅全局目标 | 使用五损失全局配置训练，`spatial_goal.enabled=false` | 空间目标、读取、proposal 残差与空间评分的整体作用 |
| 位移辅助项 | 训练时设 `delta_jepa.lambda_delta=0.05`，其余保持不变 | latent 位移监督 |
| 逆动力学辅助项 | 训练时设 `delta_jepa.lambda_ctrl=0.02`，其余保持不变 | 单步动作重建监督 |

关闭空间评分时保留空间 proposal，才能隔离评分项；最好保存并复用同一批候选。历史消融应分别训练，使用相同数据划分、seed、动作接口和优化器更新次数。旧全局配置可继续使用，空间版本包含额外参数，应使用对应配置训练并严格加载其 checkpoint。完整续训保留原训练的结构、损失权重、历史设置和有效 batch。

框架代码在 [spatial_jepa.py](../starVLA/model/framework/spatial_jepa.py)，神经模块在 [spatial_goal.py](../starVLA/model/modules/world_model/spatial_goal.py)，时间采样在 [spatial_history.py](../starVLA/dataloader/spatial_history.py)。
