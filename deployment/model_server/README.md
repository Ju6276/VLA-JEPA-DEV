# JEPA Spatial Goal 推理服务

[返回主 README](../../README.md) · [控制接口](../../docs/control_interfaces.md)

## 服务端

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/checkpoints/sonic_spatial_goal_8xa100/checkpoints/steps_40000_pytorch_model.pt \
  --cuda 0 --use_bf16 --port 10093
```

服务通过 checkpoint 配置构建模型并加载权重。部署目录应包含训练保存的配置与 `dataset_statistics.json`，配置中的 Qwen 与 V-JEPA 路径应可访问。

服务加载 V-JEPA 编码器、目标预测器、动作模块和动作条件世界模型。默认目标由当前观测计算，配置为 `subgoals_path: null`。

## 客户端

客户端使用 WebSocket 和 MessagePack 传输 NumPy 数组。下面示例读取一张 RGB 图像与一个已经归一化的 state 文件：

```python
import numpy as np
from PIL import Image
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

client = WebsocketClientPolicy(host="127.0.0.1", port=10093)
try:
    metadata = client.get_server_metadata()
    image = np.asarray(Image.open("ego.png").convert("RGB"), dtype=np.uint8)
    state = np.load("normalized_state.npy").astype(np.float32)
    assert state.shape == (metadata["state_dim"],)

    response = client.infer({
        "batch_images": [[image]],
        "instructions": ["Pick up the object and place it on the table."],
        "state": state[None, None, :],
    })
    if not response["ok"]:
        raise RuntimeError(response["error"]["message"])

    result = response["data"]
    normalized_actions = result["normalized_actions"][0]
    assert normalized_actions.shape == (
        metadata["action_horizon"], metadata["action_dim"]
    )
    print(result["goal_source"], normalized_actions.shape)
finally:
    client.close()
```

SIMPLE 输出为 `[30,36]`，SONIC 输出为 `[40,78]`。原始 state 可通过 `control_transforms.normalize_simple_state` 或 `normalize_sonic_state` 按对应参考客户端归一化。执行前使用同一 checkpoint 的统计恢复动作：SIMPLE 前 32 维按 min-max 裁剪后缩放、后 4 维按 mean/std 缩放；SONIC 全部 78 维按其 min-max 与 mask 规则处理。完整调用见 [控制接口](../../docs/control_interfaces.md#数据统计与部署)。SIMPLE 客户端将动作提交给仿真控制接口；SONIC 客户端拆分 motion token、左手、右手三个字段后发送给控制器。

## 协议

连接建立后，服务首先发送元数据：

| 字段 | 含义 |
|---|---|
| `state_dim` | 输入状态维度 |
| `action_dim` | 单步动作维度 |
| `action_horizon` | chunk 长度 |
| `learned_goal_enabled` | 自动目标是否启用 |
| `spatial_goal_enabled` | 空间目标、局部动作条件和空间评分是否启用 |
| `spatial_memory_enabled` | 是否使用真实过去观测 |
| `use_verifier` | 默认是否启用候选评分 |
| `num_subgoals` | 加载的外部 subgoal 数量 |

推理请求：

```text
{
  "type": "infer",
  "request_id": "request-001",
  "payload": {
    "batch_images": [[uint8_RGB_image]],
    "instructions": [task_text],
    "state": float32_array[B, 1, S]
  }
}
```

自动目标推理响应的 `data` 字段包含：

| 字段 | 形状 / 类型 | 含义 |
|---|---|---|
| `normalized_actions` | `[B,H,A]` float32 | 选中的动作 chunk |
| `verification_scores` | `[B]` float32 | 选中候选的分数 |
| `all_candidates` | `[B,N,H,A]` float32 | 全部候选动作 |
| `candidate_goal_progress` | `[B,N]` float32 | 每个候选的视觉目标进展 |
| `candidate_spatial_progress` | `[B,N]` float32 | 空间配置下每个候选的局部目标进展 |
| `candidate_prior_error` | `[B,N]` float32 | 每个候选的动作 prior 误差 |
| `candidate_scores` | `[B,N]` float32 | 每个候选的联合分数，与候选顺序一致 |
| `goal_source` | string | `predicted`、`images` 或 `tracker` |
| `goal_proposal_used` | bool | 是否包含目标条件 proposal |
| `subgoal_index` | integer / null | 外部 tracker 的目标位置 |
| `spatial_attention` | `[B,K,G²]` float32 | 当前任务查询的空间权重，默认 K=4、G=8 |
| `spatial_goal_attention` | `[B,K,G²]` float32 | 同一任务查询读取目标网格的权重 |
| `spatial_history_used` | `[B]` integer | 本次使用的有效过去观测数量 |

空间配置的组合公式为 `candidate_scores = candidate_goal_progress + gamma * candidate_spatial_progress - beta * candidate_prior_error`，其中 `gamma` 为 `framework.spatial_goal.score_weight`（默认 0.5），`beta` 为 `verifier_action_prior_weight`（默认 0.1）。全局配置使用全局进展与 prior 两项。分数以模型计算时的精度求得，再转成 float32 导出；复现服务端选择时直接使用 `candidate_scores`，保留混合精度的舍入结果。固定候选集合的评分对照见 [空间方法与消融](../../docs/spatial_goals.md#消融)。

## 真实观测历史

空间配置的在线服务每个连接维护一条轨迹，batch size 为 1。客户端推荐发送相机观测的 `timestamp`（有限数值，单位秒）和 `episode_id`（字符串或整数）。未提供时间戳时使用服务端的单调时钟；两种时钟来源切换时清空历史。

```python
response = client.infer({
    "batch_images": [[ego_rgb]],
    "instructions": [task],
    "state": normalized_state[None, None, :],
    "timestamp": observation_timestamp_seconds,
    "episode_id": "episode-001",
})
```

服务器缓存推理请求中的真实图像特征。默认读取约 0.8 秒、0.4 秒前的观测；历史偏移和容差与训练配置一致。新 episode、指令改变、时间回退、过长间隔以及 `client.reset()` 会清空当前连接的历史，新连接独立初始化。缺少某个时间槽时将其标记无效，`spatial_history_used` 可用于确认实际历史数量。

若相机采集频率高于策略调用频率，客户端也可以随请求提供它已经采集的过去图像。使用 `history_images: [[past_rgb_0, past_rgb_1]]`、`history_valid: [[True, True]]`、`history_ages: [[actual_age_0, actual_age_1]]`，并设置 `update_memory: False`；槽位顺序与训练的历史偏移一致，年龄为当前观测与过去观测时间戳之差。该请求使用显式历史且不更新服务器缓存。过去图像不是目标图片，也不需要标注。

采集和请求频率应覆盖训练设定的历史时间槽；执行整个 chunk 后才发请求时，缓存可能较稀疏。若改变历史偏移，应使用相同设置训练。完整的数据与时间定义见 [空间目标说明](../../docs/spatial_goals.md#数据与历史)。

## 推理控制

请求可设置 `num_candidates` 调整候选数量，或设置 `use_verifier=False` 使用普通 Action Expert 输出。普通策略响应仅包含其对应动作与条件特征字段。

默认请求只需当前图像、指令和 state，subgoal latent 由模型预测。目标图片对照实验可在 payload 中增加可选字段 `subgoal_images: [uint8_RGB_image]`；每个 batch 样本对应一张目标图。服务统一转换当前图像与目标图像，显式目标优先于 tracker 与自动预测目标。省略该字段或传入 `None` 均使用默认目标来源。

当前 learned-goal 配置使用单 ego 相机。已有多视角 checkpoint 的观察和显式目标使用相同的相机顺序，按 `[batch][view]` 组织：

```python
payload = {
    "batch_images": [[current_ego, current_wrist]],
    "subgoal_images": [[goal_ego, goal_wrist]],
    "instructions": [task],
    "state": normalized_state[None, None, :],
}
```

每个样本的视角数必须与 checkpoint 的 `num_video_views` 一致。多视角 tracker 的 `subgoals.json` 用 `paths` 列出同一目标的各相机图像：

```json
{"subgoals": [
  {"paths": ["goal0_ego.png", "goal0_wrist.png"]},
  {"paths": ["goal1_ego.png", "goal1_wrist.png"]}
]}
```

相对路径以 manifest 所在目录为基准；原单视角 `path` 格式继续支持。pickle 资产中的 `frames` 对应为 `[[goal0_view0, goal0_view1], ...]`。

示范 tracker 可由服务参数 `--subgoals_path /path/to/subgoals` 加载，此参数会在模型构建前覆盖 checkpoint 中保存的路径。tracker 每个连接独立维护一条轨迹，仅支持 batch size 1。切换任务时发送 `{"type":"reset","request_id":"reset-001"}`，或调用 `client.reset()`，重置当前连接的 tracker 与空间历史。新连接从第一个外部目标开始；自动目标模式保持 `subgoals_path: null`。
