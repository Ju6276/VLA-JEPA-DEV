# G1 推理服务

[返回主 README](../../README.md) · [控制接口](../../docs/control_interfaces.md)

## 服务端

```bash
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/checkpoints/sonic_latent_learned_goal/checkpoints/steps_40000_pytorch_model.pt \
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

SIMPLE 输出为 `[30,36]`，SONIC 输出为 `[40,78]`。执行前使用对应 checkpoint 的统计与字段变换反归一化。SIMPLE 客户端将动作提交给仿真控制接口；SONIC 客户端拆分 motion token、左手、右手三个字段后发送给控制器。

## 协议

连接建立后，服务首先发送元数据：

| 字段 | 含义 |
|---|---|
| `state_dim` | 输入状态维度 |
| `action_dim` | 单步动作维度 |
| `action_horizon` | chunk 长度 |
| `learned_goal_enabled` | 自动目标是否启用 |
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
| `goal_source` | string | `predicted`、`images` 或 `tracker` |
| `goal_proposal_used` | bool | 是否包含目标条件 proposal |
| `subgoal_index` | integer / null | 外部 tracker 的目标位置 |

## 推理控制

请求可设置 `num_candidates` 调整候选数量，或设置 `use_verifier=False` 使用普通 Action Expert 输出。普通策略响应仅包含其对应动作与条件特征字段。

目标图片对照实验可在 payload 中增加 `subgoal_images: [uint8_RGB_image]`。显式图片优先于 tracker 与自动预测目标。

示范 tracker 可由服务参数 `--subgoals_path /path/to/subgoals` 加载。切换任务时发送 `{"type":"reset","request_id":"reset-001"}` 重置 tracker。
