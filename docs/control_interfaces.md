# 控制接口与数据字段

[返回主 README](../README.md)

本文定义 VLA 策略的观测、动作及归一化边界。SIMPLE 和 SONIC 通过各自的底层控制策略执行动作，每种接口使用独立的配置、数据统计和 checkpoint。

| 接口 | 图像 | state | action | chunk |
|---|---|---:|---:|---:|
| SIMPLE | 单 ego RGB | 32 | 36 | 30 |
| SONIC | 单 ego RGB | 46 | 78 | 40 |

## SIMPLE

数据配置为 `G1HandoverDataConfig`，Pick 任务的数据 mixture 为 `g1_pick_between_tables`。视频键 `video.rs_view` 映射到 `observation.images.egocentric`。

以下切片使用 Python 下标，从 0 开始，右端不包含。

| 字段 | state 切片 | action 切片 | 含义 |
|---|---|---|---|
| 左手 | `[0:7]` | `[0:7]` | 关节位置 / 关节目标 |
| 右手 | `[7:14]` | `[7:14]` | 关节位置 / 关节目标 |
| 左臂 | `[14:21]` | `[14:21]` | 关节位置 / 关节目标 |
| 右臂 | `[21:28]` | `[21:28]` | 关节位置 / 关节目标 |
| 腰部 RPY | `[28:31]` | `[28:31]` | 腰关节 roll/pitch/yaw / 姿态目标 |
| height | `[31:32]` | `[31:32]` | 高度命令状态 / 高度目标 |
| torso_vx | — | `[32:33]` | 平面 x 方向运动控制 |
| torso_vy | — | `[33:34]` | 平面 y 方向运动控制 |
| torso_vyaw | — | `[34:35]` | 转向控制字段 |
| target_yaw | — | `[35:36]` | 目标朝向 |

Pick 数据的腰部状态对应 `observation.leg_joints` 的 `[13,14,12]` 重排，高度字段对应 `observation.prev_height`。第 34 维按数据集 `torso_vyaw` 的连续数值读写，执行端按所使用的 SIMPLE 控制接口解释该字段。

state 与 action 前 32 维使用逐字段 min-max 归一化；action 后 4 维使用 mean/std。

## SONIC

数据配置为 `SonicLatentDataConfig`，mixture 为 `sonic_merged_dataset_001`，视频键为 `video.ego_view`。

| 字段 | state 切片 | action 切片 |
|---|---|---|
| 左腿 | `[0:6]` | motion token |
| 右腿 | `[6:12]` | motion token |
| 腰 | `[12:15]` | motion token |
| 左臂 | `[15:22]` | motion token |
| 左手 | `[22:29]` | `[64:71]` |
| 右臂 | `[29:36]` | motion token |
| 右手 | `[36:43]` | `[71:78]` |
| projected_gravity | `[43:46]` | — |
| motion_token | — | `[0:64]` |

state 由 43 维关节位置和 3 维机身坐标系重力方向构成。机器人模型将身体、左右手传感器数据映射到上述关节顺序，重力方向由机身四元数计算。SonicStar 的 G1 客户端对左手中指复制食指的硬件耦合关系进行处理，采集与部署使用相同的映射。

每一步 action 拆分为：

```python
motion_token = actions[..., :64]
left_hand = actions[..., 64:71]
right_hand = actions[..., 71:78]
```

SONIC 跟踪策略结合 motion token 和自身本体观测，产生 29 维身体动作，再通过关节缩放和默认姿态转换为目标关节位置；双手各 7 维命令通过手部接口执行。训练数据中的 token 定义应与部署使用的 SONIC 模型版本对应。

SONIC state 与 action 全部使用逐字段 min-max 归一化。

## 数据统计与部署

训练输出目录中的 `dataset_statistics.json` 保存归一化统计。服务接收归一化 state，返回 `normalized_actions`；客户端在执行前恢复动作量纲。

对于 min-max 字段：

```text
x_normalized = 2 * (x - x_min) / (x_max - x_min) - 1
x = (x_normalized + 1) / 2 * (x_max - x_min) + x_min
```

对于 mean/std 字段：

```text
x_normalized = (x - mean) / std
x = x_normalized * std + mean
```

常量维度按 [StateActionTransform](../starVLA/dataloader/gr00t_lerobot/transform/state_action.py) 的实现处理。控制客户端应使用数据配置的同一变换与字段顺序。

已加载的模型可直接获取带接口归一化模式的统计并恢复动作：

```python
action_stats = model.get_action_stats()
actions = model.unnormalize_actions(result["normalized_actions"], action_stats)
```

只运行客户端时，可用 `baseframework.get_action_stats(norm_stats=checkpoint_statistics)` 读取保存的统计。SIMPLE 和 SONIC 的统计标签会补充对应的逐维 `normalization_modes`；该字段也可显式提供。反归一化支持 `[H,A]` 和 `[B,H,A]`，保留连续动作通道；min-max 和 mean/std 字段不会统一裁剪到 `[-1,1]`。

服务在连接时发送 `state_dim`、`action_dim` 和 `action_horizon`，用于客户端选择对应的状态处理与动作执行接口。WebSocket 消息格式见 [部署文档](../deployment/model_server/README.md)。
