# VLA-JEPA 权重推理端启动 SOP

本文档用于规范：当用户提供一个训练得到的 VLA-JEPA 权重目录，例如
`/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30`，如何把该权重接入本仓库推理端并完成最小可用测试。

## 1. 核心原则

只做推理接入所必需的路径配置，不改模型结构、不改训练逻辑、不改权重内容。

推理加载入口是：

- 通用策略服务：`deployment/model_server/server_policy.py`
- SIMPLE/G1 handover 专用策略服务：`deployment/model_server/server_policy_simple_g1.py`

通用入口内部调用：

```python
baseframework.from_pretrained(args.ckpt_path)
```

因此 `--ckpt_path` 必须指向一个具体的 `.pt` 权重文件，而不是只指向权重文件夹。

## 2. 推荐权重目录结构

推荐把训练产物保持为一个完整 run 目录，不要只拷贝 `.pt` 文件。

示例：

```text
/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/
├── config.yaml
├── config.json
├── dataset_statistics.json
├── summary.jsonl                      # 可选
├── steps_40000_pytorch_model.pt       # 必需，实际权重文件
└── output_server/                     # 可选，启动服务后放日志
```

也兼容训练时常见的子目录形式：

```text
/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/
├── config.yaml
├── config.json
├── dataset_statistics.json
└── checkpoints/
    └── steps_40000_pytorch_model.pt
```

加载逻辑会从 `.pt` 所在目录和它的上一级目录中查找：

- `config.yaml`
- `dataset_statistics.json`

注意：当前 `baseframework.from_pretrained()` 实际读取的是 `config.yaml`，不是 `config.json`。`config.json` 建议保留，用于兼容其他脚本或人工检查。

## 3. 必须检查/允许改动的文件

### 3.1 权重目录内允许改动

只允许改动权重目录中的配置文件路径字段：

- `<CKPT_DIR>/config.yaml`
- `<CKPT_DIR>/config.json`，如果存在，保持和 `config.yaml` 一致

必须检查并按本机路径更新：

```yaml
framework:
  qwenvl:
    base_vlm: /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct
  vj2_model:
    base_encoder: /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256
```

字段名必须是 `framework.qwenvl.base_vlm`，不要写成 `basevlm`。

如果使用 SIMPLE/G1 专用入口，也可以不改配置文件，启动时用命令行覆盖：

```bash
python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256 \
  --port 10093 \
  --cuda 0 \
  --use_bf16
```

### 3.2 可改动的启动脚本

如果需要固定路径、端口、GPU，可以改这些脚本里的变量：

- `deployment/model_server/README.md`：只作为说明文档
- `examples/SimplerEnv/eval_files/run_policy_server.sh`：SimplerEnv 侧启动策略服务
- `examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh`：SimplerEnv 批量评测入口
- `examples/LIBERO/eval_libero.sh`：LIBERO 评测入口
- `examples/LIBERO-Plus/eval_libero_plus.sh`：LIBERO-Plus 评测入口

建议改动范围只限于：

- `your_ckpt` / `MODEL_PATH`
- `port`
- `gpu_id` / `CUDA_VISIBLE_DEVICES`
- `LIBERO_HOME` / `SimplerEnv_PATH`
- `sim_python`

## 4. 禁止改动的文件

除非明确是在修 bug 或开发新功能，否则推理接入时不要改：

- `starVLA/model/framework/VLA_JEPA.py`
- `starVLA/model/framework/base_framework.py`
- `starVLA/model/framework/share_tools.py`
- `starVLA/model/modules/**`
- `starVLA/dataloader/**`
- `starVLA/training/**`
- `.pt` 权重文件本身
- `dataset_statistics.json` 中的统计值

特别注意：

- 不要为了“能加载”而改 `strict=True`。
- 不要手动删除/重命名 state dict key。
- 不要随意改 `framework.action_model.action_dim`、`state_dim`、`action_horizon`、`future_action_window_size` 等结构参数。这些参数必须和训练时一致，否则会出现权重 shape mismatch。
- 不要把训练环境路径写进代码文件，路径只应该出现在权重目录配置或启动脚本中。

## 5. 启动通用 VLA-JEPA 推理服务

适用于 SimplerEnv、LIBERO 或通用 websocket policy server。

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

export TOKENIZERS_PARALLELISM=false

python deployment/model_server/server_policy.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt \
  --port 10093 \
  --cuda 0 \
  --use_bf16
```

成功现象：

- 日志出现 `Loading from local checkpoint path`
- 没有 `Missing config.yaml / dataset_statistics.json`
- 没有 `Missing keys` / `Unexpected keys`
- 日志出现 `server running ...`

如果启动时报：

```text
OSError: [Errno 98] error while attempting to bind ... address already in use
```

说明目标端口已有旧服务在监听。启动前可以先查端口：

```bash
lsof -iTCP:10093 -sTCP:LISTEN -P -n
```

如果确认是旧推理服务，先停止旧进程：

```bash
kill <PID>
```

也可以不停止旧服务，改用一个新的 `--port`。

## 6. 启动 SIMPLE/G1 专用推理服务

如果权重是 G1 handover / SIMPLE 机器人动作格式，优先使用专用入口：

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

export TOKENIZERS_PARALLELISM=false

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256 \
  --port 10093 \
  --cuda 0 \
  --use_bf16
```

使用这个入口的原因：

- 它会用 SIMPLE/G1 的 36D action layout 做专用反归一化。
- 它不会调用 `baseframework.unnormalize_actions()` 中针对 7D gripper 的默认硬编码逻辑。
- 它支持 `--base_vlm_path` 和 `--base_encoder_path` 直接覆盖旧机器路径。

## 7. 冒烟测试指令

服务启动后，在另一个终端运行：

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

python deployment/model_server/debug_server_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --device cuda \
  --test init
```

如果要进一步测试一次调试推理：

```bash
python deployment/model_server/debug_server_policy.py \
  --host 127.0.0.1 \
  --port 10093 \
  --device cuda \
  --test infer
```

注意：当前 `debug_server_policy.py --test infer` 会尝试读取 `assets/table.jpeg`。如果该图片不存在，脚本可能只验证到 websocket 连接和设备初始化，随后在本地调试输入阶段报错。`--test init` 才是最稳定的服务存活冒烟测试；真实任务效果必须通过 SimplerEnv/LIBERO 或实际控制端验证。
