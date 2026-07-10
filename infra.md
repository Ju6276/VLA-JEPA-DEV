# VLA-JEPA Inference Infra

This document describes how to start the VLA-JEPA inference server for the
SIMPLE_OPEN_OVEN_30 checkpoint on this machine.

## 1. Enter the Repository

```bash
cd /home/d013/桌面/VLA-JEPA
```

## 2. Activate the Runtime Environment

Use the VLA-JEPA conda environment. If multiple environments are active in the
prompt, deactivate them first to avoid running the wrong Python.

```bash
conda deactivate
conda deactivate
conda deactivate

conda activate VLA_JEPA
```

Check that the active Python can import PyTorch:

```bash
which python
python -c "import sys; print(sys.executable)"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this fails with `ModuleNotFoundError: No module named 'torch'`, the active
Python environment is wrong or PyTorch is not installed in `VLA_JEPA`.

## 3. Check Required Local Paths

The inference server expects these files and model directories to exist:

```bash
test -f /home/d013/桌面/CKPT/JEPA21/SIMPLE_OPEN_OVEN_30/steps_10000_pytorch_model.pt
test -d /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct
test -d /home/d013/桌面/VLA-JEPA/VJEPA21
```

The checkpoint directory should also contain:

```text
/home/d013/桌面/CKPT/JEPA21/SIMPLE_OPEN_OVEN_30/config.yaml
/home/d013/桌面/CKPT/JEPA21/SIMPLE_OPEN_OVEN_30/dataset_statistics.json
```

## 4. Start the SIMPLE/G1 Policy Server

Use `server_policy_simple_g1.py` for this checkpoint because the checkpoint uses
the SIMPLE/G1 action layout.

Before starting the server, check whether the target port is already occupied:

```bash
lsof -iTCP:10090 -sTCP:LISTEN -P -n
```

If a previous inference server is still listening on `10090`, stop it before
starting a new one:

```bash
kill <PID>
```

Alternatively, keep the old server running and start the new server with a
different `--port` value.

```bash
export TOKENIZERS_PARALLELISM=false

python deployment/model_server/server_policy_simple_g1.py \
  --ckpt_path /home/d013/桌面/CKPT/JEPA21/SIMPLE_OPEN_OVEN_30/steps_10000_pytorch_model.pt \
  --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
  --base_encoder_path /home/d013/桌面/VLA-JEPA/VJEPA21 \
  --port 10090 \
  --cuda 0 \
  --use_bf16
```

Successful startup should include log output similar to:

```text
Loading from local checkpoint path
server running ...
```

## 5. Smoke Test the Server

Open a second terminal, activate the same environment, and run:

```bash
cd /home/d013/桌面/VLA-JEPA
conda activate VLA_JEPA

python deployment/model_server/debug_server_policy.py \
  --host 127.0.0.1 \
  --port 10090 \
  --device cuda \
  --test init
```

Use `--test init` for the most stable liveness check. Full task behavior should
be verified through the real SIMPLE/G1 control client or evaluation workflow.
