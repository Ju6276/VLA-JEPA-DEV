
# start VLA-JEPA policy server


```bash

your_ckpt=/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --cuda 0 \
    --use_bf16
```


# start SIMPLE / G1 handover policy server

```bash
your_ckpt=/home/d013/桌面/CKPT/JEPA2/SIMPLE_OPENFAUCET__30/steps_40000_pytorch_model.pt

python deployment/model_server/server_policy_simple_g1.py \
    --ckpt_path ${your_ckpt} \
    --base_vlm_path /home/d013/桌面/VLA-JEPA/Qwen3-VL-2B-Instruct \
    --base_encoder_path /home/d013/桌面/VLA-JEPA/vjepa2-vitl-fpc64-256 \
    --port 10093 \
    --cuda 0 \
    --use_bf16
```

This entrypoint is intended for SIMPLE's `G1WholebodyHandoverTeleop-v0`.
It performs custom action de-normalization using `model.norm_stats["g1_handover"]`
and skips the default gripper hard-coding in `baseframework.unnormalize_actions()`.


# connect to policy server for debug

```bash
python deployment/model_server/debug_server_policy.py

# plus server_policy.py into your vla controler by ref to debug_server_policy.py
```
