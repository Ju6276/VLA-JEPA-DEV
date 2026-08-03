<h3 align="center" style="font-size:48px; font-weight:bold; color:#9C276A; margin: 0;">
  <a href="https://arxiv.org/abs/2602.10098" style="color:#9C276A; text-decoration: none;">
    VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model
  </a>
</h3>

<div align="center">
<p>
  <a href="https://arxiv.org/abs/2602.10098">
    <img src="https://img.shields.io/badge/Paper-PDF-orange.svg" alt="Paper PDF">
  </a>
  <a href="https://ginwind.github.io/VLA-JEPA/">
    <img src="https://img.shields.io/badge/Project-Page-Green.svg" alt="Project Page">
  </a>
  <a href="https://huggingface.co/ginwind/VLA-JEPA">
    <img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow.svg" alt="Hugging Face">
  </a>
</p>
</div>

<div align="center">
  <img src="assets/VLA-JEPA.png" width="90%" alt="VLA-JEPA overview" />
</div>

> **Development fork.** This repository extends the original
> [VLA-JEPA](https://github.com/ginwind/VLA-JEPA) with: (1) support for the
> **SonicStar / Unitree G1 humanoid latent-action** dataset, and (2) a pluggable
> **V-JEPA 2.1** world-model encoder. See [`pipeline.md`](./pipeline.md) for the
> full change log and design notes.

## Table of Contents
- [🆕 What's New in This Fork](#whats-new)
- [⚙️ Environment Setup](#environment-setup)
- [🔥 Training](#training)
  - [0️⃣ Pretrained Model Preparation](#pretrained-model-preparation)
  - [1️⃣ Data Preparation](#data-preparation)
  - [2️⃣ Start Training](#start-training)
  - [3️⃣ Custom Dataset Training](#custom-dataset-training)
- [📊 Evaluation](#evaluation)
- [🚀 Deployment](#deployment)
- [🤝 Acknowledgement](#acknowledgement)
- [📝 Citation](#citation)

<a id="whats-new"></a>
## 🆕 What's New in This Fork

- **SonicStar (Unitree G1) latent-action support.** A new data config
  (`SonicLatentDataConfig`) and mixture for the humanoid dataset whose actions are
  `motion_token(64) + left/right hand joints(7+7) = 78-dim`, with `46-dim` state.
  See `scripts/config/vlajepa_sonic_latent.yaml`.
- **Pluggable V-JEPA 2.1 encoder.** The world-model encoder can be either the
  stock HuggingFace V-JEPA 2 model **or** Meta's V-JEPA 2.1 ViT-L/384. The choice
  is driven purely by `framework.vj2_model.base_encoder`:
  - a **HF model directory** (e.g. `vjepa2-vitl-fpc64-256`) → loaded via `AutoModel` (V-JEPA 2);
  - a **`.pt` file** (e.g. `vjepa2_1_vitl_dist_vitG_384.pt`) → loaded via the vendored
    2.1 encoder adapter in `starVLA/model/modules/world_model/`.

  Both paths are interchangeable; `scripts/config/vlajepa_sonic_latent_vjepa21.yaml`
  is the V-JEPA 2.1 variant (384px).

<a id="environment-setup"></a>
## ⚙️ Environment Setup

```bash
git clone https://github.com/Ju6276/VLA-JEPA-DEV.git
cd VLA-JEPA-DEV

# Create conda environment
conda create -n VLA_JEPA python=3.10 -y
conda activate VLA_JEPA

# Install requirements
pip install -r requirements.txt

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install project
pip install -e .
```

This repository's code is based on [starVLA](https://github.com/starVLA/starVLA).

<a id="training"></a>
## 🔥 Training

<a id="pretrained-model-preparation"></a>
### 0️⃣ Pretrained Model Preparation

Download the [Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and a world-model encoder:

- **V-JEPA 2** (default): [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256)
- **V-JEPA 2.1** (optional, 384px): download the ViT-L checkpoint from Meta's CDN
  ```bash
  wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
  ```

<a id="data-preparation"></a>
### 1️⃣ Data Preparation

Robot datasets use the **LeRobot v2.1** format. Add a `modality.json` file under each
dataset's `meta/` subdirectory; templates for LIBERO, BridgeV2, Fractal, and Droid are
provided under `./examples` (BridgeV2 and Fractal under `./examples/SimplerEnv`).

<a id="start-training"></a>
### 2️⃣ Start Training

Pick a script + YAML from [`/scripts`](./scripts). In the YAML, make sure:

- `framework.qwenvl.base_vlm` and `framework.vj2_model.base_encoder` point to your
  downloaded checkpoints (a `.pt` encoder path automatically enables the V-JEPA 2.1 adapter);
- `datasets.vla_data.data_root_dir` / `data_mix` match your dataset.

Then launch, e.g. for the SonicStar (Unitree G1) runs:

```bash
# V-JEPA 2 (256px)
bash scripts/vlajepa_sonic_latent.sh

# V-JEPA 2.1 (384px)
bash scripts/vlajepa_sonic_latent_vjepa21.sh
```

> The launch scripts use `accelerate` + DeepSpeed ZeRO-2 (8 GPUs by default) and
> require `WANDB_API_KEY` to be exported when `WANDB_MODE=online`.

<a id="custom-dataset-training"></a>
### 3️⃣ Custom Dataset Training

VLA-JEPA supports both robot datasets and human video datasets.

- **Robot Data (LeRobot v2.1):**
  - Define a config class in [`data_config.py`](./starVLA/dataloader/gr00t_lerobot/data_config.py)
    (its video/state/action keys must match `modality.json`), and register it in
    `ROBOT_TYPE_CONFIG_MAP`. `SonicLatentDataConfig` is a worked example.
  - Register the mixture in [`mixtures.py`](./starVLA/dataloader/gr00t_lerobot/mixtures.py):
    the dict key maps to `datasets.vla_data.data_mix`, and each entry is
    `(subdirectory, version, robot_type)`. `robot_type` selects state/action
    normalization. Add the matching `EmbodimentTag` in
    [`embodiment_tags.py`](./starVLA/dataloader/gr00t_lerobot/embodiment_tags.py).

- **Human Video:** implement your own DataLoader and register it in `build_dataloader`
  (`./starVLA/dataloader/__init__.py`), or use the provided video dataloader and configure
  `datasets.video_data` (`video_dir`, `text_file`, `CoT_prompt`, `extensions`).

<a id="evaluation"></a>
## 📊 Evaluation

Pretrained reference checkpoints: https://huggingface.co/ginwind/VLA-JEPA

```bash
# Extra eval deps (inside the VLA_JEPA env)
pip install tyro matplotlib mediapy websockets msgpack
```

**Common configuration (all benchmarks):** in the checkpoint folder, edit `config.json`
and `config.yaml` so `framework.qwenvl.base_vlm` and `framework.vj2_model.base_encoder`
point to your local checkpoints. Each benchmark runs in its own conda environment and
talks to the model over a WebSocket policy server (see [Deployment](#deployment)).

| Benchmark | Setup | Launch |
| --- | --- | --- |
| **LIBERO** | [official repo](https://github.com/Lifelong-Robot-Learning/LIBERO); set `LIBERO_HOME`, `sim_python`, `your_ckpt` in the script | `bash ./examples/LIBERO/eval_libero.sh` (4 suites / 4 GPUs) |
| **LIBERO-Plus** | [repo](https://github.com/sylvestf/LIBERO-plus); follow `./examples/LIBERO-Plus/libero_plus_init.py` notes | `bash ./examples/LIBERO-Plus/eval_libero_plus.sh` (7 dims / 7 GPUs) |
| **SimplerEnv** | [repo](https://github.com/simpler-env/SimplerEnv); set `SimplerEnv_PATH`, `sim_python`, `MODEL_PATH` | `bash examples/SimplerEnv/eval_files/auto_eval_scripts/batch_evaluate.sh` |

For SimplerEnv, compute success rates afterwards:

```bash
# <task_suite>: pick_coke_can | move_near | drawer | long_horizon_apple_in_drawer | bridge_put_on
bash ./examples/SimplerEnv/eval_files/auto_eval_scripts/calc_success_rate.sh <task_suite> <model_path> <log_dir>
```

> Ensure every parallel process has a GPU and that all checkpoint paths are correct.
> Reduce the parallelization in the launch scripts if you have fewer GPUs.

<a id="deployment"></a>
## 🚀 Deployment

The model is served as a **WebSocket policy server**; simulators / robots connect as
clients (msgpack-numpy protocol).

```bash
# Start the server (loads config + weights from the checkpoint dir)
python -m deployment.model_server.server_policy \
  --ckpt_path /path/to/run/checkpoints/steps_XXXXX_pytorch_model.pt \
  --port 10093 --cuda 0

# Smoke-test the connection from another shell
python deployment/model_server/debug_server_policy.py --host 127.0.0.1 --port 10093 --test infer
```

`server_policy.py` rebuilds the model from the run's `config.yaml`, so the encoder
referenced by `base_encoder` (HF dir or `.pt`) must be present at deploy time.

<a id="acknowledgement"></a>
## 🤝 Acknowledgement

We extend our sincere gratitude to the [starVLA](https://github.com/starVLA/starVLA)
and [V-JEPA2](https://github.com/facebookresearch/vjepa2) projects for their invaluable
open-source contributions.

<a id="citation"></a>
## 📝 Citation

If you find our code or models useful, please cite [the paper](https://arxiv.org/abs/2602.10098):

```bibtex
@misc{vlajepa2026,
      title={VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model},
      author={Jingwen Sun and Wenyao Zhang and Zekun Qi and Shaojie Ren and Zezhi Liu and Hanxin Zhu and Guangzhong Sun and Xin Jin and Zhibo Chen},
      year={2026},
      eprint={2602.10098},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.10098},
}
```
