# Official VLA-JEPA on HumanoidArena SONIC40

This branch starts directly from `ginwind/VLA-JEPA` `main` at `ec8c70f`. It
does not contain the separate V-JEPA 2.1, SONIC latent78, music-JEPA, or
one-stage experiments from other development branches.

The benchmark contract matches GR00T N1.7 and Psi0: front RGB
`[480,640,3]`, state `[64]`, exact task prompt, and a `[30,40]`
`semantic_v3` reference-pose action chunk. Training uses the original
VLA-JEPA V-JEPA 2 encoder (`facebook/vjepa2-vitl-fpc64-256`).

Use the same converted LeRobot v2.1 data prepared by the GR00T adapter:

```bash
export DATA_ROOT=/path/to/shared/data_v2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NUM_PROCESSES=8
export GLOBAL_BATCH_SIZE=8
export WANDB_MODE=online
export WANDB_ENTITY=your-wandb-entity
# Use either an existing `wandb login` entry or export WANDB_API_KEY.

bash scripts/vlajepa_humanoidarena_sonic40.sh opendoor
```

With eight GPUs and gradient accumulation one, global batch 8 means per-device
batch 1. Each task is trained separately for 100,000 optimizer steps. Valid
task names are `opendoor`, `double_desk`, `football`, `pp_box`, `boxing`,
`sit_sofa`, and `vision_navi`.

Serve a checkpoint file such as `checkpoints/steps_100000_pytorch_model.pt`:

```bash
python examples/HumanoidArena/serve_humanoidarena.py \
  --policy-path /path/to/steps_100000_pytorch_model.pt \
  --device cuda:0 --host 127.0.0.1 --port 18080
```

HumanoidArena's parallel evaluator can start this script directly using
`SERVER_SCRIPT`, with `SONIC_VLA_ACTION_FORMAT=semantic_v3`. Use one smoke
episode first, followed by the same three seeds and 20 repeats per seed used by
the GR00T and Psi0 comparison.

For example, OpenDoor smoke evaluation is:

```bash
cd /path/to/HumanoidArena
AUTO_ACTIVATE_CONDA=0 \
EVAL_PYTHON=/path/to/unitree_sim_env/bin/python \
SERVER_PYTHON=/path/to/VLA-JEPA/env/bin/python \
SERVER_SCRIPT=/path/to/VLA-JEPA/examples/HumanoidArena/serve_humanoidarena.py \
SONIC_POLICY_ROOT=/path/to/GEAR-SONIC/release \
MODEL_PATHS_CSV=/path/to/steps_100000_pytorch_model.pt \
ENV_CONFIG_YAML=tasks/common_test_config/base_test/open_door_sonic_test.yaml \
RESULTS_DIR=/path/to/HumanoidArena/eval_results/vlajepa_opendoor_base_smoke \
SEEDS_OVERRIDE=0 REPEATS_PER_SEED=1 RESUME_LATEST=0 \
NUM_WORKERS=1 SERVER_GPU_IDS=0 ISAAC_DEVICE=cuda:0 \
HEADLESS=1 RECORD_VIDEO_EVERY_N=0 \
SONIC_VLA_ACTION_FORMAT=semantic_v3 \
bash isaaclab_twist2_g1/script/eval_scripts/sonic_pi05/HSI_open_door_run_vla_eval_parallel.sh
```

After that produces one real episode without an infrastructure error, change
to `SEEDS_OVERRIDE='0 1 2'`, `REPEATS_PER_SEED=20`, enable video with
`RECORD_VIDEO_EVERY_N=1`, and use a fresh formal-results directory.
