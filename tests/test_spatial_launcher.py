"""Exercise the real launch scripts while replacing only Accelerate."""

import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf
import pytest


ROOT = Path(__file__).resolve().parents[1]


def launch(tmp_path, interface, extra=(), **environment):
    executable = tmp_path / "accelerate"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$SPATIAL_TEST_ARGUMENTS"\n')
    executable.chmod(0o755)
    checkpoint = tmp_path / "jepa.pt"
    checkpoint.touch()
    output = tmp_path / "arguments"
    env = dict(os.environ, DATA_ROOT=str(tmp_path), VJEPA21_CKPT=str(checkpoint),
               PATH=f"{tmp_path}:{os.environ['PATH']}", SPATIAL_TEST_ARGUMENTS=str(output))
    for name in ("CONFIG_YAML", "RUN_ID", "PER_DEVICE_BATCH_SIZE", "NUM_PROCESSES"):
        env.pop(name, None)
    env.update(environment)
    subprocess.run(["bash", str(ROOT / "scripts/train_spatial_goal.sh"), interface, *extra],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    return output.read_text().splitlines()


@pytest.mark.parametrize("interface,accumulation,state,action", [("simple", "32", 32, 36), ("sonic", "4", 46, 78)])
def test_spatial_launcher_interface_defaults(tmp_path, interface, accumulation, state, action):
    arguments = launch(tmp_path, interface)
    cfg = OmegaConf.load(arguments[arguments.index("--config_yaml") + 1])
    assert cfg.framework.spatial_goal.enabled
    assert cfg.framework.action_model.state_dim == state
    assert cfg.framework.action_model.action_dim == action
    assert arguments[arguments.index("--num_processes") + 1] == "8"
    assert arguments[arguments.index("--datasets.vla_data.per_device_batch_size") + 1] == "1"
    assert arguments[arguments.index("--trainer.gradient_accumulation_steps") + 1] == accumulation
    assert cfg.datasets.vla_data.per_device_batch_size == 1
    assert cfg.trainer.gradient_accumulation_steps == int(accumulation)
    assert arguments[arguments.index("--run_id") + 1] == f"{interface}_spatial_goal_8xa100"
    assert cfg.framework.delta_jepa.subgoals_path is None


def test_spatial_launcher_preserves_final_user_overrides_and_resume(tmp_path):
    extra = ["--trainer.gradient_accumulation_steps", "2", "--trainer.resume_from_checkpoint",
             "/tmp/saved run/checkpoints/steps_10000", "--framework.spatial_goal.score_weight", "0.0"]
    arguments = launch(tmp_path, "simple", extra, RUN_ID="spatial_resume", PER_DEVICE_BATCH_SIZE="2", NUM_PROCESSES="4")
    assert arguments[-len(extra):] == extra
    positions = [i for i, key in enumerate(arguments) if key == "--trainer.gradient_accumulation_steps"]
    assert [arguments[i + 1] for i in positions] == ["32", "2"]
    assert arguments[arguments.index("--run_id") + 1] == "spatial_resume"
    assert arguments[arguments.index("--num_processes") + 1] == "4"
    assert arguments[arguments.index("--datasets.vla_data.per_device_batch_size") + 1] == "2"


@pytest.mark.parametrize("arguments", [[], ["unknown"]])
def test_spatial_launcher_rejects_missing_or_unknown_interface(arguments, tmp_path):
    process = subprocess.run(["bash", str(ROOT / "scripts/train_spatial_goal.sh"), *arguments],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 2
    assert "simple" in process.stderr and "sonic" in process.stderr
