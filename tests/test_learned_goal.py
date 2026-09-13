"""CPU integration checks with small backbones; no pretrained downloads needed."""

import importlib
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pytest
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from starVLA.model.modules.world_model.delta_jepa import LatentGoalPredictor


framework = importlib.import_module("starVLA.model.framework.VLA_JEPA")
ROOT = Path(__file__).resolve().parents[1]


class TinyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 8)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=8))
        self.processor = SimpleNamespace(tokenizer=None)

    def build_qwenvl_inputs(self, images, instructions, **kwargs):
        ids = [[4 if "right" in instruction else 3, 1, 2] for instruction in instructions]
        return {"input_ids": torch.tensor(ids, device=self.embedding.weight.device)}

    def forward(self, input_ids, **kwargs):
        # Causal mixing makes both special tokens depend on the instruction.
        return SimpleNamespace(hidden_states=[self.embedding(input_ids).cumsum(dim=1)])


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 6)
        self.config = SimpleNamespace(hidden_size=6, tubelet_size=2, image_size=32)

    @property
    def device(self):
        return self.proj.weight.device

    def get_vision_features(self, pixel_values_videos):
        # Deliberately mixes ALL frames, like a bidirectional video encoder.
        x = pixel_values_videos.mean(dim=(1, 3, 4)) / 255
        features = self.proj(x)
        return features[:, None].expand(-1, pixel_values_videos.shape[1] // 2 * 4, -1)


class TinyProcessor:
    def __call__(self, videos, **kwargs):
        return {"pixel_values_videos": torch.from_numpy(np.stack(videos)).float()}


class TinyWorldPredictor(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = nn.Linear(14, 6)
        self.inputs = []

    def forward(self, x, actions):
        self.inputs.append(x.detach().clone())
        task = actions.mean(dim=1)[:, None].expand(-1, x.shape[1], -1)
        return self.proj(torch.cat([x, task], dim=-1))


class TinyActionModel(nn.Module):
    def __init__(self, action_dim=3, horizon=4):
        super().__init__()
        self.proj = nn.Linear(8, action_dim)
        self.horizon = horizon

    def predict_action(self, tokens, state):
        return self.proj(tokens.mean(dim=1))[:, None].expand(-1, self.horizon, -1)

    def forward(self, tokens, actions, state):
        return F.mse_loss(self.predict_action(tokens, state), actions)


@pytest.fixture
def make_model(monkeypatch):
    monkeypatch.setattr(framework, "get_vlm_model", lambda config: TinyQwen())
    monkeypatch.setattr(framework, "get_action_model", lambda config: TinyActionModel(
        config.framework.action_model.action_dim, config.framework.action_model.action_horizon,
    ))
    monkeypatch.setattr(framework.AutoModel, "from_pretrained", lambda *a, **kw: TinyEncoder())
    monkeypatch.setattr(framework.AutoVideoProcessor, "from_pretrained", lambda *a, **kw: TinyProcessor())
    monkeypatch.setattr(framework, "VisionTransformerPredictorAC", TinyWorldPredictor)
    monkeypatch.setattr(framework.VLA_JEPA, "expand_tokenizer", lambda *a, **kw: (["<action>"], [1], 2))

    def build(learned=True, action_dim=3, state_dim=2, horizon=4, proposal=True):
        cfg = OmegaConf.create({
            "framework": {
                "action_model": {
                    "diffusion_model_cfg": {}, "action_horizon": horizon, "action_dim": action_dim,
                    "state_dim": state_dim, "future_action_window_size": horizon - 1, "past_action_window_size": 0,
                },
                "vj2_model": {
                    "base_encoder": "tiny", "num_video_views": 1, "num_frames": 4,
                    "depth": 1, "num_heads": 1, "special_action_token": "<action_{}>",
                    "num_action_tokens_per_timestep": 1, "num_embodied_action_tokens_per_instruction": 1,
                },
                "delta_jepa": {
                    "enabled": True, "use_verifier": True, "hidden_dim": 16,
                    "goal_action_proposal_enabled": proposal, "subgoals_path": None,
                },
            },
            "datasets": {"vla_data": {"image_size": [8, 8], "CoT_prompt": "{instruction}"}},
            "trainer": {"repeated_diffusion_steps": 1},
        })
        if learned:
            cfg.framework.delta_jepa.learned_goal_enabled = True
        return framework.VLA_JEPA(cfg)

    return build


@pytest.fixture
def examples():
    samples = []
    for index in range(2):
        video = np.zeros((1, 4, 8, 8, 3), dtype=np.uint8)
        video[:, 0, :, :, index] = 50
        video[:, 1:, :, :, 2] = 200
        samples.append({
            "image": [Image.fromarray(video[0, 0])], "video": video,
            "lang": "pick left" if index == 0 else "pick right",
            "action": np.full((4, 3), 0.2 + index * 0.1, dtype=np.float32),
            "state": np.full((1, 2), index * 0.1, dtype=np.float32),
        })
    return samples


def inference_inputs(examples):
    return {
        "batch_images": [sample["image"] for sample in examples],
        "instructions": [sample["lang"] for sample in examples],
        "state": np.stack([sample["state"] for sample in examples]),
    }


def test_goal_head_learns_task_conditioned_targets():
    torch.manual_seed(0)
    head = LatentGoalPredictor(6, 8, 2, hidden_dim=16)
    current = torch.ones(2, 6)
    task = torch.stack([torch.zeros(3, 8), torch.ones(3, 8)])
    target = torch.eye(6)[:2]
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    initial = (1 - F.cosine_similarity(head(current, task, None), target)).mean().item()
    for _ in range(80):
        loss = (1 - F.cosine_similarity(head(current, task, None), target)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < initial * 0.05
    torch.testing.assert_close(head(current, task, None).norm(dim=-1), torch.ones(2))


def test_future_is_only_a_label(make_model, examples):
    model = make_model()
    videos = np.stack([sample["video"] for sample in examples])
    images = [sample["image"] for sample in examples]
    current_a, target_a = model._encode_training_goal_pair(images, videos)
    changed = videos.copy()
    changed[:, :, 1:] = 255 - changed[:, :, 1:]
    current_b, target_b = model._encode_training_goal_pair(images, changed)
    torch.testing.assert_close(current_a, current_b)
    assert not torch.allclose(target_a, target_b)
    assert not current_a.requires_grad and not target_a.requires_grad


def test_training_backward_and_current_only_world_model(make_model, examples):
    torch.manual_seed(2)
    model = make_model()
    losses = model(examples)
    assert set(losses) == {
        "action_loss", "wm_loss", "delta_loss", "ctrl_loss", "action_prior_loss",
        "goal_proposal_loss", "goal_prediction_loss",
    }
    assert all(torch.isfinite(loss).item() for loss in losses.values())
    sum(losses.values()).backward()
    for module in (model.goal_predictor, model.goal_action_proposal, model.vj_predictor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    assert all(p.grad is None for p in model.vj_encoder.parameters())
    training_input = model.vj_predictor.inputs[0]
    model.eval().predict_action(**inference_inputs(examples), num_candidates=1)
    torch.testing.assert_close(training_input, model.vj_predictor.inputs[-1])


@pytest.mark.parametrize("learned", [True, False])
def test_all_goal_heads_share_the_deployed_current_latent(make_model, examples, learned):
    model = make_model(learned=learned).eval()
    original_encode = model.vj_encoder.get_vision_features

    def temporal_encode(pixel_values_videos):
        tokens = original_encode(pixel_values_videos).clone()
        block_size = tokens.shape[1] // model.num_temporal_frames
        for block in range(model.num_temporal_frames):
            tokens[:, block * block_size:(block + 1) * block_size, block] += 2 * (block + 1)
        return tokens

    model.vj_encoder.get_vision_features = temporal_encode
    goal_inputs, proposal_inputs, inverse_inputs = [], [], []
    hooks = [
        model.goal_action_proposal.register_forward_pre_hook(
            lambda _, args: proposal_inputs.append(args[0].detach().clone())),
        model.inv_dyn_decoder.register_forward_pre_hook(
            lambda _, args: inverse_inputs.append(args[0].detach().clone())),
    ]
    if model.goal_predictor is not None:
        hooks.append(model.goal_predictor.register_forward_pre_hook(
            lambda _, args: goal_inputs.append(args[0].detach().clone())))
    model(examples)
    model.predict_action(
        **inference_inputs(examples), num_candidates=1,
        subgoal_images=None if learned else [sample["image"][0] for sample in examples],
    )
    for hook in hooks:
        hook.remove()

    current, target = model._encode_training_goal_pair(
        [sample["image"] for sample in examples], np.stack([sample["video"] for sample in examples]),
    )
    expected_current = model._current_visual_latent(current)
    for actual in goal_inputs + proposal_inputs:
        torch.testing.assert_close(actual, expected_current)
    from starVLA.model.modules.world_model.delta_jepa import pool_vjepa_tokens
    if learned:
        torch.testing.assert_close(inverse_inputs[0], pool_vjepa_tokens(target) - expected_current)


def test_frozen_encoder_stays_in_eval_and_out_of_optimizer(make_model):
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

    model = make_model().train()
    assert not model.vj_encoder.training
    assert model.goal_predictor.training
    assert all(not p.requires_grad for p in model.vj_encoder.parameters())
    model.config.trainer.learning_rate = {"base": 1e-4, "vj_encoder": 1e-5}
    optimizer_params = {
        id(param) for group in build_param_lr_groups(model, model.config) for param in group["params"]
    }
    assert optimizer_params.isdisjoint(id(p) for p in model.vj_encoder.parameters())
    assert {id(p) for p in model.goal_predictor.parameters()} <= optimizer_params


def test_action_repeat_configuration_is_honored(make_model, examples):
    model = make_model()
    del model.config.trainer.repeated_diffusion_steps
    # Legacy checkpoints stored an unused action-model value. Preserve their
    # effective default, and let the trainer option explicitly control repeats.
    model.config.framework.action_model.repeated_diffusion_steps = 8
    batches = []
    hook = model.action_model.register_forward_pre_hook(lambda _, args: batches.append(args[1].shape[0]))
    model(examples)
    model.config.trainer.repeated_diffusion_steps = 2
    model(examples)
    hook.remove()
    assert batches == [len(examples) * 4, len(examples) * 2]


@pytest.mark.parametrize("num_candidates", [1, 3])
def test_deployment_without_external_goal(make_model, examples, num_candidates):
    model = make_model().eval()
    calls = []
    hook = model.goal_predictor.register_forward_hook(lambda *args: calls.append(1))
    result = model.predict_action(**inference_inputs(examples), num_candidates=num_candidates)
    hook.remove()
    assert result["goal_source"] == "predicted"
    assert result["subgoal_index"] is None
    assert result["goal_proposal_used"]
    assert result["normalized_actions"].shape == (2, 4, 3)
    assert result["all_candidates"].shape == (2, num_candidates, 4, 3)
    assert np.isfinite(result["verification_scores"]).all()
    assert len(calls) == 1  # All candidates must be scored against the same goal.


def test_bfloat16_single_proposal_export(make_model, examples):
    model = make_model().eval().to(torch.bfloat16)
    result = model.predict_action(**inference_inputs(examples), num_candidates=1)
    assert result["normalized_actions"].dtype == np.float32
    assert np.isfinite(result["normalized_actions"]).all()


def test_explicit_goal_override(make_model, examples, monkeypatch):
    model = make_model().eval()
    def unexpected_prediction(*args):
        raise AssertionError("An explicit image should override the predicted goal")
    monkeypatch.setattr(model.goal_predictor, "forward", unexpected_prediction)
    result = model.predict_action(
        **inference_inputs(examples), num_candidates=1,
        subgoal_images=[sample["image"][0] for sample in examples],
    )
    assert result["goal_source"] == "images"


def test_tracker_can_override_learned_goal(make_model, examples):
    from starVLA.tools.subgoal_tracker import SubgoalTracker

    model = make_model().eval()
    model.subgoal_tracker = SubgoalTracker(
        subgoal_images=[sample["image"][0] for sample in examples],
    )
    inputs = inference_inputs(examples[:1])
    result = model.predict_action(**inputs, num_candidates=1)
    assert result["goal_source"] == "tracker"
    assert result["subgoal_index"] is not None


def test_real_world_predictor_endpoint_training(make_model, examples):
    from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC

    model = make_model()
    model.vj_predictor = VisionTransformerPredictorAC(
        img_size=(32, 32), num_frames=2, tubelet_size=1,
        embed_dim=6, predictor_embed_dim=64, depth=1, num_heads=4,
        action_embed_dim=8, num_add_tokens=1,
    )
    losses = model(examples)
    sum(losses.values()).backward()
    assert model.vj_predictor.predictor_proj.weight.grad.abs().sum() > 0
    result = model.eval().predict_action(**inference_inputs(examples), num_candidates=2)
    assert result["goal_source"] == "predicted"
    assert np.isfinite(result["verification_scores"]).all()


@pytest.mark.parametrize("head_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("verify", [False, True])
def test_real_action_head_precision_and_export(make_model, examples, monkeypatch, head_dtype, verify):
    from starVLA.model.modules.action_model.GR00T_ActionHeader import DiTConfig, FlowmatchingActionHead

    monkeypatch.setitem(DiTConfig, "DiT-test", {
        "input_embedding_dim": 32, "attention_head_dim": 8, "num_attention_heads": 4,
    })
    action_cfg = OmegaConf.load(ROOT / "scripts/config/vlajepa_sonic_latent_learned_goal.yaml")
    action_cfg.framework.action_model.update({
        "action_model_type": "DiT-test", "hidden_size": 32,
        "state_dim": 2, "action_dim": 3, "action_horizon": 4,
        "future_action_window_size": 3, "num_target_vision_tokens": 2,
        "num_inference_timesteps": 2,
    })
    action_cfg.framework.action_model.diffusion_model_cfg.update({
        "cross_attention_dim": 8, "output_dim": 32, "num_layers": 2,
    })
    model = make_model().eval()
    model.qwen_vl_interface.to(torch.bfloat16)
    model.action_model = FlowmatchingActionHead(action_cfg).to(head_dtype).eval()
    output = model.predict_action(**inference_inputs(examples), use_verifier=verify, num_candidates=3)
    assert output["normalized_actions"].shape == (2, 4, 3)
    assert output["normalized_actions"].dtype == np.float32
    assert np.isfinite(output["normalized_actions"]).all()
    if verify:
        assert output["all_candidates"].shape == (2, 3, 4, 3)
        assert np.isfinite(output["verification_scores"]).all()

    # Exercise the real flow head's training boundary and preserve gradients
    # into a differently typed language backbone.
    task = torch.randn(2, 2, 8, dtype=torch.bfloat16, requires_grad=True)
    loss = model.action_model(task, torch.randn(2, 4, 3), torch.randn(2, 1, 2))
    loss.backward()
    assert torch.isfinite(loss)
    assert task.grad is not None and torch.isfinite(task.grad).all() and task.grad.abs().sum() > 0


def test_legacy_configuration_and_weights(make_model, examples):
    legacy = make_model(learned=False)
    assert legacy.goal_predictor is None
    assert not any(key.startswith("goal_predictor.") for key in legacy.state_dict())
    assert "goal_prediction_loss" not in legacy(examples)
    make_model(learned=False).load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(ValueError, match="checkpoint trained with"):
        legacy.predict_action(**inference_inputs(examples), num_candidates=1)
    result = legacy.predict_action(
        **inference_inputs(examples), num_candidates=1,
        subgoal_images=[sample["image"][0] for sample in examples],
    )
    assert result["goal_source"] == "images"


def test_original_prior_checkpoint_and_candidate_path(make_model, examples):
    original = make_model(learned=False, proposal=False).eval()
    restored = make_model(learned=False, proposal=False).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    assert restored.goal_predictor is None and restored.goal_action_proposal is None
    result = restored.predict_action(
        **inference_inputs(examples), num_candidates=3,
        subgoal_images=[sample["image"][0] for sample in examples],
    )
    assert not result["goal_proposal_used"]
    assert result["all_candidates"].shape == (2, 3, 4, 3)
    assert np.isfinite(result["verification_scores"]).all()


def test_learned_goal_checkpoint_roundtrip(make_model, examples, tmp_path):
    model = make_model().eval()
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = make_model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    for key in ("normalized_actions", "verification_scores"):
        expected = model.predict_action(**inference_inputs(examples), num_candidates=1)[key]
        actual = restored.predict_action(**inference_inputs(examples), num_candidates=1)[key]
        np.testing.assert_allclose(actual, expected)
    with pytest.raises(RuntimeError, match="goal_predictor"):
        restored.load_state_dict(make_model(learned=False).state_dict(), strict=True)


@pytest.mark.parametrize("script,action_dim,state_dim,horizon", [
    ("train_g1_delta_jepa_8xa100.sh", 36, 32, 30),
    ("train_sonic_learned_goal.sh", 78, 46, 40),
])
def test_training_launcher_needs_no_subgoal_files(tmp_path, script, action_dim, state_dim, horizon):
    executable = tmp_path / "accelerate"
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$TEST_LAUNCH_ARGS"\n')
    executable.chmod(0o755)
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.touch()
    args_file = tmp_path / "args"
    env = dict(os.environ, DATA_ROOT=str(tmp_path), VJEPA21_CKPT=str(checkpoint),
               PATH=f"{tmp_path}:{os.environ['PATH']}", TEST_LAUNCH_ARGS=str(args_file))
    env.pop("SUBGOALS_PATH", None)
    env.pop("CONFIG_YAML", None)
    env.pop("RUN_ID", None)
    subprocess.run(["bash", str(ROOT / "scripts" / script)],
                   env=env, check=True, capture_output=True, text=True)
    args = args_file.read_text().splitlines()
    cfg = OmegaConf.load(ROOT / args[args.index("--config_yaml") + 1])
    assert cfg.framework.delta_jepa.learned_goal_enabled
    assert cfg.framework.delta_jepa.subgoals_path is None
    assert "--framework.delta_jepa.subgoals_path" not in args
    assert cfg.framework.action_model.action_dim == action_dim
    assert cfg.framework.action_model.state_dim == state_dim
    assert cfg.framework.action_model.action_horizon == horizon
    assert cfg.datasets.vla_data.video_frame_offsets[-1] == horizon
    assert args[args.index("--run_id") + 1] == cfg.run_id


def test_dataloader_goal_time_alignment(monkeypatch, tmp_path):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    from starVLA.dataloader import lerobot_datasets as loader

    # Intercept only dataset I/O; exercise the actual G1 modality configuration.
    monkeypatch.setattr(loader, "LeRobotSingleDataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(loader, "LeRobotMixtureDataset", lambda datasets, **kwargs: datasets)
    for name in (
        "vlajepa_g1_pick_between_tables_ft.yaml",
        "vlajepa_g1_pick_between_tables_vjepa21.yaml",
        "vlajepa_g1_pick_between_tables_vjepa21_8xa100.yaml",
    ):
        cfg = OmegaConf.load(ROOT / "scripts/config" / name)
        cfg.datasets.vla_data.data_root_dir = str(tmp_path)
        mixture = loader.get_vla_dataset(
            cfg.datasets.vla_data,
            action_horizon=cfg.framework.action_model.action_horizon,
            video_horizon=cfg.framework.vj2_model.num_frames,
        )
        modalities = mixture[0][0]["modality_configs"]
        assert modalities["video"].delta_indices == [0, 4, 8, 12, 16, 20, 24, 30]
        assert modalities["action"].delta_indices == list(range(30))
        assert modalities["state"].delta_indices[0] == 0

    config = loader.make_LeRobotSingleDataset(
        tmp_path, "demo", "g1_pick_between_tables", video_horizon=8,
    )
    assert config["modality_configs"]["video"].delta_indices == list(range(8))
    with pytest.raises(ValueError, match="exactly"):
        loader.make_LeRobotSingleDataset(
            tmp_path, "demo", "g1_pick_between_tables", video_horizon=8, video_frame_offsets=[0, 30],
        )

    cfg = OmegaConf.load(ROOT / "scripts/config/vlajepa_sonic_latent_learned_goal.yaml")
    cfg.datasets.vla_data.data_root_dir = str(tmp_path)
    mixture = loader.get_vla_dataset(cfg.datasets.vla_data, action_horizon=40, video_horizon=8)
    modalities = mixture[0][0]["modality_configs"]
    assert modalities["action"].modality_keys == [
        "action.motion_token", "action.left_hand_joints", "action.right_hand_joints",
    ]
    assert modalities["action"].delta_indices == list(range(40))
    assert modalities["video"].delta_indices == [0, 6, 12, 18, 24, 30, 35, 40]
    assert modalities["state"].delta_indices == [0]


@pytest.mark.parametrize("action_dim,state_dim,horizon", [(36, 32, 30), (78, 46, 40)])
def test_sim_and_sonic_action_interfaces(make_model, examples, action_dim, state_dim, horizon):
    model = make_model(action_dim=action_dim, state_dim=state_dim, horizon=horizon)
    samples = [{
        **sample,
        "action": np.full((horizon, action_dim), 0.2, dtype=np.float32),
        "state": np.zeros((1, state_dim), dtype=np.float32),
    } for sample in examples]
    losses = model(samples)
    sum(losses.values()).backward()
    assert torch.isfinite(losses["goal_prediction_loss"])
    assert model.goal_action_proposal.action_head[-1].weight.grad.abs().sum() > 0
    result = model.eval().predict_action(**inference_inputs(samples), num_candidates=2)
    assert result["normalized_actions"].shape == (2, horizon, action_dim)
    assert result["goal_source"] == "predicted"
    assert np.isfinite(result["normalized_actions"]).all()
