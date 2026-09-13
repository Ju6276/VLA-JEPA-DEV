"""Past-only data flow for annotation-free spatial goal training."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from starVLA.dataloader.spatial_history import select_history_indices
from starVLA.dataloader.gr00t_lerobot import datasets as datasets_module
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset


def test_history_uses_seconds_and_only_earlier_frames():
    times = np.array([0.0, 0.19, 0.43, 0.59, 0.83, 1.02, 1.21, 1.43])
    indices, valid, ages = select_history_indices(times, 6, [-0.8, -0.4], 0.25)
    np.testing.assert_array_equal(indices, [1, 3])
    assert valid.all()
    np.testing.assert_allclose(ages, [1.02, 0.62])
    assert np.all(times[indices] <= times[6] + [-0.8, -0.4])
    # Future pixels/timestamps cannot replace past observations.
    changed = np.append(times[:7], [1.22, 1.23, 5.0])
    np.testing.assert_array_equal(select_history_indices(changed, 6, [-0.8, -0.4], 0.25)[0], indices)


def test_episode_start_and_sparse_history_are_masked():
    times = [0.0, 0.7, 1.0]
    for step, expected in [(0, [-1, -1]), (1, [-1, -1]), (2, [-1, -1])]:
        indices, valid, ages = select_history_indices(times, step, [-0.8, -0.4], 0.15)
        np.testing.assert_array_equal(indices, expected)
        assert not valid.any()
        assert not ages.any()
    indices, valid, ages = select_history_indices([0.0, 0.4, 0.8], 2, [-0.8, -0.4], 0.0)
    np.testing.assert_array_equal(indices, [0, 1])
    assert valid.all()
    np.testing.assert_allclose(ages, [0.8, 0.4])


@pytest.mark.parametrize("times,offsets,tolerance", [
    ([0.0, 0.0], [-0.4], 0.15), ([0.0, float("nan")], [-0.4], 0.15),
    ([0.0, 0.4], [0.0], 0.15), ([0.0, 0.4], [0.1], 0.15),
    ([0.0, 0.4], [-0.4, -0.8], 0.15), ([0.0, 0.4], [-0.4, -0.4], 0.15),
    ([0.0, 0.4], [-0.4], -1.0), ([0.0, 0.4], [-0.4], float("inf")),
])
def test_invalid_temporal_contract_fails(times, offsets, tolerance):
    with pytest.raises(ValueError):
        select_history_indices(times, 1, offsets, tolerance)


def make_sample_pipeline(monkeypatch, *, step=12, spatial=True, mutate_video=False):
    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    times = np.arange(21, dtype=np.float64) / 10
    dataset.curr_traj_data = pd.DataFrame({"timestamp": times})
    dataset._modality_keys = {
        "video": ["video.ego"], "language": ["annotation.task"],
        "state": ["state.joints"], "action": ["action.joints"],
    }
    dataset.video_backend, dataset.video_backend_kwargs = "decord", {}
    dataset.get_video_path = lambda episode, key: Path(f"/episode_{episode}/{key}.mp4")

    def frame(index):
        image = np.full((40, 80, 3), index * 10, dtype=np.uint8)
        image[:, 35:45] = 255
        return image

    def get_step(episode, base_index):
        assert episode == 17 and base_index == step
        return {"video.ego": np.stack([frame(step), frame(min(step + 4, 20))]),
                "annotation.task": ["pick up the cup"],
                "state.joints": np.arange(6).reshape(2, 3),
                "action.joints": np.arange(8).reshape(4, 2)}

    def transform(data):
        if mutate_video:
            data["video.ego"][:] = 0
        data["action.joints"] = data["action.joints"] / 10
        return data

    dataset.get_step_data = get_step
    dataset.transforms = transform
    decoded_calls = []

    def decode(path, timestamps, **kwargs):
        decoded_calls.append((path, np.asarray(timestamps).copy()))
        return np.stack([frame(round(time * 10)) for time in timestamps])

    monkeypatch.setattr(datasets_module, "get_frames_by_timestamps", decode)
    mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mixture.sample_step = lambda index: (dataset, 17, step)
    mixture.with_state = True
    mixture.resolution_size, mixture.video_resolution_size = 224, 384
    mixture.duplicate_single_view = False
    mixture.spatial_goal_enabled = spatial
    mixture.history_offsets_seconds = [-0.8, -0.4]
    mixture.history_tolerance_seconds = 0.15
    return mixture, decoded_calls, times


def test_actual_mixture_emits_high_resolution_current_and_real_history(monkeypatch):
    mixture, calls, times = make_sample_pipeline(monkeypatch, mutate_video=True)
    sample = mixture[0]
    assert sample["image"][0].size == (224, 224)
    assert not np.asarray(sample["image"][0]).any()
    assert sample["jepa_image"][0].size == (384, 384)
    assert all(image.size == (384, 384) for image in sample["history_images"])
    assert sample["timestamp"] == times[12]
    assert sample["history_valid"].all()
    np.testing.assert_allclose(sample["history_ages"], [0.8, 0.4])
    assert len(calls) == 1
    assert calls[0][0] == "/episode_17/ego.mp4"
    np.testing.assert_allclose(calls[0][1], [0.4, 0.8])
    assert np.all(calls[0][1] < sample["timestamp"])
    # Current high-resolution pixels are shared with the independent target
    # video, unaffected by an in-place augmentation of Qwen's input data.
    np.testing.assert_array_equal(np.asarray(sample["jepa_image"][0]), sample["video"][0, 0])
    assert np.asarray(sample["jepa_image"][0])[0, 0, 0] == 120
    assert np.asarray(sample["history_images"][0])[0, 0, 0] == 40
    assert np.asarray(sample["history_images"][1])[0, 0, 0] == 80
    np.testing.assert_allclose(sample["action"], np.arange(8).reshape(4, 2) / 10, atol=0.001)
    assert sample["state"].shape == (1, 3)
    assert "region_targets" not in sample


def test_missing_history_does_not_decode_or_cross_episode(monkeypatch):
    mixture, calls, _ = make_sample_pipeline(monkeypatch, step=0)
    sample = mixture[0]
    assert calls == []
    assert not sample["history_valid"].any()
    assert not sample["history_ages"].any()
    assert len(sample["history_images"]) == 2
    for image in sample["history_images"]:
        np.testing.assert_array_equal(np.asarray(image), np.asarray(sample["jepa_image"][0]))


def test_disabled_history_preserves_legacy_sample_and_transform(monkeypatch):
    mixture, calls, _ = make_sample_pipeline(monkeypatch, spatial=False, mutate_video=True)
    sample = mixture[0]
    assert set(sample) == {"action", "image", "lang", "video", "state"}
    assert not sample["video"].any()
    assert calls == []


@pytest.mark.parametrize("interface,state_dim,action_dim", [("simple", 32, 36), ("sonic", 46, 78)])
def test_spatial_configs_align_train_and_deploy_history(interface, state_dim, action_dim):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / f"scripts/config/vlajepa_{interface}_spatial_goal.yaml")
    data, spatial = cfg.datasets.vla_data, cfg.framework.spatial_goal
    assert data.spatial_goal_enabled and spatial.enabled
    assert data.history_offsets_seconds == spatial.history_offsets_seconds
    assert data.history_tolerance_seconds == spatial.history_tolerance_seconds
    cfg.framework.spatial_goal.history_offsets_seconds = [-1.0, -0.5]
    assert data.history_offsets_seconds == [-1.0, -0.5]
    assert cfg.framework.action_model.state_dim == state_dim
    assert cfg.framework.action_model.action_dim == action_dim
    assert cfg.framework.delta_jepa.lambda_delta == cfg.framework.delta_jepa.lambda_ctrl == 0
    assert data.require_full_horizon is False


def test_factory_forwards_resolved_history_options(monkeypatch, tmp_path):
    from starVLA.dataloader import lerobot_datasets as loader

    monkeypatch.setattr(loader, "LeRobotSingleDataset", lambda **kwargs: kwargs)
    monkeypatch.setattr(loader, "LeRobotMixtureDataset", lambda datasets, **kwargs: kwargs)
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.load(root / "scripts/config/vlajepa_simple_spatial_goal.yaml")
    cfg.datasets.vla_data.data_root_dir = str(tmp_path)
    cfg.framework.spatial_goal.history_offsets_seconds = [-1.0, -0.5]
    options = loader.get_vla_dataset(cfg.datasets.vla_data, action_horizon=30, video_horizon=8)
    assert options["spatial_goal_enabled"]
    assert options["history_offsets_seconds"] == [-1.0, -0.5]
    assert options["history_tolerance_seconds"] == 0.15
    assert options["video_resolution_size"] == 384
