"""Sample-window regression tests, without videos or pretrained models."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from starVLA.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    STEP_INDEX_CACHE_VERSION,
)
from starVLA.dataloader import lerobot_datasets


def make_dataset(tmp_path, lengths=(8,), actions=(0, 1, 2), video=(0, 3), full=True):
    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset._dataset_path = tmp_path
    dataset._dataset_name = "tiny"
    dataset._trajectory_ids = np.arange(len(lengths))
    dataset._trajectory_lengths = np.asarray(lengths)
    dataset._modality_keys = {"action": ["action.joints"], "video": ["video.ego"]}
    dataset._delta_indices = {
        "action.joints": np.asarray(actions), "video.ego": np.asarray(video),
    }
    dataset.delete_pause_frame = False
    dataset.require_full_horizon = full
    dataset.get_trajectory_data = lambda trajectory_id: pd.DataFrame({
        "timestamp": np.arange(lengths[trajectory_id]),
    })
    return dataset


@pytest.mark.parametrize("actions,video,expected", [
    ((0, 1, 2), (0, 3), list(range(5))),
    ((0, 1, 2, 3, 4), (0, 2), list(range(4))),
    ((0, 1), (-2, 0, 3), [2, 3, 4]),
    ((0,), (0, 7), [0]),
    ((0,), (0, 8), []),
])
def test_only_real_action_and_video_offsets_are_sampled(tmp_path, actions, video, expected):
    dataset = make_dataset(tmp_path, actions=actions, video=video)
    assert dataset._get_all_steps_single_process() == [(0, step) for step in expected]


def test_legacy_keeps_tail_indices_and_padding(tmp_path):
    dataset = make_dataset(tmp_path, full=False)
    assert dataset._get_all_steps_single_process() == [(0, step) for step in range(8)]
    actions = np.arange(8, dtype=np.float32)[:, None] + 1
    padded = dataset.retrieve_data_and_pad(actions, np.array([6, 7, 8]), 8, "zero")
    np.testing.assert_array_equal(padded[:, 0], [7, 8, 0])


def test_short_episodes_are_skipped_and_mixture_samples_only_valid_starts(tmp_path):
    dataset = make_dataset(tmp_path, lengths=(3, 8))
    dataset._all_steps = dataset._get_all_steps()
    assert dataset.all_steps == [(1, step) for step in range(5)]
    mixture = LeRobotMixtureDataset.__new__(LeRobotMixtureDataset)
    mixture.datasets = [dataset]
    mixture._dataset_sampling_weights = np.array([1.0])
    mixture.mode, mixture.epoch, mixture.seed = "train", 0, 7
    for index in range(100):
        _, trajectory, step = mixture.sample_step(index)
        assert trajectory == 1 and 0 <= step <= 4


def test_no_complete_window_raises_before_sampling(tmp_path):
    dataset = make_dataset(tmp_path, lengths=(2, 3))
    with pytest.raises(ValueError, match="no valid samples with require_full_horizon=True"):
        dataset._get_all_steps()
    # The same clear error holds when the empty index was cached.
    with pytest.raises(ValueError, match="no valid samples"):
        dataset._get_all_steps()


def test_metadata_longer_than_actual_rows_does_not_create_padded_windows(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.get_trajectory_data = lambda _: pd.DataFrame({"timestamp": range(5)})
    assert dataset._get_all_steps_single_process() == [(0, 0), (0, 1)]


def test_pause_filter_is_combined_with_complete_windows(tmp_path):
    dataset = make_dataset(tmp_path)
    dataset.delete_pause_frame = True
    dataset._get_position_and_gripper_values = lambda _: (
        np.ones((8, 3)), np.zeros(8),
    )
    assert dataset._get_all_steps_single_process() == [(0, step) for step in range(5)]


@pytest.mark.parametrize("legacy_name", [
    "steps_332420bad1ab.pkl", "steps_2d5a34b904d2.pkl", "steps_data_index.pkl",
])
def test_old_truncated_cache_does_not_override_padding_policy(tmp_path, legacy_name):
    meta = tmp_path / "meta"
    meta.mkdir()
    lengths = (100, 85)
    # SONIC-shaped legacy indices silently excluded the last 40 starts of each
    # episode, despite require_full_horizon=False in the current configuration.
    legacy_steps = [(episode, step) for episode, length in enumerate(lengths) for step in range(length - 40)]
    legacy_bytes = pickle.dumps({
        "config_key": "9cc61c8b0718", "delete_pause_frame": False, "steps": legacy_steps,
    })
    (meta / legacy_name).write_bytes(legacy_bytes)
    arguments = {"lengths": lengths, "actions": tuple(range(40)), "video": (0, 40)}
    padded = make_dataset(tmp_path, full=False, **arguments)
    complete = make_dataset(tmp_path, full=True, **arguments)
    assert padded._get_all_steps() == [
        (episode, step) for episode, length in enumerate(lengths) for step in range(length)
    ]
    assert complete._get_all_steps() == legacy_steps
    assert (meta / legacy_name).read_bytes() == legacy_bytes


def test_padded_and_complete_caches_are_separate_and_reusable(tmp_path):
    cached_keys = set()
    for full in (False, True):
        dataset = make_dataset(tmp_path, full=full)
        cached_keys.add(dataset._get_steps_config_key())
        assert dataset._get_all_steps() == [(0, step) for step in range(5 if full else 8)]
    assert len(cached_keys) == 2
    for full in (True, False):
        dataset = make_dataset(tmp_path, full=full)
        # Matching indices are reused without reading parquet or video again.
        dataset._get_all_steps_single_process = lambda: pytest.fail("valid cache was ignored")
        assert dataset._get_all_steps() == [(0, step) for step in range(5 if full else 8)]


@pytest.mark.parametrize("full", [False, True])
def test_changing_horizon_offsets_or_episode_lengths_invalidates_cache(tmp_path, full):
    original = make_dataset(tmp_path, full=full)
    original._get_all_steps()
    configurations = [
        make_dataset(tmp_path, video=(0, 5), full=full),
        make_dataset(tmp_path, actions=tuple(range(6)), full=full),
        make_dataset(tmp_path, lengths=(10,), full=full),
    ]
    for dataset in configurations:
        assert dataset._get_steps_config_key() != original._get_steps_config_key()
        expected = dataset._get_all_steps_single_process()
        assert dataset._get_all_steps() == expected


@pytest.mark.parametrize("full", [False, True])
def test_changing_episode_ids_or_pause_filter_invalidates_cache(tmp_path, full):
    original = make_dataset(tmp_path, full=full)
    original._get_all_steps()

    renamed = make_dataset(tmp_path, full=full)
    renamed._trajectory_ids = np.array([5])
    renamed.get_trajectory_data = lambda _: pd.DataFrame({"timestamp": range(8)})
    assert renamed._get_steps_config_key() != original._get_steps_config_key()
    assert renamed._get_all_steps() == [(5, step) for step in range(5 if full else 8)]

    moving_only = make_dataset(tmp_path, full=full)
    moving_only.delete_pause_frame = True
    motion = np.zeros((8, 3))
    motion[0] = 1
    moving_only._get_position_and_gripper_values = lambda _: (motion, np.zeros(8))
    assert moving_only._get_steps_config_key() != original._get_steps_config_key()
    assert moving_only._get_all_steps() == [(0, 0)]


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("mismatch", ["key", "missing_policy", "wrong_policy"])
def test_cache_metadata_mismatch_is_rebuilt(tmp_path, full, mismatch):
    dataset = make_dataset(tmp_path, full=full)
    config_key = dataset._get_steps_config_key()
    path = tmp_path / "meta" / f"steps_v{STEP_INDEX_CACHE_VERSION}_{config_key}.pkl"
    path.parent.mkdir()
    cache = {"config_key": config_key, "config": dataset._get_steps_config(), "steps": [(0, 7)]}
    if mismatch == "key":
        cache["config_key"] = "different"
    elif mismatch == "missing_policy":
        del cache["config"]["require_full_horizon"]
    else:
        cache["config"]["require_full_horizon"] = not full
    with path.open("wb") as f:
        pickle.dump(cache, f)
    assert dataset._get_all_steps() == [(0, step) for step in range(5 if full else 8)]


@pytest.mark.parametrize("full", [False, True])
def test_readonly_cache_location_still_computes_valid_windows(tmp_path, monkeypatch, full):
    dataset = make_dataset(tmp_path, full=full)

    def readonly(*args, **kwargs):
        raise PermissionError("dataset is read-only")

    monkeypatch.setattr("starVLA.dataloader.gr00t_lerobot.datasets.tempfile.NamedTemporaryFile", readonly)
    assert dataset._get_all_steps() == [(0, step) for step in range(5 if full else 8)]


@pytest.mark.parametrize("full", [False, True])
def test_mixture_builder_forwards_option_with_legacy_default(tmp_path, monkeypatch, full):
    seen = []
    monkeypatch.setattr(lerobot_datasets, "make_LeRobotSingleDataset", lambda *a, **kw: seen.append(kw) or object())
    monkeypatch.setattr(lerobot_datasets, "LeRobotMixtureDataset", lambda *a, **kw: None)
    monkeypatch.setitem(lerobot_datasets.DATASET_NAMED_MIXTURES, "tiny", [("tiny", 1.0, "unused")])
    config = {"data_root_dir": str(tmp_path), "data_mix": "tiny"}
    if full:
        config["require_full_horizon"] = True
    lerobot_datasets.get_vla_dataset(OmegaConf.create(config))
    assert seen[0]["require_full_horizon"] is full


def test_learned_goal_examples_preserve_legacy_padding_by_default():
    root = Path(__file__).resolve().parents[1]
    learned = []
    for path in (root / "scripts/config").glob("*.yaml"):
        config = OmegaConf.load(path)
        if OmegaConf.select(config, "framework.delta_jepa.learned_goal_enabled", default=False):
            learned.append(path)
            assert config.datasets.vla_data.require_full_horizon is False
    assert learned


class IndexOnlyMixture(LeRobotMixtureDataset):
    """Use the real mixture sampler without decoding videos in worker processes."""

    def __init__(self):
        self.datasets = [SimpleNamespace(all_steps=[(0, index) for index in range(1000)])]
        self._dataset_sampling_weights = np.array([1.0])
        self.mode, self.seed = "train", 42
        self.set_epoch(0)

    def __len__(self):
        return 24

    def __getitem__(self, index):
        _, trajectory, step = self.sample_step(index)
        return self.epoch, trajectory, step


def collate_indices(batch):
    # Index-only sampling needs no tensor IPC (nor its resource-sharer socket).
    return batch


def collect_indices(loader):
    return torch.tensor([sample for batch in loader for sample in batch])


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_persistent_workers_follow_epoch_and_match_rebuilt_loader(context):
    dataset = IndexOnlyMixture()
    loader = DataLoader(
        dataset, batch_size=4, num_workers=1, persistent_workers=True,
        multiprocessing_context=context, timeout=15, collate_fn=collate_indices,
    )
    epoch_zero = collect_indices(loader)
    assert torch.all(epoch_zero[:, 0] == 0)

    # Reusing the same loader keeps its workers alive. set_epoch must update them.
    dataset.set_epoch(3)
    epoch_three = collect_indices(loader)
    assert torch.all(epoch_three[:, 0] == 3)
    assert not torch.equal(epoch_zero[:, 2], epoch_three[:, 2])

    # A resumed run starts fresh workers at the same epoch and must sample the
    # same trajectories/steps as an uninterrupted run with persistent workers.
    resumed_dataset = IndexOnlyMixture()
    resumed_dataset.set_epoch(3)
    resumed_loader = DataLoader(
        resumed_dataset, batch_size=4, num_workers=1, persistent_workers=True,
        multiprocessing_context=context, timeout=15, collate_fn=collate_indices,
    )
    torch.testing.assert_close(collect_indices(resumed_loader), epoch_three)
