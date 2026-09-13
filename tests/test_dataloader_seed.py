"""The training seed also controls LeRobot mixture sampling."""

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("configured_seed,expected_seed", [(731, 731), (42, 42), (None, 42)])
def test_dataloader_forwards_training_seed_and_preserves_legacy_default(monkeypatch, configured_seed, expected_seed):
    monkeypatch.setenv("NO_ALBUMENTATIONS_UPDATE", "1")
    from starVLA import dataloader
    from starVLA.dataloader import lerobot_datasets

    config = OmegaConf.create({
        "framework": {"action_model": {"action_horizon": 30}, "vj2_model": {"num_frames": 8}},
        "datasets": {"vla_data": {"per_device_batch_size": 1, "num_workers": 0,
                                   "delete_pause_frame": False}},
    })
    if configured_seed is not None:
        config.seed = configured_seed
    received = {}
    samples = [{"marker": "existing sample"}]

    def make_dataset(**kwargs):
        received.update(kwargs)
        return samples

    monkeypatch.setattr(lerobot_datasets, "get_vla_dataset", make_dataset)
    # Skip rank-zero statistics writing; keep the real DataLoader construction
    # and collate path so this verifies the production training entry point.
    monkeypatch.setattr(dataloader.dist, "get_rank", lambda: 1)
    loader = dataloader.build_dataloader(config, dataset_py="lerobot_datasets")
    assert received["seed"] == expected_seed
    assert received["data_cfg"] is config.datasets.vla_data
    assert received["action_horizon"] == 30 and received["video_horizon"] == 8
    assert received["delete_pause_frame"] is False
    assert loader.dataset is samples
    assert loader.collate_fn(samples) == samples
