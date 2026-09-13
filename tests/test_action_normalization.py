"""Action decoding follows training transforms and checkpoint-specific schemas."""

from copy import deepcopy
import importlib
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from starVLA.dataloader.gr00t_lerobot.data_config import G1HandoverDataConfig, SonicLatentDataConfig
from starVLA.dataloader.gr00t_lerobot.transform.state_action import Normalizer
from starVLA.model.framework.base_framework import baseframework


def make_statistics(dimensions):
    low = np.linspace(-2, -0.2, dimensions, dtype=np.float32)
    high = low + np.linspace(0.5, 3, dimensions, dtype=np.float32)
    return {
        "min": low.tolist(), "max": high.tolist(),
        "q01": (low + 0.1).tolist(), "q99": (high - 0.1).tolist(),
        "mean": ((low + high) / 2).tolist(), "std": np.linspace(0.2, 1, dimensions).tolist(),
        "mask": [True] * dimensions,
    }


@pytest.mark.parametrize("tag,dimensions,config_type,field_widths", [
    ("g1_pick_between_tables", 36, G1HandoverDataConfig, [7, 7, 7, 7, 3, 1, 1, 1, 1, 1]),
    ("g1_handover", 36, G1HandoverDataConfig, [7, 7, 7, 7, 3, 1, 1, 1, 1, 1]),
    ("sonic_humanoid", 78, SonicLatentDataConfig, [64, 7, 7]),
])
def test_interface_roundtrip_matches_actual_training_normalizers(tag, dimensions, config_type, field_widths):
    stats = make_statistics(dimensions)
    # Include a nonzero constant min/max channel, as in the real datasets.
    stats["min"][1] = stats["max"][1] = 0.75
    model = baseframework()
    model.norm_stats = {tag: {"action": stats}}
    action_stats = model.get_action_stats()

    data_config = config_type(observation_indices=[0], action_indices=[0])
    transforms = data_config.transform().transforms
    configured_modes = {key: mode for transform in transforms
                        for key, mode in getattr(transform, "normalization_modes", {}).items()}
    expected_modes = [configured_modes[key] for key, width in zip(data_config.action_keys, field_widths)
                      for _ in range(width)]
    assert action_stats["normalization_modes"] == expected_modes
    assert "normalization_modes" not in stats

    raw = torch.linspace(-3, 3, 2 * 3 * dimensions).reshape(2, 3, dimensions)
    # Roundtrip valid min/max actions. Out-of-range deployment clipping is
    # checked independently against the SIMPLE/SONIC source implementations.
    minmax = np.asarray(expected_modes) == "min_max"
    raw[..., minmax] = torch.maximum(
        torch.minimum(raw[..., minmax], torch.tensor(stats["max"])[minmax]),
        torch.tensor(stats["min"])[minmax],
    )
    raw[..., 1] = 0.75
    raw[..., 6] = torch.linspace(stats["min"][6], stats["max"][6], 6).reshape(2, 3)
    normalized = torch.empty_like(raw)
    training_inverse = torch.empty_like(raw)
    for mode in set(expected_modes):
        indices = np.flatnonzero(np.asarray(expected_modes) == mode)
        statistics = {key: np.asarray(value)[indices].tolist() for key, value in stats.items() if key != "mask"}
        normalizer = Normalizer(mode, statistics)
        normalized[..., indices] = normalizer.forward(raw[..., indices])
        training_inverse[..., indices] = normalizer.inverse(normalized[..., indices])
    if dimensions == 36:
        assert normalized[..., 32:].abs().max() > 1
    else:
        assert torch.unique(normalized[..., 6]).numel() > 2

    original = normalized.numpy().copy()
    restored = model.unnormalize_actions(original, action_stats)
    np.testing.assert_allclose(restored, training_inverse.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored, raw.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(original, normalized.numpy())
    assert restored.shape == (2, 3, dimensions)


def test_stats_accessor_uses_each_instance_and_preserves_explicit_class_calls():
    first, second = baseframework(), baseframework()
    first.norm_stats = {"first": {"action": make_statistics(8)}}
    second.norm_stats = {"second": {"action": make_statistics(9)}}
    assert len(first.get_action_stats()["min"]) == 8
    assert len(second.get_action_stats()["min"]) == 9
    assert len(baseframework.get_action_stats("second", second.norm_stats)["min"]) == 9
    assert len(baseframework.get_action_stats(norm_stats=first.norm_stats)["min"]) == 8
    assert len(first.get_action_stats(norm_stats=second.norm_stats)["min"]) == 9
    with pytest.raises(ValueError, match="pass norm_stats explicitly"):
        baseframework.get_action_stats()
    with pytest.raises(ValueError, match="pass norm_stats explicitly"):
        baseframework().get_action_stats()


def test_legacy_quantile_mask_uses_final_axis_without_implicit_binary_channel():
    stats = {"q01": [-2.0] * 8, "q99": [2.0] * 8, "mask": [True] * 7 + [False]}
    normalized = np.full((2, 3, 8), 0.25, dtype=np.float32)
    normalized[..., 0] = 4.0
    normalized[..., -1] = 3.75
    before = normalized.copy()
    restored = baseframework.unnormalize_actions(normalized, stats)
    np.testing.assert_array_equal(restored[..., 0], 2.0)
    np.testing.assert_array_equal(restored[..., 6], 0.5)
    np.testing.assert_array_equal(restored[..., -1], 3.75)
    np.testing.assert_array_equal(normalized, before)


def test_explicit_modes_override_stored_schema_and_binary_is_opt_in():
    stats = make_statistics(3)
    stats["normalization_modes"] = ["min_max"] * 3
    stats["std"][0], stats["mean"][0] = 0.0, 4.0
    values = np.array([3.0, 0.51, -2.0], dtype=np.float32)
    restored = baseframework.unnormalize_actions(values, stats, ["mean_std", "binary", "identity"])
    np.testing.assert_array_equal(restored, [4.0, 1.0, -2.0])
    low, high = np.asarray(stats["min"], dtype=np.float32), np.asarray(stats["max"], dtype=np.float32)
    expected = (np.clip(values, -1, 1) + 1) / 2 * (high - low) + low
    np.testing.assert_allclose(baseframework.unnormalize_actions(values, stats, "min_max"), expected)


@pytest.mark.parametrize("modes,match", [(["min_max"], "one mode"), (["unknown"] * 3, "Unknown")])
def test_invalid_action_schema_fails_clearly(modes, match):
    with pytest.raises(ValueError, match=match):
        baseframework.unnormalize_actions(np.ones(3), make_statistics(3), modes)


def test_checkpoint_overrides_take_effect_before_build_without_mutating_saved_config(tmp_path, monkeypatch):
    module = importlib.import_module("starVLA.model.framework.base_framework")
    checkpoint = tmp_path / "checkpoints" / "model.pt"
    checkpoint.parent.mkdir()
    config = {
        "framework": {"delta_jepa": {"subgoals_path": "/old/missing/goals", "enabled": True}},
        "trainer": {"pretrained_checkpoint": "old-model.pt"},
    }
    OmegaConf.save(OmegaConf.create(config), tmp_path / "config.yaml")
    statistics = {"demo": {"action": make_statistics(2)}}
    (tmp_path / "dataset_statistics.json").write_text(json.dumps(statistics))
    original = torch.nn.Linear(2, 2)
    torch.save(original.state_dict(), checkpoint)
    built_configs = []

    def build_framework(cfg):
        built_configs.append(deepcopy(OmegaConf.to_container(cfg)))
        if cfg.framework.delta_jepa.subgoals_path == "/old/missing/goals":
            raise FileNotFoundError("checkpoint contains an unavailable goal path")
        return torch.nn.Linear(2, 2)

    monkeypatch.setattr(module, "build_framework", build_framework)
    with pytest.raises(FileNotFoundError, match="unavailable goal path"):
        baseframework.from_pretrained(checkpoint)
    restored = baseframework.from_pretrained(
        checkpoint, config_overrides={"framework.delta_jepa.subgoals_path": None},
    )
    assert built_configs[-1]["framework"]["delta_jepa"] == {"subgoals_path": None, "enabled": True}
    assert built_configs[-1]["trainer"]["pretrained_checkpoint"] is None
    assert restored.norm_stats == statistics
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)
    assert OmegaConf.to_container(OmegaConf.load(tmp_path / "config.yaml")) == config
