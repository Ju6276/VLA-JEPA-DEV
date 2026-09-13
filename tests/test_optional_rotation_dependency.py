"""SIMPLE/SONIC data transforms run without installing optional PyTorch3D."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
BLOCK_PYTORCH3D = """
import builtins
original_import = builtins.__import__
attempts = []
def import_without_pytorch3d(name, *args, **kwargs):
    if name == "pytorch3d" or name.startswith("pytorch3d."):
        attempts.append(name)
        raise ModuleNotFoundError("PyTorch3D deliberately unavailable", name="pytorch3d")
    return original_import(name, *args, **kwargs)
builtins.__import__ = import_without_pytorch3d
"""


def run_without_pytorch3d(code):
    environment = dict(os.environ, NO_ALBUMENTATIONS_UPDATE="1")
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(BLOCK_PYTORCH3D) + textwrap.dedent(code)],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("interface", ["simple", "sonic"])
def test_production_interface_normalization_roundtrip_without_pytorch3d(interface):
    run_without_pytorch3d(f"""
        import numpy as np
        import torch
        from starVLA.dataloader.gr00t_lerobot.data_config import G1HandoverDataConfig, SonicLatentDataConfig
        from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform

        interface = {interface!r}
        if interface == "simple":
            config = G1HandoverDataConfig(observation_indices=[0], action_indices=[0, 1])
            state_widths = [7, 7, 7, 7, 3, 1]
            action_widths = [7, 7, 7, 7, 3, 1, 1, 1, 1, 1]
        else:
            config = SonicLatentDataConfig(observation_indices=[0], action_indices=[0, 1])
            state_widths = [6, 6, 3, 7, 7, 7, 7, 3]
            action_widths = [64, 7, 7]
        transforms = config.transform()
        assert all(not transform.target_rotations for transform in transforms.transforms
                   if isinstance(transform, StateActionTransform))
        modalities = {{"state": {{}}, "action": {{}}, "video": {{}}}}
        statistics = {{"state": {{}}, "action": {{}}}}
        raw = {{}}
        for modality, keys, widths in (("state", config.state_keys, state_widths),
                                      ("action", config.action_keys, action_widths)):
            assert len(keys) == len(widths)
            for key, width in zip(keys, widths):
                short_key = key.split(".", 1)[1]
                modalities[modality][short_key] = {{"absolute": True, "continuous": True, "shape": [width]}}
                statistics[modality][short_key] = {{
                    "min": np.full(width, -2.), "max": np.full(width, 2.),
                    "mean": np.zeros(width), "std": np.full(width, 2.),
                    "q01": np.full(width, -1.8), "q99": np.full(width, 1.8),
                }}
                raw[key] = np.full((2, width), 1., dtype=np.float32)
        metadata = DatasetMetadata.model_validate({{
            "modalities": modalities, "statistics": statistics, "embodiment_tag": "new_embodiment",
        }})
        transforms.set_metadata(metadata)
        normalized = transforms({{key: value.copy() for key, value in raw.items()}})
        for key in raw:
            assert isinstance(normalized[key], torch.Tensor)
            torch.testing.assert_close(normalized[key], torch.full_like(normalized[key], .5))
        restored = transforms.unapply(normalized)
        for key in raw:
            np.testing.assert_allclose(restored[key], raw[key])
        assert attempts == [], attempts
    """)


def test_rotation_conversion_reports_optional_dependency_only_when_requested():
    run_without_pytorch3d("""
        from starVLA.dataloader.gr00t_lerobot.transform.state_action import RotationTransform
        assert attempts == []
        try:
            RotationTransform(from_rep="axis_angle", to_rep="rotation_6d")
        except ImportError as error:
            assert "requires the optional PyTorch3D dependency" in str(error)
            assert "target_rotations" in str(error)
            assert "Python, PyTorch and CUDA" in str(error)
            assert isinstance(error.__cause__, ModuleNotFoundError)
        else:
            raise AssertionError("Rotation conversion should require PyTorch3D")
        assert attempts == ["pytorch3d.transforms"], attempts
    """)
