from __future__ import annotations

from typing import Any

from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES


def _infer_expected_action_horizon_from_data_mix(data_mix: str) -> int:
    if data_mix not in DATASET_NAMED_MIXTURES:
        raise ValueError(
            f"Unknown datasets.vla_data.data_mix `{data_mix}` for LatentWorld policy contract validation."
        )

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    robot_types = {spec[2] for spec in mixture_spec}
    horizon_by_robot = {}
    for robot_type in robot_types:
        if robot_type not in ROBOT_TYPE_CONFIG_MAP:
            raise ValueError(
                f"Unknown robot_type `{robot_type}` in data_mix `{data_mix}`. "
                "Cannot infer action horizon contract."
            )
        data_cfg = ROBOT_TYPE_CONFIG_MAP[robot_type]
        action_indices = getattr(data_cfg, "action_indices", None)
        if action_indices is None:
            raise ValueError(
                f"robot_type `{robot_type}` does not expose `action_indices`; "
                "cannot infer action horizon contract."
            )
        horizon_by_robot[robot_type] = int(len(action_indices))

    unique_horizons = set(horizon_by_robot.values())
    if len(unique_horizons) != 1:
        raise ValueError(
            "LatentWorld policy requires one consistent action horizon across mixture robot types. "
            f"data_mix={data_mix}, horizon_by_robot={horizon_by_robot}."
        )

    return next(iter(unique_horizons))


def validate_policy_contract(config: Any, policy_cfg: Any) -> None:
    expected = int(policy_cfg.future_action_window_size + policy_cfg.past_action_window_size + 1)
    got = int(policy_cfg.action_horizon)
    if got != expected:
        raise ValueError(
            "Invalid policy action window contract: expected "
            "`action_horizon = future_action_window_size + past_action_window_size + 1`, "
            f"got action_horizon={got}, future_action_window_size={policy_cfg.future_action_window_size}, "
            f"past_action_window_size={policy_cfg.past_action_window_size}."
        )

    data_mix = str(config.datasets.vla_data.data_mix)
    sec_chunk = config.datasets.vla_data.get("sec_chunk", None)
    # In fixed-physical-time mode (`sec_chunk` enabled), per-dataset action length
    # is resolved at dataloader runtime from dataset fps and can differ across sources.
    # Still require the physical-time contracts to match.
    if sec_chunk is not None:
        sec_chunk_f = float(sec_chunk)
        horizon_sec = float(policy_cfg.flow_cfg.horizon_sec)
        if abs(sec_chunk_f - horizon_sec) > 1e-6:
            raise ValueError(
                "LatentWorld physical-time contract mismatch detected at startup. "
                f"datasets.vla_data.sec_chunk={sec_chunk_f}, "
                f"framework.action_model.flow_cfg.horizon_sec={horizon_sec}. "
                "These must match so dataloader action padding aligns with the flow time grid."
            )
        return

    expected_data_horizon = _infer_expected_action_horizon_from_data_mix(data_mix)
    if got != expected_data_horizon:
        raise ValueError(
            "Policy/data horizon mismatch detected at startup. "
            f"data_mix={data_mix}, expected_action_horizon={expected_data_horizon}, configured_action_horizon={got}. "
            "Please align `framework.action_model.future_action_window_size`, "
            "`framework.action_model.past_action_window_size`, and `framework.action_model.action_horizon`."
        )
