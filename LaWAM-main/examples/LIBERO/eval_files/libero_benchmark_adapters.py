import dataclasses
import json
import logging
import math
import pathlib
from pathlib import Path
from typing import Any, Callable

import numpy as np


TaskMetadata = dict[int, tuple[str | None, str]]


@dataclasses.dataclass(frozen=True)
class BenchmarkAdapter:
    variant_name: str
    default_num_trials_per_task: int
    default_output_root: str
    enable_category_aggregation_by_default: bool
    load_task_metadata: Callable[[str, str | None], TaskMetadata]
    resolve_task_meta: Callable[[int, Any, str, TaskMetadata], tuple[str | None, str]]
    get_max_steps: Callable[[str], int]
    build_env: Callable[[Any, int, int], tuple[Any, str]]


def get_benchmark_adapter(variant_name: str) -> BenchmarkAdapter:
    normalized = str(variant_name).strip().lower()
    if normalized == "libero":
        return _libero_adapter()
    if normalized == "libero_plus":
        return _libero_plus_adapter()
    raise ValueError(f"Unknown benchmark variant: {variant_name!r}.")


def safe_segment(name: str) -> str:
    return (
        str(name)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float64)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _load_libero_api() -> tuple[Any, Any]:
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    return get_libero_path, OffScreenRenderEnv


def _build_env(task: Any, resolution: int, seed: int) -> tuple[Any, str]:
    get_libero_path, offscreen_render_env = _load_libero_api()
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = offscreen_render_env(
        bddl_file_name=str(task_bddl_file),
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


def _load_no_metadata(_: str, __: str | None) -> TaskMetadata:
    return {}


def _resolve_libero_task_meta(
    task_id: int,
    task: Any,
    task_description: str,
    _: TaskMetadata,
) -> tuple[str | None, str]:
    del task_id
    task_name = getattr(task, "name", None) or safe_segment(task_description)
    return None, str(task_name)


def _load_libero_plus_task_metadata(task_suite_name: str, libero_home: str | None) -> TaskMetadata:
    if not libero_home:
        logging.warning("LIBERO_HOME is unset; LIBERO-plus category aggregation will fall back to Unknown.")
        return {}

    task_map_path = Path(libero_home) / "libero/libero/benchmark/task_classification.json"
    if not task_map_path.exists():
        logging.warning("Task mapping file not found: %s", task_map_path)
        return {}

    with task_map_path.open("r", encoding="utf-8") as f:
        all_mapping = json.load(f)

    suite_mapping = all_mapping.get(task_suite_name, [])
    if not isinstance(suite_mapping, list):
        logging.warning(
            "Invalid task mapping format for suite %s in %s",
            task_suite_name,
            task_map_path,
        )
        return {}

    id_to_meta: TaskMetadata = {}
    for item in suite_mapping:
        try:
            task_id = int(item.get("id"))
        except Exception:
            continue
        category_raw = item.get("category", "Unknown")
        category = None if category_raw is None else str(category_raw)
        task_name = str(item.get("name", f"task_{task_id}"))
        id_to_meta[task_id] = (category or "Unknown", task_name)
    return id_to_meta


def _resolve_libero_plus_task_meta(
    task_id: int,
    task: Any,
    task_description: str,
    metadata: TaskMetadata,
) -> tuple[str | None, str]:
    del task
    for candidate in (task_id + 1, task_id):
        if candidate in metadata:
            return metadata[candidate]
    return "Unknown", safe_segment(task_description)


def _get_canonical_max_steps(task_suite_name: str) -> int:
    max_steps = {
        "libero_spatial": 250,
        "libero_object": 300,
        "libero_goal": 320,
        "libero_10": 550,
        "libero_90": 420,
    }.get(task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return int(max_steps)


def _libero_adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        variant_name="libero",
        default_num_trials_per_task=50,
        default_output_root="results/eval_runs/libero",
        enable_category_aggregation_by_default=False,
        load_task_metadata=_load_no_metadata,
        resolve_task_meta=_resolve_libero_task_meta,
        get_max_steps=_get_canonical_max_steps,
        build_env=_build_env,
    )


def _libero_plus_adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        variant_name="libero_plus",
        default_num_trials_per_task=1,
        default_output_root="results/eval_runs/libero_plus",
        enable_category_aggregation_by_default=True,
        load_task_metadata=_load_libero_plus_task_metadata,
        resolve_task_meta=_resolve_libero_plus_task_meta,
        get_max_steps=_get_canonical_max_steps,
        build_env=_build_env,
    )
