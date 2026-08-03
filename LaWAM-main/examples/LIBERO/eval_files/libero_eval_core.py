import dataclasses
import datetime as dt
import json
import logging
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np

from examples.LIBERO.eval_files.libero_benchmark_adapters import (
    BenchmarkAdapter,
    get_benchmark_adapter,
    quat2axisangle,
    safe_segment,
)
from examples.LIBERO.eval_files.model2libero_interface import ModelClient
from examples.eval_utils.similarity_video import (
    extract_anchor_feature,
    fit_token_rgb_projection,
    render_pca_image,
    render_similarity_overlay,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_PARALLEL_OVERRIDE_ENV = "STARVLA_ALLOW_UNSAFE_LIBERO_PARALLEL"


@dataclasses.dataclass
class EvalArgs:
    host: str = "127.0.0.1"
    port: int = 10093
    resize_size: list[int] = dataclasses.field(default_factory=lambda: [256, 256])
    task_suite_name: str = "libero_goal"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    video_out_path: str = "experiments/libero/logs"
    save_videos: bool = False
    save_only_failure_videos: bool = False
    seed: int = 0
    pretrained_path: str = ""
    unnorm_key: Optional[str] = None
    post_process_action: bool = True
    job_name: str = "test"
    save_similarity_video: bool = False
    sim_src_row: int = 3
    sim_src_col: int = 7
    sim_vmin: float = 0.4
    sim_vmax: float = 1.0
    sim_alpha: float = 0.5
    sim_cmap: str = "jet"
    max_tasks: Optional[int] = None
    log_path: Optional[str] = None
    benchmark_variant: str = "libero"
    enable_category_aggregation: Optional[bool] = None
    num_workers: int = 1
    worker_sync_timeout_sec: float = 1.0
    worker_result_timeout_sec: float = 600.0
    eval_action_chunk_len: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class EpisodeWorkItem:
    task_id: int
    episode_idx: int
    init_state_idx: int


@dataclasses.dataclass
class EpisodeResult:
    task_id: int
    episode_idx: int
    task_description: str
    success: bool
    num_actions: int
    rollout_video_path: Optional[str]
    env_step_count: int = 0
    env_step_total_sec: float = 0.0
    env_step_max_sec: float = 0.0
    dummy_step_count: int = 0
    dummy_step_total_sec: float = 0.0
    dummy_step_max_sec: float = 0.0
    error: Optional[str] = None


@dataclasses.dataclass
class WorkerSlotState:
    slot_id: int
    phase: str = "idle"
    work_item: Optional[EpisodeWorkItem] = None
    task_description: Optional[str] = None
    obs: Optional[dict[str, Any]] = None
    t: int = 0
    step: int = 0


@dataclasses.dataclass
class WorkerEvent:
    kind: str
    slot_id: int
    task_id: Optional[int] = None
    episode_idx: Optional[int] = None
    task_description: Optional[str] = None
    obs: Optional[dict[str, Any]] = None
    t: int = 0
    step: int = 0
    done: bool = False
    phase: Optional[str] = None
    step_elapsed_sec: float = 0.0
    batch_id: Optional[int] = None
    result: Optional[EpisodeResult] = None
    error: Optional[str] = None


@dataclasses.dataclass
class BatchInFlight:
    batch_id: int
    target_slot_ids: tuple[int, ...]
    dispatched_slot_ids: tuple[int, ...]
    barrier_wait_sec: float
    partial_batch_used: bool
    received_slot_ids: set[int] = dataclasses.field(default_factory=set)
    step_durations_sec: list[tuple[int, float, int, int]] = dataclasses.field(default_factory=list)


def _get_benchmark_dict() -> dict[str, Any]:
    from libero.libero import benchmark

    return benchmark.get_benchmark_dict()


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def _concat_replay_panels(
    primary_img: np.ndarray,
    wrist_img: np.ndarray | None,
    similarity_img: np.ndarray | None = None,
    pca_img: np.ndarray | None = None,
    *,
    layout: str = "horizontal",
) -> np.ndarray:
    panels = [primary_img]
    if wrist_img is not None:
        panels.append(wrist_img)
    if similarity_img is not None:
        panels.append(similarity_img)
    if pca_img is not None:
        panels.append(pca_img)

    if any(panel.ndim != 3 for panel in panels):
        raise ValueError(f"Expected RGB replay frames with shape [H, W, C], got {[p.shape for p in panels]}.")
    if layout == "horizontal":
        panel_heights = {int(panel.shape[0]) for panel in panels}
        if len(panel_heights) != 1:
            raise ValueError(f"Replay panels must have the same height, got {[p.shape for p in panels]}.")
        return np.concatenate(panels, axis=1)
    if layout == "vertical":
        panel_widths = {int(panel.shape[1]) for panel in panels}
        if len(panel_widths) != 1:
            raise ValueError(f"Replay panels must have the same width, got {[p.shape for p in panels]}.")
        return np.concatenate(panels, axis=0)
    raise ValueError(f"Unsupported replay panel layout: {layout!r}.")


def _configure_logging() -> None:
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
        )
    else:
        root.setLevel(logging.INFO)


def _prepare_output_dir(output_dir: str | None) -> Path | None:
    if output_dir is None:
        return None
    run_output_dir = Path(output_dir).expanduser().resolve()
    run_output_dir.mkdir(parents=True, exist_ok=True)
    return run_output_dir


def _validate_similarity_args(args: EvalArgs) -> None:
    if int(args.sim_src_row) < 0 or int(args.sim_src_col) < 0:
        raise ValueError(
            f"`sim_src_row` and `sim_src_col` must be >= 0, got {(args.sim_src_row, args.sim_src_col)}."
        )
    if float(args.sim_vmax) <= float(args.sim_vmin):
        raise ValueError(f"`sim_vmax` must be > `sim_vmin`, got {args.sim_vmin}, {args.sim_vmax}.")
    if not (0.0 <= float(args.sim_alpha) <= 1.0):
        raise ValueError(f"`sim_alpha` must be in [0, 1], got {args.sim_alpha}.")


def _resolve_category_aggregation(
    args: EvalArgs,
    adapter: BenchmarkAdapter,
) -> bool:
    if args.enable_category_aggregation is not None:
        return bool(args.enable_category_aggregation)
    return bool(adapter.enable_category_aggregation_by_default)


def _configure_mujoco_backend(
    args: EvalArgs,
    adapter: BenchmarkAdapter,
) -> None:
    requested_num_workers = int(args.num_workers)
    current_backend = os.environ.get("MUJOCO_GL", "").strip().lower()

    if requested_num_workers <= 1:
        if not current_backend:
            os.environ["MUJOCO_GL"] = "egl"
            current_backend = "egl"
        if not os.environ.get("PYOPENGL_PLATFORM") and current_backend == "egl":
            os.environ["PYOPENGL_PLATFORM"] = "egl"
        return

    if not current_backend:
        os.environ["MUJOCO_GL"] = "osmesa"
        current_backend = "osmesa"

    if not os.environ.get("PYOPENGL_PLATFORM") and current_backend in {"egl", "osmesa"}:
        os.environ["PYOPENGL_PLATFORM"] = current_backend


def _resolve_effective_num_workers(
    args: EvalArgs,
    adapter: BenchmarkAdapter,
) -> int:
    requested_num_workers = int(args.num_workers)
    if requested_num_workers <= 1:
        return requested_num_workers
    if adapter.variant_name != "libero":
        return requested_num_workers

    current_backend = os.environ.get("MUJOCO_GL", "").strip().lower()
    if current_backend in {"osmesa", "glx"}:
        return requested_num_workers

    override_raw = os.environ.get(LIBERO_PARALLEL_OVERRIDE_ENV, "")
    override_enabled = override_raw.strip().lower() in {"1", "true", "yes", "on"}
    if override_enabled:
        logging.warning(
            "Unsafe parallel LIBERO rollout override enabled via %s=%r with MUJOCO_GL=%s. "
            "Worker crashes inside robosuite offscreen rendering may still occur.",
            LIBERO_PARALLEL_OVERRIDE_ENV,
            override_raw,
            current_backend or "<unset>",
        )
        return requested_num_workers

    logging.warning(
        "Parallel original LIBERO with MUJOCO_GL=%s remains unsafe because robosuite "
        "offscreen rendering can abort in worker processes. Forcing num_workers from %s "
        "to 1. Prefer MUJOCO_GL=osmesa or MUJOCO_GL=glx, or set %s=1 to re-enable at "
        "your own risk.",
        current_backend or "<unset>",
        requested_num_workers,
        LIBERO_PARALLEL_OVERRIDE_ENV,
    )
    return 1


def _validate_args(args: EvalArgs, adapter: BenchmarkAdapter) -> None:
    _validate_similarity_args(args)
    if int(args.num_trials_per_task) <= 0:
        raise ValueError(f"`num_trials_per_task` must be > 0, got {args.num_trials_per_task}.")
    if args.max_tasks is not None and int(args.max_tasks) <= 0:
        raise ValueError(f"`max_tasks` must be > 0 when set, got {args.max_tasks}.")
    if int(args.num_workers) <= 0:
        raise ValueError(f"`num_workers` must be > 0, got {args.num_workers}.")
    if float(args.worker_sync_timeout_sec) <= 0.0:
        raise ValueError(
            "`worker_sync_timeout_sec` must be > 0, "
            f"got {args.worker_sync_timeout_sec}."
        )
    if float(args.worker_result_timeout_sec) <= 0.0:
        raise ValueError(
            "`worker_result_timeout_sec` must be > 0, "
            f"got {args.worker_result_timeout_sec}."
        )
    if int(args.num_workers) > 1 and bool(args.save_similarity_video):
        raise ValueError(
            "`save_similarity_video=True` is only supported in serial mode. "
            "Set `num_workers=1` to enable similarity overlay videos."
        )
    if not bool(args.save_videos) and bool(args.save_similarity_video):
        raise ValueError("`save_similarity_video=True` requires `save_videos=True`.")
    if args.eval_action_chunk_len is not None and int(args.eval_action_chunk_len) <= 0:
        raise ValueError(
            "`eval_action_chunk_len` must be > 0 when set, "
            f"got {args.eval_action_chunk_len}."
        )


def _build_model_client(args: EvalArgs) -> ModelClient:
    root_logger = logging.getLogger()
    previous_level = root_logger.level
    if previous_level <= logging.INFO:
        root_logger.setLevel(logging.WARNING)
    try:
        return ModelClient(
            policy_ckpt_path=args.pretrained_path,
            unnorm_key=args.unnorm_key,
            host=args.host,
            port=args.port,
            image_size=args.resize_size,
            eval_action_chunk_len=args.eval_action_chunk_len,
        )
    finally:
        root_logger.setLevel(previous_level)


def _task_init_state_index(initial_states: list[Any], episode_idx: int) -> int:
    if len(initial_states) == 0:
        raise RuntimeError("Task returned empty initial states.")
    return int(episode_idx % len(initial_states))


def _build_episode_video_path(
    run_output_dir: Path,
    task_id: int,
    task_name: str,
    episode_idx: int,
    success: bool,
) -> Path:
    suffix = "success" if success else "failure"
    task_segment = safe_segment(task_name)
    return run_output_dir / f"rollout_task{task_id:02d}_{task_segment}_episode{episode_idx:03d}_{suffix}.mp4"


def _build_episode_frame_dir(rollout_video_path: Path) -> Path:
    return rollout_video_path.parent / f"{rollout_video_path.stem}_frames"


def _save_episode_frames(frame_dir: Path, frames: list[np.ndarray]) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, frame in enumerate(frames):
        frame_path = frame_dir / f"frame_{frame_idx:04d}.png"
        imageio.imwrite(frame_path, np.asarray(frame))


def _build_policy_example_from_obs(
    obs: dict[str, Any],
    task_description: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = None
    if "robot0_eye_in_hand_image" in obs:
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[-1:],
        )
    )

    example_dict = {
        "primary_image": [img],
        "lang": str(task_description),
        "state": np.expand_dims(state, axis=0).astype(np.float32, copy=False),
    }
    if wrist_img is not None:
        example_dict["wrist_image"] = [wrist_img]
    return example_dict, img, wrist_img


def _response_to_delta_action(response: dict[str, Any], task_description: str, step: int) -> np.ndarray:
    raw_action = response["raw_action"]
    world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
    rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
    open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
    gripper = invert_gripper_action(_binarize_gripper_open(open_gripper))
    if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
        logging.warning(
            "Unexpected action sizes: wv=%s, rot=%s, grip=%s. Falling back to LIBERO_DUMMY_ACTION.",
            world_vector_delta.shape,
            rotation_delta.shape,
            gripper.shape,
        )
        raise ValueError(
            f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
            f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
        )
    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
    if not np.all(np.isfinite(delta_action)):
        raise ValueError(f"Non-finite environment action encountered: {delta_action.tolist()}")
    max_abs_action = float(np.max(np.abs(delta_action[:6]))) if delta_action.size >= 6 else 0.0
    if max_abs_action > 1.5:
        logging.warning(
            "Large environment action detected before env.step: task=%s step=%s max_abs=%.4f action=%s",
            task_description,
            step,
            max_abs_action,
            delta_action.tolist(),
        )
    return delta_action


def _run_single_episode(
    *,
    args: EvalArgs,
    adapter: BenchmarkAdapter,
    task: Any,
    task_id: int,
    episode_idx: int,
    init_state: Any,
    client_model: ModelClient,
    run_output_dir: Path,
    env_seed: int,
) -> EpisodeResult:
    env, task_description = adapter.build_env(task, LIBERO_ENV_RESOLUTION, env_seed)
    try:
        client_model.reset(task_description=task_description)
        env.reset()
        obs = env.set_init_state(init_state)

        max_steps = adapter.get_max_steps(args.task_suite_name)
        t = 0
        step = 0
        replay_images: list[np.ndarray] = []
        replay_frame_images: list[np.ndarray] = []
        full_actions: list[np.ndarray] = []
        anchor_feature = None
        pca_projection = None
        current_similarity_tokens = None
        current_vision_tokens_hw = None
        debug_video_enabled = bool(args.save_videos and args.save_similarity_video)
        similarity_supported = bool(debug_video_enabled)
        done = False

        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                del reward, info
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = None
            if "robot0_eye_in_hand_image" in obs:
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)[-1:],
                )
            )

            example_dict = {
                "primary_image": [img],
                "lang": str(task_description),
                "state": np.expand_dims(state, axis=0).astype(np.float32, copy=False),
            }
            if wrist_img is not None:
                example_dict["wrist_image"] = [wrist_img]

            response = client_model.step(
                example=example_dict,
                step=step,
                return_intermediates=bool(similarity_supported),
            )

            raw_action = response["raw_action"]
            has_fresh_intermediates = "intermediates" in response
            intermediates = response.get("intermediates", None) if similarity_supported else None
            if similarity_supported and has_fresh_intermediates and intermediates is None:
                logging.warning(
                    "Similarity video requested, but no intermediates were returned. "
                    "Disabling similarity overlay for the rest of this episode."
                )
                similarity_supported = False
            elif similarity_supported and has_fresh_intermediates and intermediates is not None:
                try:
                    h_t = np.asarray(intermediates["h_t"])
                    h_t1_pred = np.asarray(intermediates["h_t1_pred"])
                    vision_tokens_hw_raw = intermediates["vision_tokens_hw"]
                    current_vision_tokens_hw = (
                        int(vision_tokens_hw_raw[0]),
                        int(vision_tokens_hw_raw[1]),
                    )
                    if anchor_feature is None:
                        anchor_feature = extract_anchor_feature(
                            h_t[0],
                            args.sim_src_row,
                            args.sim_src_col,
                            current_vision_tokens_hw,
                        )
                    if pca_projection is None:
                        pca_projection = fit_token_rgb_projection([np.asarray(h_t[0], dtype=np.float32)])
                    current_similarity_tokens = np.asarray(h_t1_pred[0], dtype=np.float32)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid similarity intermediates returned by server: {exc}"
                    ) from exc

            similarity_overlay = None
            if similarity_supported and current_similarity_tokens is not None and anchor_feature is not None:
                similarity_overlay = render_similarity_overlay(
                    current_similarity_tokens,
                    anchor_feature,
                    current_vision_tokens_hw,
                    img,
                    vmin=float(args.sim_vmin),
                    vmax=float(args.sim_vmax),
                    alpha=float(args.sim_alpha),
                    cmap=str(args.sim_cmap),
                )
            pca_image = None
            if (
                similarity_supported
                and current_similarity_tokens is not None
                and pca_projection is not None
            ):
                pca_image = render_pca_image(
                    current_similarity_tokens,
                    pca_projection,
                    current_vision_tokens_hw,
                    target_hw=img.shape[:2],
                )
            if args.save_videos:
                replay_wrist_img = None if debug_video_enabled else wrist_img
                replay_images.append(
                    _concat_replay_panels(
                        img,
                        replay_wrist_img,
                        similarity_overlay,
                        pca_image,
                        layout="horizontal",
                    )
                )
                if debug_video_enabled:
                    replay_frame_images.append(
                        _concat_replay_panels(
                            img,
                            replay_wrist_img,
                            similarity_overlay,
                            pca_image,
                            layout="vertical",
                        )
                    )

            world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
            rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
            open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
            gripper = invert_gripper_action(_binarize_gripper_open(open_gripper))
            if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                logging.warning(
                    "Unexpected action sizes: wv=%s, rot=%s, grip=%s. Falling back to LIBERO_DUMMY_ACTION.",
                    world_vector_delta.shape,
                    rotation_delta.shape,
                    gripper.shape,
                )
                raise ValueError(
                    f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                    f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                )
            delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)
            if not np.all(np.isfinite(delta_action)):
                raise ValueError(f"Non-finite environment action encountered: {delta_action.tolist()}")
            max_abs_action = float(np.max(np.abs(delta_action[:6]))) if delta_action.size >= 6 else 0.0
            if max_abs_action > 1.5:
                logging.warning(
                    "Large environment action detected before env.step: task=%s step=%s max_abs=%.4f action=%s",
                    task_description,
                    step,
                    max_abs_action,
                    delta_action.tolist(),
                )
            full_actions.append(delta_action)

            obs, reward, done, info = env.step(delta_action.tolist())
            del reward, info
            if done:
                break
            t += 1
            step += 1

        should_save_video = bool(args.save_videos) and (
            (not args.save_only_failure_videos) or (not done)
        )
        rollout_video_path = None
        if should_save_video:
            task_name = getattr(task, "name", None) or str(task_description)
            rollout_video = _build_episode_video_path(
                run_output_dir,
                task_id=task_id,
                task_name=str(task_name),
                episode_idx=episode_idx,
                success=bool(done),
            )
            imageio.mimwrite(rollout_video, [np.asarray(x) for x in replay_images], fps=25)
            if bool(args.save_similarity_video):
                _save_episode_frames(_build_episode_frame_dir(rollout_video), replay_frame_images)
            rollout_video_path = str(rollout_video)

        return EpisodeResult(
            task_id=int(task_id),
            episode_idx=int(episode_idx),
            task_description=str(task_description),
            success=bool(done),
            num_actions=int(len(full_actions)),
            rollout_video_path=rollout_video_path,
        )
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def _task_id_list(task_suite: Any, num_tasks_to_run: int) -> list[int]:
    return list(range(int(num_tasks_to_run)))


def _finalize_results(
    *,
    args: EvalArgs,
    adapter: BenchmarkAdapter,
    run_output_dir: Path,
    log_output_dir: Path,
    task_suite: Any,
    task_metadata: dict[int, tuple[str | None, str]],
    episode_results: list[EpisodeResult],
) -> None:
    category_aggregation_enabled = _resolve_category_aggregation(args, adapter)
    results_sorted = sorted(episode_results, key=lambda item: (item.task_id, item.episode_idx))
    total_episodes = len(results_sorted)
    total_successes = sum(int(result.success) for result in results_sorted)
    total_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0

    category_summary: dict[str, dict[str, int]] = {}
    episode_records: list[dict[str, Any]] = []

    for result in results_sorted:
        task = task_suite.get_task(int(result.task_id))
        category, task_name = adapter.resolve_task_meta(
            int(result.task_id),
            task,
            result.task_description,
            task_metadata,
        )
        if category_aggregation_enabled:
            normalized_category = category or "Unknown"
            if normalized_category not in category_summary:
                category_summary[normalized_category] = {"total_count": 0, "success_count": 0}
            category_summary[normalized_category]["total_count"] += 1
            category_summary[normalized_category]["success_count"] += int(result.success)
        episode_records.append(
            {
                "task_id": int(result.task_id),
                "task_description": str(result.task_description),
                "task_name": str(task_name),
                "category": category,
                "episode_idx": int(result.episode_idx),
                "success": bool(result.success),
                "num_actions": int(result.num_actions),
                "rollout_video_path": result.rollout_video_path,
                "env_step_count": int(result.env_step_count),
                "env_step_total_sec": float(result.env_step_total_sec),
                "env_step_max_sec": float(result.env_step_max_sec),
                "dummy_step_count": int(result.dummy_step_count),
                "dummy_step_total_sec": float(result.dummy_step_total_sec),
                "dummy_step_max_sec": float(result.dummy_step_max_sec),
            }
        )

    suite_json_path = None
    if category_aggregation_enabled:
        suite_json_path = log_output_dir / f"{args.task_suite_name}.json"
        with suite_json_path.open("w", encoding="utf-8") as f:
            json.dump(category_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "task_suite_name": args.task_suite_name,
        "benchmark_variant": adapter.variant_name,
        "checkpoint_path": args.pretrained_path,
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "total_success_rate": float(total_success_rate),
        "num_trials_per_task": int(args.num_trials_per_task),
        "seed": int(args.seed),
        "max_tasks": None if args.max_tasks is None else int(args.max_tasks),
        "category_summary_path": str(suite_json_path) if suite_json_path is not None else None,
    }
    with (run_output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (run_output_dir / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for record in episode_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _get_multiprocessing_context():
    return mp.get_context("spawn")


def _suite_success_rate(episode_results: list[EpisodeResult]) -> float:
    if not episode_results:
        return 0.0
    return float(sum(int(result.success) for result in episode_results)) / float(len(episode_results))


def _log_suite_progress(
    *,
    suite_name: str,
    completed_tasks: int,
    total_tasks: int,
    episode_results: list[EpisodeResult],
) -> None:
    logging.info(
        "suite=%s completed_tasks=%s/%s success_rate=%.4f",
        suite_name,
        completed_tasks,
        total_tasks,
        _suite_success_rate(episode_results),
    )


def _pack_worker_obs(obs: dict[str, Any]) -> dict[str, Any]:
    packed = {
        "agentview_image": np.ascontiguousarray(obs["agentview_image"]),
        "robot0_eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).copy(),
        "robot0_eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).copy(),
        "robot0_gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).copy(),
    }
    if "robot0_eye_in_hand_image" in obs:
        packed["robot0_eye_in_hand_image"] = np.ascontiguousarray(obs["robot0_eye_in_hand_image"])
    return packed


def _build_parallel_replay_panel(obs: dict[str, Any]) -> np.ndarray:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = None
    if "robot0_eye_in_hand_image" in obs:
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return _concat_replay_panels(img, wrist_img, None)


def _serial_eval(
    *,
    args: EvalArgs,
    adapter: BenchmarkAdapter,
    run_output_dir: Path,
    log_output_dir: Path,
    task_suite: Any,
    num_tasks_to_run: int,
    task_metadata: dict[int, tuple[str | None, str]],
) -> None:
    client_model = _build_model_client(args)
    episode_results: list[EpisodeResult] = []
    completed_tasks = 0

    try:
        for task_id in _task_id_list(task_suite, num_tasks_to_run):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            if len(initial_states) == 0:
                raise RuntimeError(f"Task {task_id} returned empty initial states.")

            for episode_idx in range(args.num_trials_per_task):
                init_state_idx = _task_init_state_index(initial_states, episode_idx)
                result = _run_single_episode(
                    args=args,
                    adapter=adapter,
                    task=task,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    init_state=initial_states[init_state_idx],
                    client_model=client_model,
                    run_output_dir=run_output_dir,
                    env_seed=int(args.seed),
                )
                episode_results.append(result)

            completed_tasks += 1
            _log_suite_progress(
                suite_name=args.task_suite_name,
                completed_tasks=completed_tasks,
                total_tasks=num_tasks_to_run,
                episode_results=episode_results,
            )
    finally:
        client_model.close()

    _finalize_results(
        args=args,
        adapter=adapter,
        run_output_dir=run_output_dir,
        log_output_dir=log_output_dir,
        task_suite=task_suite,
        task_metadata=task_metadata,
        episode_results=episode_results,
    )


def _env_worker_main(
    args: EvalArgs,
    slot_id: int,
    command_queue: Any,
    result_queue: Any,
) -> None:
    env = None
    task = None
    task_id = None
    episode_idx = None
    task_description = None
    try:
        adapter = get_benchmark_adapter(args.benchmark_variant)
        benchmark_dict = _get_benchmark_dict()
        task_suite = benchmark_dict[args.task_suite_name]()
        run_output_dir = Path(args.video_out_path).expanduser().resolve()
        episode_horizon_steps = int(adapter.get_max_steps(args.task_suite_name)) + int(args.num_steps_wait)

        obs = None
        t = 0
        step = 0
        done = False
        replay_images: list[np.ndarray] = []
        full_actions: list[np.ndarray] = []
        env_step_count = 0
        env_step_total_sec = 0.0
        env_step_max_sec = 0.0
        dummy_step_count = 0
        dummy_step_total_sec = 0.0
        dummy_step_max_sec = 0.0

        def _close_env() -> None:
            nonlocal env
            if env is None:
                return
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
            env = None

        def _reset_runtime_state() -> None:
            nonlocal task, task_id, episode_idx, task_description, obs, t, step, done
            nonlocal replay_images, full_actions
            nonlocal env_step_count, env_step_total_sec, env_step_max_sec
            nonlocal dummy_step_count, dummy_step_total_sec, dummy_step_max_sec
            task = None
            task_id = None
            episode_idx = None
            task_description = None
            obs = None
            t = 0
            step = 0
            done = False
            replay_images = []
            full_actions = []
            env_step_count = 0
            env_step_total_sec = 0.0
            env_step_max_sec = 0.0
            dummy_step_count = 0
            dummy_step_total_sec = 0.0
            dummy_step_max_sec = 0.0

        while True:
            command = command_queue.get()
            command_type = str(command.get("type"))

            if command_type == "shutdown":
                _close_env()
                return

            try:
                if command_type == "activate_episode":
                    work_item = command["work_item"]
                    _close_env()
                    _reset_runtime_state()
                    task_id = int(work_item.task_id)
                    episode_idx = int(work_item.episode_idx)
                    initial_states = task_suite.get_task_init_states(task_id)
                    if len(initial_states) == 0:
                        raise RuntimeError(f"Task {task_id} returned empty initial states.")
                    task = task_suite.get_task(task_id)
                    env, task_description = adapter.build_env(task, LIBERO_ENV_RESOLUTION, int(args.seed))
                    env.reset()
                    obs = env.set_init_state(initial_states[int(work_item.init_state_idx)])
                    result_queue.put(
                        WorkerEvent(
                            kind="activated",
                            slot_id=slot_id,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            task_description=str(task_description),
                            obs=_pack_worker_obs(obs),
                            t=int(t),
                            step=int(step),
                        )
                    )
                    continue

                if env is None or task_id is None or episode_idx is None or task_description is None or obs is None:
                    raise RuntimeError("Worker received a step/finalize command without an active episode.")

                if command_type == "warmup_step":
                    if done or t >= episode_horizon_steps:
                        result_queue.put(
                            WorkerEvent(
                                kind="step",
                                slot_id=slot_id,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                obs=_pack_worker_obs(obs),
                                t=int(t),
                                step=int(step),
                                done=True,
                                phase="warmup",
                                step_elapsed_sec=0.0,
                            )
                        )
                        continue
                    step_start = time.perf_counter()
                    obs, reward, done_flag, info = env.step(LIBERO_DUMMY_ACTION)
                    step_elapsed = time.perf_counter() - step_start
                    del reward, info
                    done = bool(done_flag)
                    t += 1
                    dummy_step_count += 1
                    dummy_step_total_sec += float(step_elapsed)
                    dummy_step_max_sec = max(dummy_step_max_sec, float(step_elapsed))
                    result_queue.put(
                        WorkerEvent(
                            kind="step",
                            slot_id=slot_id,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            obs=_pack_worker_obs(obs),
                            t=int(t),
                            step=int(step),
                            done=done,
                            phase="warmup",
                            step_elapsed_sec=float(step_elapsed),
                        )
                    )
                    continue

                if command_type == "policy_step":
                    batch_id = int(command["batch_id"])
                    delta_action = np.asarray(command["action"], dtype=np.float32).reshape(-1)
                    if done or t >= episode_horizon_steps:
                        result_queue.put(
                            WorkerEvent(
                                kind="step",
                                slot_id=slot_id,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                obs=_pack_worker_obs(obs),
                                t=int(t),
                                step=int(step),
                                done=True,
                                phase="policy",
                                step_elapsed_sec=0.0,
                                batch_id=batch_id,
                            )
                        )
                        continue
                    if args.save_videos:
                        replay_images.append(_build_parallel_replay_panel(obs))
                    full_actions.append(delta_action.copy())
                    step_start = time.perf_counter()
                    obs, reward, done_flag, info = env.step(delta_action.tolist())
                    step_elapsed = time.perf_counter() - step_start
                    del reward, info
                    done = bool(done_flag)
                    t += 1
                    step += 1
                    env_step_count += 1
                    env_step_total_sec += float(step_elapsed)
                    env_step_max_sec = max(env_step_max_sec, float(step_elapsed))
                    result_queue.put(
                        WorkerEvent(
                            kind="step",
                            slot_id=slot_id,
                            task_id=task_id,
                            episode_idx=episode_idx,
                            obs=_pack_worker_obs(obs),
                            t=int(t),
                            step=int(step),
                            done=done,
                            phase="policy",
                            step_elapsed_sec=float(step_elapsed),
                            batch_id=batch_id,
                        )
                    )
                    continue

                if command_type == "finalize_episode":
                    should_save_video = bool(args.save_videos) and (
                        (not args.save_only_failure_videos) or (not done)
                    )
                    rollout_video_path = None
                    if should_save_video:
                        task_name = getattr(task, "name", None) or str(task_description)
                        rollout_video = _build_episode_video_path(
                            run_output_dir,
                            task_id=int(task_id),
                            task_name=str(task_name),
                            episode_idx=int(episode_idx),
                            success=bool(done),
                        )
                        imageio.mimwrite(rollout_video, [np.asarray(x) for x in replay_images], fps=25)
                        if bool(args.save_similarity_video):
                            _save_episode_frames(_build_episode_frame_dir(rollout_video), replay_images)
                        rollout_video_path = str(rollout_video)
                    result = EpisodeResult(
                        task_id=int(task_id),
                        episode_idx=int(episode_idx),
                        task_description=str(task_description),
                        success=bool(done),
                        num_actions=int(len(full_actions)),
                        rollout_video_path=rollout_video_path,
                        env_step_count=int(env_step_count),
                        env_step_total_sec=float(env_step_total_sec),
                        env_step_max_sec=float(env_step_max_sec),
                        dummy_step_count=int(dummy_step_count),
                        dummy_step_total_sec=float(dummy_step_total_sec),
                        dummy_step_max_sec=float(dummy_step_max_sec),
                    )
                    _close_env()
                    _reset_runtime_state()
                    result_queue.put(WorkerEvent(kind="finalized", slot_id=slot_id, result=result))
                    continue

                raise ValueError(f"Unknown worker command type: {command_type!r}.")
            except Exception as exc:
                result_queue.put(
                    WorkerEvent(
                        kind="error",
                        slot_id=slot_id,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                raise
    except Exception as exc:
        try:
            result_queue.put(
                WorkerEvent(
                    kind="error",
                    slot_id=slot_id,
                    task_id=task_id,
                    episode_idx=episode_idx,
                    error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
                )
            )
        except Exception:
            pass
        raise
    finally:
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


def _terminate_processes(processes: list[Any]) -> None:
    for proc in processes:
        join_fn = getattr(proc, "join", None)
        if callable(join_fn):
            proc.join(timeout=1.0)
    for proc in processes:
        if getattr(proc, "is_alive", lambda: False)():
            proc.terminate()
    for proc in processes:
        join_fn = getattr(proc, "join", None)
        if callable(join_fn):
            proc.join(timeout=1.0)


def _check_worker_processes(processes: list[Any], slot_states: dict[int, WorkerSlotState]) -> None:
    for slot_id, proc in enumerate(processes):
        exitcode = getattr(proc, "exitcode", None)
        if exitcode is None:
            continue
        if int(exitcode) == 0:
            if slot_states[slot_id].phase == "idle":
                continue
        work_item = slot_states[slot_id].work_item
        raise RuntimeError(
            "LIBERO env worker exited unexpectedly: "
            f"slot_id={slot_id}, exitcode={exitcode}, "
            f"task_id={None if work_item is None else int(work_item.task_id)}, "
            f"episode_idx={None if work_item is None else int(work_item.episode_idx)}"
        )


def _coordinated_parallel_eval(
    *,
    args: EvalArgs,
    adapter: BenchmarkAdapter,
    run_output_dir: Path,
    log_output_dir: Path,
    task_suite: Any,
    num_tasks_to_run: int,
    task_metadata: dict[int, tuple[str | None, str]],
) -> None:
    client_model = _build_model_client(args)
    ctx = _get_multiprocessing_context()
    result_queue = ctx.Queue()
    command_queues = [ctx.Queue() for _ in range(int(args.num_workers))]
    processes = [
        ctx.Process(
            target=_env_worker_main,
            args=(args, slot_id, command_queues[slot_id], result_queue),
        )
        for slot_id in range(int(args.num_workers))
    ]
    for proc in processes:
        proc.start()

    total_tasks = int(num_tasks_to_run)
    work_items: list[EpisodeWorkItem] = []
    for task_id in _task_id_list(task_suite, num_tasks_to_run):
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) == 0:
            raise RuntimeError(f"Task {task_id} returned empty initial states.")
        for episode_idx in range(args.num_trials_per_task):
            work_items.append(
                EpisodeWorkItem(
                    task_id=int(task_id),
                    episode_idx=int(episode_idx),
                    init_state_idx=_task_init_state_index(initial_states, episode_idx),
                )
            )

    slot_states = {
        slot_id: WorkerSlotState(slot_id=slot_id)
        for slot_id in range(int(args.num_workers))
    }
    slot_stats: dict[int, dict[str, float | int]] = {
        slot_id: {
            "env_step_count": 0,
            "env_step_total_sec": 0.0,
            "env_step_max_sec": 0.0,
            "dummy_step_count": 0,
            "dummy_step_total_sec": 0.0,
            "dummy_step_max_sec": 0.0,
        }
        for slot_id in range(int(args.num_workers))
    }
    next_work_idx = 0
    next_batch_id = 0
    pending_batches: dict[int, BatchInFlight] = {}
    barrier_target_slot_ids: set[int] | None = None
    barrier_start_monotonic: float | None = None
    episode_results: list[EpisodeResult] = []
    completed_episodes_by_task: dict[int, int] = defaultdict(int)
    last_worker_event_monotonic = time.monotonic()
    last_status_log_monotonic = last_worker_event_monotonic
    round_policy_count = 0
    partial_batch_round_count = 0
    round_sync_wait_total_sec = 0.0
    round_sync_wait_max_sec = 0.0
    round_estimated_idle_total_sec = 0.0
    round_estimated_idle_max_sec = 0.0
    round_max_skew_ratio = 0.0
    slowest_slot_record = None
    slowest_slot_sec = 0.0

    def _completed_tasks() -> int:
        return sum(int(count >= int(args.num_trials_per_task)) for count in completed_episodes_by_task.values())

    def _assign_idle_slots() -> None:
        nonlocal next_work_idx
        for slot_id, state in slot_states.items():
            if state.phase != "idle" or next_work_idx >= len(work_items):
                continue
            work_item = work_items[next_work_idx]
            next_work_idx += 1
            state.phase = "activating"
            state.work_item = work_item
            state.task_description = None
            state.obs = None
            state.t = 0
            state.step = 0
            command_queues[slot_id].put({"type": "activate_episode", "work_item": work_item})

    def _record_batch_completion(batch: BatchInFlight) -> None:
        nonlocal round_policy_count, partial_batch_round_count
        nonlocal round_sync_wait_total_sec, round_sync_wait_max_sec
        nonlocal round_estimated_idle_total_sec, round_estimated_idle_max_sec
        nonlocal round_max_skew_ratio, slowest_slot_record, slowest_slot_sec
        round_policy_count += 1
        if batch.partial_batch_used:
            partial_batch_round_count += 1
        round_sync_wait_total_sec += float(batch.barrier_wait_sec)
        round_sync_wait_max_sec = max(round_sync_wait_max_sec, float(batch.barrier_wait_sec))
        if not batch.step_durations_sec:
            return
        round_secs = [item[1] for item in batch.step_durations_sec]
        round_sum_sec = float(sum(round_secs))
        round_max_sec = float(max(round_secs))
        round_mean_sec = round_sum_sec / float(len(round_secs))
        estimated_idle_sec = float(len(round_secs)) * round_max_sec - round_sum_sec
        round_estimated_idle_total_sec += estimated_idle_sec
        round_estimated_idle_max_sec = max(round_estimated_idle_max_sec, estimated_idle_sec)
        skew_ratio = round_max_sec / round_mean_sec if round_mean_sec > 1e-12 else 0.0
        round_max_skew_ratio = max(round_max_skew_ratio, skew_ratio)
        slowest_slot_id, slowest_sec_candidate, slowest_step, slowest_task_id = max(
            batch.step_durations_sec,
            key=lambda item: item[1],
        )
        if slowest_sec_candidate >= slowest_slot_sec:
            slowest_slot_sec = float(slowest_sec_candidate)
            slowest_slot_record = {
                "slot_id": int(slowest_slot_id),
                "task_id": int(slowest_task_id),
                "step": int(slowest_step),
                "env_step_sec": float(slowest_sec_candidate),
            }

    def _log_status_if_needed(*, force: bool = False) -> None:
        nonlocal last_status_log_monotonic
        now_monotonic = time.monotonic()
        if not force and now_monotonic - last_status_log_monotonic < 60.0:
            return
        _log_suite_progress(
            suite_name=args.task_suite_name,
            completed_tasks=_completed_tasks(),
            total_tasks=total_tasks,
            episode_results=episode_results,
        )
        last_status_log_monotonic = now_monotonic

    _assign_idle_slots()
    try:
        while (
            next_work_idx < len(work_items)
            or any(state.phase != "idle" for state in slot_states.values())
            or pending_batches
            or barrier_target_slot_ids
        ):
            _check_worker_processes(processes, slot_states)
            _assign_idle_slots()

            for slot_id, state in slot_states.items():
                if state.phase == "ready" and state.t < int(args.num_steps_wait):
                    state.phase = "warmup_pending"
                    command_queues[slot_id].put({"type": "warmup_step"})

            if barrier_target_slot_ids is None:
                ready_policy_slots = [
                    slot_id
                    for slot_id, state in slot_states.items()
                    if state.phase == "ready" and state.t >= int(args.num_steps_wait)
                ]
                if ready_policy_slots:
                    barrier_target_slot_ids = {
                        slot_id
                        for slot_id, state in slot_states.items()
                        if state.t >= int(args.num_steps_wait) and state.phase in {"ready", "policy_pending"}
                    }
                    barrier_start_monotonic = time.monotonic()

            if barrier_target_slot_ids is not None:
                current_target = {
                    slot_id
                    for slot_id in barrier_target_slot_ids
                    if slot_states[slot_id].phase in {"ready", "policy_pending"}
                }
                if not current_target:
                    barrier_target_slot_ids = None
                    barrier_start_monotonic = None
                else:
                    current_ready = sorted(
                        slot_id
                        for slot_id in current_target
                        if slot_states[slot_id].phase == "ready"
                    )
                    now_monotonic = time.monotonic()
                    barrier_elapsed = (
                        0.0
                        if barrier_start_monotonic is None
                        else now_monotonic - float(barrier_start_monotonic)
                    )
                    should_dispatch = (
                        bool(current_ready)
                        and (
                            len(current_ready) == len(current_target)
                            or barrier_elapsed >= float(args.worker_sync_timeout_sec)
                        )
                    )
                    if should_dispatch:
                        selected_slot_ids = tuple(current_ready)
                        partial_batch = len(current_ready) != len(current_target)
                        batch_wait_sec = float(barrier_elapsed)
                        batch_target_snapshot = tuple(sorted(current_target))
                        barrier_target_slot_ids = None
                        barrier_start_monotonic = None
                        examples = [
                            _build_policy_example_from_obs(
                                slot_states[slot_id].obs,
                                str(slot_states[slot_id].task_description),
                            )[0]
                            for slot_id in selected_slot_ids
                        ]
                        responses = client_model.step_batch(
                            examples,
                            cache_keys=list(selected_slot_ids),
                            return_intermediates=False,
                        )
                        next_batch_id += 1
                        batch_id = next_batch_id
                        pending_batches[batch_id] = BatchInFlight(
                            batch_id=batch_id,
                            target_slot_ids=batch_target_snapshot,
                            dispatched_slot_ids=selected_slot_ids,
                            barrier_wait_sec=batch_wait_sec,
                            partial_batch_used=partial_batch,
                        )
                        for slot_id, response in zip(selected_slot_ids, responses):
                            state = slot_states[slot_id]
                            delta_action = _response_to_delta_action(
                                response,
                                str(state.task_description),
                                int(state.step),
                            )
                            state.phase = "policy_pending"
                            command_queues[slot_id].put(
                                {
                                    "type": "policy_step",
                                    "action": delta_action,
                                    "batch_id": batch_id,
                                }
                            )

            wait_timeout = 0.1
            if barrier_target_slot_ids is not None and barrier_start_monotonic is not None:
                remaining = max(
                    0.0,
                    float(args.worker_sync_timeout_sec) - (time.monotonic() - float(barrier_start_monotonic)),
                )
                wait_timeout = min(wait_timeout, remaining)

            try:
                event = result_queue.get(timeout=wait_timeout)
            except queue.Empty:
                _check_worker_processes(processes, slot_states)
                now_monotonic = time.monotonic()
                _log_status_if_needed(force=False)
                if now_monotonic - last_worker_event_monotonic >= float(args.worker_result_timeout_sec):
                    active_summary = {
                        slot_id: {
                            "phase": state.phase,
                            "task_id": None if state.work_item is None else int(state.work_item.task_id),
                            "episode_idx": None if state.work_item is None else int(state.work_item.episode_idx),
                            "t": int(state.t),
                            "step": int(state.step),
                        }
                        for slot_id, state in slot_states.items()
                        if state.phase != "idle"
                    }
                    raise RuntimeError(
                        "Timed out while waiting for LIBERO worker progress. "
                        f"suite={args.task_suite_name} completed_tasks={_completed_tasks()}/{total_tasks} "
                        f"success_rate={_suite_success_rate(episode_results):.4f} "
                        f"active={active_summary}"
                    )
                continue

            last_worker_event_monotonic = time.monotonic()
            if event.error:
                raise RuntimeError(
                    "LIBERO env worker failed: "
                    f"slot_id={event.slot_id}, task_id={event.task_id}, "
                    f"episode_idx={event.episode_idx}, error={event.error}"
                )

            slot_state = slot_states[int(event.slot_id)]
            if event.kind == "activated":
                slot_state.phase = "ready"
                slot_state.task_description = str(event.task_description)
                slot_state.obs = event.obs
                slot_state.t = int(event.t)
                slot_state.step = int(event.step)
                client_model.reset(task_description=slot_state.task_description, slot_key=int(event.slot_id))
                continue

            if event.kind == "step":
                slot_state.obs = event.obs
                slot_state.t = int(event.t)
                slot_state.step = int(event.step)
                if event.phase == "warmup":
                    slot_stats[int(event.slot_id)]["dummy_step_count"] = int(
                        slot_stats[int(event.slot_id)]["dummy_step_count"]
                    ) + 1
                    slot_stats[int(event.slot_id)]["dummy_step_total_sec"] = float(
                        slot_stats[int(event.slot_id)]["dummy_step_total_sec"]
                    ) + float(event.step_elapsed_sec)
                    slot_stats[int(event.slot_id)]["dummy_step_max_sec"] = max(
                        float(slot_stats[int(event.slot_id)]["dummy_step_max_sec"]),
                        float(event.step_elapsed_sec),
                    )
                elif event.phase == "policy":
                    slot_stats[int(event.slot_id)]["env_step_count"] = int(
                        slot_stats[int(event.slot_id)]["env_step_count"]
                    ) + 1
                    slot_stats[int(event.slot_id)]["env_step_total_sec"] = float(
                        slot_stats[int(event.slot_id)]["env_step_total_sec"]
                    ) + float(event.step_elapsed_sec)
                    slot_stats[int(event.slot_id)]["env_step_max_sec"] = max(
                        float(slot_stats[int(event.slot_id)]["env_step_max_sec"]),
                        float(event.step_elapsed_sec),
                    )
                    if event.batch_id is not None and int(event.batch_id) in pending_batches:
                        batch = pending_batches[int(event.batch_id)]
                        batch.received_slot_ids.add(int(event.slot_id))
                        batch.step_durations_sec.append(
                            (
                                int(event.slot_id),
                                float(event.step_elapsed_sec),
                                int(event.step),
                                int(event.task_id) if event.task_id is not None else -1,
                            )
                        )
                        if len(batch.received_slot_ids) >= len(batch.dispatched_slot_ids):
                            _record_batch_completion(batch)
                            pending_batches.pop(int(event.batch_id), None)

                if event.done:
                    if slot_state.phase != "finalizing":
                        slot_state.phase = "finalizing"
                        command_queues[int(event.slot_id)].put({"type": "finalize_episode"})
                    else:
                        slot_state.phase = "finalizing"
                else:
                    slot_state.phase = "ready"
                continue

            if event.kind == "finalized":
                result = event.result
                if result is None:
                    raise RuntimeError(f"Worker slot {event.slot_id} returned a finalize event without result.")
                episode_results.append(result)
                task_episode_count = completed_episodes_by_task[int(result.task_id)] + 1
                completed_episodes_by_task[int(result.task_id)] = task_episode_count
                slot_state.phase = "idle"
                slot_state.work_item = None
                slot_state.task_description = None
                slot_state.obs = None
                slot_state.t = 0
                slot_state.step = 0
                if task_episode_count >= int(args.num_trials_per_task):
                    _log_suite_progress(
                        suite_name=args.task_suite_name,
                        completed_tasks=_completed_tasks(),
                        total_tasks=total_tasks,
                        episode_results=episode_results,
                    )
                    last_status_log_monotonic = time.monotonic()
                _assign_idle_slots()
                continue

            raise RuntimeError(f"Unknown worker event kind: {event.kind!r}.")
    finally:
        for command_queue in command_queues:
            try:
                command_queue.put({"type": "shutdown"})
            except Exception:
                pass
        _terminate_processes(processes)
        client_model.close()

    timing_summary = {
        "round_policy_count": int(round_policy_count),
        "partial_batch_round_count": int(partial_batch_round_count),
        "mean_round_sync_wait_sec": (
            float(round_sync_wait_total_sec) / float(round_policy_count) if round_policy_count > 0 else 0.0
        ),
        "max_round_sync_wait_sec": float(round_sync_wait_max_sec),
        "mean_estimated_barrier_idle_sec": (
            float(round_estimated_idle_total_sec) / float(round_policy_count) if round_policy_count > 0 else 0.0
        ),
        "max_estimated_barrier_idle_sec": float(round_estimated_idle_max_sec),
        "max_skew_ratio": float(round_max_skew_ratio),
        "slowest_slot_record": slowest_slot_record,
        "slot_stats": slot_stats,
    }
    with (run_output_dir / "timing_summary.json").open("w", encoding="utf-8") as f:
        json.dump(timing_summary, f, ensure_ascii=False, indent=2)

    _finalize_results(
        args=args,
        adapter=adapter,
        run_output_dir=run_output_dir,
        log_output_dir=log_output_dir,
        task_suite=task_suite,
        task_metadata=task_metadata,
        episode_results=episode_results,
    )


def eval_libero(args: EvalArgs) -> None:
    _configure_logging()
    adapter = get_benchmark_adapter(args.benchmark_variant)
    _configure_mujoco_backend(args, adapter)
    effective_num_workers = _resolve_effective_num_workers(args, adapter)
    if effective_num_workers != int(args.num_workers):
        args.num_workers = effective_num_workers
    _validate_args(args, adapter)

    run_output_dir = _prepare_output_dir(args.video_out_path)
    if run_output_dir is None:
        raise ValueError("`video_out_path` must not be empty.")
    log_output_dir = _prepare_output_dir(args.log_path) if args.log_path else run_output_dir
    if log_output_dir is None:
        raise ValueError("Failed to prepare `log_output_dir`.")

    np.random.seed(args.seed)

    benchmark_dict = _get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = int(task_suite.n_tasks)

    num_tasks_to_run = num_tasks_in_suite
    if args.max_tasks is not None:
        num_tasks_to_run = min(num_tasks_in_suite, int(args.max_tasks))
    _log_suite_progress(
        suite_name=args.task_suite_name,
        completed_tasks=0,
        total_tasks=num_tasks_to_run,
        episode_results=[],
    )

    task_metadata = adapter.load_task_metadata(args.task_suite_name, os.environ.get("LIBERO_HOME"))

    if int(args.num_workers) == 1:
        _serial_eval(
            args=args,
            adapter=adapter,
            run_output_dir=run_output_dir,
            log_output_dir=log_output_dir,
            task_suite=task_suite,
            num_tasks_to_run=num_tasks_to_run,
            task_metadata=task_metadata,
        )
        return

    _coordinated_parallel_eval(
        args=args,
        adapter=adapter,
        run_output_dir=run_output_dir,
        log_output_dir=log_output_dir,
        task_suite=task_suite,
        num_tasks_to_run=num_tasks_to_run,
        task_metadata=task_metadata,
    )
