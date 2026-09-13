"""Spatial goal training and causal observation memory for VLA-JEPA."""

import math
from numbers import Integral, Real
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from starVLA.model.modules.world_model.spatial_goal import (
    SpatialActionAdapter, SpatialGoalPredictor, TaskSpatialReader, fixed_spatial_grid,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images


class SpatialJEPAMixin:
    def _init_spatial_goal(self):
        cfg = self.config.framework.get("spatial_goal", {})
        self.use_spatial_goal = bool(cfg.get("enabled", False))
        self.use_spatial_memory = self.use_spatial_goal and bool(cfg.get("memory_enabled", True))
        self.spatial_memory = {}
        self.spatial_training_metrics = {}
        if not self.use_spatial_goal:
            return
        if not (self.use_learned_goal and self.use_goal_action_proposal and self.num_video_views == 1):
            raise ValueError("Spatial goals require learned goals, an action proposal, and one ego view")
        self.spatial_grid_size = int(cfg.get("grid_size", 8))
        self.spatial_goal_weight = float(cfg.get("goal_weight", 1.0))
        self.spatial_score_weight = float(cfg.get("score_weight", 0.5))
        self.spatial_history_offsets = list(cfg.get("history_offsets_seconds", [-0.8, -0.4]))
        self.spatial_history_tolerance = float(cfg.get("history_tolerance_seconds", 0.15))
        self.spatial_memory_max_gap = float(cfg.get("memory_max_gap_seconds", 2.0))
        self.spatial_memory_max_frames = int(cfg.get("memory_max_frames", 256))
        self.spatial_history_dropout = float(cfg.get("history_dropout", 0.0))
        if not math.isfinite(self.spatial_history_dropout) or not 0 <= self.spatial_history_dropout <= 1:
            raise ValueError("history_dropout must be between zero and one")
        offsets = self.spatial_history_offsets
        if not offsets or any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                              not math.isfinite(v) or v >= 0 for v in offsets):
            raise ValueError("history_offsets_seconds must contain finite negative seconds")
        if any(a >= b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("history_offsets_seconds must be strictly increasing")
        for name, value in (("goal_weight", self.spatial_goal_weight),
                            ("score_weight", self.spatial_score_weight),
                            ("history_tolerance_seconds", self.spatial_history_tolerance)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"spatial_goal.{name} must be finite and nonnegative")
        if not math.isfinite(self.spatial_memory_max_gap) or self.spatial_memory_max_gap <= 0:
            raise ValueError("memory_max_gap_seconds must be finite and positive")
        if self.spatial_memory_max_frames < 2:
            raise ValueError("memory_max_frames must be at least two")
        data = self.config.datasets.vla_data
        if bool(data.get("spatial_goal_enabled", False)):
            if list(data.get("history_offsets_seconds", offsets)) != offsets:
                raise ValueError("Data and model history_offsets_seconds must match")
            if float(data.get("history_tolerance_seconds", self.spatial_history_tolerance)) != self.spatial_history_tolerance:
                raise ValueError("Data and model history_tolerance_seconds must match")
        latent = self.vj_encoder.config.hidden_size
        task = self.qwen_vl_interface.model.config.hidden_size
        state = self.config.framework.action_model.state_dim
        hidden = int(cfg.get("hidden_dim", 256))
        heads = int(cfg.get("num_heads", 4))
        queries = int(cfg.get("num_queries", 4))
        self.spatial_goal_predictor = SpatialGoalPredictor(
            latent, task, state, self.spatial_grid_size, hidden, heads)
        self.spatial_reader = TaskSpatialReader(latent, task, state, hidden, queries, heads)
        self.spatial_action_adapter = SpatialActionAdapter(
            latent, self.config.framework.action_model.action_dim, self.chunk_len, hidden, queries)

    def reset_spatial_memory(self):
        self.spatial_memory.clear()

    def _spatial_images(self, batch_images):
        size = int(self.config.framework.vj2_model.get("image_size", self.vj_encoder.config.image_size))
        return resize_images(batch_images, target_size=[size, size])

    def _spatial_grid(self, tokens):
        return fixed_spatial_grid(tokens, self.spatial_grid_size)

    def _current_spatial_grid(self, tokens):
        return self._spatial_grid(tokens[:, -(tokens.shape[1] // self.num_temporal_frames):])

    def _encode_spatial_goal_images(self, images):
        batch = [list(x) if isinstance(x, (list, tuple)) else [x] for x in images]
        embeddings = self._encode_video_batch(self._images_to_video_batch(self._spatial_images(batch)))
        return self._current_spatial_grid(embeddings)

    def _encode_spatial_training_pair(self, examples, videos):
        if videos.ndim != 6 or videos.shape[1] != 1 or videos.shape[2] < 2:
            raise ValueError("Spatial training needs a current/future video from one ego camera")
        if any("jepa_image" not in x for x in examples):
            raise ValueError("Spatial training requires jepa_image; enable datasets.vla_data.spatial_goal_enabled")
        current = self._encode_video_batch(self._images_to_video_batch(
            self._spatial_images([x["jepa_image"] for x in examples])))
        future = [[Image.fromarray(v[0, -1].astype(np.uint8))] for v in videos]
        target = self._encode_video_batch(self._images_to_video_batch(self._spatial_images(future)))
        return current, target[:, -(target.shape[1] // self.num_temporal_frames):]

    def _encode_spatial_history(self, images, valid, ages, current_grid):
        """Encode only valid past stills, independently of current/future images."""
        batch, _, dim = current_grid.shape
        count = len(self.spatial_history_offsets)
        valid = torch.as_tensor(np.asarray(valid), device=current_grid.device, dtype=torch.bool)
        ages = torch.as_tensor(np.asarray(ages), device=current_grid.device, dtype=torch.float32)
        if valid.shape != (batch, count) or ages.shape != valid.shape:
            raise ValueError("History validity/ages must have shape [B, number of history offsets]")
        if len(images) != batch or any(len(row) != count for row in images):
            raise ValueError("history_images must contain one image per configured past offset")
        if not torch.isfinite(ages).all() or (ages[valid] <= 0).any():
            raise ValueError("Valid history observations must have finite positive ages")
        history = current_grid.new_zeros(batch, count, self.spatial_grid_size ** 2, dim)
        positions = valid.nonzero().tolist()
        if positions:
            batch_images = [[images[b][m]] for b, m in positions]
            encoded = self._encode_video_batch(self._images_to_video_batch(self._spatial_images(batch_images)))
            grids = self._current_spatial_grid(encoded).to(history)
            for index, (b, m) in enumerate(positions):
                history[b, m] = grids[index]
        return history, valid, torch.where(valid, ages, 0)

    def _spatial_train_outputs(self, examples, current_tokens, target_tokens, task_tokens, state):
        current = self._current_spatial_grid(current_tokens).detach()
        target = self._spatial_grid(target_tokens).detach()
        history = valid = ages = None
        if self.use_spatial_memory:
            required = ("history_images", "history_valid", "history_ages")
            if any(any(key not in x for key in required) for x in examples):
                raise ValueError("Spatial memory training needs past images, validity and ages from the dataloader")
            history, valid, ages = self._encode_spatial_history(
                [x["history_images"] for x in examples], [x["history_valid"] for x in examples],
                [x["history_ages"] for x in examples], current)
            if self.training and self.spatial_history_dropout:
                valid = valid & (torch.rand(valid.shape, device=valid.device) >= self.spatial_history_dropout)
        predicted = self.spatial_goal_predictor(current, task_tokens, state, history, valid, ages)
        query = self.spatial_reader.make_query(task_tokens, state)
        current_local, _ = self.spatial_reader(current, query)
        # Raw frozen features/goal predictions may be detached. The reader must
        # remain on the action-loss gradient path in both goal training modes.
        target_local, _ = self.spatial_reader(target, query)
        predicted_local, _ = self.spatial_reader(predicted.detach(), query)
        return {
            "loss": F.l1_loss(predicted.float(), target.float()),
            "true_residual": self.spatial_action_adapter(current_local, target_local),
            "predicted_residual": self.spatial_action_adapter(current_local, predicted_local),
        }

    def _spatial_history_for_inference(self, current, instructions, timestamp, episode_id,
                                     history_images, history_valid, history_ages, update_memory):
        batch = current.shape[0]
        if not self.use_spatial_memory:
            if any(x is not None for x in (history_images, history_valid, history_ages)):
                raise ValueError("Explicit history requires spatial memory to be enabled")
            return None, None, None, None
        if history_images is not None:
            if history_valid is None or history_ages is None:
                raise ValueError("Explicit history requires history_valid and history_ages")
            if update_memory:
                raise ValueError("Explicit history requires update_memory=False")
            return (*self._encode_spatial_history(history_images, history_valid, history_ages, current), None)
        if history_valid is not None or history_ages is not None:
            raise ValueError("history_valid/history_ages require history_images")
        if not update_memory:
            return None, None, None, None
        if batch != 1:
            raise ValueError("Online spatial memory supports one trajectory per connection; use explicit history or update_memory=False for batches")
        if timestamp is not None and (isinstance(timestamp, (bool, np.bool_)) or
                not isinstance(timestamp, Real) or not math.isfinite(timestamp)):
            raise ValueError("timestamp must be finite numeric seconds")
        if episode_id is not None and (isinstance(episode_id, (bool, np.bool_)) or not isinstance(episode_id, (str, Integral))):
            raise ValueError("episode_id must be a string or integer")
        source = "client" if timestamp is not None else "server"
        now = float(timestamp) if timestamp is not None else time.monotonic()
        key = (tuple(instructions), episode_id, source)
        memory = self.spatial_memory
        last = memory.get("timestamp")
        reset = memory.get("key") != key or (last is not None and
                (now <= last or now - last > self.spatial_memory_max_gap))
        # Build a transaction; failed inference never advances or clears memory.
        entries = [] if reset else memory.get("entries", [])
        count = len(self.spatial_history_offsets)
        history = current.new_zeros(1, count, current.shape[1], current.shape[2])
        valid = torch.zeros(1, count, device=current.device, dtype=torch.bool)
        ages = torch.zeros(1, count, device=current.device)
        if entries:
            from starVLA.dataloader.spatial_history import select_history_indices
            times = [entry[0] for entry in entries] + [now]
            indices, mask, deltas = select_history_indices(
                times, len(entries), self.spatial_history_offsets, self.spatial_history_tolerance)
            for i in range(count):
                if mask[i]:
                    history[0, i] = entries[indices[i]][1][0].to(current)
                    valid[0, i] = True
                    ages[0, i] = float(deltas[i])
        keep_after = now + min(self.spatial_history_offsets) - self.spatial_history_tolerance
        pending = {
            "key": key, "timestamp": now,
            "entries": ([e for e in entries if e[0] >= keep_after] + [(now, current.detach().clone())])[-self.spatial_memory_max_frames:],
        }
        return history, valid, ages, pending

    def _spatial_inference_context(self, current_tokens, task_tokens, state, instructions,
                                   goal_source, subgoal_images, timestamp=None, episode_id=None,
                                   history_images=None, history_valid=None, history_ages=None,
                                   update_memory=True):
        current = self._current_spatial_grid(current_tokens)
        history, valid, ages, pending = self._spatial_history_for_inference(
            current, instructions, timestamp, episode_id, history_images, history_valid, history_ages, update_memory)
        if goal_source == "predicted":
            goal = self.spatial_goal_predictor(current, task_tokens, state, history, valid, ages)
        else:
            images = subgoal_images if goal_source == "images" else self.subgoal_tracker.current_subgoal_images(current.shape[0])
            goal = self._encode_spatial_goal_images(images)
        query = self.spatial_reader.make_query(task_tokens, state)
        current_local, attention = self.spatial_reader(current, query)
        goal_local, goal_attention = self.spatial_reader(goal, query)
        residual = self.spatial_action_adapter(current_local, goal_local)
        return {
            "query": query, "current_local": current_local, "goal_local": goal_local,
            "residual": residual, "attention": attention, "goal_attention": goal_attention,
            "history_used": (valid.sum(1) if valid is not None else torch.zeros(current.shape[0], device=current.device)),
            "pending_memory": pending,
        }

    def _spatial_progress(self, context, predicted_tokens):
        future, _ = self.spatial_reader(self._spatial_grid(predicted_tokens), context["query"])
        goal = context["goal_local"].float()
        # Same target and query for all candidates. Cosine is a feature-space
        # progress heuristic, not a calibrated probability of task success.
        return (F.cosine_similarity(future.float(), goal, dim=-1) -
                F.cosine_similarity(context["current_local"].float(), goal, dim=-1)).mean(-1)
