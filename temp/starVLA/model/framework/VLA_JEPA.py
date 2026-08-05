# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025].
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Privileged end-to-end VLA-JEPA framework.

Training uses a deployable student path plus a teacher-only privileged branch:
  - student: obs/lang/state_0 -> z_student -> u_student_hat_T -> action
  - teacher: visual/state endpoints -> z_teacher -> u_teacher_hat_T

Latent displacements are grounded by reconstructing the complete action chunk.
Knowledge insulation prevents the flow action objective from crossing the
future/latent-action token interface into the student/world-model dynamics.

Inference keeps only the student path and never runs V-JEPA.
"""
from contextlib import nullcontext
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoVideoProcessor

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, get_action_model
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model.delta_action_decoder import MultiStepDeltaActionDecoder
from starVLA.model.modules.world_model.privileged_latent import (
    SharedWorldDecoder,
    StateDeltaPredictor,
    StudentCurrentAdapter,
    StudentPredictor,
    TeacherEncoder,
    apply_knowledge_insulation,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


def _cuda_autocast(dtype: torch.dtype):
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denom


@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    End-to-end privileged latent world model.

    V-JEPA21 is a frozen training-only target encoder. The deployed student path
    consumes only current images, language, and state_0.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token,
        )

        qwen_cfg = self.qwen_vl_interface.model.config
        qwen_text_cfg = getattr(qwen_cfg, "text_config", None)
        qwen_hidden_size = getattr(qwen_text_cfg, "hidden_size", None)
        if qwen_hidden_size is None:
            qwen_hidden_size = getattr(qwen_cfg, "hidden_size", None)
        if qwen_hidden_size is None:
            raise AttributeError("Cannot resolve Qwen hidden_size from model config.")
        qwen_hidden_dim = int(qwen_hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = qwen_hidden_dim
        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)

        self.future_action_window_size = int(config.framework.action_model.future_action_window_size)
        self.past_action_window_size = int(config.framework.action_model.past_action_window_size)
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        latent_cfg = self.config.framework.get("privileged_latent", {})
        self.load_vjepa = bool(_cfg_get(latent_cfg, "load_vjepa", True))
        self.vj_encoder = None
        self.vj_processor = None
        self.vj_dim = int(self.config.framework.vj2_model.get("hidden_size", 1024))
        self.vj_tubelet_size = int(self.config.framework.vj2_model.get("tubelet_size", 2))
        if self.load_vjepa:
            self._load_vjepa_encoder()
        self._freeze_vjepa21()

        self.latent_dim = int(_cfg_get(latent_cfg, "latent_dim", 32))
        self.lambda_act = float(_cfg_get(latent_cfg, "lambda_act", 1.0))
        self.lambda_current_align = float(_cfg_get(latent_cfg, "lambda_current_align", 0.1))
        self.lambda_student_wm = float(
            _cfg_get(latent_cfg, "lambda_student_wm", _cfg_get(latent_cfg, "lambda_wm", 0.5))
        )
        self.lambda_teacher_wm = float(_cfg_get(latent_cfg, "lambda_teacher_wm", 0.1))
        self.lambda_distill = float(_cfg_get(latent_cfg, "lambda_distill", 0.1))
        self.lambda_latent = float(_cfg_get(latent_cfg, "lambda_latent", 0.0))
        self.lambda_state = float(_cfg_get(latent_cfg, "lambda_state", 0.0))
        self.lambda_ldad_gt = float(_cfg_get(latent_cfg, "lambda_ldad_gt", 0.05))
        self.lambda_ldad_pred = float(_cfg_get(latent_cfg, "lambda_ldad_pred", 0.05))
        self.knowledge_insulation = bool(_cfg_get(latent_cfg, "knowledge_insulation", True))
        self.delta_action_grounding = bool(_cfg_get(latent_cfg, "delta_action_grounding", True))

        vj_dim = int(self.vj_dim)
        state_dim = int(self.config.framework.action_model.get("state_dim", 0) or 0)
        if state_dim <= 0:
            raise ValueError("Privileged VLA-JEPA requires `framework.action_model.state_dim > 0`.")

        hidden_dim = int(_cfg_get(latent_cfg, "hidden_dim", qwen_hidden_dim))
        decoder_layers = int(_cfg_get(latent_cfg, "decoder_layers", 6))
        decoder_heads = int(_cfg_get(latent_cfg, "decoder_heads", 16))
        decoder_dropout = float(_cfg_get(latent_cfg, "decoder_dropout", 0.1))

        self._latent_cfg = latent_cfg

        self.student_current_adapter = StudentCurrentAdapter(
            qwen_dim=qwen_hidden_dim,
            vj_dim=vj_dim,
            hidden_dim=hidden_dim,
            dropout=float(_cfg_get(latent_cfg, "dropout", 0.0)),
        )
        self.student_predictor = StudentPredictor(
            qwen_dim=qwen_hidden_dim,
            state_dim=state_dim,
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
        )
        self.teacher_encoder = (
            TeacherEncoder(
                vj_dim=vj_dim,
                state_dim=state_dim,
                latent_dim=self.latent_dim,
                hidden_dim=int(_cfg_get(latent_cfg, "teacher_hidden_dim", vj_dim)),
                num_layers=int(_cfg_get(latent_cfg, "teacher_layers", 4)),
                num_heads=int(_cfg_get(latent_cfg, "teacher_heads", 16)),
                dropout=float(_cfg_get(latent_cfg, "teacher_dropout", 0.0)),
            )
            if self.load_vjepa
            else None
        )
        self.shared_world_decoder = SharedWorldDecoder(
            vj_dim=vj_dim,
            latent_dim=self.latent_dim,
            context_dim=int(_cfg_get(latent_cfg, "decoder_context_dim", vj_dim)),
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=decoder_dropout,
            grid_hw=tuple(_cfg_get(latent_cfg, "decoder_grid_hw", (1, 1))),
        )
        self.state_delta_predictor = (
            StateDeltaPredictor(
                latent_dim=self.latent_dim,
                state_dim=state_dim,
                hidden_dim=int(_cfg_get(latent_cfg, "state_hidden_dim", vj_dim)),
            )
            if self.lambda_state != 0.0
            else None
        )
        self.delta_action_decoder = (
            MultiStepDeltaActionDecoder(
                latent_dim=vj_dim,
                action_dim=int(self.config.framework.action_model.action_dim),
                action_horizon=self.chunk_len,
                hidden_dim=int(_cfg_get(latent_cfg, "delta_decoder_hidden_dim", 512)),
                num_layers=int(_cfg_get(latent_cfg, "delta_decoder_layers", 3)),
                num_heads=int(_cfg_get(latent_cfg, "delta_decoder_heads", 8)),
                dropout=float(_cfg_get(latent_cfg, "delta_decoder_dropout", 0.0)),
                state_dim=state_dim,
                use_state=bool(_cfg_get(latent_cfg, "delta_decoder_use_state", False)),
            )
            if self.delta_action_grounding
            else None
        )
        self.future_latent_to_qwen = nn.Linear(vj_dim, qwen_hidden_dim)
        self.latent_action_to_qwen = nn.Linear(self.latent_dim, qwen_hidden_dim)

        self.replace_prompt = "".join(
            [
                each * self.config.framework.vj2_model.num_action_tokens_per_timestep
                for each in action_tokens[
                    : self.config.framework.vj2_model.num_frames // self.vj_tubelet_size - 1
                ]
            ]
        )
        self.embodied_replace_prompt = "".join(
            [embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction]
        )

    def _load_vjepa_encoder(self) -> None:
        if self.vj_encoder is not None:
            return
        base_encoder = self.config.framework.vj2_model.base_encoder
        if str(base_encoder).endswith(".pt"):
            from starVLA.model.modules.world_model.vjepa21_encoder import (
                build_vjepa21_processor,
                load_vjepa21_encoder,
            )

            vj21_img_size = self.config.framework.vj2_model.get("image_size", 384)
            self.vj_encoder = load_vjepa21_encoder(
                checkpoint_path=base_encoder,
                arch=self.config.framework.vj2_model.get("arch", "vit_large"),
                img_size=vj21_img_size,
            )
            self.vj_processor = build_vjepa21_processor(img_size=vj21_img_size)
        else:
            self.vj_encoder = AutoModel.from_pretrained(base_encoder)
            self.vj_processor = AutoVideoProcessor.from_pretrained(base_encoder)
        self.vj_dim = int(self.vj_encoder.config.hidden_size)
        self.vj_tubelet_size = int(self.vj_encoder.config.tubelet_size)
        self._freeze_vjepa21()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_vjepa21()
        return self

    def _freeze_vjepa21(self) -> None:
        if self.vj_encoder is None:
            return
        self.vj_encoder.eval()
        for param in self.vj_encoder.parameters():
            param.requires_grad = False

    def expand_tokenizer(
        self,
        tokenizer: AutoTokenizer,
        special_action_token: str = "<|action_{}|>",
        max_action_tokens: int = 32,
        embodied_action_token: str = "<|embodied_action|>",
    ):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added action_token_i: {action_token_i}.")
            action_token_ids.append(tokenizer.convert_tokens_to_ids(action_token_i))

        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def _extract_token_block(self, hidden: torch.Tensor, input_ids: torch.Tensor, token_ids: list[int], name: str):
        mask = torch.isin(input_ids, torch.tensor(token_ids, device=input_ids.device))
        indices = mask.nonzero(as_tuple=True)
        batch_size, _, hidden_dim = hidden.shape
        if indices[0].numel() % batch_size != 0:
            raise ValueError(f"{name} token count must divide batch size, got {indices[0].numel()} for B={batch_size}.")
        return hidden[indices[0], indices[1], :].view(batch_size, -1, hidden_dim)

    def _run_student_context(self, batch_images: List[List[Image.Image]], instructions: List[str], training: bool):
        prompt_cfg = self.config.datasets.vla_data if training else self.config.datasets.get("vla_data", {})
        prompt_template = prompt_cfg.get("CoT_prompt", "") if hasattr(prompt_cfg, "get") else ""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={"{actions}": self.replace_prompt, "{e_actions}": self.embodied_replace_prompt},
            prompt_template=prompt_template,
        )

        with _cuda_autocast(torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            embodied_tokens = self._extract_token_block(
                hidden=last_hidden,
                input_ids=qwen_inputs["input_ids"],
                token_ids=[self.embodied_action_token_id],
                name="embodied_action",
            )
        current_context = embodied_tokens.mean(dim=1)
        return last_hidden, embodied_tokens, current_context

    def _states_from_examples(self, examples: List[dict], hidden_ref: torch.Tensor):
        if "state" not in examples[0]:
            raise ValueError("Privileged VLA-JEPA training/inference requires `state` / state_0.")

        state = torch.as_tensor(np.array([example["state"] for example in examples]), device=hidden_ref.device)
        state = state.to(dtype=hidden_ref.dtype)
        if state.dim() == 2:
            state = state.unsqueeze(1)
        state_0 = state[:, 0, :]

        if "state_0" in examples[0]:
            state_0 = torch.as_tensor(np.array([example["state_0"] for example in examples]), device=hidden_ref.device)
            state_0 = state_0.to(dtype=hidden_ref.dtype)
            if state_0.dim() == 3:
                state_0 = state_0[:, 0, :]

        if "state_T" in examples[0]:
            state_T = torch.as_tensor(np.array([example["state_T"] for example in examples]), device=hidden_ref.device)
            state_T = state_T.to(dtype=hidden_ref.dtype)
            if state_T.dim() == 3:
                state_T = state_T[:, -1, :]
        elif state.shape[1] >= 2:
            state_T = state[:, -1, :]
        else:
            raise ValueError("Training batch must include `state_T` or a multi-step `state` ending at the action chunk.")

        return state_0, state_T, state_0.unsqueeze(1)

    def _actions_from_examples(self, examples: List[dict], hidden_ref: torch.Tensor):
        actions = torch.as_tensor(np.array([example["action"] for example in examples]), device=hidden_ref.device)
        actions = actions.to(dtype=hidden_ref.dtype)
        return actions[:, -(self.future_action_window_size + 1) :, :]

    def _mask_from_examples(
        self,
        examples: List[dict],
        key: str,
        hidden_ref: torch.Tensor,
        *,
        target_len: int | None = None,
    ) -> torch.Tensor | None:
        if key not in examples[0]:
            return None
        mask = torch.as_tensor(np.array([example[key] for example in examples]), device=hidden_ref.device)
        mask = mask.to(dtype=torch.bool)
        if mask.dim() > 2:
            mask = mask.squeeze(-1)
        if target_len is not None and mask.dim() == 2:
            mask = mask[:, -target_len:]
        return mask

    def _encode_video_endpoints(self, batch_videos: List[np.ndarray], hidden_ref: torch.Tensor):
        if self.vj_encoder is None or self.vj_processor is None:
            raise RuntimeError(
                "V-JEPA encoder is not loaded. Set `framework.privileged_latent.load_vjepa=true` for training."
            )
        videos = np.stack(batch_videos)  # [B, V, T, H, W, 3]
        videos = videos.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, 3, H, W]
        videos = videos[:, 0]  # first-version single-view endpoint latent
        batch_size, raw_t, channels, height, width = videos.shape
        del channels, height, width
        input_videos = self.vj_processor(
            videos=[videos[i] for i in range(batch_size)], return_tensors="pt"
        )["pixel_values_videos"].to(self.vj_encoder.device)

        with torch.no_grad():
            video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)

        tubelet_size = int(self.vj_tubelet_size)
        latent_t = raw_t // tubelet_size
        if latent_t < 2:
            raise ValueError(
                f"V-JEPA endpoint latent requires T_latent >= 2, got raw_T={raw_t}, tubelet_size={tubelet_size}."
            )
        if video_embeddings.shape[1] % latent_t != 0:
            raise ValueError(
                "Cannot reshape V-JEPA tokens into temporal endpoints: "
                f"tokens={video_embeddings.shape[1]}, T={latent_t}."
            )

        spatial_tokens = video_embeddings.shape[1] // latent_t
        video_embeddings = video_embeddings.view(batch_size, latent_t, spatial_tokens, -1)
        # Both endpoints are frozen V-JEPA targets.  The privileged teacher
        # consumes both, while the deployable student only sees the current
        # image through Qwen.
        u_0_target = video_embeddings[:, 0].mean(dim=1).to(
            device=hidden_ref.device,
            dtype=hidden_ref.dtype,
        )
        u_T_target = video_embeddings[:, -1].mean(dim=1).to(
            device=hidden_ref.device,
            dtype=hidden_ref.dtype,
        )
        return u_0_target.detach(), u_T_target.detach()

    def _student_latents(self, current_context: torch.Tensor, state_0: torch.Tensor):
        u_student = self.student_current_adapter(current_context)
        z_student = self.student_predictor(current_context, state_0)
        u_student_hat_T = self.shared_world_decoder(u_student, z_student)
        return u_student, z_student, u_student_hat_T

    def _append_latent_action_tokens(
        self,
        embodied_action_tokens: torch.Tensor,
        u_student_hat_T: torch.Tensor,
        z_student: torch.Tensor,
    ) -> torch.Tensor:
        u_for_action, z_for_action = apply_knowledge_insulation(
            future_latent=u_student_hat_T,
            latent_action=z_student,
            enabled=self.knowledge_insulation,
        )
        u_token = self.future_latent_to_qwen(u_for_action).unsqueeze(1)
        z_token = self.latent_action_to_qwen(z_for_action).unsqueeze(1)
        return torch.cat([embodied_action_tokens, u_token, z_token], dim=1)

    def _compute_ldad_losses(
        self,
        *,
        u_0_target: torch.Tensor,
        u_T_target: torch.Tensor,
        u_student: torch.Tensor,
        u_student_hat_T: torch.Tensor,
        state_0: torch.Tensor,
        actions_target: torch.Tensor,
        action_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.delta_action_decoder is None:
            zero = actions_target.new_zeros((), dtype=torch.float32)
            return zero, zero

        delta_gt = u_T_target - u_0_target
        delta_pred = u_student_hat_T - u_student
        actions_from_gt_delta = self.delta_action_decoder(delta_gt, state_0)
        actions_from_pred_delta = self.delta_action_decoder(delta_pred, state_0)

        with _cuda_autocast(torch.float32):
            ldad_gt_per_dim = F.smooth_l1_loss(
                actions_from_gt_delta.float(),
                actions_target.float(),
                reduction="none",
            )
            ldad_pred_per_dim = F.smooth_l1_loss(
                actions_from_pred_delta.float(),
                actions_target.float(),
                reduction="none",
            )
            ldad_gt_loss = _masked_mean(ldad_gt_per_dim, action_mask)
            ldad_pred_loss = _masked_mean(ldad_pred_per_dim, action_mask)
        return ldad_gt_loss, ldad_pred_loss

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        del kwargs
        if not examples:
            raise ValueError("VLA_JEPA.forward requires a non-empty list of examples.")

        batch_images = [example["image"] for example in examples]
        batch_videos = [example["video"] for example in examples]
        instructions = [example["lang"] for example in examples]
        has_actions = "action" in examples[0]

        _, embodied_action_tokens, current_context = self._run_student_context(
            batch_images=batch_images,
            instructions=instructions,
            training=has_actions,
        )

        state_0, state_T, state_0_for_action = self._states_from_examples(examples, current_context)
        u_student, z_student, u_student_hat_T = self._student_latents(
            current_context=current_context,
            state_0=state_0,
        )

        u_0_target, u_T_target = self._encode_video_endpoints(
            batch_videos=batch_videos,
            hidden_ref=current_context,
        )
        action_mask = self._mask_from_examples(
            examples,
            "action_mask",
            current_context,
            target_len=self.future_action_window_size + 1,
        )
        video_mask = self._mask_from_examples(examples, "video_mask", current_context)
        state_mask = self._mask_from_examples(examples, "state_mask", current_context)
        video_endpoint_mask = self._mask_from_examples(examples, "video_endpoint_mask", current_context)
        if video_endpoint_mask is None and video_mask is not None and video_mask.dim() == 2:
            video_endpoint_mask = video_mask[:, 0] & video_mask[:, -1]
        if video_endpoint_mask is not None:
            video_endpoint_mask = video_endpoint_mask.reshape(-1).to(dtype=torch.bool)
            if not torch.any(video_endpoint_mask):
                raise ValueError("All samples in this batch have invalid video endpoints; resample a valid batch.")

        if self.teacher_encoder is None:
            raise RuntimeError(
                "TeacherEncoder is disabled. Training requires "
                "`framework.privileged_latent.load_vjepa=true`."
            )
        z_teacher = self.teacher_encoder(
            u_0_target,
            u_T_target,
            state_0,
            state_T,
        )
        u_teacher_hat_T = self.shared_world_decoder(u_0_target, z_teacher)

        with _cuda_autocast(torch.float32):
            current_align_per_dim = F.smooth_l1_loss(
                u_student.float(),
                u_0_target.float(),
                reduction="none",
            )
            student_wm_per_dim = F.smooth_l1_loss(
                u_student_hat_T.float(),
                u_T_target.float(),
                reduction="none",
            )
            teacher_wm_per_dim = F.smooth_l1_loss(
                u_teacher_hat_T.float(),
                u_T_target.float(),
                reduction="none",
            )
            current_align_loss = _masked_mean(current_align_per_dim, video_endpoint_mask)
            student_wm_loss = _masked_mean(student_wm_per_dim, video_endpoint_mask)
            teacher_wm_loss = _masked_mean(teacher_wm_per_dim, video_endpoint_mask)
            distill_loss = F.smooth_l1_loss(
                z_student.float(),
                z_teacher.detach().float(),
                reduction="mean",
            )
            latent_loss = (z_student.float() ** 2).mean()
            # Monitoring metrics only (not part of the training objective).
            z_teacher_norm = z_teacher.detach().float().norm(dim=-1).mean()
            z_student_norm = z_student.detach().float().norm(dim=-1).mean()
            z_cosine = F.cosine_similarity(
                z_student.detach().float(),
                z_teacher.detach().float(),
                dim=-1,
            ).mean()
            state_loss = torch.zeros((), device=current_context.device, dtype=torch.float32)
            if self.lambda_state != 0.0 and self.state_delta_predictor is not None:
                delta_hat = self.state_delta_predictor(z_student, state_0)
                delta_target = state_T - state_0
                state_loss_raw = F.smooth_l1_loss(delta_hat.float(), delta_target.float(), reduction="none")
                state_endpoint_mask = None
                if state_mask is not None and state_mask.dim() == 2:
                    state_endpoint_mask = state_mask[:, 0] & state_mask[:, -1]
                state_loss = _masked_mean(state_loss_raw, state_endpoint_mask)

            if not has_actions:
                total_loss = (
                    self.lambda_current_align * current_align_loss
                    + self.lambda_student_wm * student_wm_loss
                    + self.lambda_teacher_wm * teacher_wm_loss
                    + self.lambda_distill * distill_loss
                    + self.lambda_latent * latent_loss
                    + self.lambda_state * state_loss
                )
                return {
                    "loss_total": total_loss,
                    "current_align_loss": current_align_loss,
                    "student_wm_loss": student_wm_loss,
                    "wm_loss": student_wm_loss,
                    "teacher_wm_loss": teacher_wm_loss,
                    "distill_loss": distill_loss,
                    "latent_loss": latent_loss,
                    "state_loss": state_loss,
                    "z_teacher_norm": z_teacher_norm,
                    "z_student_norm": z_student_norm,
                    "z_pred_norm": z_student_norm,
                    "z_cosine": z_cosine,
                }

            actions_target = self._actions_from_examples(examples, current_context)
            ldad_gt_loss, ldad_pred_loss = self._compute_ldad_losses(
                u_0_target=u_0_target,
                u_T_target=u_T_target,
                u_student=u_student,
                u_student_hat_T=u_student_hat_T,
                state_0=state_0,
                actions_target=actions_target,
                action_mask=action_mask,
            )
            repeated_diffusion_steps = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            action_condition = self._append_latent_action_tokens(
                embodied_action_tokens,
                u_student_hat_T,
                z_student,
            )
            action_condition_repeated = action_condition.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = state_0_for_action.repeat(repeated_diffusion_steps, 1, 1)
            action_mask_repeated = (
                action_mask.repeat(repeated_diffusion_steps, 1) if action_mask is not None else None
            )
            action_loss = self.action_model(
                action_condition_repeated,
                actions_target_repeated,
                state_repeated,
                actions_mask=action_mask_repeated,
            )

            total_loss = (
                self.lambda_act * action_loss
                + self.lambda_current_align * current_align_loss
                + self.lambda_student_wm * student_wm_loss
                + self.lambda_teacher_wm * teacher_wm_loss
                + self.lambda_distill * distill_loss
                + self.lambda_latent * latent_loss
                + self.lambda_state * state_loss
                + self.lambda_ldad_gt * ldad_gt_loss
                + self.lambda_ldad_pred * ldad_pred_loss
            )

        return {
            "loss_total": total_loss,
            "action_loss": action_loss,
            "current_align_loss": current_align_loss,
            "student_wm_loss": student_wm_loss,
            "wm_loss": student_wm_loss,
            "teacher_wm_loss": teacher_wm_loss,
            "distill_loss": distill_loss,
            "ldad_gt_loss": ldad_gt_loss,
            "ldad_pred_loss": ldad_pred_loss,
            "latent_loss": latent_loss,
            "state_loss": state_loss,
            "z_teacher_norm": z_teacher_norm,
            "z_student_norm": z_student_norm,
            "z_pred_norm": z_student_norm,
            "z_cosine": z_cosine,
        }

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        **kwargs: str,
    ) -> np.ndarray:
        del kwargs
        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        _, embodied_action_tokens, current_context = self._run_student_context(
            batch_images=batch_images,
            instructions=instructions,
            training=False,
        )

        if state is None:
            state_0 = None
            state_0_for_action = None
        else:
            state_0 = torch.as_tensor(np.array(state), device=current_context.device, dtype=current_context.dtype)
            if state_0.dim() == 2:
                state_0_for_action = state_0.unsqueeze(1)
            elif state_0.dim() == 3:
                state_0_for_action = state_0[:, 0:1, :]
                state_0 = state_0[:, 0, :]
            else:
                raise ValueError(f"`state` must be [B,D] or [B,1,D], got {tuple(state_0.shape)}")

        if state_0 is None:
            raise ValueError("Privileged VLA-JEPA predict_action requires current `state`.")

        _, z_student, u_student_hat_T = self._student_latents(
            current_context=current_context,
            state_0=state_0,
        )
        action_condition = self._append_latent_action_tokens(
            embodied_action_tokens,
            u_student_hat_T,
            z_student,
        )

        with _cuda_autocast(torch.float32):
            pred_actions = self.action_model.predict_action(action_condition, state_0_for_action)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {
            "normalized_actions": normalized_actions,
            "z_student": z_student.to(dtype=torch.float32).detach().cpu().numpy(),
            "u_student_hat_T": u_student_hat_T.to(dtype=torch.float32).detach().cpu().numpy(),
            # Backward-compatible aliases used by the existing deployment
            # adapters and visualization scripts.
            "z_pred": z_student.to(dtype=torch.float32).detach().cpu().numpy(),
            "u_hat_T": u_student_hat_T.to(dtype=torch.float32).detach().cpu().numpy(),
        }
