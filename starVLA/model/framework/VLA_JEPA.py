# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoVideoProcessor, AutoModel, AutoTokenizer, VJEPA2VideoProcessor

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC
from starVLA.model.modules.world_model.delta_jepa import (
    CandidateActionEncoder,
    LatentInverseDynamics,
    latent_progress_score,
    pool_last_temporal_frame,
    pool_vjepa_tokens,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.tools.subgoal_tracker import SubgoalTracker

@FRAMEWORK_REGISTRY.register("VLA_JEPA")
class VLA_JEPA(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen VL interface for fused language/vision token embeddings
      - DiT diffusion head for future action sequence modeling
      - JEPA world model for future frame prediction

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        embodied_action_token = self.config.framework.vj2_model.get("embodied_action_token", "<|embodied_action|>")
        action_tokens, self.action_token_ids, self.embodied_action_token_id = self.expand_tokenizer(
            tokenizer=self.qwen_vl_interface.processor.tokenizer,
            special_action_token=self.config.framework.vj2_model.special_action_token,
            max_action_tokens=self.config.framework.action_model.action_horizon * 4,
            embodied_action_token=embodied_action_token
        )

        # TODO speical tokens

        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        base_encoder = self.config.framework.vj2_model.base_encoder
        if str(base_encoder).endswith(".pt"):
            # V-JEPA 2.1: Meta raw checkpoint loaded via the vendored encoder adapter
            # (the installed transformers does not support the 2.1 architecture).
            from starVLA.model.modules.world_model.vjepa21_encoder import (
                load_vjepa21_encoder,
                build_vjepa21_processor,
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

        tubelet_size = self.vj_encoder.config.tubelet_size
        self.num_video_views = self.config.framework.vj2_model.get("num_video_views", 2)
        self.vj_predictor = VisionTransformerPredictorAC(
            num_frames=self.config.framework.vj2_model.num_frames//tubelet_size,
            img_size=((self.vj_encoder.config.image_size, self.vj_encoder.config.image_size)),
            tubelet_size=1,
            depth=self.config.framework.vj2_model.depth,
            num_heads=self.config.framework.vj2_model.num_heads,
            embed_dim=self.vj_encoder.config.hidden_size * self.num_video_views,
            action_embed_dim=self.qwen_vl_interface.model.config.hidden_size,
            num_add_tokens=self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )
        self.replace_prompt = "".join(
            [each * self.config.framework.vj2_model.num_action_tokens_per_timestep for each in
             action_tokens[:self.config.framework.vj2_model.num_frames//tubelet_size - 1]]
        )

        self.embodied_replace_prompt = "".join([embodied_action_token * self.config.framework.vj2_model.num_embodied_action_tokens_per_instruction])

        # Delta-JEPA: control-aware latent displacement + inference-time verifier support
        delta_cfg = self.config.framework.get("delta_jepa", {})
        self.use_delta_jepa = bool(delta_cfg.get("enabled", False))
        self.delta_jepa_blend_vlm_tokens = bool(delta_cfg.get("blend_vlm_tokens", True))
        self.lambda_wm = float(delta_cfg.get("lambda_wm", 0.1))
        self.lambda_delta = float(delta_cfg.get("lambda_delta", 0.05))
        self.lambda_ctrl = float(delta_cfg.get("lambda_ctrl", 0.02))
        self.ctrl_action_step = delta_cfg.get("ctrl_action_step", "first")
        self.verifier_num_candidates = int(delta_cfg.get("verifier_num_candidates", 8))
        self.use_verifier_default = bool(delta_cfg.get("use_verifier", False))
        self.subgoal_tracking_mode = delta_cfg.get("subgoal_tracking_mode", "sequential")
        self.subgoal_epsilon = float(delta_cfg.get("subgoal_epsilon", 0.15))
        self.subgoals_path = delta_cfg.get("subgoals_path", None)

        self.num_temporal_frames = self.config.framework.vj2_model.num_frames // tubelet_size
        self.num_predictor_action_tokens = max(
            1,
            (self.num_temporal_frames - 1)
            * self.config.framework.vj2_model.num_action_tokens_per_timestep,
        )

        if self.use_delta_jepa:
            vlm_hidden_dim = self.qwen_vl_interface.model.config.hidden_size
            action_dim = self.config.framework.action_model.action_dim
            state_dim = self.config.framework.action_model.state_dim
            jepa_embed_dim = self.vj_encoder.config.hidden_size * self.num_video_views
            hidden_dim = int(delta_cfg.get("hidden_dim", 512))

            self.candidate_action_encoder = CandidateActionEncoder(
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                output_dim=vlm_hidden_dim,
                num_output_tokens=self.num_predictor_action_tokens,
            )
            self.inv_dyn_decoder = LatentInverseDynamics(
                latent_dim=jepa_embed_dim,
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
            logger.info(
                "Delta-JEPA enabled: lambda_wm=%s lambda_delta=%s lambda_ctrl=%s",
                self.lambda_wm,
                self.lambda_delta,
                self.lambda_ctrl,
            )
        else:
            self.candidate_action_encoder = None
            self.inv_dyn_decoder = None

        self.subgoal_tracker: Optional[SubgoalTracker] = None
        if self.subgoals_path:
            self.load_subgoal_tracker(self.subgoals_path, precompute_latents=False)

    def _encode_subgoal_batch(self, goal_images: List[Image.Image]) -> torch.Tensor:
        """Encode a list of subgoal images into pooled latents [M, D]."""
        goal_embeddings = self._encode_goal_images(goal_images)
        return goal_embeddings

    def load_subgoal_tracker(self, subgoals_path: str, precompute_latents: bool = True) -> None:
        """Load demo-derived subgoals for online tracking."""
        self.subgoals_path = subgoals_path
        encode_fn = self._encode_subgoal_batch if precompute_latents else None
        self.subgoal_tracker = SubgoalTracker.from_path(
            subgoals_path,
            mode=self.subgoal_tracking_mode,
            epsilon=self.subgoal_epsilon,
            encode_fn=encode_fn,
        )
        logger.info(
            "Loaded %s subgoals from %s (mode=%s)",
            self.subgoal_tracker.num_subgoals,
            subgoals_path,
            self.subgoal_tracking_mode,
        )

    def reset_subgoal_tracker(self) -> None:
        if self.subgoal_tracker is not None:
            self.subgoal_tracker.reset()

    def _get_delta_jepa_action_target(self, actions_target: torch.Tensor) -> torch.Tensor:
        """Pick the action label used by inverse-dynamics supervision."""
        if self.ctrl_action_step == "mean":
            return actions_target.mean(dim=1)
        if self.ctrl_action_step == "last":
            return actions_target[:, -1, :]
        return actions_target[:, 0, :]

    def _build_predictor_action_cond(
        self,
        action_tokens: torch.Tensor,
        actions_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build predictor conditioning tokens (VLM tokens, action chunks, or both)."""
        if not self.use_delta_jepa or actions_target is None:
            return action_tokens

        action_encoder_param = next(self.candidate_action_encoder.parameters())
        actions_target = actions_target.to(
            device=action_encoder_param.device,
            dtype=action_encoder_param.dtype,
        )
        candidate_cond = self.candidate_action_encoder(actions_target)
        if candidate_cond.shape[1] != action_tokens.shape[1]:
            candidate_cond = F.interpolate(
                candidate_cond.transpose(1, 2),
                size=action_tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        if self.delta_jepa_blend_vlm_tokens:
            return action_tokens + candidate_cond
        return candidate_cond

    def _compute_delta_jepa_losses(
        self,
        input_states: torch.Tensor,
        gt_states: torch.Tensor,
        predicted_states: torch.Tensor,
        actions_target: torch.Tensor,
        state: Optional[torch.Tensor],
        tokens_per_frame: int,
    ) -> dict:
        """Delta-JEPA displacement + control-aware inverse dynamics losses."""
        z_t = pool_last_temporal_frame(input_states, tokens_per_frame)
        z_tk = pool_vjepa_tokens(gt_states)
        delta_z_gt = z_tk - z_t

        z_hat = pool_vjepa_tokens(predicted_states)
        delta_z_pred = z_hat - z_t.detach()

        delta_loss = F.mse_loss(delta_z_pred, delta_z_gt.detach())

        if state is None:
            ctrl_loss = torch.zeros((), device=predicted_states.device, dtype=predicted_states.dtype)
        else:
            if state.dim() == 2:
                state = state.unsqueeze(1)
            a_gt = self._get_delta_jepa_action_target(actions_target)
            a_hat_gt = self.inv_dyn_decoder(delta_z_gt.detach(), state)
            a_hat_pred = self.inv_dyn_decoder(delta_z_pred, state)
            ctrl_loss = F.mse_loss(a_hat_gt, a_gt) + F.mse_loss(a_hat_pred, a_gt)

        return {"delta_loss": delta_loss, "ctrl_loss": ctrl_loss}

    def _encode_video_batch(self, batch_videos: np.ndarray) -> torch.Tensor:
        """Encode a numpy video batch with the frozen V-JEPA encoder. Returns [B, N, D]."""
        batch_videos = batch_videos.transpose(0, 1, 2, 5, 3, 4)  # [B, V, T, 3, H, W]
        bsz, num_views, num_frames, channels, height, width = batch_videos.shape
        flat_videos = batch_videos.reshape(bsz * num_views, num_frames, channels, height, width)
        input_videos = self.vj_processor(
            videos=[flat_videos[i] for i in range(flat_videos.shape[0])],
            return_tensors="pt",
        )["pixel_values_videos"]
        encoder_param = next(self.vj_encoder.parameters())
        input_videos = input_videos.to(
            device=encoder_param.device,
            dtype=encoder_param.dtype,
        )
        use_cuda_autocast = (
            encoder_param.is_cuda
            and encoder_param.dtype in (torch.float16, torch.bfloat16)
        )
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=encoder_param.dtype,
            enabled=use_cuda_autocast,
        ):
            video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
            if num_views > 1:
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=num_views, dim=0), dim=2)
        return video_embeddings

    def _images_to_video_batch(
        self,
        batch_images: List[List[Image.Image]],
        num_frames: Optional[int] = None,
    ) -> np.ndarray:
        """Convert current images into a short video clip by repeating frames."""
        num_frames = num_frames or self.config.framework.vj2_model.num_frames
        videos = []
        for sample_images in batch_images:
            frame = np.array(sample_images[0]).astype(np.uint8)
            view_video = np.stack([frame] * num_frames, axis=0)  # [T, H, W, 3]
            videos.append(view_video[None, ...])  # [1, T, H, W, 3]
        return np.stack(videos, axis=0)  # [B, 1, T, H, W, 3]

    def _encode_goal_images(self, goal_images: List[Image.Image]) -> torch.Tensor:
        """Encode subgoal images into pooled JEPA latents. Returns [B, D]."""
        goal_batch = self._images_to_video_batch([[img] for img in goal_images])
        goal_embeddings = self._encode_video_batch(goal_batch)
        tokens_per_frame = max(1, goal_embeddings.shape[1] // self.num_temporal_frames)
        return pool_vjepa_tokens(goal_embeddings[:, -tokens_per_frame:, :])

    def _predict_future_latent(
        self,
        video_embeddings: torch.Tensor,
        action_cond: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run predictor and return (input_states, gt_states, predicted_states)."""
        num_temporal = self.num_temporal_frames
        tokens_per_step = video_embeddings.shape[1] // num_temporal
        input_states = video_embeddings[:, :-tokens_per_step, :]
        gt_states = video_embeddings[:, tokens_per_step:, :]
        predictor_param = next(self.vj_predictor.parameters())
        input_states = input_states.to(
            device=predictor_param.device,
            dtype=predictor_param.dtype,
        )
        action_cond = action_cond.to(
            device=predictor_param.device,
            dtype=predictor_param.dtype,
        )
        use_cuda_autocast = (
            predictor_param.is_cuda
            and predictor_param.dtype in (torch.float16, torch.bfloat16)
        )
        with torch.autocast(
            device_type="cuda",
            dtype=predictor_param.dtype,
            enabled=use_cuda_autocast,
        ):
            predicted_states = self.vj_predictor(input_states, action_cond)
        return input_states, gt_states, predicted_states

    def _get_vlm_action_tokens(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        include_embodied: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            prompt_replace_dict={
                "{actions}": self.replace_prompt,
                "{e_actions}": self.embodied_replace_prompt,
            },
            prompt_template=self.config.datasets.vla_data.get("CoT_prompt", ""),
        )
        action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor(self.action_token_ids, device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)
        embodied_action_indices = torch.isin(
            qwen_inputs["input_ids"],
            torch.tensor([self.embodied_action_token_id], device=qwen_inputs["input_ids"].device),
        ).nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]
            batch_size = last_hidden.shape[0]
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(batch_size, -1, last_hidden.shape[-1])
            embodied_action_tokens = None
            if include_embodied:
                embodied_action_tokens = last_hidden[
                    embodied_action_indices[0], embodied_action_indices[1], :
                ].view(batch_size, -1, last_hidden.shape[-1])
        return action_tokens, embodied_action_tokens

    def sample_action_candidates(
        self,
        embodied_action_tokens: torch.Tensor,
        state: Optional[torch.Tensor],
        num_candidates: int,
    ) -> torch.Tensor:
        """Sample multiple action chunks from the flow-matching head."""
        candidates = []
        with torch.autocast("cuda", dtype=torch.float32):
            for _ in range(num_candidates):
                candidates.append(self.action_model.predict_action(embodied_action_tokens, state))
        return torch.stack(candidates, dim=1)  # [B, N, T, action_dim]

    def expand_tokenizer(self, 
                         tokenizer: AutoTokenizer,
                         special_action_token: str = "<|action_{}|>",
                         max_action_tokens: int = 32,
                         embodied_action_token: str = "<|embodied_action|>"):
        action_tokens, action_token_ids = [], []
        for i in range(0, max_action_tokens):
            action_token_i = special_action_token.format(i)
            action_tokens.append(action_token_i)
            if action_token_i not in tokenizer.get_vocab():
                added = tokenizer.add_tokens([action_token_i], special_tokens=True)
                if added == 0:
                    logger.warning(f"Warning: 0 tokens added (they may already exist) action_token_i: {action_token_i}.")
            action_token_id = tokenizer.convert_tokens_to_ids(action_token_i)    
            action_token_ids.append(action_token_id)
        
        if embodied_action_token not in tokenizer.get_vocab():
            added = tokenizer.add_tokens([embodied_action_token], special_tokens=True)
            if added == 0:
                logger.warning(f"Warning: 0 tokens added (they may already exist) embodied_action_token: {embodied_action_token}.")
        embodied_action_token_id = tokenizer.convert_tokens_to_ids(embodied_action_token)

        vla_embedding_size = self.qwen_vl_interface.model.get_input_embeddings().weight.size(0)
        if vla_embedding_size < len(tokenizer):
            # 2) resize embeddings of vla
            self.qwen_vl_interface.model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Model embedding size: {vla_embedding_size} ;tokenizer.vocab_size: {len(tokenizer)}")
        return action_tokens, action_token_ids, embodied_action_token_id

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL.Image]]
        batch_videos = [example["video"] for example in examples]  #  [B, V, T, H, W, 3]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"]for example in examples] if "action" in examples[0] else None # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        """
        if self.action_model.device == torch.device("cuda:0") and "action" in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(actions[0].shape) # [T-1, action_dim]
            print(state[0].shape) if state is not None else print("No state") #[state_dim]
            print(len(batch_videos), len(instructions), len(actions), len(state) if state is not None else "No state")
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "data_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "data_view_1.mp4")
            batch_images[0][0].save("data_image_view_0.png")
            batch_images[0][1].save("data_image_view_1.png")
            #print(self.action_tokens)
            print(self.replace_prompt)
            print(self.action_token_ids)
        elif self.action_model.device == torch.device("cuda:0") and "action" not in examples[0]:
            print(batch_videos[0].shape) #[V, T, H, W, 3]
            print(instructions[0])
            print(len(batch_videos), len(instructions))
            from diffusers.utils import export_to_video
            export_to_video(batch_videos[0][0]/255.0, "video_view_0.mp4")
            export_to_video(batch_videos[0][1]/255.0, "video_view_1.mp4")
            batch_images[0][0].save("video_image_view_0.png")
        exit()
        """
        
        

        #[print(each.shape, end=";") for each in batch_videos]
        batch_videos = np.stack(batch_videos)  #  [B, V, T, H, W, 3]
        batch_videos = batch_videos.transpose(0,1,2,5,3,4)  # [B, V, T, 3, H, W]

        # Step 1: QWenVL input format
        if actions is not None:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt},
                prompt_template=self.config.datasets.vla_data.get("CoT_prompt", "")) 
        else:
            qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
                images=batch_images, 
                instructions=instructions,
                prompt_replace_dict={"{actions}":self.replace_prompt},
                prompt_template=self.config.datasets.video_data.get("CoT_prompt", ""))
        
        action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        action_indices = action_indices.nonzero(as_tuple=True)

        # TODO action condition tokens
        #embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)
        
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            action_tokens = last_hidden[action_indices[0], action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)  # [B, action_len, H]
            #print(action_tokens.shape, last_hidden.shape, embodied_action_tokens.shape)
            #exit()
        
            # Step 2: JEPA Encoder
            B, V, T, C, H, W = batch_videos.shape
            batch_videos = batch_videos.reshape(B*V, T, C, H, W)  # [B*V, T, C, H, W]
            input_videos = self.vj_processor(
                videos=[batch_videos[i] for i in range(B*V)], return_tensors="pt"
            )["pixel_values_videos"].to(self.vj_encoder.device)  # [B*V, T, C, H, W]
            with torch.no_grad():
                video_embeddings = self.vj_encoder.get_vision_features(pixel_values_videos=input_videos)
                video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
            #print(video_embeddings.shape) # [B, T//tubelet_size * dim_per_frame, V*embed_dim]
        
            # Step 3: VJ Predictor
            T = T // self.vj_encoder.config.tubelet_size
            tokens_per_step = video_embeddings.shape[1] // T
            input_states = video_embeddings[:, :-tokens_per_step, :]
            gt_states = video_embeddings[:, tokens_per_step:, :]

            action_cond = action_tokens
            actions_tensor = None
            if actions is not None and self.use_delta_jepa:
                actions_tensor = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )
                actions_target_for_predictor = actions_tensor[
                    :, -(self.future_action_window_size + 1) :, :
                ]
                action_cond = self._build_predictor_action_cond(action_tokens, actions_target_for_predictor)

            predicted_states = self.vj_predictor(
                input_states,
                action_cond
            )

            teacher_forcing_wm_loss = F.l1_loss(
                predicted_states,
                gt_states,
                reduction="mean"
            )
        
        if "action" not in examples[0]:
            wm_weight = self.lambda_wm if self.use_delta_jepa else 1.0
            return {"wm_loss": teacher_forcing_wm_loss * wm_weight}

        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            if actions_tensor is None:
                actions_tensor = torch.tensor(
                    np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype
                )
            actions_target = actions_tensor[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            embodied_action_repeated = embodied_action_tokens.repeat(repeated_diffusion_steps, 1, 1)
            
            state_repeated = None
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=last_hidden.dtype
                )
                state_repeated = state_tensor.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(embodied_action_repeated, actions_target_repeated, state_repeated)

        wm_weight = self.lambda_wm if self.use_delta_jepa else 0.1
        output = {
            "action_loss": action_loss,
            "wm_loss": teacher_forcing_wm_loss * wm_weight,
        }

        if self.use_delta_jepa:
            delta_losses = self._compute_delta_jepa_losses(
                input_states=input_states,
                gt_states=gt_states,
                predicted_states=predicted_states,
                actions_target=actions_target,
                state=state_tensor,
                tokens_per_frame=tokens_per_step,
            )
            output["delta_loss"] = delta_losses["delta_loss"] * self.lambda_delta
            output["ctrl_loss"] = delta_losses["ctrl_loss"] * self.lambda_ctrl

        return output

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: List[List[Image.Image]],  # Batch of PIL Image list as [view1, view2]
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        use_verifier: Optional[bool] = None,
        reset_subgoals: bool = False,
        num_candidates: Optional[int] = None,
        subgoal_images: Optional[List[Image.Image]] = None,
        **kwargs,
    ) -> dict:
        """
        Predict actions. When verifier mode is enabled, runs best-of-N latent verification.
        """
        if reset_subgoals:
            self.reset_subgoal_tracker()

        should_verify = self.use_verifier_default if use_verifier is None else bool(use_verifier)
        if should_verify and self.use_delta_jepa:
            return self.predict_action_verified(
                batch_images=batch_images,
                instructions=instructions,
                state=state,
                subgoal_images=subgoal_images,
                num_candidates=num_candidates,
                **kwargs,
            )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, 
            instructions=instructions,
            prompt_replace_dict={"{actions}":self.replace_prompt, "{e_actions}":self.embodied_replace_prompt})
        
        embodied_action_indices = torch.isin(qwen_inputs['input_ids'], torch.tensor([self.embodied_action_token_id], device=qwen_inputs['input_ids'].device))
        #embodied_action_indices = ~torch.isin(qwen_inputs['input_ids'], torch.tensor(self.action_token_ids, device=qwen_inputs['input_ids'].device))
        embodied_action_indices = embodied_action_indices.nonzero(as_tuple=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]
            B, _, H = last_hidden.shape
            embodied_action_tokens = last_hidden[embodied_action_indices[0], embodied_action_indices[1], :].view(B, -1, H)

        state = torch.from_numpy(np.array(state)).to(last_hidden.device, dtype=last_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(embodied_action_tokens, state)  # (B, chunk_len, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions, "embodied_action_tokens": embodied_action_tokens.to(dtype=torch.float32).detach().cpu().numpy()}

    @torch.inference_mode()
    def predict_action_verified(
        self,
        batch_images: List[List[Image.Image]],
        instructions: List[str],
        state: Optional[np.ndarray] = None,
        subgoal_images: Optional[List[Image.Image]] = None,
        num_candidates: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """
        Best-of-N inference with JEPA latent action verification.

        Samples multiple candidate action chunks, rolls each forward in latent space,
        and selects the candidate with the best predicted progress toward the subgoal.
        """
        if not self.use_delta_jepa:
            raise RuntimeError("predict_action_verified requires framework.delta_jepa.enabled=true")

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        num_candidates = num_candidates or self.verifier_num_candidates
        video_batch = self._images_to_video_batch(batch_images)
        video_embeddings = self._encode_video_batch(video_batch)
        tokens_per_frame = max(1, video_embeddings.shape[1] // self.num_temporal_frames)
        z_current = pool_last_temporal_frame(video_embeddings, tokens_per_frame)

        if self.subgoal_tracker is not None:
            self.subgoal_tracker.maybe_precompute_latents(self._encode_subgoal_batch)
            self.subgoal_tracker.update(z_current[0] if z_current.shape[0] == 1 else z_current)
            subgoal_images = self.subgoal_tracker.current_subgoal_images(len(batch_images))
        elif subgoal_images is None:
            raise ValueError(
                "subgoal_images is required unless framework.delta_jepa.subgoals_path is configured"
            )

        action_tokens, embodied_action_tokens = self._get_vlm_action_tokens(
            batch_images=batch_images,
            instructions=instructions,
            include_embodied=True,
        )

        state_tensor = (
            torch.from_numpy(np.array(state)).to(embodied_action_tokens.device, dtype=embodied_action_tokens.dtype)
            if state is not None
            else None
        )

        candidates = self.sample_action_candidates(
            embodied_action_tokens=embodied_action_tokens,
            state=state_tensor,
            num_candidates=num_candidates,
        )

        z_goal = self._encode_goal_images(subgoal_images)

        best_scores = torch.full((candidates.shape[0],), -1e9, device=candidates.device)
        best_actions = candidates[:, 0]

        for idx in range(num_candidates):
            candidate_chunk = candidates[:, idx]
            candidate_cond = self._build_predictor_action_cond(action_tokens, candidate_chunk)
            _, _, predicted_states = self._predict_future_latent(video_embeddings, candidate_cond)
            z_predicted = pool_vjepa_tokens(predicted_states)
            scores = latent_progress_score(z_current, z_predicted, z_goal)
            better = scores > best_scores
            best_scores = torch.where(better, scores, best_scores)
            best_actions = torch.where(better.unsqueeze(-1).unsqueeze(-1), candidate_chunk, best_actions)

        return {
            "normalized_actions": best_actions.detach().cpu().numpy(),
            "verification_scores": best_scores.detach().cpu().numpy(),
            "all_candidates": candidates.detach().cpu().numpy(),
            "subgoal_index": (
                self.subgoal_tracker.current_index if self.subgoal_tracker is not None else None
            ),
            "subgoal_state": (
                self.subgoal_tracker.state_dict() if self.subgoal_tracker is not None else None
            ),
        }



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
     
    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
