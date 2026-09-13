# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""
import warnings

# 全局忽略所有警告
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.runtime import (
    TrainingProgress,
    action_error_metrics,
    configure_training_accelerator,
    data_iterator_at_progress,
    load_training_checkpoint,
    resolve_resume_path,
    save_training_checkpoint,
)

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def load_fast_tokenizer():
    fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    return fast_tokenizer


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    # New goal/action/world-model heads must use the configured initialization
    # seed. Per-rank training RNG is initialized later in prepare_training().
    set_seed(cfg.get("seed", 3047))
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, DataLoader]:
    """prepare training data"""
    # VLA data loader
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        configure_training_accelerator(self.accelerator, cfg.trainer)
        self.writer = (
            SummaryWriter(log_dir=os.path.join(cfg.run_root_dir, cfg.run_id, "tensorboard"))
            if accelerator.is_main_process else None
        )

        # training status tracking
        self.progress = TrainingProgress()
        self._last_saved_step = None
        self.total_batch_size = self._calculate_total_batch_size()

    @property
    def completed_steps(self):
        return self.progress.completed_steps

    @completed_steps.setter
    def completed_steps(self, value):
        self.progress.completed_steps = value

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self.resume_path = resolve_resume_path(self.config)
        # Weight initialization starts a new run. A full resume restores the model
        # together with optimizer/scheduler state after Accelerate preparation.
        if not self.resume_path and getattr(self.config.trainer, "pretrained_checkpoint", None):
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )

        # This scheduler counts optimizer updates, not workers or microbatches.
        # Keep the existing raw scheduler and register it exactly once for saving.
        self.progress.batches_per_epoch = len(self.vla_train_dataloader)
        self.progress.world_size = self.accelerator.num_processes
        self.progress.gradient_accumulation_steps = self.accelerator.gradient_accumulation_steps
        if self.progress.batches_per_epoch == 0:
            raise ValueError("The training dataloader contains no batches")
        self.accelerator.register_for_checkpointing(self.lr_scheduler, self.progress)

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        if self.resume_path:
            self._load_checkpoint(self.resume_path)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        load_training_checkpoint(self.accelerator, self.progress, checkpoint_path)
        self.accelerator.print(
            f"Resumed step {self.completed_steps} from {checkpoint_path}; "
            f"data epoch {self.progress.data_epoch}, batch {self.progress.batches_in_epoch}. "
            "Worker augmentation/prefetch RNG is not replayed."
        )

    def _save_checkpoint(self):
        """save current training state"""

        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
        save_training_checkpoint(self.accelerator, self.model, self.progress, checkpoint_path)
        self._last_saved_step = self.completed_steps
        if self.accelerator.is_main_process:
            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if self.accelerator.is_main_process:
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(
                    self.progress.data_epoch + self.progress.batches_in_epoch / len(self.vla_train_dataloader), 2
                )

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = data_iterator_at_progress(self.accelerator, self.vla_train_dataloader, self.progress)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.progress.data_epoch += 1
            self.progress.batches_in_epoch = 0
            self._create_data_iterators()
            batch_vla = next(self.vla_iter)
        self.progress.batches_in_epoch += 1
        return batch_vla

    import torch

    def compare_state_dict(self, sd1, sd2, verbose=True):
        # 1. key 完全一致
        keys1 = set(sd1.keys())
        keys2 = set(sd2.keys())

        if keys1 != keys2:
            missing_1 = keys2 - keys1
            missing_2 = keys1 - keys2
            if verbose:
                if missing_1:
                    print("❌ sd1 缺少 keys:", missing_1)
                if missing_2:
                    print("❌ sd2 缺少 keys:", missing_2)
            return False

        # 2. 逐 tensor 比较
        for k in keys1:
            t1 = sd1[k]
            t2 = sd2[k]

            # 允许 Parameter
            if isinstance(t1, torch.nn.Parameter):
                t1 = t1.data
            if isinstance(t2, torch.nn.Parameter):
                t2 = t2.data

            # shape
            if t1.shape != t2.shape:
                if verbose:
                    print(f"❌ [{k}] shape 不一致: {t1.shape} vs {t2.shape}")
                return False

            # dtype
            if t1.dtype != t2.dtype:
                if verbose:
                    print(f"❌ [{k}] dtype 不一致: {t1.dtype} vs {t2.dtype}")
                return False

            # device 无所谓，统一搬到 CPU 比
            t1_cpu = t1.detach().cpu()
            t2_cpu = t2.detach().cpu()

            # 数值完全一致（bit 级）
            if not torch.equal(t1_cpu, t2_cpu):
                if verbose:
                    max_diff = (t1_cpu - t2_cpu).abs().max().item()
                    print(f"❌ [{k}] 数值不一致, max diff = {max_diff}")
                return False

        if verbose:
            print("✅ 两个 state_dict 完全一致")

        return True


    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        # main training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            # update progress
            did_update = self.accelerator.sync_gradients and not self.accelerator.optimizer_step_was_skipped
            if did_update:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                        {
                            "data_times": f"{t_end_data - t_start_data:.3f}",
                            "model_times": f"{t_end_model - t_start_model:.3f}",
                        }
                    )

            # evaluate model
            if did_update and self.completed_steps % self.config.trainer.eval_interval == 0:
                step_metrics = self.eval_action_model(step_metrics, examples=batch_vla)

            # record metrics
            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            if did_update:
                self._log_metrics(step_metrics)

            # save checkpoint
            if did_update and self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

            # check termination condition
            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        # training end processing
        progress_bar.close()
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None, *, examples) -> dict:
        """Action error diagnostic on the current training batch on every rank.

        Reusing this batch avoids advancing only rank zero's training iterator.
        This is a training diagnostic, not a held-out success-rate evaluation.
        """
        step_metrics = {} if step_metrics is None else step_metrics
        step_metrics.update(action_error_metrics(self.accelerator, self.model, examples))
        if self.writer is not None:
            for name in ("mae_score", "mse_score"):
                self.writer.add_scalar(name, step_metrics[name], self.completed_steps)
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)

                total_loss = sum(output_dict.values())

            # VLA backward propagation
            self.accelerator.backward(total_loss)

            # gradient clipping
            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # optimizer step
            self.optimizer.step()
            if self.accelerator.sync_gradients and not self.accelerator.optimizer_step_was_skipped:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            result_dict = {k: v.item() for k, v in output_dict.items()}
            # Components are diagnostics, already included in goal_prediction_loss.
            # Keep them outside output_dict so they are never summed twice.
            unwrapped = self.accelerator.unwrap_model(self.model)
            result_dict.update({k: v.item() for k, v in
                                getattr(unwrapped, "spatial_training_metrics", {}).items()})

        return result_dict

    def _finalize_training(self):
        """training end processing"""
        if self._last_saved_step != self.completed_steps:
            self._save_checkpoint()
        # save final model
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()
            self.writer.close()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)
    # build model
    vla = build_model(cfg)
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
