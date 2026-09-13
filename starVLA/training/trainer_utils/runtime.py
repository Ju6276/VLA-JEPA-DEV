"""Evaluation and restart state shared by the step-based VLA trainer."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import torch


def configure_training_accelerator(accelerator, trainer_config, *, deepspeed_plugin=None):
    """Apply trainer settings before prepare() creates the distributed optimizer.

    DeepSpeed performs accumulation and clipping inside engine.backward/step;
    changing only Accelerate's counters or calling clip_grad_norm_ afterwards
    cannot configure those operations.
    """
    steps = trainer_config.get("gradient_accumulation_steps", 1)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be a positive integer")
    clipping = trainer_config.get("gradient_clipping")
    if clipping is not None and (
        isinstance(clipping, bool) or not isinstance(clipping, (int, float))
        or not math.isfinite(clipping) or clipping < 0
    ):
        raise ValueError("trainer.gradient_clipping must be a nonnegative finite number or null")
    if deepspeed_plugin is None:
        deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    accelerator.gradient_accumulation_steps = steps
    # This is a step-based loop over a continuous stream. Partial epochs must
    # not reset Accelerate's boundary while DeepSpeed continues accumulating.
    accelerator.gradient_state.plugin_kwargs["sync_with_dataloader"] = False
    # ZeRO-2 partitions gradients during every backward pass and rejects no_sync.
    accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = deepspeed_plugin is not None
    if deepspeed_plugin is not None:
        deepspeed_plugin.deepspeed_config["gradient_accumulation_steps"] = steps
        # DeepSpeed represents disabled clipping as 0, rather than None.
        deepspeed_plugin.deepspeed_config["gradient_clipping"] = float(clipping or 0)


@contextmanager
def evaluating(model):
    """Disable dropout and autograd, restoring even mixed child modes on error."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        # model.train(was_training) alone would overwrite frozen children in eval.
        for module, training in modes:
            module.training = training


def action_error_metrics(accelerator, model, examples):
    """All ranks evaluate their current batch, then reduce sums and element count."""
    model = accelerator.unwrap_model(model)
    with evaluating(model), accelerator.autocast():
        output = model.predict_action(
            batch_images=[example["image"] for example in examples],
            instructions=[example["lang"] for example in examples],
            state=[example["state"] for example in examples] if "state" in examples[0] else None,
            use_ddim=True,
            num_ddim_steps=20,
        )
    predicted = output["normalized_actions"]
    if torch.is_tensor(predicted):
        predicted = predicted.detach().float().cpu().numpy()
    target = np.asarray([example["action"] for example in examples])
    predicted = np.asarray(predicted, dtype=np.float64)
    if predicted.shape != target.shape:
        raise ValueError(f"Action diagnostic shape mismatch: predicted={predicted.shape}, target={target.shape}")
    error = predicted - target
    totals = torch.tensor(
        [np.abs(error).sum(), np.square(error).sum(), error.size],
        dtype=torch.float64,
        device=accelerator.device,
    )
    totals = accelerator.reduce(totals, reduction="sum")
    return {"mae_score": (totals[0] / totals[2]).item(), "mse_score": (totals[1] / totals[2]).item()}


@dataclass
class TrainingProgress:
    completed_steps: int = 0
    data_epoch: int = 0
    batches_in_epoch: int = 0
    batches_per_epoch: int = 0
    world_size: int = 1
    gradient_accumulation_steps: int = 1

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state):
        for name in ("batches_per_epoch", "world_size", "gradient_accumulation_steps"):
            if state[name] != getattr(self, name):
                raise ValueError(
                    f"Cannot resume with changed {name}: checkpoint={state[name]}, "
                    f"current={getattr(self, name)}. Use pretrained_checkpoint for weight initialization."
                )
        for name in ("completed_steps", "data_epoch", "batches_in_epoch"):
            value = int(state[name])
            if value < 0:
                raise ValueError(f"Invalid checkpoint {name}: {value}")
            setattr(self, name, value)
        if self.batches_in_epoch > self.batches_per_epoch:
            raise ValueError("Checkpoint batch cursor exceeds the dataloader length")


def resolve_resume_path(config):
    """A full-state directory is independent of optional weight initialization."""
    trainer = config.trainer
    path = trainer.get("resume_from_checkpoint") or config.get("resume_from_checkpoint")
    if trainer.get("is_resume", False) and not path:
        raise ValueError("trainer.is_resume requires trainer.resume_from_checkpoint (a full-state directory)")
    if not path:
        return None
    if not isinstance(path, (str, Path)):
        raise ValueError("trainer.resume_from_checkpoint must be a full-state directory path")
    path = Path(path).expanduser()
    if not path.is_dir():
        raise ValueError(
            f"Resume directory does not exist: {path}. A .pt export is weight-only; "
            "load it with trainer.pretrained_checkpoint instead."
        )
    return str(path)


def save_training_checkpoint(accelerator, model, progress, directory):
    """All ranks must enter: ZeRO optimizer/model checkpointing uses collectives.

    The scheduler and TrainingProgress must be registered for checkpointing before
    this function. Accelerator also saves optimizer, RNG and precision state.
    The JSON marker is written last, so partially written directories cannot resume.
    """
    directory = Path(directory)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "trainer_state.json").unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    accelerator.save_state(str(directory), safe_serialization=False)
    # get_state_dict can itself be collective (e.g. ZeRO-3), even for one export.
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        torch.save(state_dict, str(directory) + "_pytorch_model.pt")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        metadata = {"format_version": 1, **progress.state_dict()}
        (directory / "trainer_state.json").write_text(json.dumps(metadata, indent=2) + "\n")
    accelerator.wait_for_everyone()


def load_training_checkpoint(accelerator, progress, directory):
    directory = Path(directory)
    marker = directory / "trainer_state.json"
    if not marker.is_file():
        raise ValueError(f"{directory} is not a complete training checkpoint (missing trainer_state.json)")
    metadata = json.loads(marker.read_text())
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported trainer checkpoint format")
    # Accelerate 1.5 logs and ignores RNG loading failures, so catch missing
    # per-rank state before a seemingly successful but incomplete resume.
    for rank in range(metadata["world_size"]):
        if not (directory / f"random_states_{rank}.pkl").is_file():
            raise ValueError(f"Incomplete training checkpoint: missing RNG state for rank {rank}")
    # Validate data topology before mutating model or optimizer state.
    probe = TrainingProgress(**progress.state_dict())
    probe.load_state_dict(metadata)
    accelerator.wait_for_everyone()
    accelerator.load_state(str(directory))
    if progress.state_dict() != probe.state_dict():
        raise ValueError("Checkpoint progress does not match its completion marker")
    accelerator.wait_for_everyone()


def data_iterator_at_progress(accelerator, dataloader, progress):
    """Resume the batch position; worker augmentation/prefetch RNG is not serialized.

    Epoch-aware samplers and map-style datasets recover their index position.
    This is not a bitwise replay of stochastic preprocessing or iterable datasets.
    """
    if progress.batches_in_epoch == len(dataloader):
        progress.data_epoch += 1
        progress.batches_in_epoch = 0
    if callable(getattr(dataloader, "set_epoch", None)):
        dataloader.set_epoch(progress.data_epoch)
    elif callable(getattr(dataloader.sampler, "set_epoch", None)):
        dataloader.sampler.set_epoch(progress.data_epoch)
    # Some Accelerate shards update either the sampler or the dataset. A mixture
    # with epoch-conditioned sampling needs both, including persistent workers.
    dataset = getattr(dataloader, "dataset", None)
    if callable(getattr(dataset, "set_epoch", None)):
        dataset.set_epoch(progress.data_epoch)
    remaining = dataloader
    if progress.batches_in_epoch:
        remaining = accelerator.skip_first_batches(dataloader, progress.batches_in_epoch)
        # skip_first_batches constructs a new shard; carry over its epoch too.
        if callable(getattr(remaining, "set_epoch", None)):
            remaining.set_epoch(progress.data_epoch)
    return iter(remaining)
