"""W&B identity and configuration for fresh and resumed policy training."""

import json
import os
from pathlib import Path
import tempfile

from omegaconf import OmegaConf
import wandb


def initialize_wandb(config, *, resume_path=None):
    """Initialize on the main rank; distributed error handling lives in the trainer."""
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "disabled" or "wandb" not in config.get("trackers", ["wandb"]):
        return None

    output_dir = Path(config.output_dir)
    identity_path = output_dir / "wandb_run.json"
    previous = {}
    if resume_path:
        # A checkpoint may be resumed into a different output directory.
        source_identity = Path(resume_path).parent.parent / "wandb_run.json"
        if source_identity.is_file():
            previous = json.loads(source_identity.read_text())
            if not isinstance(previous, dict) or not previous.get("id") or not previous.get("project"):
                raise ValueError(f"Invalid W&B run identity in {source_identity}")

    run_id = os.environ.get("WANDB_RUN_ID") or previous.get("id") or wandb.util.generate_id()
    project = os.environ.get("WANDB_PROJECT") or previous.get("project") or config.get("wandb_project", "SPATIAL_JEPA")
    entity = os.environ.get("WANDB_ENTITY") or previous.get("entity") or config.get("wandb_entity")
    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "id": run_id,
        "name": config.run_id,
        "dir": str(wandb_dir),
        "project": project,
        "entity": entity,
        "group": "vla-train",
        "mode": mode,
        "config": OmegaConf.to_container(config, resolve=True),
        "allow_val_change": True,
    }
    if mode not in ("offline", "dryrun", "disabled"):
        options["resume"] = "allow" if resume_path else "never"
        options["force"] = True
    run = wandb.init(**options)
    # SDK history stays monotonic even when restarting from an older checkpoint.
    # The plots use the actual optimizer update count as their horizontal axis.
    run.define_metric("optimizer_step")
    run.define_metric("*", step_metric="optimizer_step")

    identity = {
        "id": run.id,
        "project": run.project or project,
        "entity": run.entity or entity,
    }
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_dir,
            prefix=".wandb_run.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(identity, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, identity_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return run
