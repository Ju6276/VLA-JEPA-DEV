"""Training run identity, resolved configuration, and rollback-safe W&B history."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from starVLA.training.trainer_utils import wandb_tracking


@pytest.fixture
def fake_sdk(monkeypatch):
    for name in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_RUN_ID", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)
    calls = []
    generated = []

    def generate_id():
        generated.append(True)
        return "new-random-id"

    def initialize(**options):
        metrics = []
        run = SimpleNamespace(
            id=options["id"], project=options["project"], entity=options["entity"],
            define_metric=lambda *args, **kwargs: metrics.append((args, kwargs)),
            metrics=metrics,
        )
        calls.append((options, run))
        return run

    monkeypatch.setattr(wandb_tracking.wandb.util, "generate_id", generate_id)
    monkeypatch.setattr(wandb_tracking.wandb, "init", initialize)
    return SimpleNamespace(calls=calls, generated=generated)


def config_at(output_dir):
    return OmegaConf.create({
        "output_dir": str(output_dir), "run_id": "spatial_training", "seed": 731,
        "trackers": ["wandb"], "wandb_project": "configured-project",
        "wandb_entity": "configured-team",
        "framework": {"name": "VLA_JEPA", "spatial_goal": {"enabled": True}},
        "datasets": {"vla_data": {"seed": "${seed}"}},
        "trainer": {"learning_rate": 3e-5, "gradient_accumulation_steps": 32},
    })


def write_identity(directory, **overrides):
    directory.mkdir(parents=True, exist_ok=True)
    identity = {"id": "previous-id", "project": "previous-project", "entity": "previous-team"}
    identity.update(overrides)
    (directory / "wandb_run.json").write_text(json.dumps(identity))
    return identity


def test_fresh_run_uses_new_id_complete_resolved_config_and_optimizer_axis(tmp_path, fake_sdk):
    config = config_at(tmp_path / "new-output")
    run = wandb_tracking.initialize_wandb(config)
    options, recorded_run = fake_sdk.calls[0]
    assert run is recorded_run
    assert fake_sdk.generated == [True]
    assert options["id"] == "new-random-id"
    assert options["project"] == "configured-project" and options["entity"] == "configured-team"
    assert options["name"] == "spatial_training" and options["resume"] == "never"
    assert options["mode"] == "online" and options["force"] is True
    assert options["config"] == OmegaConf.to_container(config, resolve=True)
    assert options["config"]["datasets"]["vla_data"]["seed"] == 731
    assert options["config"]["trainer"]["gradient_accumulation_steps"] == 32
    assert run.metrics == [(("optimizer_step",), {}), (("*",), {"step_metric": "optimizer_step"})]
    assert Path(options["dir"]).is_dir()
    assert json.loads((Path(config.output_dir) / "wandb_run.json").read_text()) == {
        "id": "new-random-id", "project": "configured-project", "entity": "configured-team",
    }
    assert not list(Path(config.output_dir).glob(".wandb_run.*.tmp"))


def test_fresh_run_does_not_accidentally_resume_stale_output_identity(tmp_path, fake_sdk):
    write_identity(tmp_path)
    wandb_tracking.initialize_wandb(config_at(tmp_path))
    options, _ = fake_sdk.calls[0]
    assert options["id"] == "new-random-id" and options["resume"] == "never"
    assert json.loads((tmp_path / "wandb_run.json").read_text())["id"] == "new-random-id"


@pytest.mark.parametrize("same_output", [False, True])
def test_resume_restores_identity_even_into_new_output_directory(tmp_path, fake_sdk, same_output):
    source = tmp_path / "original"
    output = source if same_output else tmp_path / "resumed"
    identity = write_identity(source)
    wandb_tracking.initialize_wandb(config_at(output), resume_path=source / "checkpoints" / "steps_100.pt")
    options, _ = fake_sdk.calls[0]
    assert {key: options[key] for key in identity} == identity
    assert options["resume"] == "allow" and not fake_sdk.generated
    assert json.loads((output / "wandb_run.json").read_text()) == identity


def test_source_checkpoint_identity_takes_priority_over_unrelated_output_run(tmp_path, fake_sdk):
    source, output = tmp_path / "original", tmp_path / "resumed"
    expected = write_identity(source)
    write_identity(output, id="unrelated-output-id")
    wandb_tracking.initialize_wandb(config_at(output), resume_path=source / "checkpoints" / "steps_100.pt")
    assert fake_sdk.calls[0][0]["id"] == expected["id"]


def test_legacy_checkpoint_without_metadata_starts_trackable_run(tmp_path, fake_sdk):
    config = config_at(tmp_path / "resumed")
    wandb_tracking.initialize_wandb(config, resume_path=tmp_path / "legacy" / "checkpoints" / "steps_100.pt")
    options, _ = fake_sdk.calls[0]
    assert options["id"] == "new-random-id" and options["resume"] == "allow"
    assert options["project"] == "configured-project"
    assert (Path(config.output_dir) / "wandb_run.json").is_file()


def test_legacy_checkpoint_never_adopts_unrelated_target_directory_identity(tmp_path, fake_sdk):
    output = tmp_path / "resumed"
    write_identity(output, id="unrelated-id", project="unrelated-project")
    wandb_tracking.initialize_wandb(config_at(output), resume_path=tmp_path / "legacy" / "checkpoints" / "steps_100.pt")
    options, _ = fake_sdk.calls[0]
    assert options["id"] == "new-random-id" and options["project"] == "configured-project"
    assert json.loads((output / "wandb_run.json").read_text())["id"] == "new-random-id"


@pytest.mark.parametrize("resuming", [False, True])
def test_explicit_wandb_environment_overrides_config_and_resumed_identity(tmp_path, fake_sdk, monkeypatch, resuming):
    write_identity(tmp_path)
    for name, value in {"WANDB_PROJECT": "environment-project", "WANDB_ENTITY": "environment-team",
                        "WANDB_RUN_ID": "explicit-run-id"}.items():
        monkeypatch.setenv(name, value)
    wandb_tracking.initialize_wandb(
        config_at(tmp_path), resume_path=tmp_path / "checkpoints" / "steps_100.pt" if resuming else None,
    )
    options, _ = fake_sdk.calls[0]
    assert (options["project"], options["entity"], options["id"]) == (
        "environment-project", "environment-team", "explicit-run-id",
    )
    assert not fake_sdk.generated


@pytest.mark.parametrize("mode", ["offline", "dryrun"])
@pytest.mark.parametrize("resuming", [False, True])
def test_offline_modes_do_not_request_online_resume(tmp_path, fake_sdk, monkeypatch, mode, resuming):
    monkeypatch.setenv("WANDB_MODE", mode)
    if resuming:
        write_identity(tmp_path)
    wandb_tracking.initialize_wandb(
        config_at(tmp_path), resume_path=tmp_path / "checkpoints" / "steps_100.pt" if resuming else None,
    )
    options, _ = fake_sdk.calls[0]
    assert options["mode"] == mode and "resume" not in options and "force" not in options
    assert options["id"] == ("previous-id" if resuming else "new-random-id")


@pytest.mark.parametrize("resuming", [False, True])
def test_disabled_mode_preserves_real_identity_without_starting_dummy_run(tmp_path, fake_sdk, monkeypatch, resuming):
    identity = write_identity(tmp_path)
    previous_bytes = (tmp_path / "wandb_run.json").read_bytes()
    monkeypatch.setenv("WANDB_MODE", "disabled")
    result = wandb_tracking.initialize_wandb(
        config_at(tmp_path), resume_path=tmp_path / "checkpoints" / "steps_100.pt" if resuming else None,
    )
    assert result is None and not fake_sdk.calls and not fake_sdk.generated
    assert (tmp_path / "wandb_run.json").read_bytes() == previous_bytes
    assert json.loads(previous_bytes) == identity
    assert not (tmp_path / "wandb").exists()


def test_unselected_tracker_has_no_sdk_or_filesystem_side_effect(tmp_path, fake_sdk):
    output = tmp_path / "unused"
    config = config_at(output)
    config.trackers = ["json"]
    assert wandb_tracking.initialize_wandb(config) is None
    assert not fake_sdk.calls and not fake_sdk.generated and not output.exists()


@pytest.mark.parametrize("contents", ["{truncated", '[]', '{"id":"old"}'])
def test_invalid_saved_identity_fails_before_sdk_initialization(tmp_path, fake_sdk, contents):
    (tmp_path / "wandb_run.json").write_text(contents)
    with pytest.raises((json.JSONDecodeError, ValueError)):
        wandb_tracking.initialize_wandb(config_at(tmp_path), resume_path=tmp_path / "checkpoints" / "steps_100.pt")
    assert not fake_sdk.calls


def test_atomic_identity_publication_failure_retains_previous_file_and_cleans_temporary(tmp_path, fake_sdk, monkeypatch):
    previous = write_identity(tmp_path)

    def fail_replace(source, destination):
        assert json.loads(Path(source).read_text())["id"] == "new-random-id"
        assert json.loads(Path(destination).read_text()) == previous
        raise OSError("simulated atomic publication failure")

    monkeypatch.setattr(wandb_tracking.os, "replace", fail_replace)
    with pytest.raises(OSError, match="publication failure"):
        wandb_tracking.initialize_wandb(config_at(tmp_path))
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == previous
    assert not list(tmp_path.glob(".wandb_run.*.tmp"))


def test_real_offline_sdk_keeps_both_history_rows_when_optimizer_step_rolls_back(tmp_path):
    """Exercise the installed SDK in isolation and inspect its saved protobuf log."""
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    script = r'''
from pathlib import Path
import sys
from omegaconf import OmegaConf
from starVLA.training.trainer_utils.wandb_tracking import initialize_wandb

output = Path(sys.argv[1])
config = OmegaConf.create({
    "output_dir": str(output), "run_id": "offline-rollback-test", "seed": 731,
    "trackers": ["wandb"], "wandb_project": "offline-spatial-tests",
    "framework": {"spatial_goal": {"enabled": True}},
    "datasets": {"seed": "${seed}"},
})
run = initialize_wandb(config)
run.log({"optimizer_step": 100, "train/loss": 1.25})
run.log({"optimizer_step": 50, "train/loss": 2.5})
run.finish()
'''
    env = {name: value for name, value in os.environ.items() if not name.startswith("WANDB_")}
    env.update({
        "WANDB_MODE": "offline", "WANDB_SILENT": "true", "WANDB_CONSOLE": "off",
        "WANDB_DISABLE_CODE": "true", "WANDB_DISABLE_GIT": "true",
        "WANDB_DISABLE_JOB_CREATION": "true", "WANDB_X_DISABLE_STATS": "true",
        "WANDB_CONFIG_DIR": str(tmp_path / "settings"), "WANDB_CACHE_DIR": str(tmp_path / "cache"),
    })
    output = tmp_path / "offline-output"
    result = subprocess.run(
        [sys.executable, "-c", script, str(output)], env=env,
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr[-6000:]
    # Read only after SDK process shutdown has flushed its offline datastore.
    # finish() alone can return before the last raw protobuf records are visible.
    logs = list(output.rglob("run-*.wandb"))
    assert len(logs) == 1, logs
    reader = DataStore()
    reader.open_for_scan(str(logs[0]))
    history, saved_config, metrics, run_ids = [], {}, [], []
    try:
        while True:
            data = reader.scan_data()
            if data is None:
                break
            record = Record()
            record.ParseFromString(data)
            kind = record.WhichOneof("record_type")
            if kind == "history":
                history.append({item.key or ".".join(item.nested_key): json.loads(item.value_json)
                                for item in record.history.item})
            elif kind == "run":
                run_ids.append(record.run.run_id)
                saved_config.update({item.key: json.loads(item.value_json) for item in record.run.config.update})
            elif kind == "config":
                saved_config.update({item.key: json.loads(item.value_json) for item in record.config.update})
            elif kind == "metric":
                metrics.append({"name": record.metric.name, "glob_name": record.metric.glob_name,
                                "step_metric": record.metric.step_metric})
    finally:
        reader.close()
    report = {
        "history": history, "config": saved_config, "metrics": metrics, "run_ids": run_ids,
        "identity": json.loads((output / "wandb_run.json").read_text()),
        "log_bytes": logs[0].stat().st_size,
    }
    (output / "inspection.json").write_text(json.dumps(report))
    rows = [row for row in report["history"] if "optimizer_step" in row]
    assert [row["optimizer_step"] for row in rows] == [100, 50]
    assert [row["train/loss"] for row in rows] == [1.25, 2.5]
    assert [row["_step"] for row in rows] == [0, 1]
    assert report["config"]["framework"] == {"spatial_goal": {"enabled": True}}
    assert report["config"]["datasets"] == {"seed": 731}
    assert report["identity"]["project"] == "offline-spatial-tests"
    assert report["identity"]["id"] in report["run_ids"] and report["log_bytes"] > 0
    assert any(metric["step_metric"] == "optimizer_step" for metric in report["metrics"])
