"""Real Gloo communication for the production trainer's W&B initialization.

Only the external W&B initializer is replaced. No SDK login, credentials,
network upload, model construction, or CUDA initialization is needed.
"""

import ast
from datetime import timedelta
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _production_wandb_init(initializer):
    """Compile the unchanged production method without entrypoint side effects."""
    path = Path(__file__).resolve().parents[1] / "starVLA/training/train_starvla.py"
    tree = ast.parse(path.read_text())
    trainer = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VLATrainer"
    )
    method = next(
        node for node in trainer.body
        if isinstance(node, ast.FunctionDef) and node.name == "_init_wandb"
    )
    namespace = {"dist": dist, "initialize_wandb": initializer}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_init_wandb"]


def _wandb_initialization_worker(rank, directory, fail_initialization):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        calls = []
        config = SimpleNamespace(run_id="distributed_wandb_test")
        resume_path = f"{directory}/checkpoints/steps_10000"
        initialized_run = object()

        def initialize_wandb(actual_config, *, resume_path):
            assert actual_config is config
            calls.append({"rank": rank, "resume_path": resume_path})
            if fail_initialization:
                raise PermissionError("simulated W&B authentication failure")
            return initialized_run

        trainer = SimpleNamespace(
            accelerator=SimpleNamespace(is_main_process=rank == 0),
            config=config,
            resume_path=resume_path,
        )
        initialize = _production_wandb_init(initialize_wandb)
        training_entered = False
        error = None
        try:
            initialize(trainer)
            # The next collective represents entering distributed training.
            # An initialization failure must prevent both ranks reaching it.
            training_entered = True
            peers = torch.ones((), dtype=torch.int64)
            dist.all_reduce(peers)
            assert peers.item() == 2
        except RuntimeError as exception:
            error = str(exception)

        result = {
            "rank": rank,
            "calls": calls,
            "training_entered": training_entered,
            "error": error,
            "has_initialized_run": trainer.wandb_run is initialized_run,
            "run_is_none": trainer.wandb_run is None,
        }
        (Path(directory) / f"rank_{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fail_initialization", [False, True], ids=["success", "auth_failure"])
def test_two_rank_wandb_initialization(tmp_path, fail_initialization):
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local sockets are blocked by the sandbox; Gloo requires loopback networking")

    mp.spawn(
        _wandb_initialization_worker,
        args=(str(tmp_path), fail_initialization),
        nprocs=2,
        join=True,
    )
    results = [json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(2)]
    assert results[0]["calls"] == [{
        "rank": 0,
        "resume_path": str(tmp_path / "checkpoints/steps_10000"),
    }]
    assert results[1]["calls"] == []
    assert results[1]["run_is_none"]

    if fail_initialization:
        expected = "W&B initialization failed: PermissionError: simulated W&B authentication failure"
        assert [result["error"] for result in results] == [expected, expected]
        assert not any(result["training_entered"] for result in results)
        assert all(result["run_is_none"] for result in results)
    else:
        assert [result["error"] for result in results] == [None, None]
        assert all(result["training_entered"] for result in results)
        assert results[0]["has_initialized_run"]
