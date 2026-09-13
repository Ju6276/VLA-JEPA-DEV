"""CPU runtime regressions using real Accelerate state and small torch models."""

from copy import deepcopy
import os
import random
import socket

from accelerate import Accelerator
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from starVLA.training.trainer_utils.runtime import (
    TrainingProgress,
    action_error_metrics,
    data_iterator_at_progress,
    evaluating,
    load_training_checkpoint,
    resolve_resume_path,
    save_training_checkpoint,
)


def test_evaluation_disables_dropout_and_restores_mixed_modes_on_error():
    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.8), nn.Sequential(nn.Dropout(0.5)))
    model.train()
    model[2].eval()
    original = [part.training for part in model.modules()]
    x = torch.ones(3, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="prediction failed"):
        with evaluating(model):
            assert not any(part.training for part in model.modules())
            assert not torch.is_grad_enabled()
            output = model(x)
            assert not output.requires_grad
            assert torch.equal(output, model(x))
            raise RuntimeError("prediction failed")
    assert [part.training for part in model.modules()] == original
    assert torch.is_grad_enabled()


def test_action_diagnostic_computes_actual_mae_and_mse_with_bf16_output():
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(()))
            self.dropout = nn.Dropout(0.9)

        def predict_action(self, **kwargs):
            assert not self.training and not torch.is_grad_enabled()
            values = self.dropout(torch.tensor([[[1.0, 3.0]], [[2.0, 4.0]]]))
            return {"normalized_actions": values.to(torch.bfloat16)}

    accelerator = Accelerator(cpu=True)
    policy = accelerator.prepare(Policy())
    examples = [{"image": None, "lang": "pick", "action": np.zeros((1, 2))} for _ in range(2)]
    metrics = action_error_metrics(accelerator, policy, examples)
    assert metrics == {"mae_score": 2.5, "mse_score": 7.5}
    assert policy.training and policy.dropout.training
    assert len(examples) == 2


def test_full_checkpoint_restores_optimizer_scheduler_rng_and_next_update(tmp_path):
    accelerator = Accelerator(cpu=True)
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.3), nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    loader = DataLoader(TensorDataset(torch.arange(12)), batch_size=2)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    progress = TrainingProgress(completed_steps=0, batches_per_epoch=len(loader))
    accelerator.register_for_checkpointing(scheduler, progress)
    x, y = torch.ones(4, 3), torch.zeros(4, 2)

    def update():
        optimizer.zero_grad()
        loss = (model(x) - y).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        progress.completed_steps += 1
        return loss.detach().clone()

    update()
    update()
    progress.data_epoch, progress.batches_in_epoch = 3, 2
    optimizer_before = deepcopy(optimizer.state_dict())
    scheduler_before = deepcopy(scheduler.state_dict())
    checkpoint = tmp_path / "steps_2"
    save_training_checkpoint(accelerator, model, progress, checkpoint)
    expected_rng = (random.random(), np.random.random(), torch.rand(4))
    expected_loss = update()
    expected_weights = deepcopy(model.state_dict())

    for _ in range(3):
        update()
    progress.data_epoch = 99
    random.seed(111)
    np.random.seed(222)
    torch.manual_seed(333)
    load_training_checkpoint(accelerator, progress, checkpoint)

    assert progress.completed_steps == 2
    assert (progress.data_epoch, progress.batches_in_epoch) == (3, 2)
    assert scheduler.state_dict() == scheduler_before
    assert optimizer.param_groups[0]["lr"] == optimizer_before["param_groups"][0]["lr"]
    for key, state in optimizer_before["state"].items():
        for name, value in state.items():
            assert torch.equal(optimizer.state_dict()["state"][key][name], value)
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    assert torch.equal(torch.rand(4), expected_rng[2])
    assert torch.equal(update(), expected_loss)
    for name, value in expected_weights.items():
        assert torch.equal(model.state_dict()[name], value)
    export = torch.load(tmp_path / "steps_2_pytorch_model.pt", weights_only=True)
    assert export.keys() == expected_weights.keys()


def test_resume_epoch_and_batch_cursor_on_prepared_loader():
    accelerator = Accelerator(cpu=True)
    dataset = TensorDataset(torch.arange(24))
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, seed=31, shuffle=True)
    loader = accelerator.prepare(DataLoader(dataset, batch_size=3, sampler=sampler))
    loader.set_epoch(4)
    expected = list(loader)
    progress = TrainingProgress(data_epoch=4, batches_in_epoch=3, batches_per_epoch=len(loader))
    remaining = list(data_iterator_at_progress(accelerator, loader, progress))
    assert len(remaining) == len(expected) - 3
    for actual, target in zip(remaining, expected[3:]):
        assert torch.equal(actual[0], target[0])

    progress.batches_in_epoch = len(loader)
    next_epoch = list(data_iterator_at_progress(accelerator, loader, progress))
    assert progress.data_epoch == 5 and progress.batches_in_epoch == 0
    loader.set_epoch(5)
    assert torch.equal(next_epoch[0][0], next(iter(loader))[0])


def test_resume_config_separates_weights_from_training_state(tmp_path):
    assert resolve_resume_path(OmegaConf.create({"trainer": {"pretrained_checkpoint": "old.pt"}})) is None
    config = OmegaConf.create({"trainer": {"resume_from_checkpoint": str(tmp_path)}})
    assert resolve_resume_path(config) == str(tmp_path)
    config = OmegaConf.create({"trainer": {}, "resume_from_checkpoint": str(tmp_path)})
    assert resolve_resume_path(config) == str(tmp_path)
    with pytest.raises(ValueError, match="requires"):
        resolve_resume_path(OmegaConf.create({"trainer": {"is_resume": True}}))
    weights = tmp_path / "weights.pt"
    weights.touch()
    with pytest.raises(ValueError, match="weight-only"):
        resolve_resume_path(OmegaConf.create({"trainer": {"resume_from_checkpoint": str(weights)}}))


def test_reject_incomplete_checkpoint_and_changed_data_topology(tmp_path):
    accelerator = Accelerator(cpu=True)
    progress = TrainingProgress(batches_per_epoch=4)
    with pytest.raises(ValueError, match="missing trainer_state"):
        load_training_checkpoint(accelerator, progress, tmp_path)
    for name in ("batches_per_epoch", "world_size", "gradient_accumulation_steps"):
        state = progress.state_dict()
        state[name] += 1
        with pytest.raises(ValueError, match=f"changed {name}"):
            progress.load_state_dict(state)


def _distributed_resume_worker(rank, directory):
    """A real two-rank run catches misplaced main-process checkpoint guards."""
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank), LOCAL_WORLD_SIZE="2")
    dist.init_process_group("gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2)
    try:
        accelerator = Accelerator(cpu=True)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
        model, optimizer = accelerator.prepare(model, optimizer)
        progress = TrainingProgress(completed_steps=1, batches_per_epoch=5, world_size=2)
        accelerator.register_for_checkpointing(scheduler, progress)
        loss = model(torch.ones(2, 2)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        before = deepcopy(accelerator.unwrap_model(model).state_dict())
        random.seed(rank + 87)
        save_training_checkpoint(accelerator, model, progress, f"{directory}/steps_1")
        expected_random = random.random()
        with torch.no_grad():
            for param in model.parameters():
                param.add_(rank + 10)
        progress.completed_steps = 99
        load_training_checkpoint(accelerator, progress, f"{directory}/steps_1")
        assert progress.completed_steps == 1
        assert random.random() == expected_random
        for name, value in before.items():
            assert torch.equal(accelerator.unwrap_model(model).state_dict()[name], value)
        # Sum/count aggregation, including unequal local values.
        totals = accelerator.reduce(torch.tensor([rank + 1.0, 1.0]), reduction="sum")
        assert totals.tolist() == [3.0, 2.0]
    finally:
        dist.destroy_process_group()


def test_two_rank_checkpoint_roundtrip(tmp_path):
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("Local sockets are blocked by the sandbox; Gloo requires loopback networking")
    mp.spawn(_distributed_resume_worker, args=(str(tmp_path),), nprocs=2, join=True)
