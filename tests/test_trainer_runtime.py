"""CPU runtime regressions using real Accelerate state and small torch models."""

from copy import deepcopy
import ast
import os
from pathlib import Path
import random
import socket
from types import SimpleNamespace

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed
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
    configure_training_accelerator,
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


def _trainer_function(name, namespace, *, class_name=None):
    """Run a production function without importing the CUDA entrypoint globals."""
    path = Path(__file__).resolve().parents[1] / "starVLA/training/train_starvla.py"
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name:
        body = next(node.body for node in body if isinstance(node, ast.ClassDef) and node.name == class_name)
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_model_initialization_is_controlled_by_configured_seed():
    config = OmegaConf.create({"seed": 3047, "framework": {"qwenvl": {"base_vlm": "tiny"}}})
    build = _trainer_function("build_model", {
        "torch": torch,
        "set_seed": set_seed,
        "logger": SimpleNamespace(info=lambda *args: None),
        "build_framework": lambda config: nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 1)),
    })
    torch.manual_seed(7)
    first = build(config).state_dict()
    torch.manual_seed(8)
    second = build(config).state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
    config.seed += 1
    third = build(config).state_dict()
    assert any(not torch.equal(first[key], third[key]) for key in first)


def test_configure_actual_accelerate_and_deepspeed_accumulation_and_clipping():
    accelerator = Accelerator(cpu=True)
    ds_config = Path(__file__).resolve().parents[1] / "starVLA/config/deepseeds/ds_config.yaml"
    plugin = DeepSpeedPlugin(hf_ds_config=str(ds_config))
    config = OmegaConf.create({"gradient_accumulation_steps": 2, "gradient_clipping": 0.5})
    configure_training_accelerator(accelerator, config, deepspeed_plugin=plugin)
    assert accelerator.gradient_accumulation_steps == 2
    assert not accelerator.gradient_state.sync_with_dataloader
    assert accelerator.gradient_state.plugin_kwargs["sync_each_batch"]
    # These are the actual config processing operations used by Accelerate.prepare.
    plugin.fill_match("gradient_accumulation_steps", must_match=False,
                      gradient_accumulation_steps=accelerator.gradient_accumulation_steps)
    plugin.deepspeed_config_process(
        must_match=False, gradient_clipping=1.0,
        train_micro_batch_size_per_gpu=1, train_batch_size=2,
    )
    assert plugin.deepspeed_config["gradient_accumulation_steps"] == 2
    assert plugin.deepspeed_config["gradient_clipping"] == 0.5
    config.gradient_clipping = None
    configure_training_accelerator(accelerator, config, deepspeed_plugin=plugin)
    assert plugin.deepspeed_config["gradient_clipping"] == 0.0


@pytest.mark.parametrize("values", [
    {"gradient_accumulation_steps": 0},
    {"gradient_accumulation_steps": 1.5},
    {"gradient_clipping": -1},
    {"gradient_clipping": float("nan")},
])
def test_invalid_accumulation_or_clipping_is_rejected(values):
    with pytest.raises(ValueError, match="trainer\\."):
        configure_training_accelerator(Accelerator(cpu=True), OmegaConf.create(values))


def test_accumulated_updates_cross_epoch_and_resume_without_scheduler_drift(tmp_path):
    class LossModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1, bias=False)
            nn.init.zeros_(self.linear.weight)

        def forward(self, batch):
            return {"loss": (self.linear(batch[0]) - 1).square().mean()}

    accelerator = Accelerator(cpu=True)
    trainer_config = OmegaConf.create({"gradient_accumulation_steps": 2, "gradient_clipping": None})
    configure_training_accelerator(accelerator, trainer_config)
    model = LossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loader = DataLoader(TensorDataset(torch.tensor([[1.], [2.], [3.]])), batch_size=1)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    progress = TrainingProgress(batches_per_epoch=len(loader), gradient_accumulation_steps=2)
    accelerator.register_for_checkpointing(scheduler, progress)
    trainer = SimpleNamespace(
        accelerator=accelerator, model=model, optimizer=optimizer, lr_scheduler=scheduler,
        config=SimpleNamespace(trainer=trainer_config),
    )
    train_step = _trainer_function("_train_step", {"torch": torch}, class_name="VLATrainer")

    def run_batches(count):
        iterator = data_iterator_at_progress(accelerator, loader, progress)
        updates = []
        for _ in range(count):
            try:
                batch = next(iterator)
            except StopIteration:
                progress.data_epoch += 1
                progress.batches_in_epoch = 0
                iterator = data_iterator_at_progress(accelerator, loader, progress)
                batch = next(iterator)
            progress.batches_in_epoch += 1
            train_step(trainer, batch)
            did_update = accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped
            progress.completed_steps += int(did_update)
            updates.append(did_update)
        return updates

    assert run_batches(2) == [False, True]
    checkpoint = tmp_path / "steps_1"
    save_training_checkpoint(accelerator, model, progress, checkpoint)
    assert run_batches(4) == [False, True, False, True]
    expected_weights = deepcopy(model.state_dict())
    expected_scheduler = deepcopy(scheduler.state_dict())
    assert progress.completed_steps == 3 and scheduler.last_epoch == 3

    load_training_checkpoint(accelerator, progress, checkpoint)
    assert run_batches(4) == [False, True, False, True]
    assert scheduler.state_dict() == expected_scheduler
    assert progress.completed_steps == 3
    for key, value in expected_weights.items():
        assert torch.equal(model.state_dict()[key], value)

    reference = LossModel()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.9)
    for pair in ([1., 2.], [3., 1.], [2., 3.]):
        reference_optimizer.zero_grad()
        reference((torch.tensor(pair).reshape(2, 1),))["loss"].backward()
        reference_optimizer.step()
        reference_scheduler.step()
    assert torch.allclose(model.linear.weight, reference.linear.weight, atol=1e-7)


def test_invalid_learning_rate_module_path_does_not_silently_use_base_rate():
    from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
    config = OmegaConf.create({"trainer": {"learning_rate": {"base": 1e-4, "misspelled_head": 1e-3}}})
    with pytest.raises(ValueError, match="misspelled_head"):
        build_param_lr_groups(nn.Linear(2, 2), config)
