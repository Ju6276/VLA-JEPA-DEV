"""Train/deploy checks for spatial goals using frozen, spatially distinct features."""

import copy
from types import MethodType

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from test_learned_goal import examples, inference_inputs, make_model


def spatial_config(**overrides):
    return {
        "enabled": True, "memory_enabled": True, "grid_size": 2,
        "hidden_dim": 16, "num_heads": 2, "num_queries": 2,
        "history_offsets_seconds": [-0.8, -0.4],
        "history_tolerance_seconds": 0.15,
        "memory_max_gap_seconds": 2.0, **overrides,
    }


def patch_spatial_encoder(model, monkeypatch):
    def encode(self, pixel_values_videos):
        # Mix all temporal frames, but preserve four different spatial cells.
        # This detects future leakage and makes reader gradients meaningful.
        rgb = F.adaptive_avg_pool2d(pixel_values_videos.mean(1), (2, 2))
        rgb = rgb.flatten(2).transpose(1, 2) / 255
        return self.proj(rgb).repeat(1, pixel_values_videos.shape[1] // 2, 1)

    monkeypatch.setattr(model.vj_encoder, "get_vision_features", MethodType(encode, model.vj_encoder))


@pytest.fixture
def spatial_examples(examples):
    result = copy.deepcopy(examples)
    for index, sample in enumerate(result):
        current = np.zeros((8, 8, 3), dtype=np.uint8)
        current[:4, :4] = [200, 30, 10]
        current[:4, 4:] = [10, 180, 70]
        current[4:, :4] = [25, 50, 210]
        current[4:, 4:] = [130, 100, 20 + index * 80]
        sample["image"] = [Image.fromarray(current)]
        sample["jepa_image"] = [sample["image"][0].resize((32, 32))]
        sample["video"][0, 0] = current
        sample["video"][0, 1:] = np.roll(current, 3, axis=1)
        sample["history_images"] = [
            Image.fromarray(np.roll(current, 1, axis=0)),
            Image.fromarray(np.roll(current, 2, axis=0)),
        ]
        sample["history_valid"] = np.array([index == 1, True])
        sample["history_ages"] = np.array([0.8 if index == 1 else 0.0, 0.4], dtype=np.float32)
    return result


def diagnostic_inputs(samples):
    return {
        **inference_inputs(samples),
        "batch_images": [sample["jepa_image"] for sample in samples],
        "history_images": [sample["history_images"] for sample in samples],
        "history_valid": [sample["history_valid"] for sample in samples],
        "history_ages": [sample["history_ages"] for sample in samples],
        "update_memory": False,
    }


def assert_memory_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in actual.keys() - {"entries"}:
        assert actual[key] == expected[key]
    if "entries" in actual:
        assert len(actual["entries"]) == len(expected["entries"])
        for (actual_time, actual_grid), (expected_time, expected_grid) in zip(actual["entries"], expected["entries"]):
            assert actual_time == expected_time
            torch.testing.assert_close(actual_grid, expected_grid)


def test_action_supervision_trains_reader_and_adapter_but_not_detached_goal(make_model, monkeypatch, spatial_examples):
    torch.manual_seed(10)
    model = make_model(spatial=spatial_config())
    patch_spatial_encoder(model, monkeypatch)
    model.lambda_delta = model.lambda_ctrl = 0.0
    losses = model(spatial_examples)
    assert set(losses) == {
        "action_loss", "wm_loss", "delta_loss", "ctrl_loss", "action_prior_loss",
        "goal_proposal_loss", "goal_prediction_loss",
    }
    assert losses["delta_loss"].item() == losses["ctrl_loss"].item() == 0
    assert all(torch.isfinite(loss).all() for loss in losses.values())
    torch.testing.assert_close(losses["goal_prediction_loss"].detach(), sum(model.spatial_training_metrics.values()))

    reader_gradients = torch.autograd.grad(
        losses["goal_proposal_loss"], tuple(model.spatial_reader.parameters()),
        retain_graph=True, allow_unused=True,
    )
    assert sum(gradient.abs().sum() for gradient in reader_gradients if gradient is not None) > 1e-8
    goal_gradients = torch.autograd.grad(
        losses["goal_proposal_loss"], tuple(model.spatial_goal_predictor.parameters()),
        retain_graph=True, allow_unused=True,
    )
    assert all(gradient is None for gradient in goal_gradients)

    sum(losses.values()).backward()
    for module in (model.spatial_reader, model.spatial_action_adapter, model.spatial_goal_predictor):
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in module.parameters())
    assert all(parameter.grad is None for parameter in model.vj_encoder.parameters())
    assert model.spatial_memory == {}


def test_future_changes_targets_without_changing_current_or_spatial_goal(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    predictions = []
    hook = model.spatial_goal_predictor.register_forward_hook(lambda module, args, output: predictions.append(output.detach().clone()))
    first = model(spatial_examples)
    first_world_input = model.vj_predictor.inputs[-1].clone()
    changed = copy.deepcopy(spatial_examples)
    for sample in changed:
        sample["video"][:, 1:] = 255 - sample["video"][:, 1:]
    second = model(changed)
    hook.remove()

    torch.testing.assert_close(first_world_input, model.vj_predictor.inputs[-1])
    torch.testing.assert_close(predictions[0], predictions[1])
    assert not torch.allclose(first["goal_prediction_loss"], second["goal_prediction_loss"])
    assert not torch.allclose(first["wm_loss"], second["wm_loss"])


def test_history_dropout_trains_current_only_condition_without_changing_targets(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config(history_dropout=1.0)).train()
    patch_spatial_encoder(model, monkeypatch)
    predictions = []
    hook = model.spatial_goal_predictor.register_forward_hook(
        lambda module, args, output: predictions.append(output.detach().clone()))
    first = model(spatial_examples)
    changed = copy.deepcopy(spatial_examples)
    for sample in changed:
        sample["history_images"] = [Image.new("RGB", (32, 32), "white")] * 2
    second = model(changed)
    hook.remove()
    torch.testing.assert_close(predictions[0], predictions[1])
    torch.testing.assert_close(first["goal_prediction_loss"], second["goal_prediction_loss"])


def test_training_diagnostic_uses_explicit_history_without_touching_live_memory(make_model, monkeypatch, spatial_examples):
    from accelerate import Accelerator
    from starVLA.training.trainer_utils.runtime import action_error_metrics
    model = make_model(spatial=spatial_config()).train()
    patch_spatial_encoder(model, monkeypatch)
    model.spatial_memory["sentinel"] = "live connection"
    metrics = action_error_metrics(Accelerator(cpu=True), model, spatial_examples)
    assert np.isfinite(list(metrics.values())).all()
    assert set(metrics) == {"mae_score", "mse_score"}
    assert model.training and not model.vj_encoder.training
    assert model.spatial_memory == {"sentinel": "live connection"}


def test_batched_diagnostics_match_training_history_and_keep_jepa_resolution(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    predictions, residuals, encoded_sizes = [], [], []
    goal_hook = model.spatial_goal_predictor.register_forward_hook(lambda module, args, output: predictions.append(output.detach().clone()))
    action_hook = model.spatial_action_adapter.register_forward_hook(lambda module, args, output: residuals.append(output.detach().clone()))
    processor = model.vj_processor

    def record_processor(*, videos, **kwargs):
        encoded_sizes.extend(video.shape[-2:] for video in videos)
        return processor(videos=videos, **kwargs)

    monkeypatch.setattr(model, "vj_processor", record_processor)
    model.spatial_memory.update({"sentinel": "unrelated-live-session"})
    model(spatial_examples)
    train_residual = residuals[-1]
    result = model.predict_action(**diagnostic_inputs(spatial_examples), num_candidates=2)
    goal_hook.remove()
    action_hook.remove()

    torch.testing.assert_close(predictions[0], predictions[1])
    torch.testing.assert_close(train_residual, residuals[-1])
    assert encoded_sizes and set(encoded_sizes) == {(32, 32)}
    assert result["spatial_history_used"].tolist() == [1, 2]
    assert model.spatial_memory == {"sentinel": "unrelated-live-session"}


def test_candidate_scores_share_query_and_select_matching_actions(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    query_ids = []
    hook = model.spatial_reader.register_forward_pre_hook(lambda module, args: query_ids.append(id(args[1])))
    result = model.predict_action(**diagnostic_inputs(spatial_examples), num_candidates=3)
    hook.remove()

    assert len(query_ids) == 5  # Current, goal, and all three candidate futures.
    assert len(set(query_ids)) == 1
    expected = result["candidate_goal_progress"] - model.verifier_action_prior_weight * result["candidate_prior_error"]
    expected += model.spatial_score_weight * result["candidate_spatial_progress"]
    np.testing.assert_allclose(result["candidate_scores"], expected, atol=1e-6)
    selected = expected.argmax(axis=1)
    np.testing.assert_allclose(result["normalized_actions"], result["all_candidates"][np.arange(2), selected])
    np.testing.assert_allclose(result["verification_scores"], expected[np.arange(2), selected])
    np.testing.assert_allclose(result["spatial_attention"].sum(-1), 1, atol=1e-6)


def test_online_memory_stores_only_actual_observations(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = inference_inputs(spatial_examples[:1])
    for timestamp, expected_history in [(1.0, 0), (1.4, 1), (1.8, 2)]:
        result = model.predict_action(**payload, timestamp=timestamp, episode_id="trial", num_candidates=3)
        assert result["spatial_history_used"].tolist() == [expected_history]
    actual = model._current_spatial_grid(model._encode_video_batch(model._images_to_video_batch(model._spatial_images(payload["batch_images"]))))
    assert len(model.spatial_memory["entries"]) == 3
    for timestamp, cached in model.spatial_memory["entries"]:
        assert timestamp in (1.0, 1.4, 1.8)
        torch.testing.assert_close(cached, actual)
        assert not cached.requires_grad


@pytest.mark.parametrize("change", ["instruction", "episode", "rollback", "gap", "clock_source", "reset"])
def test_online_memory_resets_on_trajectory_boundaries(make_model, monkeypatch, spatial_examples, change):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = inference_inputs(spatial_examples[:1])
    model.predict_action(**payload, timestamp=1.0, episode_id="trial", num_candidates=1)
    model.predict_action(**payload, timestamp=1.4, episode_id="trial", num_candidates=1)
    options = {"timestamp": 1.8, "episode_id": "trial", "num_candidates": 1}
    if change == "instruction":
        payload["instructions"] = ["carry the cup"]
    elif change == "episode":
        options["episode_id"] = "next-trial"
    elif change == "rollback":
        options["timestamp"] = 1.4
    elif change == "gap":
        options["timestamp"] = 5.0
    elif change == "clock_source":
        options.pop("timestamp")
    else:
        options["reset_subgoals"] = True
    result = model.predict_action(**payload, **options)
    assert result["spatial_history_used"].tolist() == [0]
    assert len(model.spatial_memory["entries"]) == 1


def test_failed_rollout_does_not_commit_pending_memory(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = inference_inputs(spatial_examples[:1])
    model.predict_action(**payload, timestamp=1.0, episode_id="trial", num_candidates=1)
    before = copy.deepcopy(model.spatial_memory)

    def fail(*args, **kwargs):
        raise ValueError("candidate readout failed")

    monkeypatch.setattr(model, "_spatial_progress", fail)
    payload["instructions"] = ["new instruction"]
    with pytest.raises(ValueError, match="candidate readout failed"):
        model.predict_action(**payload, timestamp=1.4, episode_id="new-trial", num_candidates=1)
    assert_memory_equal(model.spatial_memory, before)


def test_nonfinite_actions_cannot_advance_spatial_memory(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = inference_inputs(spatial_examples[:1])
    model.predict_action(**payload, timestamp=1.0, episode_id="trial", num_candidates=1)
    before = copy.deepcopy(model.spatial_memory)
    original = model.goal_action_proposal.forward
    monkeypatch.setattr(model.goal_action_proposal, "forward", lambda *args: original(*args) * float("nan"))
    with pytest.raises(ValueError, match="finite"):
        model.predict_action(**payload, timestamp=1.4, episode_id="trial", num_candidates=1)
    assert_memory_equal(model.spatial_memory, before)


def test_explicit_history_cannot_enable_disabled_memory(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config(memory_enabled=False)).eval()
    patch_spatial_encoder(model, monkeypatch)
    with pytest.raises(ValueError, match="memory"):
        model.predict_action(**diagnostic_inputs(spatial_examples), num_candidates=1)


def test_explicit_history_requires_nonmutating_diagnostics(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = diagnostic_inputs(spatial_examples)
    payload["update_memory"] = True
    with pytest.raises(ValueError, match="update_memory=False"):
        model.predict_action(**payload, num_candidates=1)
    assert model.spatial_memory == {}


def test_wire_history_images_reach_model_without_request_mutation(make_model, monkeypatch, spatial_examples):
    from deployment.model_server.tools import msgpack_numpy
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    payload = diagnostic_inputs(spatial_examples[:1])
    payload["batch_images"] = [[np.asarray(image) for image in sample] for sample in payload["batch_images"]]
    payload["history_images"] = [[np.asarray(image) for image in sample] for sample in payload["history_images"]]
    payload["num_candidates"] = 1
    request = msgpack_numpy.unpackb(msgpack_numpy.packb({"payload": payload}))
    response = WebsocketPolicyServer(model)._route_message(request)

    assert response["ok"] is True
    assert response["data"]["spatial_history_used"].tolist() == [1]
    assert isinstance(request["payload"]["history_images"][0][0], np.ndarray)
    assert model.spatial_memory == {}


def test_external_goal_uses_same_frozen_spatial_target(make_model, monkeypatch, spatial_examples):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    targets = [Image.fromarray(sample["video"][0, -1]) for sample in spatial_examples]
    read_grids = []
    hook = model.spatial_reader.register_forward_pre_hook(lambda module, args: read_grids.append(args[0].detach().clone()))

    def unexpected(*args, **kwargs):
        raise AssertionError("Explicit goal must bypass learned spatial goal prediction")

    monkeypatch.setattr(model.spatial_goal_predictor, "forward", unexpected)
    result = model.predict_action(**inference_inputs(spatial_examples), subgoal_images=targets, update_memory=False, num_candidates=1)
    hook.remove()
    assert result["goal_source"] == "images"
    torch.testing.assert_close(read_grids[1], model._encode_spatial_goal_images(targets))


def test_spatial_checkpoint_roundtrip_excludes_live_memory(make_model, monkeypatch, spatial_examples, tmp_path):
    model = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(model, monkeypatch)
    model.predict_action(**inference_inputs(spatial_examples[:1]), timestamp=1.0, num_candidates=1)
    path = tmp_path / "spatial.pt"
    torch.save(model.state_dict(), path)
    state = torch.load(path, weights_only=True)
    assert not any(key.startswith("spatial_memory") for key in state)
    restored = make_model(spatial=spatial_config()).eval()
    patch_spatial_encoder(restored, monkeypatch)
    restored.load_state_dict(state, strict=True)
    assert restored.spatial_memory == {}
    first = model.predict_action(**diagnostic_inputs(spatial_examples), num_candidates=2)
    second = restored.predict_action(**diagnostic_inputs(spatial_examples), num_candidates=2)
    np.testing.assert_allclose(first["normalized_actions"], second["normalized_actions"], atol=1e-6)
    np.testing.assert_allclose(first["candidate_scores"], second["candidate_scores"], atol=1e-6)


def test_disabled_spatial_branch_keeps_legacy_checkpoint_and_predictions(make_model, spatial_examples):
    torch.manual_seed(42)
    original = make_model().eval()
    torch.manual_seed(42)
    disabled = make_model(spatial={"enabled": False}).eval()
    disabled.load_state_dict(original.state_dict(), strict=True)
    assert list(original.state_dict()) == list(disabled.state_dict())
    first = original.predict_action(**inference_inputs(spatial_examples), num_candidates=2)
    second = disabled.predict_action(**inference_inputs(spatial_examples), num_candidates=2)
    np.testing.assert_array_equal(first["normalized_actions"], second["normalized_actions"])
    np.testing.assert_array_equal(first["candidate_scores"], second["candidate_scores"])
