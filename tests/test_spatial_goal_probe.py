"""Independent checks of frozen-feature spatial-head evaluation."""

import sys

import pytest
import torch

from scripts import probe_spatial_goal as probe


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_split(prefix, count=4, *, seed=1):
    generator = torch.Generator().manual_seed(seed)
    current = torch.randn(count, 4, 8, generator=generator)
    return {
        "current": current,
        "target": current + 0.3,
        "task": torch.randn(count, 2, 6, generator=generator),
        "state": torch.randn(count, 3, generator=generator),
        "history": torch.randn(count, 2, 4, 8, generator=generator),
        "valid": torch.ones(count, 2, dtype=torch.bool),
        "ages": torch.tensor([[0.8, 0.4]]).repeat(count, 1),
        "episode_ids": [f"{prefix}-episode-{i // 2}" for i in range(count)],
        "sample_ids": [f"{prefix}-sample-{i}" for i in range(count)],
    }


def make_cache():
    return {
        "schema_version": 1,
        "model_kwargs": dict(latent_dim=8, task_dim=6, state_dim=3, grid_size=2,
                             hidden_dim=16, num_heads=2),
        "splits": {name: make_split(name, seed=seed) for seed, name in enumerate(("train", "val", "test"))},
    }


def test_valid_episode_disjoint_cache_and_index_alignment():
    cache = make_cache()
    assert probe.validate_cache(cache) is cache
    split = cache["splits"]["train"]
    selected = probe.index_split(split, torch.tensor([3, 1]))
    assert selected["sample_ids"] == [split["sample_ids"][3], split["sample_ids"][1]]
    assert selected["episode_ids"] == [split["episode_ids"][3], split["episode_ids"][1]]
    for field in probe.TENSOR_FIELDS:
        torch.testing.assert_close(selected[field], split[field][[3, 1]])


@pytest.mark.parametrize("overlap", ["episode", "cross_split_sample", "within_split_sample"])
def test_cache_rejects_train_val_test_leakage(overlap):
    cache = make_cache()
    train, val, test = (cache["splits"][name] for name in ("train", "val", "test"))
    if overlap == "episode":
        val["episode_ids"][0] = train["episode_ids"][0]
    elif overlap == "cross_split_sample":
        test["sample_ids"][0] = val["sample_ids"][0]
    else:
        train["sample_ids"][1] = train["sample_ids"][0]
    with pytest.raises(ValueError, match="duplicate samples|overlapping episodes"):
        probe.validate_cache(cache)


@pytest.mark.parametrize("invalid", ["mask_dtype", "mask_shape", "future_history", "nonfinite", "empty_task"])
def test_cache_rejects_invalid_features_and_history(invalid):
    cache = make_cache()
    split = cache["splits"]["train"]
    if invalid == "mask_dtype":
        split["valid"] = split["valid"].float()
    elif invalid == "mask_shape":
        split["valid"] = split["valid"][:, :1]
    elif invalid == "future_history":
        split["ages"][0, 0] = -0.1
    elif invalid == "nonfinite":
        split["target"][0, 0, 0] = float("nan")
    else:
        split["task"] = split["task"][:, :0]
    with pytest.raises(ValueError):
        probe.validate_cache(cache)


def test_median_delta_baseline_uses_only_training_pairs():
    coordinate = torch.arange(8).reshape(1, 4, 2).float()
    train = {"current": torch.ones(3, 4, 2),
             "target": torch.ones(3, 4, 2) + coordinate + torch.tensor([1.0, 3.0, 100.0])[:, None, None]}
    heldout = {"current": torch.full((2, 4, 2), 10.0), "target": torch.zeros(2, 4, 2)}
    result = probe.baseline_predictions(train, heldout)
    torch.testing.assert_close(result["copy_current"], heldout["current"])
    torch.testing.assert_close(result["training_median_delta"], heldout["current"] + coordinate + 3.0)
    heldout["target"].fill_(10000)
    for name, values in probe.baseline_predictions(train, heldout).items():
        torch.testing.assert_close(values, result[name])


def test_metric_foreground_mask_is_shared_and_episode_means_are_equal_weight():
    current = torch.zeros(3, 4, 2)
    target = current.clone()
    target[:, 0] = torch.tensor([4.0, 8.0, 12.0])[:, None]
    episodes = ["short-episode-a", "short-episode-a", "short-episode-b"]
    metrics = probe.prediction_metrics(current, target, current, episodes)
    assert metrics["sample_mean"]["l1"] == pytest.approx(2.0)
    assert metrics["episode_mean"]["l1"] == pytest.approx(2.25)
    assert metrics["sample_mean"]["top_change_l1"] == pytest.approx(8.0)
    assert metrics["episode_mean"]["top_change_l1"] == pytest.approx(9.0)
    assert metrics["per_episode"]["short-episode-a"]["samples"] == 2
    assert metrics["per_episode"]["short-episode-b"]["samples"] == 1
    # A predictor's own huge changes in a static patch must not alter the
    # evaluation region: every predictor is judged on the same GT-change mask.
    wrong_static = current.clone()
    wrong_static[:, 3] = 100
    second = probe.prediction_metrics(wrong_static, target, current, episodes)
    assert second["episode_mean"]["top_change_l1"] == metrics["episode_mean"]["top_change_l1"]
    assert second["episode_mean"]["l1"] > metrics["episode_mean"]["l1"]
    perfect = probe.prediction_metrics(target, target, current, episodes)
    assert perfect["sample_mean"]["l1"] == 0
    assert perfect["sample_mean"]["top_change_l1"] == 0


def test_metrics_validate_episode_alignment_and_finite_values():
    tensor = torch.ones(3, 4, 2)
    with pytest.raises(ValueError):
        probe.prediction_metrics(tensor, tensor, tensor, ["only-one"])
    bad = tensor.clone()
    bad[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        probe.prediction_metrics(bad, tensor, tensor, ["a", "a", "b"])


def test_real_tiny_cpu_fit_reduces_loss_and_restores_validation_selected_weights():
    cache = make_cache()
    train = cache["splits"]["train"]
    # Overfit check intentionally uses the same tiny examples for monitoring;
    # the separate held-out experiment uses disjoint cached episodes.
    model, fit = probe.train_head(cache["model_kwargs"], train, train, torch.device("cpu"),
                                 steps=60, batch_size=4, learning_rate=0.01,
                                 history_dropout=0.0, eval_every=10)
    curve = fit["curve"]
    assert min(row["train_l1"] for row in curve[1:]) < curve[0]["train_l1"] * 0.5
    assert fit["best_validation_l1"] == min(row["validation_l1"] for row in curve)
    result = probe.prediction_metrics(probe.predict(model, train, torch.device("cpu")),
                                      train["target"], train["current"], train["episode_ids"])
    assert result["episode_mean"]["l1"] == pytest.approx(fit["best_validation_l1"], abs=1e-6)


def test_nonfinal_validation_optimum_restores_earlier_head(monkeypatch):
    cache = make_cache()
    train, val = cache["splits"]["train"], cache["splits"]["val"]
    real_metrics = probe.prediction_metrics
    validation_predictions = []
    prescribed = [3.0, 1.0, 4.0]

    def controlled_validation(predicted, target, current, episodes):
        result = real_metrics(predicted, target, current, episodes)
        if episodes[0].startswith("val-"):
            validation_predictions.append(predicted.clone())
            result["episode_mean"]["l1"] = prescribed[len(validation_predictions) - 1]
        return result

    monkeypatch.setattr(probe, "prediction_metrics", controlled_validation)
    model, fit = probe.train_head(cache["model_kwargs"], train, val, torch.device("cpu"),
                                 steps=4, batch_size=4, learning_rate=0.01,
                                 history_dropout=0.0, eval_every=2)
    assert fit["best_step"] == 2 and fit["best_validation_l1"] == 1.0
    final_prediction = probe.predict(model, val, torch.device("cpu"), batch_size=4)
    torch.testing.assert_close(final_prediction, validation_predictions[1])
    assert not torch.allclose(final_prediction, validation_predictions[-1])


def test_no_history_intervention_masks_only_history():
    split = make_split("test")
    calls = []

    class Capture(torch.nn.Module):
        def forward(self, current, task, state, history, valid, ages):
            calls.append((current.clone(), task.clone(), state.clone(), history.clone(), valid.clone(), ages.clone()))
            return current

    model = Capture()
    indices = torch.arange(4)
    probe.model_forward(model, split, indices, torch.device("cpu"))
    probe.model_forward(model, split, indices, torch.device("cpu"), intervention="no_history")
    for index in (0, 1, 2, 3, 5):
        torch.testing.assert_close(calls[0][index], calls[1][index])
    assert calls[0][4].all() and not calls[1][4].any()
    assert split["valid"].all()


@pytest.mark.parametrize("option,value", [
    ("--steps", "0"), ("--overfit-steps", "-1"), ("--overfit-samples", "0"),
    ("--batch-size", "0"), ("--eval-every", "0"), ("--cpu-threads", "0"),
    ("--learning-rate", "nan"), ("--learning-rate", "-0.1"),
    ("--history-dropout", "1.1"), ("--history-dropout", "nan"),
])
def test_command_rejects_invalid_experiment_parameters(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["probe_spatial_goal.py", "--features", "unused.pt",
                                      "--output", "unused", option, value])
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code == 2
