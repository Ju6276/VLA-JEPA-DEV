"""Source-disjoint frozen-feature sampling without loading pretrained models."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from scripts import extract_spatial_goal_features as extraction
from starVLA.dataloader.spatial_history import select_history_indices


def test_weight_manifest_includes_shard_index_and_each_unique_shard(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors",
                                              "c": "part-1.safetensors"}}))
    for name in ("part-1.safetensors", "part-2.safetensors"):
        (tmp_path / name).write_bytes(name.encode())
    files = extraction.model_weight_files(tmp_path)
    assert files == [index, tmp_path / "part-1.safetensors", tmp_path / "part-2.safetensors"]
    assert all(len(extraction.sha256(path)) == 64 for path in files)


def test_standard_huggingface_shard_symlinks_can_be_hashed(tmp_path):
    snapshot = tmp_path / "snapshots/model-revision"
    snapshot.mkdir(parents=True)
    blob = tmp_path / "blobs/weight-hash"
    blob.parent.mkdir()
    blob.write_bytes(b"weight-shard")
    shard = snapshot / "part-1.safetensors"
    shard.symlink_to("../../blobs/weight-hash")
    index = snapshot / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"parameter": shard.name}}))
    files = extraction.model_weight_files(snapshot)
    assert files[0] == index and len(files) == 2
    assert extraction.sha256(files[1]) == extraction.sha256(blob)


@pytest.mark.parametrize("shard_name", ["missing.safetensors", "../outside.safetensors"])
def test_missing_or_traversing_weight_index_shards_are_rejected(tmp_path, shard_name):
    model = tmp_path / "model"
    model.mkdir()
    (tmp_path / "outside.safetensors").write_bytes(b"outside")
    (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": shard_name}}))
    with pytest.raises(ValueError):
        extraction.model_weight_files(model)


def write_sources(path, rows):
    (path / "meta").mkdir(parents=True, exist_ok=True)
    (path / "meta/source_episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n")


def source_row(merged, original, session="session-a"):
    return dict(episode_index=merged, source_dataset=session, source_episode_index=original)


def test_merged_copies_share_source_group_and_audit(tmp_path):
    write_sources(tmp_path, [source_row(0, 7), source_row(1, 7), source_row(2, 7, "session-b")])
    groups, rows, audit = extraction.source_episode_groups(tmp_path, [0, 1, 2])
    assert len(groups) == 2
    assert rows[0]["source_group"] == rows[1]["source_group"]
    assert rows[0]["source_group"] != rows[2]["source_group"]
    assert groups[rows[0]["source_group"]] == [0, 1]
    assert audit["source_mapping_present"]
    assert audit["merged_episodes"] == 3 and audit["unique_source_episodes"] == 2
    assert audit["duplicate_source_groups"] == {rows[0]["source_group"]: [0, 1]}


def test_plain_dataset_without_source_mapping_uses_its_own_episode_ids(tmp_path):
    groups, rows, audit = extraction.source_episode_groups(tmp_path, [7, 8])
    assert len(groups) == 2
    assert rows[7]["source_episode_index"] == 7
    assert rows[8]["source_dataset"] == tmp_path.name
    assert not audit["source_mapping_present"]


def test_present_source_mapping_must_cover_all_dataset_episodes(tmp_path):
    write_sources(tmp_path, [source_row(0, 7)])
    with pytest.raises(ValueError, match="cover"):
        extraction.source_episode_groups(tmp_path, [0, 1])


def test_duplicate_merged_episode_in_source_mapping_is_rejected(tmp_path):
    write_sources(tmp_path, [source_row(0, 7), source_row(0, 8)])
    with pytest.raises(ValueError, match="Duplicate merged episode"):
        extraction.source_episode_groups(tmp_path, [0])


@pytest.mark.parametrize("field,value,remove", [
    ("source_dataset", None, True), ("source_episode_index", None, True),
    ("source_dataset", "", False), ("source_dataset", 123, False),
    ("source_episode_index", True, False), ("source_episode_index", -1, False),
    ("episode_index", False, False), ("episode_index", "0", False),
])
def test_malformed_source_provenance_cannot_silently_create_new_groups(tmp_path, field, value, remove):
    row = source_row(0, 7)
    if remove:
        del row[field]
    else:
        row[field] = value
    write_sources(tmp_path, [row])
    with pytest.raises(ValueError):
        extraction.source_episode_groups(tmp_path, [0])


class FakeDataset:
    def __init__(self, path, lengths, timestamps=None):
        self.dataset_path = path
        self.trajectory_ids = np.array(list(lengths))
        self.trajectory_lengths = np.array(list(lengths.values()))
        self.frames = {episode: pd.DataFrame({"timestamp": (
            timestamps[episode] if timestamps is not None else np.arange(length) / 10)
        }) for episode, length in lengths.items()}
        self.read_episodes = []

    def get_trajectory_data(self, episode):
        self.read_episodes.append(episode)
        return self.frames[episode]


def planning_args():
    cfg = OmegaConf.create({
        "framework": {"spatial_goal": {"history_offsets_seconds": [-0.8, -0.4],
                                        "history_tolerance_seconds": 0.15}},
        "datasets": {"vla_data": {"video_frame_offsets": [0, 2, 4]}},
    })
    args = SimpleNamespace(train_episodes=1, val_episodes=1, test_episodes=1,
                           anchors_per_episode=3, seed=42)
    return cfg, args


def test_plan_uses_one_longest_representative_and_complete_causal_windows(tmp_path):
    write_sources(tmp_path, [source_row(episode, episode // 2) for episode in range(6)])
    # Group 0's lowest merged index is too short, although its copied source
    # has a valid longer representative. Other groups exercise length/tie rules.
    dataset = FakeDataset(tmp_path, {0: 9, 1: 45, 2: 40, 3: 30, 4: 40, 5: 40})
    cfg, args = planning_args()
    plans, audit = extraction.plan_splits(dataset, cfg, args)
    assert set(dataset.read_episodes) == {1, 2, 4}
    group_sets = [{row["episode_id"] for row in plans[split]} for split in ("train", "val", "test")]
    assert all(len(groups) == 1 for groups in group_sets)
    assert not group_sets[0] & group_sets[1] and not group_sets[0] & group_sets[2] and not group_sets[1] & group_sets[2]
    all_rows = [row for rows in plans.values() for row in rows]
    assert len(all_rows) == len({row["sample_id"] for row in all_rows}) == 9
    for rows in plans.values():
        assert len(rows) == args.anchors_per_episode
        episode = rows[0]["episode_index"]
        times = dataset.frames[episode]["timestamp"].to_numpy()
        eligible_starts = np.arange(len(times) - 4)
        eligible_starts = eligible_starts[times[eligible_starts] >= times[0] + 0.8]
        expected = eligible_starts[np.rint(np.linspace(0, len(eligible_starts) - 1, 3)).astype(int)]
        assert [row["frame_index"] for row in rows] == expected.tolist()
        for row in rows:
            frame = row["frame_index"]
            assert row["future_frame_index"] == frame + 4 < len(times)
            assert row["future_timestamp"] == times[frame + 4]
            history_indices, valid, ages = select_history_indices(times, frame, [-0.8, -0.4], 0.15)
            assert row["history_frame_indices"] == history_indices.tolist()
            assert all(index < frame for index in row["history_frame_indices"])
            assert row["history_valid"] == valid.tolist() == [True, True]
            np.testing.assert_allclose(row["history_ages"], ages)
    again, _ = extraction.plan_splits(dataset, cfg, args)
    assert again == plans
    assert audit["selected_episode_ids"] == {
        split: list({row["episode_id"] for row in rows}) for split, rows in plans.items()}


def test_source_duplicates_do_not_count_as_extra_independent_episodes(tmp_path):
    write_sources(tmp_path, [source_row(episode, 0) for episode in range(6)])
    dataset = FakeDataset(tmp_path, {episode: 40 for episode in range(6)})
    cfg, args = planning_args()
    with pytest.raises(ValueError, match="independent source episodes"):
        extraction.plan_splits(dataset, cfg, args)


def old_split_manifest(tmp_path, *, group_count=12, old_train=1):
    write_sources(tmp_path, [source_row(episode, episode // 2) for episode in range(group_count * 2)])
    dataset = FakeDataset(tmp_path, {episode: 60 for episode in range(group_count * 2)})
    cfg, args = planning_args()
    args.train_episodes = old_train
    plans, audit = extraction.plan_splits(dataset, cfg, args)
    manifest = tmp_path / "old.manifest.json"
    manifest.write_text(json.dumps({"split_audit": audit}))
    return dataset, cfg, args, plans, audit, manifest


def test_split_expansion_preserves_source_assignments_and_is_deterministic(tmp_path):
    dataset, cfg, args, old_plans, old_audit, manifest = old_split_manifest(tmp_path)
    args.split_manifest = manifest
    args.train_episodes, args.val_episodes, args.test_episodes = 6, 2, 2
    args.anchors_per_episode = 5
    expanded, audit = extraction.plan_splits(dataset, cfg, args)
    selected = audit["selected_episode_ids"]
    for split, count in (("train", 6), ("val", 2), ("test", 2)):
        assert len(selected[split]) == count
        assert set(old_audit["selected_episode_ids"][split]).issubset(selected[split])
        assert len(expanded[split]) == count * 5
        for other in {"train", "val", "test"} - {split}:
            assert not set(selected[split]) & set(selected[other])
            assert not set(selected[split]) & set(old_audit["selected_episode_ids"][other])
    repeated, repeated_audit = extraction.plan_splits(dataset, cfg, args)
    assert repeated == expanded
    assert repeated_audit["selected_episode_ids"] == selected
    # The original artifact is an immutable source of assignments.
    assert json.loads(manifest.read_text())["split_audit"] == old_audit


def test_split_lock_is_source_identity_even_if_representative_copy_changes(tmp_path):
    dataset, cfg, args, old_plans, old_audit, manifest = old_split_manifest(tmp_path)
    args.split_manifest = manifest
    # Previously tied copies used the smallest merged episode; now each second
    # copy is longer. Locking must survive changing that implementation detail.
    for position, episode in enumerate(dataset.trajectory_ids):
        if episode % 2:
            dataset.trajectory_lengths[position] = 70
            dataset.frames[int(episode)] = pd.DataFrame({"timestamp": np.arange(70) / 10})
    new_plans, new_audit = extraction.plan_splits(dataset, cfg, args)
    assert new_audit["selected_episode_ids"] == old_audit["selected_episode_ids"]
    for split in old_plans:
        assert {row["episode_id"] for row in new_plans[split]} == {row["episode_id"] for row in old_plans[split]}
        assert all(row["episode_index"] % 2 == 1 for row in new_plans[split])
        assert all(row["episode_index"] % 2 == 0 for row in old_plans[split])


def test_no_split_manifest_preserves_original_seeded_plan(tmp_path):
    dataset, cfg, args, old_plans, old_audit, _ = old_split_manifest(tmp_path)
    args.split_manifest = None
    new_plans, new_audit = extraction.plan_splits(dataset, cfg, args)
    assert new_plans == old_plans
    assert new_audit["selected_episode_ids"] == old_audit["selected_episode_ids"]


@pytest.mark.parametrize("invalid", ["cross_split_overlap", "within_split_duplicate", "unknown_group", "missing_split"])
def test_invalid_locked_source_assignments_are_rejected(tmp_path, invalid):
    dataset, cfg, args, _, audit, manifest = old_split_manifest(tmp_path)
    modified = copy.deepcopy(audit)
    selected = modified["selected_episode_ids"]
    if invalid == "cross_split_overlap":
        selected["val"] = list(selected["train"])
    elif invalid == "within_split_duplicate":
        selected["train"] *= 2
        args.train_episodes = 2
    elif invalid == "unknown_group":
        selected["train"] = ["unknown-source::episode_000000"]
    else:
        del selected["test"]
    manifest.write_text(json.dumps({"split_audit": modified}))
    args.split_manifest = manifest
    with pytest.raises(ValueError):
        extraction.plan_splits(dataset, cfg, args)


def test_expansion_cannot_reduce_an_existing_split_quota(tmp_path):
    dataset, cfg, args, _, _, manifest = old_split_manifest(tmp_path, old_train=2)
    args.split_manifest, args.train_episodes = manifest, 1
    with pytest.raises(ValueError):
        extraction.plan_splits(dataset, cfg, args)


def test_expansion_requires_enough_new_independent_source_groups(tmp_path):
    dataset, cfg, args, _, _, manifest = old_split_manifest(tmp_path, group_count=5)
    args.split_manifest, args.train_episodes = manifest, 4
    with pytest.raises(ValueError):
        extraction.plan_splits(dataset, cfg, args)


def test_sparse_timestamps_cannot_claim_requested_history_exists(tmp_path):
    times = np.arange(40) * 0.3
    dataset = FakeDataset(tmp_path, {0: 40, 1: 40, 2: 40},
                          timestamps={episode: times for episode in range(3)})
    cfg, args = planning_args()
    with pytest.raises(ValueError, match="past observations"):
        extraction.plan_splits(dataset, cfg, args)


def test_metadata_length_mismatch_rejected_before_encoding(tmp_path):
    dataset = FakeDataset(tmp_path, {0: 40, 1: 40, 2: 40})
    for episode in dataset.frames:
        dataset.frames[episode] = dataset.frames[episode].iloc[:-1]
    cfg, args = planning_args()
    with pytest.raises(ValueError, match="length mismatch"):
        extraction.plan_splits(dataset, cfg, args)


class PinnedMixture:
    def __init__(self, result=None, error=None):
        self.sample_step = lambda index: ("unmodified", index, 999)
        self.result, self.error = result, error
        self.observed = []

    def __getitem__(self, index):
        # Simulate a loader retry changing its sample index: pinning must keep
        # the original requested source frame on both attempts.
        self.observed.extend([self.sample_step(index), self.sample_step(index + 123)])
        if self.error is not None:
            raise self.error
        return self.result


def sample_plan():
    row = {"episode_index": 17, "frame_index": 12, "timestamp": 1.2,
           "history_valid": [True, True], "history_ages": [0.8, 0.4]}
    sample = {key: row[key] for key in ("timestamp", "history_valid", "history_ages")}
    return row, sample


@pytest.mark.parametrize("failure", [None, "decode", "timestamp", "history"])
def test_deterministic_sample_restores_sampling_even_when_read_or_validation_fails(failure):
    row, sample = sample_plan()
    if failure == "timestamp":
        sample["timestamp"] = 1.3
    elif failure == "history":
        sample["history_ages"] = [0.6, 0.2]
    mixture = PinnedMixture(sample, IOError("decode failed") if failure == "decode" else None)
    original, single = mixture.sample_step, object()
    if failure is None:
        assert extraction.deterministic_sample(mixture, single, row) is sample
    else:
        with pytest.raises(IOError if failure == "decode" else ValueError):
            extraction.deterministic_sample(mixture, single, row)
    assert mixture.sample_step is original
    assert mixture.observed == [(single, 17, 12), (single, 17, 12)]


def test_extract_batch_passes_only_current_images_to_qwen():
    calls = {}

    class FakeModel:
        def _encode_spatial_training_pair(self, samples, videos):
            calls["jepa_samples"] = samples
            return torch.ones(2, 4, 8), torch.full((2, 4, 8), 2.0)

        def _current_spatial_grid(self, tokens):
            return tokens

        def _spatial_grid(self, tokens):
            return tokens

        def _encode_spatial_history(self, images, valid, ages, current):
            calls["past_images"] = images
            return torch.full((2, 2, 4, 8), 3.0), torch.tensor(valid), torch.tensor(ages)

        def _get_vlm_action_tokens(self, images, instructions, include_embodied):
            calls["qwen_images"], calls["instructions"] = images, instructions
            assert include_embodied
            return None, torch.ones(2, 2, 6)

    samples = [{"image": [f"current-{i}"], "jepa_image": [f"current-jepa-{i}"],
                "video": np.full((1, 2, 8, 8, 3), i, dtype=np.uint8),
                "history_images": [f"past-old-{i}", f"past-new-{i}"],
                "history_valid": [True, True], "history_ages": [0.8, 0.4],
                "lang": f"task-{i}", "state": np.ones((1, 3), dtype=np.float32) * i} for i in range(2)]
    output = extraction.extract_batch(FakeModel(), samples, torch.bfloat16)
    assert calls["qwen_images"] == [["current-0"], ["current-1"]]
    assert calls["instructions"] == ["task-0", "task-1"]
    assert calls["past_images"] == [sample["history_images"] for sample in samples]
    assert output["current"].dtype == output["target"].dtype == torch.bfloat16
    assert output["valid"].dtype == torch.bool
    assert output["state"].dtype == output["ages"].dtype == torch.float32
    assert all(value.device.type == "cpu" for value in output.values())
