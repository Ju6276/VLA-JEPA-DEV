"""Cold-start statistics caches are complete for concurrent training ranks."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import numpy as np
import pandas as pd
import pytest

from starVLA.dataloader.gr00t_lerobot import datasets as datasets_module
from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, _write_statistics_cache
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


@pytest.mark.parametrize("existing", [False, True])
def test_concurrent_statistics_writers_never_publish_partial_json(monkeypatch, tmp_path, existing):
    output = tmp_path / "stats_gr00t.json"
    previous = {"previous": [1.0, 2.0]}
    if existing:
        output.write_text(json.dumps(previous))
    original_dump = json.dump
    partially_written = threading.Barrier(3)
    finish_writing = threading.Event()

    def slow_dump(value, stream, **kwargs):
        stream.write('{"partial":')
        stream.flush()
        partially_written.wait(timeout=10)
        assert finish_writing.wait(timeout=10)
        stream.seek(0)
        stream.truncate()
        original_dump(value, stream, **kwargs)

    monkeypatch.setattr(datasets_module.json, "dump", slow_dump)
    new_values = [{"rank": 0, "mean": [3.0, 4.0]}, {"rank": 1, "mean": [3.0, 4.0]}]
    with ThreadPoolExecutor(max_workers=2) as pool:
        writers = [pool.submit(_write_statistics_cache, output, value) for value in new_values]
        try:
            partially_written.wait(timeout=10)
            # Both writers are paused with incomplete JSON in their own files.
            # A reader must never observe either incomplete prefix at output.
            for _ in range(100):
                if existing:
                    assert json.loads(output.read_text()) == previous
                else:
                    assert not output.exists()
        finally:
            finish_writing.set()
        for writer in writers:
            writer.result(timeout=10)
    assert json.loads(output.read_text()) in new_values
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("failure", ["serialize", "publish"])
def test_statistics_write_failure_preserves_old_cache_and_cleans_temporary_file(monkeypatch, tmp_path, failure):
    output = tmp_path / "stats_gr00t.json"
    old_bytes = b'{"previous": [1, 2]}'
    output.write_bytes(old_bytes)

    if failure == "serialize":
        def fail_dump(value, stream, **kwargs):
            stream.write('{"partial":')
            stream.flush()
            raise OSError("disk full")
        monkeypatch.setattr(datasets_module.json, "dump", fail_dump)
    else:
        def fail_replace(source, destination):
            raise OSError("cannot publish")
        monkeypatch.setattr(datasets_module.os, "replace", fail_replace)
    with pytest.raises(OSError):
        _write_statistics_cache(output, {"next": [3, 4]})
    assert output.read_bytes() == old_bytes
    assert list(tmp_path.iterdir()) == [output]


def make_metadata_dataset(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    modality = {
        "state": {"joints": {"start": 0, "end": 2, "dtype": "float32", "original_key": "observation.state"}},
        "action": {"joints": {"start": 0, "end": 2, "dtype": "float32", "original_key": "action"}},
        "video": {"ego": {"original_key": "observation.images.ego"}},
    }
    (meta / "modality.json").write_text(json.dumps(modality))
    (meta / "info.json").write_text(json.dumps({
        "fps": 50, "features": {"observation.images.ego": {
            "shape": [8, 8, 3], "names": ["height", "width", "channels"],
        }},
    }))
    data = tmp_path / "data/chunk-000"
    data.mkdir(parents=True)
    pd.DataFrame({
        "observation.state": [[1.0, 2.0], [3.0, 4.0]],
        "action": [[5.0, 6.0], [7.0, 8.0]],
        "timestamp": [0.0, 0.02],
    }).to_parquet(data / "episode_000000.parquet")
    dataset = LeRobotSingleDataset.__new__(LeRobotSingleDataset)
    dataset._dataset_path, dataset._dataset_name = tmp_path, "tiny"
    return dataset


@pytest.mark.parametrize("initial_content", [None, '{"observation.state":', '{"observation.state": {"mean": [0]}}'])
def test_production_metadata_rebuilds_missing_truncated_or_invalid_statistics(tmp_path, initial_content):
    dataset = make_metadata_dataset(tmp_path)
    stats_path = tmp_path / "meta/stats_gr00t.json"
    if initial_content is not None:
        stats_path.write_text(initial_content)
    metadata = dataset._get_metadata(EmbodimentTag.NEW_EMBODIMENT)
    np.testing.assert_allclose(metadata.statistics.state["joints"].mean, [2.0, 3.0])
    np.testing.assert_allclose(metadata.statistics.action["joints"].mean, [6.0, 7.0])
    saved = json.loads(stats_path.read_text())
    assert saved["observation.state"]["mean"] == [2.0, 3.0]
    assert saved["action"]["min"] == [5.0, 6.0]
    assert sorted(path.name for path in stats_path.parent.iterdir()) == ["info.json", "modality.json", "stats_gr00t.json"]


def test_production_metadata_reuses_valid_statistics_without_recalculation(monkeypatch, tmp_path):
    dataset = make_metadata_dataset(tmp_path)
    dataset._get_metadata(EmbodimentTag.NEW_EMBODIMENT)
    stats_path = tmp_path / "meta/stats_gr00t.json"
    previous_bytes, previous_mtime = stats_path.read_bytes(), stats_path.stat().st_mtime_ns

    def unexpected_recompute(paths):
        pytest.fail("An existing valid cache must be reused")

    monkeypatch.setattr(datasets_module, "calculate_dataset_statistics", unexpected_recompute)
    metadata = dataset._get_metadata(EmbodimentTag.NEW_EMBODIMENT)
    np.testing.assert_allclose(metadata.statistics.state["joints"].max, [3.0, 4.0])
    assert stats_path.read_bytes() == previous_bytes
    assert stats_path.stat().st_mtime_ns == previous_mtime
