#!/usr/bin/env python3
"""Cache frozen spatial-goal inputs with source-episode-disjoint splits.

Only current RGB and the instruction enter Qwen. JEPA current, future target,
and past observations are encoded independently using the training helpers.
The original dataset normalization statistics are used without recomputation.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf


FEATURE_FIELDS = ('current', 'target', 'task', 'state', 'history', 'valid', 'ages')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def metadata_snapshot(dataset_path):
    return {str(path.relative_to(dataset_path)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in (dataset_path / 'meta').iterdir() if path.is_file()}


def model_weight_files(model_path):
    """Identify local Hugging Face weights, including index and every shard."""
    model_path = Path(model_path).resolve()
    for single_name, index_name in (
        ('model.safetensors', 'model.safetensors.index.json'),
        ('pytorch_model.bin', 'pytorch_model.bin.index.json'),
    ):
        single_path = model_path / single_name
        if single_path.is_file():
            return [single_path]
        index_path = model_path / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text())
            weight_map = index.get('weight_map')
            require(isinstance(weight_map, dict) and weight_map, f'Invalid weight_map in {index_name}')
            require(all(isinstance(value, str) and value for value in weight_map.values()),
                    'Weight shard names must be nonempty strings')
            shards = []
            for name in sorted(set(weight_map.values())):
                relative = Path(name)
                require(not relative.is_absolute() and '..' not in relative.parts,
                        f'Weight shard index must use paths within the model directory: {name}')
                # Hugging Face snapshots contain symlinks into ../../blobs.
                # Validate the lexical index path while allowing those links.
                shard = model_path / relative
                require(shard.is_file(), f'Missing weight shard: {name}')
                shards.append(shard)
            return [index_path, *shards]
    raise FileNotFoundError(f'No local model weights or shard index found in {model_path}')


def source_episode_groups(dataset_path, episode_ids):
    """Merged copies of one source episode always share a single split key."""
    source_path = dataset_path / 'meta/source_episodes.jsonl'
    sources = {}
    if source_path.exists():
        for line in source_path.read_text().splitlines():
            row = json.loads(line)
            require(isinstance(row, dict) and
                    {'episode_index', 'source_dataset', 'source_episode_index'}.issubset(row),
                    'Source metadata requires episode_index, source_dataset and source_episode_index')
            require(isinstance(row['source_dataset'], str) and bool(row['source_dataset'].strip()),
                    'source_dataset must be a nonempty string')
            for field in ('episode_index', 'source_episode_index'):
                require(isinstance(row[field], int) and not isinstance(row[field], bool) and row[field] >= 0,
                        f'{field} must be a nonnegative integer, excluding bool')
            episode = int(row['episode_index'])
            require(episode not in sources, f'Duplicate merged episode {episode} in source metadata')
            sources[episode] = row
        require(set(episode_ids).issubset(sources), 'Source metadata does not cover every dataset episode')
    groups = defaultdict(list)
    rows = {}
    for episode in episode_ids:
        source = sources.get(episode, {})
        source_dataset = str(source.get('source_dataset', dataset_path.name))
        source_episode = int(source.get('source_episode_index', episode))
        key = f'{dataset_path.name}::{source_dataset}::episode_{source_episode:06d}'
        groups[key].append(int(episode))
        rows[int(episode)] = {'source_dataset': source_dataset, 'source_episode_index': source_episode,
                              'source_group': key}
    audit = {'merged_episodes': len(episode_ids), 'unique_source_episodes': len(groups),
             'source_session_counts': dict(Counter(row['source_dataset'] for row in rows.values())),
             'duplicate_source_groups': {key: value for key, value in groups.items() if len(value) > 1},
             'source_mapping_present': bool(sources),
             'split_unit': 'original source episode; one representative of a duplicated source is selected',
             'generalization_scope': 'held-out episodes; collection sessions may occur in multiple splits'}
    return groups, rows, audit


def plan_splits(single, cfg, args):
    """Choose episodes before seeing any encoded feature or prediction error."""
    from starVLA.dataloader.spatial_history import select_history_indices
    ids = list(map(int, single.trajectory_ids))
    lengths = dict(zip(ids, map(int, single.trajectory_lengths)))
    groups, source_rows, audit = source_episode_groups(single.dataset_path, ids)
    offsets = list(map(float, cfg.framework.spatial_goal.history_offsets_seconds))
    tolerance = float(cfg.framework.spatial_goal.history_tolerance_seconds)
    future_offset = int(max(cfg.datasets.vla_data.video_frame_offsets))
    quotas = dict(train=args.train_episodes, val=args.val_episodes, test=args.test_episodes)
    needed = sum(quotas.values())
    # Prefer a complete copy over a shorter merged fragment of the same source.
    # Ties use the smallest merged ID, making the representative deterministic.
    representatives = {key: min(episodes, key=lambda episode: (-lengths[episode], episode))
                       for key, episodes in groups.items()}
    eligible = sorted(key for key, episode in representatives.items()
                      if lengths[episode] > future_offset + args.anchors_per_episode)
    require(len(eligible) >= needed, f'Need {needed} independent source episodes, found {len(eligible)}')
    assignments = {split: [] for split in quotas}
    split_manifest = getattr(args, 'split_manifest', None)
    locked = set()
    if split_manifest is not None:
        reference = json.loads(Path(split_manifest).read_text())
        prior = reference.get('split_audit', {}).get('selected_episode_ids')
        require(isinstance(prior, dict) and all(split in prior for split in quotas),
                'Split manifest must contain split_audit.selected_episode_ids for train, val and test')
        for split, quota in quotas.items():
            keys = prior[split]
            require(isinstance(keys, list) and all(isinstance(key, str) and key for key in keys),
                    f'Locked {split} episode IDs must be a list of source-group strings')
            require(len(keys) == len(set(keys)) and not (set(keys) & locked),
                    f'Duplicate or overlapping source episodes in locked {split} split')
            require(set(keys).issubset(eligible), f'Locked {split} contains unknown or ineligible source episodes')
            require(len(keys) <= quota, f'Requested {split} quota is smaller than its locked episode count')
            assignments[split] = list(keys)
            locked.update(keys)
        audit['split_lock'] = {
            'path': str(Path(split_manifest).resolve()), 'sha256': sha256(split_manifest),
            'locked_episode_ids': {split: list(keys) for split, keys in assignments.items()},
        }
    remaining = [key for key in eligible if key not in locked]
    random.Random(args.seed).shuffle(remaining)
    cursor = 0
    for split, quota in quotas.items():
        additional = quota - len(assignments[split])
        require(cursor + additional <= len(remaining), 'Insufficient unassigned source episodes for requested split quotas')
        assignments[split].extend(remaining[cursor:cursor + additional])
        cursor += additional
    plans = {}
    for split, keys in assignments.items():
        plans[split] = []
        for key in keys:
            episode = representatives[key]
            frame_data = single.get_trajectory_data(episode)
            timestamps = frame_data['timestamp'].to_numpy(dtype=np.float64)
            require(len(timestamps) == lengths[episode], f'Episode {episode} length mismatch')
            # Full future and full observed history; no terminal-image padding.
            starts = np.arange(max(0, len(timestamps) - future_offset), dtype=np.int64)
            starts = starts[timestamps[starts] >= timestamps[0] - min(offsets)]
            require(len(starts) >= args.anchors_per_episode,
                    f'Episode {episode} is too short for complete future and history')
            indices = np.rint(np.linspace(0, len(starts) - 1, args.anchors_per_episode)).astype(np.int64)
            anchors = starts[indices]
            require(len(set(anchors.tolist())) == args.anchors_per_episode, 'Repeated anchor indices')
            for anchor in anchors:
                history_indices, valid, ages = select_history_indices(timestamps, int(anchor), offsets, tolerance)
                require(bool(valid.all()), f'Episode {episode} frame {anchor} lacks requested past observations')
                sample_id = f'{key}::merged_{episode:06d}::frame_{int(anchor):06d}'
                plans[split].append({
                    'sample_id': sample_id, 'episode_id': key, 'episode_index': episode,
                    **source_rows[episode], 'frame_index': int(anchor), 'timestamp': float(timestamps[anchor]),
                    'future_frame_index': int(anchor + future_offset),
                    'future_timestamp': float(timestamps[anchor + future_offset]),
                    'history_frame_indices': list(map(int, history_indices)),
                    'history_valid': valid.tolist(), 'history_ages': ages.tolist(),
                })
    audit['selected_episode_ids'] = assignments
    audit['selected_session_counts'] = {
        split: dict(Counter(source_rows[representatives[key]]['source_dataset'] for key in keys))
        for split, keys in assignments.items()
    }
    return plans, audit


def deterministic_sample(mixture, single, row):
    """Reuse production transforms while pinning the requested episode/frame."""
    original = mixture.sample_step
    mixture.sample_step = lambda _index: (single, row['episode_index'], row['frame_index'])
    try:
        sample = mixture[0]
    finally:
        mixture.sample_step = original
    require(abs(sample['timestamp'] - row['timestamp']) < 1e-6, 'Sampler changed the requested frame')
    require(np.array_equal(sample['history_valid'], row['history_valid']), 'History validity differs from plan')
    require(np.allclose(sample['history_ages'], row['history_ages'], atol=1e-5), 'History ages differ from plan')
    return sample


def validate_reuse_cache(cache, metadata, model_kwargs):
    """Validate provenance, CPU tensor contracts and row alignment before reuse.

    Live dataset instructions must additionally be checked by
    verify_reused_instructions(), including caches predating language hashes.
    """
    require(isinstance(cache, dict) and cache.get('schema_version') == 1, 'Expected reusable cache schema_version=1')
    require(cache.get('model_kwargs') == model_kwargs, 'Reusable cache model dimensions/configuration differ')
    old_metadata = cache.get('metadata', {})
    for key in ('seed', 'source_sha256', 'weights', 'config', 'metadata_sha256', 'feature_storage_dtype'):
        require(key in old_metadata and key in metadata and old_metadata[key] == metadata[key],
                f'Reusable cache encoding provenance differs: {key}')
    for key in ('dataset_path', 'encoding_contract', 'state_normalization'):
        require(old_metadata.get(key) == metadata.get(key), f'Reusable cache provenance differs: {key}')
    if 'language_metadata_sha256' in old_metadata:
        require(old_metadata['language_metadata_sha256'] == metadata.get('language_metadata_sha256'),
                'Reusable cache language metadata differs')
    storage = metadata['feature_storage_dtype']
    require(storage in ('bfloat16', 'float32'), 'Unsupported reusable feature storage dtype')
    feature_dtype = torch.bfloat16 if storage == 'bfloat16' else torch.float32
    splits = cache.get('splits', {})
    rows_by_split = old_metadata.get('samples', {})
    require(set(splits) == {'train', 'val', 'test'} and set(rows_by_split) == set(splits),
            'Reusable cache needs aligned train/val/test tensor and manifest splits')
    cfg = metadata['config']
    framework = cfg.get('framework', {}) if isinstance(cfg, dict) else {}
    expected_task_tokens = framework.get('vj2_model', {}).get('num_embodied_action_tokens_per_instruction')
    history_offsets = framework.get('spatial_goal', {}).get('history_offsets_seconds')
    history_count = len(history_offsets) if history_offsets is not None else None
    seen_samples, seen_episodes = set(), set()
    for name in ('train', 'val', 'test'):
        split, rows = splits[name], rows_by_split[name]
        samples, episodes = split.get('sample_ids'), split.get('episode_ids')
        require(isinstance(samples, list) and isinstance(episodes, list) and
                len(samples) > 0 and len(samples) == len(episodes), f'{name}: invalid reusable IDs')
        require(all(isinstance(item, str) and item for item in samples + episodes), f'{name}: reusable IDs must be strings')
        count = len(samples)
        require(len(set(samples)) == count and not (set(samples) & seen_samples), f'{name}: duplicate reusable samples')
        require(not (set(episodes) & seen_episodes), f'{name}: source episodes overlap reusable splits')
        seen_samples.update(samples)
        seen_episodes.update(episodes)
        require(isinstance(rows, list) and len(rows) == count, f'{name}: reusable manifest length differs')
        for index, row in enumerate(rows):
            require(isinstance(row, dict) and row.get('sample_id') == samples[index] and
                    row.get('episode_id') == episodes[index], f'{name}: reusable tensor/manifest rows are misaligned')
            require(row.get('source_group', row['episode_id']) == row['episode_id'], f'{name}: inconsistent source group')
        for key in FEATURE_FIELDS:
            value = split.get(key)
            require(isinstance(value, torch.Tensor) and value.device.type == 'cpu', f'{name}.{key}: expected a CPU tensor')
            expected_dtype = torch.bool if key == 'valid' else (torch.float32 if key in ('state', 'ages') else feature_dtype)
            require(value.dtype == expected_dtype, f'{name}.{key}: storage dtype differs from its manifest')
            require(torch.isfinite(value).all().item(), f'{name}.{key}: nonfinite reusable features')
        grid_shape = (count, model_kwargs['grid_size'] ** 2, model_kwargs['latent_dim'])
        require(split['current'].shape == grid_shape and split['target'].shape == grid_shape,
                f'{name}: reusable current/target grid shapes differ')
        task = split['task']
        require(task.ndim == 3 and task.shape[0] == count and task.shape[1] > 0 and
                task.shape[2] == model_kwargs['task_dim'], f'{name}: invalid reusable task tokens')
        if expected_task_tokens is not None:
            require(task.shape[1] == expected_task_tokens, f'{name}: task token count differs from encoding config')
        require(split['state'].shape == (count, model_kwargs['state_dim']), f'{name}: reusable state shape differs')
        history, valid, ages = split['history'], split['valid'], split['ages']
        require(history.ndim == 4 and history.shape[0] == count and history.shape[1] > 0 and
                history.shape[2:] == grid_shape[1:], f'{name}: invalid reusable history shape')
        if history_count is None:
            history_count = history.shape[1]
        require(history.shape[1] == history_count and valid.shape == history.shape[:2] and ages.shape == valid.shape,
                f'{name}: reusable history, validity and age shapes differ')
        require(bool((ages[valid] > 0).all()), f'{name}: valid history must precede current frames')
        expected_valid = torch.as_tensor([row['history_valid'] for row in rows], dtype=torch.bool)
        expected_ages = torch.as_tensor([row['history_ages'] for row in rows], dtype=torch.float32)
        require(torch.equal(valid, expected_valid) and torch.equal(ages, expected_ages),
                f'{name}: reusable history tensors differ from manifest rows')
    return cache


def plan_reuse(cache, plans):
    """Map new indices to exact old rows; maintain source splits even for new frames."""
    require(set(plans) == {'train', 'val', 'test'}, 'Reuse plans need train, val and test')
    locations, source_splits = {}, {}
    for split, old_rows in cache['metadata']['samples'].items():
        for index, row in enumerate(old_rows):
            sample, group = row['sample_id'], row['episode_id']
            require(sample not in locations, 'Duplicate samples in reusable cache')
            require(group not in source_splits or source_splits[group] == split,
                    'Reusable source episode crosses split boundaries')
            locations[sample] = (split, index, row)
            source_splits[group] = split
    mapping = {split: {} for split in plans}
    seen_samples, new_source_splits = set(), {}
    for split, rows in plans.items():
        for index, row in enumerate(rows):
            sample, group = row['sample_id'], row['episode_id']
            require(row.get('source_group', group) == group, 'New plan source group disagrees with episode ID')
            require(sample not in seen_samples, 'New reuse plan repeats a sample')
            seen_samples.add(sample)
            require(group not in new_source_splits or new_source_splits[group] == split,
                    'New source episode crosses split boundaries')
            new_source_splits[group] = split
            require(group not in source_splits or source_splits[group] == split,
                    'Reuse cannot move any old source episode into another split, even with new frame IDs')
            if sample not in locations:
                continue
            old_split, old_index, old_row = locations[sample]
            require(old_split == split, 'Reuse cannot move an existing sample into another split')
            require(set(row) - {'instruction'} == set(old_row) - {'instruction'},
                    'Reusable sample row fields differ')
            require(all(key in old_row and old_row[key] == value for key, value in row.items()),
                    'Reusable sample frame, timestamp, history or instruction differs')
            mapping[split][index] = old_index
    return mapping


def verify_reused_instructions(single, cache, plans, mapping):
    """Read current task text without decoding video before reusing Qwen tokens."""
    verified = 0
    language_key = single.modality_keys['language'][0]
    for split, indices in mapping.items():
        for new_index, old_index in indices.items():
            row = plans[split][new_index]
            old_row = cache['metadata']['samples'][split][old_index]
            require(isinstance(old_row.get('instruction'), str), 'Reusable Qwen features need a recorded instruction')
            single.curr_traj_data = single.get_trajectory_data(row['episode_index'])
            instruction = single.get_language(row['episode_index'], language_key, row['frame_index'])[0]
            require(instruction == old_row['instruction'],
                    f'Live dataset instruction changed for reusable sample {row["sample_id"]}')
            row['instruction'] = instruction
            verified += 1
    return verified


def save_feature_cache(cache, output):
    """Publish a fully written and safely reloaded cache atomically, without overwrite."""
    descriptor, name = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(cache, temporary)
        loaded = torch.load(temporary, map_location='cpu', weights_only=True)
        validate_reuse_cache(loaded, cache['metadata'], cache['model_kwargs'])
        # A hard link publishes the completed bytes and fails if output exists.
        # Both paths are in the same directory/filesystem.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


@torch.inference_mode()
def extract_batch(model, samples, storage_dtype):
    videos = np.stack([sample['video'] for sample in samples])
    current_tokens, target_tokens = model._encode_spatial_training_pair(samples, videos)
    current = model._current_spatial_grid(current_tokens)
    target = model._spatial_grid(target_tokens)
    history, valid, ages = model._encode_spatial_history(
        [sample['history_images'] for sample in samples], [sample['history_valid'] for sample in samples],
        [sample['history_ages'] for sample in samples], current)
    # This helper's only visual inputs are current 224px Qwen observations.
    _, task = model._get_vlm_action_tokens(
        [sample['image'] for sample in samples], [sample['lang'] for sample in samples], include_embodied=True)
    state = torch.as_tensor(np.stack([sample['state'][-1] for sample in samples]), dtype=torch.float32)
    outputs = dict(current=current, target=target, task=task, state=state,
                   history=history, valid=valid, ages=ages)
    for key, tensor in outputs.items():
        require(torch.isfinite(tensor).all().item(), f'Nonfinite extracted {key}')
        dtype = torch.bool if key == 'valid' else (torch.float32 if key in ('state', 'ages') else storage_dtype)
        outputs[key] = tensor.detach().to(device='cpu', dtype=dtype).contiguous()
    return outputs


def run(args):
    for artifact in (args.output, args.output.with_suffix('.manifest.json')):
        if artifact.exists():
            raise FileExistsError(f'Refusing to overwrite existing experiment artifact: {artifact}')
    repo = args.repo_root.resolve()
    if not (repo / 'starVLA').is_dir() and (Path.cwd() / 'starVLA').is_dir():
        repo = Path.cwd()
    require((repo / 'starVLA').is_dir(), 'Pass --repo-root pointing to the source repository')
    sys.path.insert(0, str(repo))
    sys.dont_write_bytecode = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for key, value in {'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                       'NO_ALBUMENTATIONS_UPDATE': '1', 'WANDB_MODE': 'disabled',
                       'TOKENIZERS_PARALLELISM': 'false',
                       'TORCH_EXTENSIONS_DIR': str(args.output.parent / 'torch_extensions'),
                       'TRITON_CACHE_DIR': str(args.output.parent / 'triton')}.items():
        os.environ[key] = value
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
    from starVLA.model.framework.VLA_JEPA import VLA_JEPA
    require(torch.cuda.is_available(), 'Feature extraction requires the CUDA model environment')
    torch.cuda.set_device(args.device)
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cfg = OmegaConf.load(args.config)
    cfg.framework.qwenvl.base_vlm = str(args.qwen_model.resolve())
    cfg.framework.vj2_model.base_encoder = str(args.vjepa_checkpoint.resolve())
    cfg.datasets.vla_data.data_root_dir = str(args.data_root.resolve())
    require(bool(cfg.framework.spatial_goal.enabled), 'Config must enable spatial goals')
    mixture_spec = DATASET_NAMED_MIXTURES[cfg.datasets.vla_data.data_mix]
    require(len(mixture_spec) == 1, 'This held-out episode probe currently accepts one dataset per cache')
    dataset_path = Path(cfg.datasets.vla_data.data_root_dir) / mixture_spec[0][0]
    before_metadata = metadata_snapshot(dataset_path)
    mixture = get_vla_dataset(cfg.datasets.vla_data, mode='val', seed=args.seed,
                              action_horizon=cfg.framework.action_model.action_horizon,
                              video_horizon=cfg.framework.vj2_model.num_frames, delete_pause_frame=False)
    single = mixture.datasets[0]
    plans, split_audit = plan_splits(single, cfg, args)
    manifest = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'seed': args.seed,
        'config': OmegaConf.to_container(cfg, resolve=True),
        'dataset_path': str(dataset_path.resolve()), 'split_audit': split_audit,
        'samples': plans, 'state_normalization': 'Existing production dataset statistics; not refitted for this probe',
        'feature_storage_dtype': args.storage_dtype,
        'encoding_contract': 'Frozen eval Qwen receives current image+instruction only; current/future/history JEPA images encoded independently using production helpers',
        'initialized_special_tokens': 'Production VLA_JEPA adds its action tokens using the pretrained Qwen initialization; no VLA finetuning checkpoint is loaded',
        'weights': {
            'vjepa21': {'path': str(args.vjepa_checkpoint.resolve()),
                        'bytes': args.vjepa_checkpoint.stat().st_size, 'sha256': sha256(args.vjepa_checkpoint)},
            'qwen': {'path': str(args.qwen_model.resolve()),
                     'files': {str(path.relative_to(args.qwen_model.resolve())):
                               {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                               for path in model_weight_files(args.qwen_model)}},
        },
        'source_sha256': {str(path.relative_to(repo)): sha256(path) for path in (
            repo / 'starVLA/model/framework/VLA_JEPA.py', repo / 'starVLA/model/framework/spatial_jepa.py',
            repo / 'starVLA/model/modules/world_model/spatial_goal.py',
            repo / 'starVLA/dataloader/gr00t_lerobot/datasets.py', repo / 'starVLA/dataloader/spatial_history.py')},
        'metadata_sha256': {path.name: sha256(path) for path in (dataset_path / 'meta').iterdir()
                           if path.name in ('episodes.jsonl', 'source_episodes.jsonl', 'info.json', 'stats_gr00t.json', 'modality.json')},
        'language_metadata_sha256': {path.name: sha256(path) for path in (dataset_path / 'meta').iterdir()
                                    if path.name in ('tasks.jsonl', 'tasks.json', 'lang_map.json')},
        'extraction_arguments': {'batch_size': args.batch_size, 'num_threads': args.num_threads,
                                 'train_episodes': args.train_episodes, 'val_episodes': args.val_episodes,
                                 'test_episodes': args.test_episodes, 'anchors_per_episode': args.anchors_per_episode},
        'extractor_sha256': sha256(Path(__file__)),
        'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'cuda': str(torch.version.cuda),
    }
    with args.output.with_suffix('.manifest.json').open('x') as stream:
        stream.write(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'event': 'split_ready', 'audit': split_audit}, ensure_ascii=False), flush=True)
    model = VLA_JEPA(cfg).to('cuda').eval().requires_grad_(False)
    require(not model.qwen_vl_interface.training and not model.vj_encoder.training, 'Backbones must be eval')
    spatial_cfg = cfg.framework.spatial_goal
    model_kwargs = {'latent_dim': int(model.vj_encoder.config.hidden_size),
                    'task_dim': int(model.qwen_vl_interface.model.config.hidden_size),
                    'state_dim': int(cfg.framework.action_model.state_dim),
                    'grid_size': int(spatial_cfg.grid_size), 'hidden_dim': int(spatial_cfg.hidden_dim),
                    'num_heads': int(spatial_cfg.num_heads)}
    reusable = None
    reuse_map = {split: {} for split in plans}
    reuse_path = getattr(args, 'reuse_features', None)
    if reuse_path is not None:
        reusable = torch.load(reuse_path, map_location='cpu', weights_only=True)
        validate_reuse_cache(reusable, manifest, model_kwargs)
        reuse_map = plan_reuse(reusable, plans)
        verified = verify_reused_instructions(single, reusable, plans, reuse_map)
        manifest['reuse_features'] = {
            'path': str(Path(reuse_path).resolve()), 'sha256': sha256(reuse_path),
            'reused_samples': {split: len(indices) for split, indices in reuse_map.items()},
            'new_samples': {split: len(plans[split]) - len(indices) for split, indices in reuse_map.items()},
            'live_instructions_verified': verified,
            'legacy_language_hash_missing': 'language_metadata_sha256' not in reusable['metadata'],
        }
        print(json.dumps({'event': 'reuse_ready', **manifest['reuse_features']}), flush=True)
    dtype = torch.bfloat16 if args.storage_dtype == 'bfloat16' else torch.float32
    splits = {}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    encoded_count = 0
    for split, rows in plans.items():
        batches = {key: [None] * len(rows) for key in FEATURE_FIELDS}
        for new_index, old_index in reuse_map[split].items():
            for key in FEATURE_FIELDS:
                batches[key][new_index] = reusable['splits'][split][key][old_index:old_index + 1]
        pending = [index for index in range(len(rows)) if index not in reuse_map[split]]
        for start in range(0, len(pending), args.batch_size):
            indices = pending[start:start + args.batch_size]
            batch_rows = [rows[index] for index in indices]
            samples = [deterministic_sample(mixture, single, row) for row in batch_rows]
            for row, sample in zip(batch_rows, samples):
                row['instruction'] = sample['lang']
            encoded = extract_batch(model, samples, dtype)
            for key, value in encoded.items():
                for batch_index, index in enumerate(indices):
                    batches[key][index] = value[batch_index:batch_index + 1]
            encoded_count += len(indices)
            print(json.dumps({'event': 'extracted', 'split': split, 'samples': len(reuse_map[split]) + min(start + args.batch_size, len(pending)),
                              'total': len(rows), 'new_samples_encoded': encoded_count,
                              'seconds': time.perf_counter() - started}), flush=True)
        splits[split] = {key: torch.cat(values) for key, values in batches.items()}
        splits[split]['episode_ids'] = [row['episode_id'] for row in rows]
        splits[split]['sample_ids'] = [row['sample_id'] for row in rows]
    require(metadata_snapshot(dataset_path) == before_metadata, 'Dataset metadata changed during extraction')
    manifest['extraction_seconds'] = time.perf_counter() - started
    manifest['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
    manifest['dataset_metadata_unchanged'] = True
    manifest['new_samples_encoded'] = encoded_count
    manifest['reused_samples_total'] = sum(map(len, reuse_map.values()))
    cache = {
        'schema_version': 1,
        'model_kwargs': model_kwargs,
        'metadata': manifest, 'splits': splits,
    }
    # Plain dicts, strings and CPU tensors support safe weights_only loading.
    save_feature_cache(cache, args.output)
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'event': 'complete', 'output': str(args.output), 'bytes': args.output.stat().st_size,
                      'seconds': manifest['extraction_seconds'], 'samples': {key: len(value['episode_ids']) for key, value in splits.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--qwen-model', type=Path, required=True)
    parser.add_argument('--vjepa-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repo-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-episodes', type=int, default=12)
    parser.add_argument('--val-episodes', type=int, default=4)
    parser.add_argument('--test-episodes', type=int, default=4)
    parser.add_argument('--split-manifest', type=Path,
                        help='Lock all source episodes to their prior train/val/test split before allocating additional episodes')
    parser.add_argument('--reuse-features', type=Path,
                        help='Reuse matching frozen features from a completed cache after strict provenance, split and live-instruction checks')
    parser.add_argument('--anchors-per-episode', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-threads', type=int, default=4)
    parser.add_argument('--storage-dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    args = parser.parse_args()
    for key in ('train_episodes', 'val_episodes', 'test_episodes', 'anchors_per_episode', 'batch_size', 'num_threads'):
        require(getattr(args, key) > 0, f'{key} must be positive')
    run(args)


if __name__ == '__main__':
    main()
