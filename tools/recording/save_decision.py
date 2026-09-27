# -*- coding: utf-8 -*-
"""Save a single-batch action-prior trace without contacting a robot or model."""
from pathlib import Path
import json
import re
import time
import numpy as np
from PIL import Image


def save_decision(root, episode_id, decision_id, request, response, metadata,
                  *, controller_commands=None, command_normalization=None):
    """Record a response from the R0 trace patch or the R1 built-in trace.

    Run logging separately from inference-rate profiling. This helper performs
    synchronous I/O; a deployed control loop should enqueue it to a disk worker.
    Metadata should identify task, checkpoint, code commit, normalization stats,
    random seed, executed prefix K, command rate, and observation timestamp.
    """
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(episode_id)):
        raise ValueError('episode_id must be a simple directory name')
    if not isinstance(decision_id, int) or decision_id < 0:
        raise ValueError('decision_id must be a nonnegative integer')
    data = response.get('data', response)
    trace = data.get('candidate_trace')
    if trace is None:
        raise ValueError('Enable verifier and return_candidate_trace in the patched model')
    candidates = np.asarray(data['all_candidates'])
    if candidates.ndim != 4 or candidates.shape[0] != 1:
        raise ValueError('This example records batch size 1 only')
    if not np.isfinite(candidates).all():
        raise ValueError('Candidate commands must be finite')
    n = candidates.shape[1]
    scores = {name: np.asarray(trace[name], dtype=np.float32)
              for name in ['goal_progress', 'prior_energy', 'joint_score']}
    if any(v.shape != (1, n) for v in scores.values()):
        raise ValueError('Candidate scores must match [batch, candidate]')
    if not all(np.isfinite(v).all() for v in scores.values()) or not np.isfinite(trace['beta']):
        raise ValueError('Candidate scores and beta must be finite')
    idx = int(np.asarray(trace['selected_index'])[0])
    if not 0 <= idx < n:
        raise ValueError('Selected index is outside the candidate set')
    selected = np.asarray(data['normalized_actions'])
    if selected.shape != candidates[:, 0].shape or not np.allclose(selected[0], candidates[0, idx]):
        raise ValueError('Selected action and candidate index disagree')
    if idx != int(np.argmax(scores['joint_score'][0])):
        raise ValueError('Selected candidate is not the first maximum of the recorded score')
    if not np.allclose(scores['joint_score'], scores['goal_progress'] -
                       float(trace['beta']) * scores['prior_energy'],
                       rtol=1e-5, atol=1e-6, equal_nan=True):
        raise ValueError('Joint scores do not match V - beta E')
    record = {'schema_version': 2, 'episode_id': str(episode_id),
              'decision_id': decision_id, 'recorded_unix_time': time.time(),
              'instructions': request.get('instructions'),
              'subgoal_index': data.get('subgoal_index'),
              'selected_index': idx, 'beta': float(trace['beta']),
              'finite_scores': all(bool(np.isfinite(v).all()) for v in scores.values()),
              'metadata': metadata,
              'outcome': None,
              'outcome_note': 'Not observed by this recorder; attach actual execution or replay outcomes separately.'}
    extra_arrays = {}
    if 'prior_group_energy' in trace:
        groups = trace['command_groups']
        components = np.asarray(trace['prior_group_energy'])
        weights = np.asarray(trace['command_group_weights'], dtype=float)
        if (components.shape != (1, n, len(groups)) or not np.isfinite(components).all()
            or weights.shape != (len(groups),) or not np.isfinite(weights).all()
            or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError('Invalid group residuals or weights')
        if not np.allclose((components * (weights / weights.sum())).sum(-1),
                           scores['prior_energy'], rtol=1e-5, atol=1e-6):
            raise ValueError('Group residuals do not reproduce prior energy')
        extra_arrays['prior_group_energy'] = components
    record['execution_contract'] = data.get('execution_contract')
    record['prior_type'] = trace.get('prior_type', 'joint')
    record['command_groups'] = trace.get('command_groups')
    record['command_group_weights'] = trace.get('command_group_weights')
    record['visual_target_mode'] = trace.get('visual_target_mode', 'legacy_shift')
    record['candidate_identity'] = 'normalized_only'
    if controller_commands is not None:
        canonical = np.asarray(controller_commands)
        if canonical.shape != candidates.shape or not np.isfinite(canonical).all():
            raise ValueError('Controller command bank must match finite [B,N,H,D] candidates')
        required = {'source_normalization_id', 'controller_id', 'decoder_id',
                    'coordinate_order', 'center', 'scale'}
        if not isinstance(command_normalization, dict) or not required <= command_normalization.keys():
            raise ValueError('Record normalization, controller/decoder identity, and coordinate order')
        if (len(command_normalization['coordinate_order']) != candidates.shape[-1]
            or len(set(command_normalization['coordinate_order'])) != candidates.shape[-1]):
            raise ValueError('Coordinate order must name each action coordinate exactly once')
        center = np.asarray(command_normalization['center'])
        scale = np.asarray(command_normalization['scale'])
        if (center.shape != (candidates.shape[-1],) or scale.shape != center.shape
            or not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError('Invalid affine normalization')
        if not np.allclose(candidates * scale + center, canonical, rtol=1e-5, atol=1e-6):
            raise ValueError('Normalized candidates do not reproduce the controller commands')
        extra_arrays['controller_commands'] = canonical
        record['command_normalization'] = command_normalization
        record['candidate_identity'] = 'controller_commands_with_affine_map'
    serialized = json.dumps(record, ensure_ascii=False, indent=2)
    destination = Path(root) / str(episode_id) / f'decision_{decision_id:06d}'
    destination.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(destination / 'decision.npz',
        candidates=candidates, selected_actions=selected,
        state=np.asarray(request.get('state', [])),
        selected_index=np.asarray(trace['selected_index']),
        current_latent=np.asarray(trace['current_latent']),
        goal_latent=np.asarray(trace['goal_latent']), **extra_arrays, **scores)
    Image.fromarray(np.asarray(trace['observation_rgb'][0],dtype=np.uint8)).save(destination/'observation.png')
    Image.fromarray(np.asarray(trace['goal_rgb'][0],dtype=np.uint8)).save(destination/'goal.png')
    (destination/'record.json').write_text(serialized+'\n',encoding='utf-8')
    return destination
