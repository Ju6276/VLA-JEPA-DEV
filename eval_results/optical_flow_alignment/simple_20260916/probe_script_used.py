#!/usr/bin/env python3
"""No-training RAFT alignment probe on cached, frozen JEPA spatial grids.

Uses test episodes only. No future image, goal labels, or action model is used.
The shuffled control pairs each sample with the same anchor/history lag from
another episode. All methods use the identical intersection validity mask.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

# Initialize PyTorch before decord loads its bundled native libraries.
import torch
import cv2
import decord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


def sample_grid(flow, height, width):
    """Current -> source displacement in input pixels, sampled at cell centers."""
    image_h, image_w = flow.shape[-2:]
    small = F.adaptive_avg_pool2d(flow, (height, width))
    yy, xx = torch.meshgrid(torch.arange(height, device=flow.device),
                            torch.arange(width, device=flow.device), indexing='ij')
    x = xx[None] + small[:, 0] * width / image_w
    y = yy[None] + small[:, 1] * height / image_h
    valid = (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
    grid = torch.stack((2 * (x + .5) / width - 1, 2 * (y + .5) / height - 1), -1)
    return grid, valid


def warp(source, flow):
    grid, valid = sample_grid(flow, *source.shape[-2:])
    return F.grid_sample(source, grid, align_corners=False, padding_mode='border'), valid


def consistent_mask(backward, forward):
    sampled, bounds = warp(forward, backward)
    # Standard relative + absolute forward/backward cycle threshold (pixels).
    residual = (backward + sampled).square().sum(1)
    scale = backward.square().sum(1) + sampled.square().sum(1)
    return bounds & (residual <= .01 * scale + .5)


def geometry_check():
    image = torch.arange(100.).reshape(1, 1, 10, 10)
    zero = torch.zeros(1, 2, 10, 10)
    assert torch.allclose(warp(image, zero)[0], image, atol=1e-5)
    shift = zero.clone(); shift[:, 0] = 1
    shifted, valid = warp(image, shift)
    assert torch.allclose(shifted[..., :-1], image[..., 1:], atol=1e-5)
    assert not valid[..., -1].any()
    mask = consistent_mask(shift, -shift)
    assert mask[..., :-1].all() and not mask[..., -1].any()
    coarse = torch.arange(25.).reshape(1, 1, 5, 5)
    shift[:, 0] = 2
    assert torch.allclose(warp(coarse, shift)[0][..., :-1], coarse[..., 1:], atol=1e-5)


def main(args):
    geometry_check()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    cache = torch.load(args.features, map_location='cpu', weights_only=True, mmap=True)
    metadata, split = cache['metadata'], cache['splits']['test']
    by_id = {row['sample_id']: row for row in metadata['samples']['test']}
    episodes = sorted(set(split['episode_ids']))[:args.episodes]
    assert len(episodes) == args.episodes
    dataset = Path(metadata['dataset_path'])
    info = json.loads((dataset / 'meta/info.json').read_text())
    modality = json.loads((dataset / 'meta/modality.json').read_text())
    camera = next(iter(modality['video'].values()))['original_key']
    pairs = []
    for episode in episodes:
        indices = sorted([i for i, e in enumerate(split['episode_ids']) if e == episode],
                         key=lambda i: by_id[split['sample_ids'][i]]['frame_index'])
        assert len(indices) >= args.anchors
        selected = np.linspace(0, len(indices) - 1, args.anchors).round().astype(int)
        row0 = by_id[split['sample_ids'][indices[0]]]
        ep = row0['episode_index']
        video = dataset / info['video_path'].format(episode_index=ep,
            episode_chunk=ep // info.get('chunks_size', 1000), video_key=camera)
        reader = decord.VideoReader(str(video), num_threads=1)
        timestamps = np.asarray(reader.get_frame_timestamp(range(len(reader))))[:, 0]
        for anchor_rank, selected_index in enumerate(selected):
            index = indices[selected_index]
            row = by_id[split['sample_ids'][index]]
            requests = [row['timestamp']] + [row['timestamp'] - age for age in row['history_ages']]
            decoded = np.abs(timestamps[:, None] - np.asarray(requests)[None]).argmin(0)
            assert np.max(np.abs(timestamps[decoded] - requests)) < .025
            images = [cv2.resize(im, (384, 384), interpolation=cv2.INTER_LINEAR)
                      for im in reader.get_batch(decoded).asnumpy()]
            for h, age in enumerate(row['history_ages']):
                assert row['history_valid'][h]
                pairs.append(dict(episode=ep, anchor_rank=anchor_rank, history_slot=h,
                    age=float(age), frame=row['frame_index'], sample_id=row['sample_id'],
                    current_rgb=images[0], history_rgb=images[h + 1],
                    current=split['current'][index].float().T.reshape(1, -1, 8, 8),
                    history=split['history'][index, h].float().T.reshape(1, -1, 8, 8)))
    print(f'Decoded {len(pairs)} pairs', flush=True)
    weights = Raft_Large_Weights.DEFAULT
    started = time.time()
    if args.flow_cache is not None:
        flow_cache = torch.load(args.flow_cache, map_location='cpu', weights_only=True)
        assert len(flow_cache['records']) == len(pairs)
        for i, pair in enumerate(pairs):
            saved = flow_cache['records'][i]
            assert (saved['sample_id'], saved['history_slot']) == (pair['sample_id'], pair['history_slot'])
            pair['flow'] = flow_cache['flows'][i:i+1].float()
            pair['valid'] = flow_cache['valid'][i:i+1]
    else:
        model = raft_large(weights=weights).cuda().eval().requires_grad_(False)
        print('RAFT loaded', flush=True)
        transform = weights.transforms()
        with torch.inference_mode():
            for i, pair in enumerate(pairs):
                current = torch.from_numpy(pair['current_rgb'].copy()).permute(2, 0, 1)[None].cuda()
                history = torch.from_numpy(pair['history_rgb'].copy()).permute(2, 0, 1)[None].cuda()
                a, b = transform(torch.cat((current, history)), torch.cat((history, current)))
                flows = model(a, b, num_flow_updates=12)[-1]
                pair['flow'] = flows[:1].float().cpu()
                pair['valid'] = consistent_mask(flows[:1], flows[1:]).cpu()
                if (i + 1) % 20 == 0:
                    print(f'RAFT {i + 1}/{len(pairs)}, {time.time()-started:.1f}s', flush=True)
        del model
        torch.cuda.empty_cache()
    records, visuals = [], []
    span = args.anchors * 2
    for i, pair in enumerate(pairs):
        wrong = pairs[(i + span) % len(pairs)]  # different episode, same anchor rank and lag
        assert wrong['episode'] != pair['episode'] and wrong['history_slot'] == pair['history_slot']
        warped, bounds = warp(pair['history'], pair['flow'])
        shuffled, wrong_bounds = warp(pair['history'], wrong['flow'])
        valid = (F.adaptive_avg_pool2d(pair['valid'].float()[:, None], (8, 8))[:, 0] >= .7)
        common = valid & bounds & wrong_bounds
        current_rgb = torch.from_numpy(pair['current_rgb'].copy()).permute(2, 0, 1)[None].float()/255
        history_rgb = torch.from_numpy(pair['history_rgb'].copy()).permute(2, 0, 1)[None].float()/255
        warped_rgb, rgb_bounds = warp(history_rgb, pair['flow'])
        shuffled_rgb, wrong_rgb_bounds = warp(history_rgb, wrong['flow'])
        common_rgb = pair['valid'] & rgb_bounds & wrong_rgb_bounds
        smooth = F.avg_pool2d(F.interpolate(pair['history'], scale_factor=2, mode='bilinear', align_corners=False), 2)
        smooth_rgb = F.avg_pool2d(F.interpolate(history_rgb, scale_factor=2, mode='bilinear', align_corners=False), 2)
        record = {k: pair[k] for k in ('episode', 'anchor_rank', 'history_slot', 'age', 'frame', 'sample_id')}
        record.update(shuffled_episode=wrong['episode'],
            pixel_valid_fraction=float(pair['valid'].float().mean()),
            common_pixel_fraction=float(common_rgb.float().mean()),
            common_grid_fraction=float(common.float().mean()),
            flow_magnitude_pixels=float(pair['flow'].square().sum(1).sqrt().mean()))
        maps = []
        for name, feature, rgb in [('identity', pair['history'], history_rgb),
                                    ('flow', warped, warped_rgb), ('shuffled', shuffled, shuffled_rgb),
                                    ('smoothing', smooth, smooth_rgb)]:
            cosine = 1-F.cosine_similarity(pair['current'], feature, dim=1)
            l1 = (pair['current']-feature).abs().mean(1)
            record[f'{name}_cosine'] = float(cosine[common].mean()) if common.any() else None
            record[f'{name}_l1'] = float(l1[common].mean()) if common.any() else None
            rgb_error = (current_rgb-rgb).abs().mean(1)
            record[f'{name}_rgb_l1'] = float(rgb_error[common_rgb].mean()) if common_rgb.any() else None
            if name != 'smoothing':
                maps.append(cosine[0].masked_fill(~common[0], float('nan')).numpy())
        records.append(record)
        # Preselected positions, never selected by measured improvement.
        if pair['episode'] in [pairs[0]['episode'], pairs[span]['episode']] and pair['anchor_rank'] in [4, 10, 16] and pair['history_slot']==0:
            visuals.append((record, pair['history_rgb'], pair['current_rgb'],
                warped_rgb[0].permute(1, 2, 0).numpy(), maps))
    with (output/'pairs.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    rng = np.random.default_rng(42)
    summaries = {}
    for label, rows in [('all', records), ('0.8s', [r for r in records if r['history_slot']==0]),
                         ('0.4s', [r for r in records if r['history_slot']==1])]:
        usable = [r for r in rows if r['flow_cosine'] is not None]
        means = {f'{method}_{metric}': float(np.mean([r[f'{method}_{metric}'] for r in usable]))
                 for method in ('identity', 'flow', 'shuffled', 'smoothing') for metric in ('cosine','l1','rgb_l1')}
        per_episode = []
        for episode in sorted({r['episode'] for r in usable}):
            selected = [r for r in usable if r['episode']==episode]
            per_episode.append({'episode':episode, **{key:float(np.mean([r[key] for r in selected])) for key in means}})
        differences = np.array([r['identity_cosine']-r['flow_cosine'] for r in per_episode])
        bootstrap = rng.choice(differences, (10000,len(differences)), replace=True).mean(1)
        against_controls = {}
        for control in ('shuffled', 'smoothing'):
            diff = np.array([r[f'{control}_cosine']-r['flow_cosine'] for r in per_episode])
            boot = rng.choice(diff, (10000,len(diff)), replace=True).mean(1)
            against_controls[control] = dict(improved_episodes=int((diff>0).sum()),
                cosine_improvement_ci95=np.quantile(boot,[.025,.975]).tolist())
        summaries[label] = dict(against_controls=against_controls, means=means, pairs=len(rows), usable_pairs=len(usable),
            mean_common_grid_fraction=float(np.mean([r['common_grid_fraction'] for r in rows])),
            mean_pixel_valid_fraction=float(np.mean([r['pixel_valid_fraction'] for r in rows])),
            improved_episodes=int((differences>0).sum()), total_episodes=len(differences),
            cosine_relative_improvement_percent=100*(means['identity_cosine']-means['flow_cosine'])/means['identity_cosine'],
            episode_bootstrap_cosine_improvement_ci95=np.quantile(bootstrap,[.025,.975]).tolist(),
            per_episode=per_episode)
    report = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        flow_cache=str(args.flow_cache) if args.flow_cache else None,
        smoothing_control='Bilinear 2x upsampling then 2x average pooling; no motion information. Not an exact matched-interpolation control.',
        scope='No-training correspondence diagnostic; not action success or future-goal prediction.',
        cache=str(args.features.resolve()), dataset=str(dataset),
        selection='First sorted 10 test episodes; 20 evenly spaced cached anchors each, both past lags; no result-based selection.',
        feature_space='Existing frozen V-JEPA 2.1 raw 8x8 spatial grids; bilinear backward warping.',
        preprocessing='RGB OpenCV INTER_LINEAR 384x384, matching feature extraction.',
        flow_model=str(weights), weight_url=weights.url, torchvision=torchvision.__version__,
        flow_direction='current -> history; reverse history -> current only for cycle consistency',
        validity='Cycle residual squared <= .01*(backward squared+sampled forward squared)+.5; grid cell >=70% valid; same intersection of true/shuffled in-bounds grid for all methods.',
        control='Different episode, same anchor rank and history slot. Flow is pooled to 8x8; displacements scaled with grid dimensions.',
        confidence_interval='Paired episode-cluster bootstrap, 10000 draws, seed42; exploratory, not policy-success evidence.',
        seconds=time.time()-started, summaries=summaries)
    (output/'report.json').write_text(json.dumps(report,indent=2))
    fig, axes = plt.subplots(len(visuals), 6, figsize=(18, 3*len(visuals)), squeeze=False)
    vmax = np.nanpercentile(np.concatenate([np.concatenate([m.ravel() for m in v[4]]) for v in visuals]),95)
    for row,(record,hist,cur,aligned,maps) in enumerate(visuals):
        for col,im in enumerate((hist,cur,aligned)):
            axes[row,col].imshow(im); axes[row,col].axis('off')
        for col,m in enumerate(maps,3):
            axes[row,col].imshow(m,vmin=0,vmax=vmax,cmap='magma'); axes[row,col].axis('off')
        axes[row,0].set_title(f"ep{record['episode']} frame{record['frame']} (-0.8s)")
    for col,title in enumerate(('History RGB','Current RGB','Flow-warped history','Identity cosine error','Flow cosine error','Shuffled cosine error')):
        axes[0,col].set_title(title+'\n'+axes[0,col].get_title())
    fig.suptitle('Frozen JEPA history alignment — lower error is better; blank cells excluded for ALL methods')
    fig.colorbar(axes[0,3].images[0], ax=axes[:,3:].ravel().tolist(), shrink=.4, label='Cosine distance')
    fig.savefig(output/'alignment.png',dpi=140); plt.close(fig)
    torch.save({'flows':torch.cat([p['flow'] for p in pairs]).half(),
                'valid':torch.cat([p['valid'] for p in pairs]),'records':records},output/'flows.pt')
    print(json.dumps({k:{key:val for key,val in s.items() if key!='per_episode'} for k,s in summaries.items()},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--flow-cache',type=Path,default=None)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--anchors',type=int,default=20)
    args = parser.parse_args()
    if args.episodes < 2 or args.anchors < 17:
        parser.error('Use at least 2 episodes and 17 anchors for the fixed comparison/figure selection')
    main(args)
