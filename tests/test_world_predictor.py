"""Behavioral checks for the predictor's optional attention modes."""

import pytest
import torch

from starVLA.model.modules.world_model.vj2_predictor import VisionTransformerPredictorAC


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("extrinsics", [True, False])
def test_predictor_attention_modes_and_backward(causal, extrinsics):
    torch.manual_seed(17)
    model = VisionTransformerPredictorAC(
        img_size=32, num_frames=2, tubelet_size=1, embed_dim=6,
        predictor_embed_dim=64, depth=1, num_heads=4, action_embed_dim=8,
        num_add_tokens=1, is_frame_causal=causal, use_extrinsics=extrinsics,
    )
    context = torch.randn(2, 8, 6)
    actions = torch.randn(2, 2, 8)
    cameras = torch.randn(2, 2, 7) if extrinsics else None
    output = model(context, actions, cameras)
    assert output.shape == context.shape
    output.square().mean().backward()
    assert torch.isfinite(model.predictor_proj.weight.grad).all()

    if causal:
        # Future action/camera changes must not alter an earlier frame's output.
        changed_actions = actions.clone()
        changed_actions[:, 1] += 100
        changed_cameras = cameras.clone() if cameras is not None else None
        if changed_cameras is not None:
            changed_cameras[:, 1] -= 100
        changed = model(context, changed_actions, changed_cameras)
        torch.testing.assert_close(changed[:, :4], output[:, :4])
