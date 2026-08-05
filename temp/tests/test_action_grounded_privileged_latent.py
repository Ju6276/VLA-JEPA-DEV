"""Focused tests for the one-stage action-grounded latent path."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from starVLA.model.modules.world_model.delta_action_decoder import (
    MultiStepDeltaActionDecoder,
)
from starVLA.model.modules.world_model.privileged_latent import (
    SharedWorldDecoder,
    StudentCurrentAdapter,
    StudentPredictor,
    TeacherEncoder,
    apply_knowledge_insulation,
)


def _has_nonzero_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in module.parameters()
    )


class ActionGroundedPrivilegedLatentTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_teacher_uses_future_visual_endpoint(self) -> None:
        teacher = TeacherEncoder(
            vj_dim=16,
            state_dim=6,
            latent_dim=8,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        ).eval()
        u_0 = torch.randn(2, 16)
        u_t = torch.randn(2, 16)
        state_0 = torch.randn(2, 6)
        state_t = torch.randn(2, 6)

        with torch.no_grad():
            z_first = teacher(u_0, u_t, state_0, state_t)
            z_changed_future = teacher(
                u_0,
                u_t + torch.randn_like(u_t),
                state_0,
                state_t,
            )

        self.assertEqual(tuple(z_first.shape), (2, 8))
        self.assertFalse(torch.allclose(z_first, z_changed_future))

    def test_delta_decoder_outputs_full_chunk_and_backpropagates(self) -> None:
        decoder = MultiStepDeltaActionDecoder(
            latent_dim=16,
            action_dim=7,
            action_horizon=5,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
        )
        delta = torch.randn(2, 16, requires_grad=True)
        actions = decoder(delta)
        actions.square().mean().backward()

        self.assertEqual(tuple(actions.shape), (2, 5, 7))
        self.assertIsNotNone(delta.grad)
        self.assertGreater(torch.count_nonzero(delta.grad).item(), 0)
        self.assertTrue(_has_nonzero_gradient(decoder))

    def test_knowledge_insulation_stops_only_dynamics_gradients(self) -> None:
        future = torch.randn(2, 16, requires_grad=True)
        latent_action = torch.randn(2, 8, requires_grad=True)
        future_projection = torch.nn.Linear(16, 12)
        action_projection = torch.nn.Linear(8, 12)

        insulated_future, insulated_action = apply_knowledge_insulation(
            future,
            latent_action,
            enabled=True,
        )
        loss = (
            future_projection(insulated_future).square().mean()
            + action_projection(insulated_action).square().mean()
        )
        loss.backward()

        self.assertIsNone(future.grad)
        self.assertIsNone(latent_action.grad)
        self.assertTrue(_has_nonzero_gradient(future_projection))
        self.assertTrue(_has_nonzero_gradient(action_projection))

        passthrough_future, passthrough_action = apply_knowledge_insulation(
            future,
            latent_action,
            enabled=False,
        )
        self.assertIs(passthrough_future, future)
        self.assertIs(passthrough_action, latent_action)

    def test_joint_one_stage_objective_has_expected_gradient_routes(self) -> None:
        batch_size = 2
        qwen_dim = 12
        vj_dim = 16
        state_dim = 6
        latent_dim = 8
        horizon = 5
        action_dim = 7

        current_adapter = StudentCurrentAdapter(qwen_dim, vj_dim, hidden_dim=16)
        student = StudentPredictor(qwen_dim, state_dim, latent_dim, hidden_dim=16)
        teacher = TeacherEncoder(
            vj_dim,
            state_dim,
            latent_dim,
            hidden_dim=16,
            num_layers=1,
            num_heads=4,
        )
        world = SharedWorldDecoder(
            vj_dim,
            latent_dim,
            context_dim=16,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            grid_hw=(1, 1),
        )
        delta_decoder = MultiStepDeltaActionDecoder(
            vj_dim,
            action_dim,
            horizon,
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
        )

        current_context = torch.randn(batch_size, qwen_dim)
        u_0_target = torch.randn(batch_size, vj_dim)
        u_t_target = torch.randn(batch_size, vj_dim)
        state_0 = torch.randn(batch_size, state_dim)
        state_t = torch.randn(batch_size, state_dim)
        action_target = torch.randn(batch_size, horizon, action_dim)

        u_student = current_adapter(current_context)
        z_student = student(current_context, state_0)
        u_student_hat_t = world(u_student, z_student)
        z_teacher = teacher(u_0_target, u_t_target, state_0, state_t)
        u_teacher_hat_t = world(u_0_target, z_teacher)
        action_from_delta = delta_decoder(
            u_student_hat_t - u_student,
            state_0,
        )

        loss = (
            F.smooth_l1_loss(u_student, u_0_target)
            + F.smooth_l1_loss(u_student_hat_t, u_t_target)
            + F.smooth_l1_loss(u_teacher_hat_t, u_t_target)
            + F.smooth_l1_loss(z_student, z_teacher.detach())
            + F.smooth_l1_loss(action_from_delta, action_target)
        )
        # LaWAM's AdaLN gates intentionally start at zero.  The first update
        # opens those gates; the second verifies that the teacher then receives
        # the expected world-model gradient through the shared decoder.
        modules = [current_adapter, student, teacher, world, delta_decoder]
        optimizer = torch.optim.SGD(
            [parameter for module in modules for parameter in module.parameters()],
            lr=1e-2,
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        u_student = current_adapter(current_context)
        z_student = student(current_context, state_0)
        u_student_hat_t = world(u_student, z_student)
        z_teacher = teacher(u_0_target, u_t_target, state_0, state_t)
        u_teacher_hat_t = world(u_0_target, z_teacher)
        action_from_delta = delta_decoder(u_student_hat_t - u_student, state_0)
        second_loss = (
            F.smooth_l1_loss(u_student, u_0_target)
            + F.smooth_l1_loss(u_student_hat_t, u_t_target)
            + F.smooth_l1_loss(u_teacher_hat_t, u_t_target)
            + F.smooth_l1_loss(z_student, z_teacher.detach())
            + F.smooth_l1_loss(action_from_delta, action_target)
        )
        second_loss.backward()

        self.assertTrue(_has_nonzero_gradient(current_adapter))
        self.assertTrue(_has_nonzero_gradient(student))
        self.assertTrue(_has_nonzero_gradient(teacher))
        self.assertTrue(_has_nonzero_gradient(world))
        self.assertTrue(_has_nonzero_gradient(delta_decoder))


if __name__ == "__main__":
    unittest.main()
