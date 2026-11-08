"""Physical-frame invariance, memory ownership and hard endpoint semantics."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_world_model_fixed8_ppo import make_fixed8_agent
from test_world_model_bptt import ToyActor, ToyCritic, ToyModel, ToyAdapter
from utils.wm_env import quat_mul, quat_apply, quat_inverse_apply, from_euler, from_rotvec
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer, imagination_objective
from utils.world_model.representation import rotation_6d, rotation_vector


class AbsorbingAdapter(ToyAdapter):
    hard_absorbing = True

    def task_outcome(self, previous, state, action, context):
        ended = state[:, 1] >= 1
        zeros = torch.zeros_like(ended)
        return dict(completed=ended & (context[:, 0] == 0),
                    failed=ended & (context[:, 0] == 1), timeout=ended & (context[:, 0] == 2))


class AdapterChecks:
    device = "cpu"

    def mixed_endpoints(self, spec):
        state, context = spec.capture()
        kinds = torch.arange(len(state), device=state.device) % 4
        root = spec.get(state, "robot").clone()
        angles = state.new_zeros(len(state), 3)
        angles[kinds == 2, 0] = 1.  # hard failure mixed with live/success/timeout
        root[:, 3:7] = from_euler(angles)
        root[:, 7:] = 0
        context = context.clone()
        context[:, 1] = 0
        state = spec.replace(state, robot=root, dof_vel=torch.zeros_like(spec.get(state, "dof_vel")),
                             table_height=state.new_full((len(state), 1), .3))
        _, obj, ee, base, _, _, arm = spec._geometry(state)
        obj = obj.clone()
        obj[:, :3] = ee[:, :3] + state.new_tensor([.02, 0., 0.])
        obj[:, 7:] = 0
        actions = state.new_zeros(len(state), 9)
        actions[:, 6] = (kinds != 1).to(state.dtype)
        hold = (kinds == 1).to(state.dtype)[:, None] * (spec.hold_steps - 1)
        completed = state.new_zeros(len(state), 1)
        completed[-3] = 1  # an already completed row must remain unchanged
        previous = spec.replace(state, object=obj, goal_pos=quat_inverse_apply(base, ee[:, :3]-arm),
                                prev_action=actions, hold=hold, best_hold=hold,
                                completed=completed,
                                progress=torch.where(kinds == 3, spec.max_steps - 1, 20)[:, None])
        endpoint = spec.replace(previous, dof_pos=spec.get(previous, "dof_pos") + .001)
        active = torch.arange(len(state), device=state.device) < len(state) // 2
        return previous, endpoint, actions, context, active

    def test_local_reward_outcome_shares_one_geometry_and_preserves_gradients(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        previous, endpoint, actions, context, _ = self.mixed_endpoints(spec)

        def evaluate(shared):
            state = endpoint.detach().requires_grad_(True)
            action = actions.detach().requires_grad_(True)
            if shared:
                reward, outcome = spec.reward_and_outcome(previous, state, action, context)
            else:
                reward = spec.reward(previous, state, action, context)
                outcome = spec.task_outcome(previous, state, action, context)
            loss = reward.sum() + outcome["distance"].sum() + outcome["height"].sum()
            gradient = torch.autograd.grad(loss, (state, action))
            return reward, outcome, gradient

        expected = evaluate(False)
        with patch.object(spec, "_geometry", wraps=spec._geometry) as geometry:
            actual = evaluate(True)
            self.assertEqual(geometry.call_count, 1)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        for name in expected[1]:
            torch.testing.assert_close(actual[1][name], expected[1][name], rtol=0, atol=0)
        for actual_gradient, expected_gradient in zip(actual[2], expected[2]):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-6, atol=2e-6)

    def test_finish_high_shared_task_preserves_mixed_absorption_history_and_gradients(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        initial, endpoint, actions, context, active = self.mixed_endpoints(spec)

        def evaluate(shared):
            previous = initial.detach().requires_grad_(True)
            state = endpoint.detach().requires_grad_(True)
            action = actions.detach().requires_grad_(True)
            if shared:
                result, reward, outcome = spec.finish_high(previous, state, action, context,
                                                          return_task=True, active=active)
            else:
                result = spec.finish_high(previous, state, action, context)
                result = torch.where(active[:, None], result, previous)
                reward = spec.reward(previous, result, action, context)
                outcome = spec.task_outcome(previous, result, action, context)
            weights = torch.linspace(.01, .1, result.shape[-1], device=self.device)
            loss = (result * weights).sum() + reward.sum() + outcome["distance"].sum()
            gradients = torch.autograd.grad(loss, (previous, state, action))
            return result, reward, outcome, gradients

        expected = evaluate(False)
        self.assertTrue(bool(expected[2]["success"].any()))
        self.assertTrue(bool(expected[2]["failed"].any()))
        self.assertTrue(bool(expected[2]["timeout"].any()))
        with patch.object(spec, "_geometry", wraps=spec._geometry) as geometry:
            actual = evaluate(True)
            self.assertEqual(geometry.call_count, 1)
        for actual_tensor, expected_tensor in zip(actual[:2], expected[:2]):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        torch.testing.assert_close(actual[0][~active], initial[~active], rtol=0, atol=0)
        for name in expected[2]:
            torch.testing.assert_close(actual[2][name], expected[2][name], rtol=0, atol=0)
        for actual_gradient, expected_gradient in zip(actual[3], expected[3]):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-6, atol=2e-6)

    def test_imagination_shared_endpoint_matches_separate_path_and_actor_gradient(self):
        agent = make_fixed8_agent(self.device)
        trainer, spec = agent.wm_imagination, agent.wm_adapter
        state, _, _, context, _ = self.mixed_endpoints(spec)
        # The fallback reproduces the previous step -> absorb -> reward ->
        # task_outcome path, including terminal observations and tail values.
        with patch.object(trainer.model, "step_with_task", None), \
                patch.object(spec, "reward_and_outcome", None):
            expected = trainer.imagine(state, context)
            expected_grad = torch.autograd.grad(imagination_objective(expected, trainer.cfg.gamma),
                                               tuple(trainer.actor.parameters()), allow_unused=True)
        actual = trainer.imagine(state, context)
        actual_grad = torch.autograd.grad(imagination_objective(actual, trainer.cfg.gamma),
                                         tuple(trainer.actor.parameters()), allow_unused=True)
        for name in ("states", "actions", "rewards", "termination", "survival", "timeouts",
                     "timeout_values", "tail_values"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)
        for actual_gradient, expected_gradient in zip(actual_grad, expected_grad):
            if expected_gradient is None:
                self.assertIsNone(actual_gradient)
            else:
                torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-6)

    def test_rot6d_state_and_command_features_ignore_quaternion_sign(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, _ = spec.capture()
        opposite = {}
        for name in ("robot", "object", "table"):
            body = spec.get(state, name)
            opposite[name] = torch.cat((body[:, :3], -body[:, 3:7], body[:, 7:]), -1)
        changed = spec.replace(state, **opposite)
        features = spec.low_features(state)
        self.assertEqual(features.shape[-1], 81)
        torch.testing.assert_close(spec.low_features(changed), features, rtol=0, atol=0)
        action = torch.cat((spec.get(state, "low_actions")[:, :12],
                            spec.get(state, "held_ee_target"),
                            spec.get(state, "held_gripper_target")), -1)
        action_other = action.clone()
        action_other[:, 15:19].neg_()
        self.assertEqual(action.shape[-1], 20)
        self.assertEqual(spec.low_action_features(state, action).shape[-1], 22)
        torch.testing.assert_close(spec.low_action_features(changed, action_other),
                                   spec.low_action_features(state, action), rtol=0, atol=0)

    def test_rotation_log_is_geodesic_and_has_finite_identity_gradients(self):
        # Known axis-angle examples distinguish log from the previous chord
        # surrogate, including the shortest rotation across quaternion signs.
        angles = torch.tensor([0., 1e-8, .4, 2.6, 3.5], device=self.device, dtype=torch.float64)
        q = torch.zeros(len(angles), 4, device=self.device, dtype=torch.float64)
        q[:, 0], q[:, 3] = (angles / 2).sin(), (angles / 2).cos()
        q.requires_grad_(True)
        expected = torch.where(angles > torch.pi, angles - 2 * torch.pi, angles)
        output = rotation_vector(q)
        torch.testing.assert_close(output[:, 0], expected, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]), rtol=0, atol=0)
        torch.testing.assert_close(rotation_vector(-q), output, rtol=0, atol=0)
        gradient, = torch.autograd.grad(output.sum(), q)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        torch.testing.assert_close(gradient[0, :3], torch.full_like(gradient[0, :3], 2.))
        # rot6d uses row-major first two columns, matching the reproduction.
        z90 = torch.tensor([[0., 0., 2**-.5, 2**-.5]], device=self.device)
        torch.testing.assert_close(rotation_6d(z90), z90.new_tensor([[0., -1., 1., 0., 0., 0.]]),
                                   atol=2e-7, rtol=0)

    def test_compact_critic_retains_physics_and_task_without_raw_memory_shortcuts(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, context = spec.capture()
        features = spec.critic_features(state, context)
        self.assertEqual(features.shape, (len(state), spec.critic_feature_dim))
        self.assertEqual(spec.critic_feature_dim, 129 + spec.context_feature_dim)
        excluded = ("prev_action", "low_history", "low_actions", "ekf_poses", "ekf_ages",
                    "ekf_count", "ee", "held_ee_target", "last_dof_vel", "clock")
        for name in excluded:
            changed = spec.replace(state, **{name: state[:, spec.slices[name]] + 100.})
            torch.testing.assert_close(spec.critic_features(changed, context), features, rtol=0, atol=0)
        # The rollout and both policies still keep their histories; this
        # compact value approximation retains physical goals and task phase.
        for name in ("robot", "object", "dof_pos", "dof_vel", "gait", "gait_wait", "walking",
                     "foot_contacts", "goal_pos", "goal_rpy", "commands",
                     "held_gripper_target", "closest", "highest",
                     "hold", "best_hold", "completed", "progress", "table_height"):
            changed = spec.replace(state, **{name: spec.get(state, name) + .1})
            self.assertGreater(float((spec.critic_features(changed, context) - features).abs().sum()), 0., name)
        changed_context = context.clone()
        changed_context[:, 3] = 1 - changed_context[:, 3]
        self.assertGreater(float((spec.critic_features(state, changed_context) - features).abs().sum()), 0.)
        differentiable = state.detach().requires_grad_(True)
        # Retained physical/task gradients remain available to BPTT, while
        # value has no direct input gradient on the discarded history fields.
        features = spec.critic_features(differentiable, context)
        weights = torch.linspace(.1, 1., features.shape[-1], device=self.device)
        gradient, = torch.autograd.grad((features * weights).sum(), differentiable)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        for name in ("robot", "object", "dof_pos", "dof_vel", "held_gripper_target",
                     "goal_pos", "goal_rpy", "commands", "hold", "progress"):
            self.assertGreater(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)
        for name in excluded:
            self.assertEqual(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)

    def test_compact_critic_goal_rotation_is_periodic(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, context = spec.capture()
        changed = spec.replace(state, goal_rpy=spec.get(state, "goal_rpy") + 2 * torch.pi)
        torch.testing.assert_close(spec.critic_features(changed, context),
                                   spec.critic_features(state, context), rtol=0, atol=8e-7)

    def test_physical_error_vector_preserves_scales_without_group_weights(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, _ = spec.capture()
        changes = {}
        for name in ("robot", "object"):
            old = spec.get(state, name)
            increment = from_rotvec(old.new_full((len(old), 3), .1))
            changes[name] = torch.cat((old[:, :3] + .05,
                                       quat_mul(old[:, 3:7], increment),
                                       old[:, 7:10] + .5, old[:, 10:13] + 1.), -1)
        changes["dof_pos"] = spec.get(state, "dof_pos") + .1
        changes["dof_vel"] = spec.get(state, "dof_vel") + 2.
        predicted = spec.replace(state, **changes).detach().requires_grad_(True)
        errors = spec.physical_error_vector(predicted, state)
        self.assertEqual(errors.shape, (len(state), 62))
        torch.testing.assert_close(errors, torch.ones_like(errors), rtol=2e-5, atol=2e-5)
        expected_legacy = torch.cat([g * (62 / (5 * g.shape[-1]))**.5
                                    for g in errors.split((12, 12, 24, 12, 2), -1)], -1)
        torch.testing.assert_close(spec.error_vector(predicted, state), expected_legacy, rtol=0, atol=0)
        gradient, = torch.autograd.grad(errors.square().sum(), predicted)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        for name in ("robot", "object", "dof_pos", "dof_vel"):
            self.assertGreater(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)
        for name in ("table", "ee", "low_history", "prev_action"):
            self.assertEqual(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)

    def test_representation_contract_rejects_old_snapshot_schema(self):
        agent = make_fixed8_agent(self.device)
        metadata = agent.wm_adapter.metadata()
        self.assertEqual(metadata["version"], 5)
        self.assertEqual(metadata["low_feature_dim"], 81)
        self.assertEqual(metadata["low_action_feature_dim"], 22)
        old = dict(metadata, version=4, schema="dq-sgs-ekf-fixed8-physical64-v4")
        with self.assertRaisesRegex(ValueError, "schema/task"):
            agent._check_metadata(old)
        old_critic = dict(metadata, critic_features="canonical_physics_context_and_explicit_memory_fixed_scales_v1")
        with self.assertRaisesRegex(ValueError, "schema/task"):
            agent._check_metadata(old_critic)

    def test_canonical_features_and_residuals_rotate_consistently(self):
        agent = make_fixed8_agent(self.device)
        spec = agent.wm_adapter
        state, context = spec.capture()
        angle = state.new_tensor([0., 0., .7]).expand(len(state), -1)
        rotation = from_euler(angle)

        def rotate(s):
            fields = {}
            for name in ("robot", "object", "table"):
                b = spec.get(s, name)
                fields[name] = torch.cat((quat_apply(rotation, b[:, :3]),
                    quat_mul(rotation, b[:, 3:7]), quat_apply(rotation, b[:, 7:10]),
                    quat_apply(rotation, b[:, 10:13])), -1)
            return spec.replace(s, **fields)

        rotated = rotate(state)
        torch.testing.assert_close(spec.low_features(rotated), spec.low_features(state), atol=2e-6, rtol=2e-5)
        delta = torch.randn(len(state), 62, device=self.device) * .1
        action = state.new_zeros(len(state), 20)
        duration = state.new_full((len(state), 1), .02)
        next_state = spec.integrate_low(state, delta, action, context, duration)
        rotated_next = spec.integrate_low(rotated, delta, action, context, duration)
        expected = rotate(next_state)
        for name in ("robot", "object", "table", "dof_pos", "dof_vel"):
            torch.testing.assert_close(spec.get(rotated_next, name), spec.get(expected, name), atol=2e-6, rtol=2e-5)
        # Reward curriculum and initial heading must not leak into physics.
        changed = context.clone()
        changed[:, 1] += 1.4
        changed[:, 3] = 1 - changed[:, 3]
        torch.testing.assert_close(spec.normalize_context(context), spec.normalize_context(changed), rtol=0, atol=0)

    def test_no_history_table_or_independent_ee_regression(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, context = spec.capture()
        changed = spec.replace(state, low_history=spec.get(state, "low_history") + 3,
                               table=spec.get(state, "table") + .1,
                               ee=state[:, spec.slices["ee"]] + 10,
                               gait=spec.get(state, "gait") + .2)
        self.assertEqual((spec.physical_dim, spec.delta_dim), (64, 62))
        self.assertEqual(spec.error_vector(changed, state).shape[-1], 62)
        self.assertLess(float(spec.error_vector(changed, state).abs().max()), 1e-6)
        torch.testing.assert_close(spec.get(changed, "ee"), spec.get(state, "ee"), rtol=0, atol=0)
        history_only = spec.replace(state, low_history=spec.get(state, "low_history") + 1)
        torch.testing.assert_close(spec.low_features(state), spec.low_features(history_only), rtol=0, atol=0)

    def test_low_feedback_retains_input_and_history_gradients(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        initial, context = spec.capture()
        state = initial.detach().requires_grad_(True)
        after, action = spec.low_control(state, context)
        loss = action[:, :12].square().sum()
        gradient, = torch.autograd.grad(loss, state)
        self.assertGreater(float(gradient[:, spec.slices["low_history"]].abs().sum()), 0.)
        self.assertGreater(float(gradient[:, spec.slices["dof_pos"]].abs().sum()), 0.)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in spec.low_actor.parameters()))
        current, _ = spec.low_observation(state, context)
        history = spec.get(after, "low_history").reshape(-1, 10, 71)
        torch.testing.assert_close(history[:, -1], current[:, :71])
        torch.testing.assert_close(history[:, :-1], spec.get(state, "low_history").reshape(-1, 10, 71)[:, 1:])

    def test_success_failure_and_timeout_each_absorb_without_losing_final_reward(self):
        actor, critic = ToyActor().to(self.device), ToyCritic().to(self.device)
        trainer = ImaginationTrainer(ToyModel().to(self.device), AbsorbingAdapter(reward_constant=2.),
            actor, critic, ImaginationConfig(horizon=4, gamma=1., dt=.16))
        state = torch.tensor([[1., 0.]] * 3, device=self.device)
        context = torch.arange(3, device=self.device)[:, None]
        rollout = trainer.imagine(state, context)
        torch.testing.assert_close(rollout.rewards[0], torch.full((3,), 2., device=self.device))
        self.assertEqual(float(rollout.rewards[1:].abs().max()), 0.)
        self.assertEqual(float(rollout.survival[1:].abs().max()), 0.)
        torch.testing.assert_close(rollout.states[1:], rollout.states[1:2].expand_as(rollout.states[1:]))
        self.assertEqual(float(imagination_objective(rollout, 1.)), 2.)


class CPUAdapterTests(AdapterChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAAdapterTests(AdapterChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
