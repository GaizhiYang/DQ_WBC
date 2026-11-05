"""Physical-frame invariance, memory ownership and hard endpoint semantics."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_world_model_fixed8_ppo import make_fixed8_agent
from test_world_model_bptt import ToyActor, ToyCritic, ToyModel, ToyAdapter
from utils.wm_env import quat_mul, quat_apply, from_euler
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

    def test_privileged_critic_retains_memory_and_excludes_ee_cache(self):
        spec = make_fixed8_agent(self.device).wm_adapter
        state, context = spec.capture()
        features = spec.critic_features(state, context)
        self.assertEqual(features.shape, (len(state), spec.critic_feature_dim))
        self.assertEqual(spec.critic_feature_dim, 946 + spec.context_feature_dim)
        changed = spec.replace(state, ee=state[:, spec.slices["ee"]] + 100.)
        torch.testing.assert_close(spec.critic_features(changed, context), features, rtol=0, atol=0)
        # These fields can change future controller decisions or return even
        # when the current physical state is identical.
        for name in ("low_history", "low_actions", "gait", "clock", "gait_wait", "walking",
                     "foot_contacts", "goal_pos", "goal_rpy", "commands", "prev_action",
                     "held_gripper_target", "ekf_ages", "ekf_count", "closest", "highest",
                     "hold", "best_hold", "completed", "progress", "table_height"):
            changed = spec.replace(state, **{name: spec.get(state, name) + .1})
            self.assertGreater(float((spec.critic_features(changed, context) - features).abs().sum()), 0., name)
        changed_context = context.clone()
        changed_context[:, 3] = 1 - changed_context[:, 3]
        self.assertGreater(float((spec.critic_features(state, changed_context) - features).abs().sum()), 0.)
        differentiable = state.detach().requires_grad_(True)
        # A linear value probe checks that no adapter-side detach blocks its
        # dependence on history, physical motion and held targets.
        features = spec.critic_features(differentiable, context)
        weights = torch.linspace(.1, 1., features.shape[-1], device=self.device)
        gradient, = torch.autograd.grad((features * weights).sum(), differentiable)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        for name in ("robot", "object", "dof_pos", "dof_vel", "low_history", "held_ee_target",
                     "held_gripper_target", "ekf_poses", "goal_pos", "prev_action"):
            self.assertGreater(float(gradient[:, spec.slices[name]].abs().sum()), 0., name)
        self.assertEqual(float(gradient[:, spec.slices["ee"]].abs().sum()), 0.)

    def test_representation_contract_rejects_old_snapshot_schema(self):
        agent = make_fixed8_agent(self.device)
        metadata = agent.wm_adapter.metadata()
        self.assertEqual(metadata["version"], 5)
        self.assertEqual(metadata["low_feature_dim"], 81)
        self.assertEqual(metadata["low_action_feature_dim"], 22)
        old = dict(metadata, version=4, schema="dq-sgs-ekf-fixed8-physical64-v4")
        with self.assertRaisesRegex(ValueError, "schema/task"):
            agent._check_metadata(old)

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
