"""Physical-state and reset-context contracts using real B1Z1 field names."""

import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_v2.state import StateCapture
from supermimic_v2.context import Context


class StateChecks:
    device = 'cpu'

    def tensor(self, value):
        return torch.tensor(value, device=self.device, dtype=torch.float32)

    def environment(self):
        robot = torch.zeros(3, 13, device=self.device)
        obj = torch.zeros_like(robot)
        robot[:, :3] = self.tensor([[1., 2., .6], [3., 4., .7], [5., 6., .8]])
        obj[:, :3] = self.tensor([[7., 8., .2], [9., 10., .3], [11., 12., .4]])
        robot[:, 6] = 1
        obj[:, 6] = 1
        obj[:, 7:13] = torch.arange(18, device=self.device).reshape(3, 6)
        return SimpleNamespace(
            num_envs=3, num_actions=9, device=self.device,
            _robot_root_states=robot, _cube_root_states=obj,
            dof_pos_gripper=self.tensor([[.1, .3], [.2, .4], [.3, .5]]),
            dof_vel_gripper=self.tensor([[1., 3.], [2., 4.], [3., 5.]]),
            progress_buf=torch.zeros(3, dtype=torch.long, device=self.device),
            max_episode_length=20,
            table_heights=self.tensor([.1, .2, .3]), init_height=self.tensor([.05, .1, .15]),
            lifted_success_threshold=.2, object_mass=self.tensor([[.5], [1.], [2.]]),
            sim_params=SimpleNamespace(gravity=SimpleNamespace(x=0., y=0., z=-9.81)),
        )

    def test_capture_uses_real_actor_and_measured_gripper_views(self):
        env = self.environment()
        capture = StateCapture(env, self.device)
        # Public aliases, if present, must not override the actual actor views.
        env.root_states = torch.full((9, 13), 99., device=self.device)
        env.robot_indices = torch.arange(3, device=self.device)
        env.object_indices = torch.arange(3, device=self.device)
        state = capture.capture()
        self.assertEqual(tuple(state.shape), (3, 34))
        torch.testing.assert_close(state[:, :7], env._robot_root_states[:, :7])
        torch.testing.assert_close(state[:, 7:10], env._cube_root_states[:, :3])
        torch.testing.assert_close(state[:, 16:22], env._cube_root_states[:, 7:13])
        torch.testing.assert_close(state[:, 22], self.tensor([.2, .3, .4]))
        torch.testing.assert_close(state[:, 23], self.tensor([2., 3., 4.]))
        torch.testing.assert_close(state[:, 10:16], self.tensor([[1, 0, 0, 0, 1, 0]]).expand(3, -1))

    def test_explicit_actor_indices_and_gripper_joint_metadata_are_supported(self):
        env = self.environment()
        expected = StateCapture(env, self.device).capture()
        roots = torch.zeros(9, 13, device=self.device)
        env.robot_indices = torch.arange(0, 9, 3, device=self.device)
        env.cube_indices = torch.arange(2, 9, 3, device=self.device)
        roots[env.robot_indices] = env._robot_root_states
        roots[env.cube_indices] = env._cube_root_states
        env._root_states = roots
        env.num_gripper_joints = 2
        env._dof_pos = torch.cat((torch.ones(3, 5, device=self.device), env.dof_pos_gripper), -1)
        env._dof_vel = torch.cat((torch.ones(3, 5, device=self.device), env.dof_vel_gripper), -1)
        for name in ('_robot_root_states', '_cube_root_states', 'dof_pos_gripper', 'dof_vel_gripper'):
            delattr(env, name)
        torch.testing.assert_close(StateCapture(env, self.device).capture(), expected)

    def test_missing_physical_fields_raise_instead_of_producing_zero_measurements(self):
        for name in ('_robot_root_states', '_cube_root_states', 'dof_pos_gripper', 'dof_vel_gripper'):
            with self.subTest(name=name):
                env = self.environment()
                delattr(env, name)
                with self.assertRaises(AttributeError):
                    StateCapture(env, self.device).capture()
        env = self.environment()
        env.num_actions = 8
        with self.assertRaisesRegex(ValueError, 'requires 9 actions'):
            StateCapture(env, self.device)
        env = self.environment()
        env.max_episode_length = 0
        with self.assertRaisesRegex(ValueError, 'positive max_episode_length'):
            StateCapture(env, self.device)

    def test_xyzw_rot6d_and_inverse_support_identity_quarter_and_half_turns(self):
        capture = StateCapture(self.environment(), self.device)
        s = math.sqrt(.5)
        quat = self.tensor([[0, 0, 0, 1], [0, 0, s, s], [1, 0, 0, 0],
                            [0, 1, 0, 0], [0, 0, 1, 0], [s, s, 0, 0]])
        rot6d = capture._quat_to_rot6d(quat)
        torch.testing.assert_close(rot6d[:2], self.tensor([[1, 0, 0, 0, 1, 0],
                                                          [0, 1, 0, -1, 0, 0]]), atol=1e-6, rtol=1e-6)
        restored = capture._rot6d_to_quat(rot6d)
        self.assertTrue(torch.isfinite(restored).all().item())
        torch.testing.assert_close((restored * quat).sum(-1).abs(), torch.ones(6, device=self.device),
                                   atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(capture._quat_to_rot6d(-quat), rot6d)

    def test_object_pose_roundtrip_preserves_real_root_orientation(self):
        env = self.environment()
        env._cube_root_states[:, 3:7] = self.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        capture = StateCapture(env, self.device)
        pos, quat = capture.get_object_pose(capture.capture())
        torch.testing.assert_close(pos, env._cube_root_states[:, :3])
        torch.testing.assert_close((quat * env._cube_root_states[:, 3:7]).sum(-1).abs(),
                                   torch.ones(3, device=self.device))

    def test_terminal_clock_and_previous_action_survive_until_actual_reset(self):
        env = self.environment()
        capture = StateCapture(env, self.device)
        initial = capture.capture()
        action = torch.arange(27, device=self.device, dtype=torch.float32).reshape(3, 9).requires_grad_()
        capture.update_action(action)
        env.progress_buf[:] = torch.tensor([5, 20, 7], device=self.device)
        capture.increment_timestep(torch.tensor([[0], [1], [0]], device=self.device))
        terminal = capture.capture()
        torch.testing.assert_close(terminal[:, 33], self.tensor([.25, 1., .35]))
        torch.testing.assert_close(terminal[:, 24:33], action.detach())
        self.assertFalse(terminal.requires_grad)
        self.assertFalse(capture.prev_action.requires_grad)
        self.assertEqual(initial[:, 24:34].count_nonzero().item(), 0)
        env.progress_buf[1] = 0  # The simulator performs the real reset first.
        capture.reset_episodes(torch.tensor([1], device=self.device))
        reset_state = capture.capture()
        self.assertEqual(reset_state[1, 24:34].count_nonzero().item(), 0)
        torch.testing.assert_close(reset_state[[0, 2]], terminal[[0, 2]])
        self.assertEqual(terminal[1, 33].item(), 1.)

    def test_clock_fallback_advances_once_and_reset_is_explicit(self):
        env = self.environment()
        del env.progress_buf
        capture = StateCapture(env, self.device)
        capture.increment_timestep(torch.ones(3, 1, device=self.device, dtype=torch.long))
        torch.testing.assert_close(capture.capture()[:, 33], self.tensor([.05] * 3))
        capture.reset_episodes([0, 2])
        torch.testing.assert_close(capture.capture()[:, 33], self.tensor([0., .05, 0.]))

    def test_context_uses_actual_goal_height_mass_gravity_and_episode_limit(self):
        env = self.environment()
        result = Context(env, self.device).get_context()
        self.assertEqual(tuple(result.shape), (3, 16))
        torch.testing.assert_close(result[:, :2], env._cube_root_states[:, :2])
        target = env.table_heights + env.init_height + env.lifted_success_threshold
        torch.testing.assert_close(result[:, 2], target)
        torch.testing.assert_close(result[:, 11], target)
        torch.testing.assert_close(result[:, 3:6], self.tensor([[0., 0., -1.]]).expand(3, -1))
        torch.testing.assert_close(result[:, 6], env.object_mass.flatten() / 10)
        torch.testing.assert_close(result[:, 7:10], env._robot_root_states[:, :3])
        torch.testing.assert_close(result[:, 10], self.tensor([.2] * 3))
        torch.testing.assert_close(result[:, 12], self.tensor([.02] * 3))

    def test_partial_context_reset_updates_only_selected_env_and_keeps_old_transition(self):
        env = self.environment()
        context = Context(env, self.device)
        old = context.get_context()
        original = old.clone()
        env._cube_root_states[:, :2] += 10
        env._robot_root_states[:, :3] += 20
        env.table_heights += .4
        env.object_mass *= 2
        context.reset_context(torch.tensor([1], device=self.device))
        current = context.get_context()
        torch.testing.assert_close(current[[0, 2]], original[[0, 2]])
        torch.testing.assert_close(old, original)
        torch.testing.assert_close(current[1, :2], env._cube_root_states[1, :2])
        torch.testing.assert_close(current[1, 7:10], env._robot_root_states[1, :3])
        self.assertAlmostEqual(current[1, 6].item(), .2)
        self.assertAlmostEqual(current[1, 11].item(), .9)
        current.zero_()
        self.assertGreater(context.get_context().abs().sum().item(), 0)

    def test_actual_actor_mass_is_read_when_environment_has_no_mass_tensor(self):
        env = self.environment()
        del env.object_mass
        env.envs = ['env0', 'env1', 'env2']
        env.cube_handles = ['cube0', 'cube1', 'cube2']
        getter = Mock(side_effect=lambda handle, actor: [SimpleNamespace(mass=int(handle[-1]) + .5),
                                                       SimpleNamespace(mass=.25)])
        env.gym = SimpleNamespace(get_actor_rigid_body_properties=getter)
        context = Context(env, self.device)
        torch.testing.assert_close(context.get_context()[:, 6], self.tensor([.075, .175, .275]))
        self.assertEqual(getter.call_count, 3)
        context.reset_context([2])
        self.assertEqual(getter.call_count, 4)
        getter.assert_called_with('env2', 'cube2')

    def test_explicit_goal_scalar_mass_and_config_gravity_are_supported(self):
        env = self.environment()
        env.goal_pos = self.tensor([1., 2., 3.])
        env.object_mass = 2.
        env.cfg = {'sim': {'gravity': [0., -2., 0.]}, 'env': {'maxEpisodeLength': 40}}
        del env.sim_params
        del env.max_episode_length
        result = Context(env, self.device).get_context()
        torch.testing.assert_close(result[:, :3], env.goal_pos.expand(3, -1))
        torch.testing.assert_close(result[:, 3:6], self.tensor([[0., -1., 0.]]).expand(3, -1))
        torch.testing.assert_close(result[:, 6], self.tensor([.2] * 3))
        torch.testing.assert_close(result[:, 12], self.tensor([.04] * 3))
        self.assertEqual(StateCapture(env, self.device).max_episode_length, 40)

    def test_missing_context_sources_are_explicit_errors(self):
        for name in ('object_mass', 'sim_params', 'table_heights'):
            with self.subTest(name=name):
                env = self.environment()
                delattr(env, name)
                with self.assertRaises(AttributeError):
                    Context(env, self.device)


class CPUStateTests(StateChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CUDAStateTests(StateChecks, unittest.TestCase):
    device = 'cuda:0'


if __name__ == '__main__':
    unittest.main()
