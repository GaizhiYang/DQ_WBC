"""TSIL reward, clock and reset semantics through real task methods on CPU.

Physics is not required for these invariants: the existing SGS stand-in runs
the production reset and success predicates, with the production TSIL hooks.
"""

import ast
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from test_sgs_env import NAMESPACE, TestEnvironment, method_node, source_method
from utils.tsil import (TemporalState, geometry_descriptors, GEOMETRY_DESCRIPTOR_SCHEMA,
                        LOOKUP_NEAREST, LOOKUP_OUT_OF_SUPPORT, LOOKUP_CONFIGURATION)


class TemporalEnvironment(TestEnvironment):
    _reset_tsil_state = source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_reset_tsil_state")
    _tsil_observations = source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_tsil_observations")
    _record_tsil_transition = source_method("b1z1_pickmulti.py", "B1Z1PickMulti", "_record_tsil_transition")
    _prepare_reward_function = source_method("b1z1_base.py", "B1Z1Base", "_prepare_reward_function")
    compute_reward = source_method("b1z1_base.py", "B1Z1Base", "compute_reward")

    def __init__(self, num_envs=2):
        super().__init__(num_envs=num_envs)
        self.tsil_enabled = True
        self.tsil_time_reference = 16.0
        self.tsil_elapsed = torch.zeros(num_envs)
        self.tsil_episode_D = torch.full((num_envs,), 16.0)
        self.tsil_episode_id = torch.full((num_envs,), -1, dtype=torch.long)
        self.tsil_lookup_status = torch.full((num_envs,), -1, dtype=torch.long)
        self.tsil_reward_fixed = torch.zeros(num_envs)
        self.tsil_reward_command_raw = torch.zeros(num_envs)
        self.tsil_state = SimpleNamespace(targets=torch.tensor([6., 8., 10., 12., 14., 16.]))
        self.dt = .02
        self.control_freq_low = 8
        self.train_reward_strict = 50
        self.rew_buf = torch.zeros(num_envs)
        self.cfg["reward"] = {"only_positive_rewards": False}
        with redirect_stdout(io.StringIO()):
            self.initialize(tuple(range(num_envs)))


command_reward = source_method("reward_vec_task.py", "RewardVecTask", "_reward_command_reward")
command_reward.__globals__["COMMAND_SMOOTH_THRESHOLD"] = .2


class TemporalEnvironmentTests(unittest.TestCase):
    def test_targets_belong_to_configuration_and_stay_fixed_until_reset(self):
        env = TemporalEnvironment()
        torch.testing.assert_close(env.tsil_episode_D, torch.tensor([6., 8.]))
        self.assertEqual(env.tsil_episode_id.tolist(), [0, 0])
        self.assertEqual(env.tsil_lookup_status.tolist(), [LOOKUP_CONFIGURATION] * 2)
        env.tsil_elapsed[:] = torch.tensor([1., 2.])
        env.tsil_state.targets[0] = 3.
        torch.testing.assert_close(env.tsil_episode_D, torch.tensor([6., 8.]))
        # An administrative reset discards this episode: no fabricated SGS
        # outcome. It still changes serial, even if the same config is sampled.
        counts = env.sgs_sampler.count.clone()
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([0])):
            env.reset_idx(torch.tensor([0]))
        torch.testing.assert_close(env.sgs_sampler.count, counts)
        torch.testing.assert_close(env.tsil_episode_D, torch.tensor([3., 8.]))
        torch.testing.assert_close(env.tsil_elapsed, torch.tensor([0., 2.]))
        self.assertEqual(env.tsil_episode_id.tolist(), [1, 0])

    def test_ordinary_reset_assigns_inferred_target_once_and_exposes_actual_reason(self):
        robot = torch.zeros(2, 13)
        obj = torch.zeros(2, 13)
        robot[:, 6] = obj[:, 6] = 1.
        robot[:, 2] = .515  # saved, grounded training references
        obj[:, :3] = torch.tensor([2., 0., .61])
        height = torch.full((2,), .55)
        state = TemporalState(2, 16, descriptors=geometry_descriptors(robot, obj, height),
                              descriptor_schema=GEOMETRY_DESCRIPTOR_SCHEMA)
        state.observe_success([0, 1], [6., 10.])
        state.commit_targets()
        env = SimpleNamespace(tsil_enabled=True, sgs_enabled=False,
                              tsil_elapsed=torch.ones(2), tsil_episode_id=torch.zeros(2, dtype=torch.long),
                              tsil_episode_D=torch.full((2,), 16.), tsil_state=state,
                              tsil_lookup_status=torch.full((2,), -1, dtype=torch.long),
                              _robot_root_states=robot, _cube_root_states=obj,
                              table_heights=height, _dof_pos=torch.randn(2, 19))
        # The normal reset starts above the floor; this cannot reject lookup.
        robot[:, 2] = .6
        obj[1, 0] = 4.  # genuinely outside the supported task distance
        TemporalEnvironment._reset_tsil_state(env, torch.arange(2))
        torch.testing.assert_close(env.tsil_episode_D, torch.tensor([8., 16.]))
        torch.testing.assert_close(env.tsil_elapsed, torch.zeros(2))
        self.assertEqual(env.tsil_lookup_status.tolist(), [LOOKUP_NEAREST, LOOKUP_OUT_OF_SUPPORT])
        self.assertEqual(env.tsil_episode_id.tolist(), [1, 1])
        state.targets[:] = 3.
        torch.testing.assert_close(env.tsil_episode_D, torch.tensor([8., 16.]))

    def test_slow_success_remains_success_with_soft_time_bonus(self):
        env = TemporalEnvironment()
        env.tsil_episode_D[:] = 6.
        env.tsil_elapsed[:] = torch.tensor([6., 8.])
        env.pick_counter.fill_(env.hold_steps - 1)
        reward = env.grasp_step([True, True])
        torch.testing.assert_close(50. * reward, torch.tensor([100., 87.5]))
        self.assertTrue(env.episode_success.all())
        self.assertTrue(env.reset_buf.bool().all())
        # Reward is paid only for the new success transition.
        torch.testing.assert_close(env.grasp_step([True, True]), torch.zeros(2))

    def test_relabel_components_preserve_negative_reward_and_ungated_command(self):
        env = TemporalEnvironment()
        env.tsil_elapsed[:] = torch.tensor([6., 8.])
        env.tsil_episode_D.fill_(6.)
        env.pick_success[:] = torch.tensor([True, False])
        env.reward_scales = {"fixed": 1., "command_reward": .05, "pick_up": 50.}
        calls = []

        def fixed_reward():
            calls.append("fixed")
            return torch.tensor([-.25, -.5]), torch.zeros(2)

        env._reward_fixed = fixed_reward
        env.commands[:, 0] = torch.tensor([0., .5])
        positions = env.arm_base.clone()
        env._reward_command_reward = lambda: command_reward(env, positions)
        env._prepare_reward_function()
        for step, gate in ((49, 0.), (50, 1.)):
            env.global_step_counter = step
            env.compute_reward()
            expected_command = .05 * torch.exp(-torch.tensor([0., .5]))
            torch.testing.assert_close(env.tsil_reward_command_raw, expected_command)
            torch.testing.assert_close(env.tsil_reward_fixed, torch.tensor([-.25, -.5]))
            expected = env.tsil_reward_fixed + gate * expected_command + torch.tensor([100., 0.])
            torch.testing.assert_close(env.rew_buf, expected)
            self.assertLess(env.rew_buf[1].item(), 0.)
        self.assertEqual(calls, ["fixed", "fixed"])

    def test_recorded_info_owns_old_episode_after_reset(self):
        env = TemporalEnvironment()
        env.tsil_elapsed.fill_(2.)
        env.tsil_reward_fixed[:] = torch.tensor([.25, -.5])
        env.tsil_reward_command_raw.fill_(.05)
        env.pick_success[0] = True
        env._record_tsil_transition()
        before = env.extras["tsil"]
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([4])):
            env.reset_idx(torch.tensor([0]))
        env.tsil_reward_fixed.zero_()
        self.assertEqual(before["config_id"].tolist(), [0, 1])
        self.assertEqual(before["episode_id"].tolist(), [0, 0])
        self.assertEqual(before["success"].tolist(), [True, False])
        torch.testing.assert_close(before["elapsed"], torch.tensor([2., 2.]))
        torch.testing.assert_close(before["elapsed_before"], torch.tensor([1.84, 1.84]))
        torch.testing.assert_close(before["reward_fixed"], torch.tensor([.25, -.5]))
        self.assertFalse(before["command_gate"])

    def test_physical_clock_advances_before_rewards_and_observations(self):
        step = source_method("b1z1_base.py", "B1Z1Base", "post_physics_step")
        elapsed_at_reward, elapsed_at_observation = [], []
        env = SimpleNamespace(
            tsil_enabled=True, tsil_elapsed=torch.zeros(2), dt=.02,
            control_freq_low=7, progress_buf=torch.zeros(2, dtype=torch.long),
            global_step_counter=0, local_step_counter=0,
            last_actions=torch.zeros(2, 9), actions=torch.zeros(2, 9),
            enable_camera=False, debug_vis=False, extras={},
            _terminate_buf=torch.zeros(2), rl_device="cpu")
        for method in ("_refresh_sim_tensors", "update_roboinfo",
                       "_update_object_pose_prediction", "check_termination"):
            setattr(env, method, lambda: None)
        env.compute_reward = lambda: elapsed_at_reward.append(env.tsil_elapsed.clone())
        env._compute_observations = lambda: elapsed_at_observation.append(env.tsil_elapsed.clone())
        step(env)
        env.control_freq_low = 8
        step(env)
        torch.testing.assert_close(elapsed_at_reward[0], torch.full((2,), .14))
        torch.testing.assert_close(elapsed_at_reward[1], torch.full((2,), .30))
        torch.testing.assert_close(elapsed_at_observation[1], torch.full((2,), .30))

    def test_time_channels_use_fixed_reference_scale(self):
        env = TemporalEnvironment()
        env.tsil_elapsed[:] = torch.tensor([.14, 8.])
        env.tsil_episode_D[:] = torch.tensor([6., 12.])
        expected = torch.tensor([[.14 / 16., 6. / 16.], [.5, .75]])
        torch.testing.assert_close(env._tsil_observations(torch.arange(2)), expected)

    def test_observation_dimensions_add_two_without_changing_student_size(self):
        class Base:
            def _setup_obs_and_action_info(self, **kwargs):
                self.dimensions = kwargs

        node = ast.ClassDef(name="Task", bases=[ast.Name(id="Base", ctx=ast.Load())],
                            keywords=[], decorator_list=[], body=[method_node(
                                "b1z1_pickmulti.py", "B1Z1PickMulti", "_setup_obs_and_action_info")])
        namespace = dict(NAMESPACE, Base=Base)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     __file__, "exec"), namespace)
        task = namespace["Task"]()
        task.no_feature = False
        task.num_features = 1024
        task.cfg = {"env": {"noVelocityObs": True, "posePrediction": {
            "enabled": True, "horizons": [.6, .9]}, "tsil": {"enabled": False}}}
        task._setup_obs_and_action_info()
        old = dict(task.dimensions)
        self.assertEqual(old["num_obs"] + 9 + 24, 1284)
        task.cfg["env"]["tsil"]["enabled"] = True
        task._setup_obs_and_action_info()
        self.assertEqual(task.dimensions["num_obs"] + 9 + 24, 1286)
        self.assertEqual(task.dimensions["num_obs"] - task.dimensions["removed_dim"],
                         old["num_obs"] - old["removed_dim"])
        self.assertEqual(task.tsil_time_indices, (-191, -190))

    def test_observation_inserts_time_after_ekf_and_preserves_grasp_action_tail(self):
        class Base:
            def _compute_observations(self, ids):
                return torch.arange(64, dtype=torch.float).expand(len(ids), -1)

        node = ast.ClassDef(name="Task", bases=[ast.Name(id="Base", ctx=ast.Load())],
                            keywords=[], decorator_list=[], body=[method_node(
                                "b1z1_pickmulti.py", "B1Z1PickMulti", "_compute_observations")])
        namespace = dict(NAMESPACE, Base=Base,
                         quat_conjugate=lambda q: q * torch.tensor([-1., -1., -1., 1.]),
                         quat_to_euler_zyx=lambda q: torch.zeros_like(q[..., :3]),
                         quat_rotate_inverse_batched=NAMESPACE["quat_rotate_inverse"])
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                     __file__, "exec"), namespace)
        task = namespace["Task"]()
        task.no_feature = False
        task.cfg = {"env": {}}
        task._robot_root_states = torch.zeros(1, 13)
        task._robot_root_states[:, 6] = 1.
        task.arm_base = torch.zeros(1, 3)
        task.feature_obs = torch.arange(1024, dtype=torch.float)[None]
        task.obs_buf = torch.zeros(1, 1286)
        task.action_history_buf = torch.arange(9, dtype=torch.float)[None, None]
        task.get_grasp_world = lambda pose, ids: (torch.zeros(1, 30, 3), torch.zeros(1, 30, 3))
        ekf = torch.arange(13, dtype=torch.float)[None]
        task._future_object_observations = lambda ids, pose: ekf
        task._tsil_observations = lambda ids: torch.tensor([[.25, .75]])
        task._compute_observations(torch.tensor([0]))
        torch.testing.assert_close(task.obs_buf[:, :1024], task.feature_obs)
        torch.testing.assert_close(task.obs_buf[:, -204:-191], ekf)
        torch.testing.assert_close(task.obs_buf[:, -191:-189], torch.tensor([[.25, .75]]))
        torch.testing.assert_close(task.obs_buf[:, -189:-9], torch.zeros(1, 180))
        torch.testing.assert_close(task.obs_buf[:, -9:], task.action_history_buf[:, -1])


if __name__ == "__main__":
    unittest.main()
