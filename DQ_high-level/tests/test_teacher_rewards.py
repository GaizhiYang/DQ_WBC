"""Behavioral regressions for the teacher's rewards and terminal success labels.

Run the production methods through the existing CPU SGS environment stand-in.
No simulator is created; geometry, reward values, failures, hold completion and
the resulting SGS outcome attribution use the actual repository methods.
"""

import unittest
from unittest.mock import patch

import torch

from test_sgs_env import (
    TestEnvironment, quat_from_euler_xyz, quat_rotate_inverse, source_method,
)


class RewardEnvironment(TestEnvironment):
    _reward_base_dir = source_method("reward_vec_task.py", "RewardVecTask", "_reward_base_dir")
    _reward_base_height = source_method("reward_vec_task.py", "RewardVecTask", "_reward_base_height")
    _reward_grasp_base_height = source_method(
        "b1z1_pickmulti.py", "B1Z1PickMulti", "_reward_grasp_base_height")
    _reward_limit_yaw_rotation_penalty = source_method(
        "b1z1_pickmulti.py", "B1Z1PickMulti", "_reward_limit_yaw_rotation_penalty")


def make_environment(heights, yaw_degrees=None):
    n = len(heights)
    env = RewardEnvironment(num_envs=n)
    env.initialize(tuple(range(n)))
    env.cfg["reward"] = {"base_height_target": .55}
    env._robot_root_states[:, 2] = .55
    yaw = torch.deg2rad(torch.tensor(yaw_degrees or [0.] * n))
    env._robot_root_states[:, 3:7] = quat_from_euler_xyz(
        torch.zeros(n), torch.zeros(n), yaw)
    env._update_base_yaw_quat()
    env._cube_root_states[:, 2] = env.table_heights + env.init_height + torch.tensor(heights)
    env._rigid_body_state[:, 1, :3] = env._cube_root_states[:, :3]
    env._rigid_body_state[:, 1, 0] -= .03
    env.update_roboinfo()
    env.curr_ee_goal_cart[:] = quat_rotate_inverse(
        env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
    return env


class TeacherRewardTests(unittest.TestCase):
    def test_stance_reward_starts_at_task_lift_gate_not_legacy_drop_gate(self):
        env = make_environment([.08, .18, .199, .21])
        env.check_termination()
        # Preserve the old drop hysteresis, but do not pay its lower threshold.
        self.assertEqual(env.lifted_now.tolist(), [False, True, True, True])
        self.assertEqual(env.lifted_object.tolist(), [False, False, False, True])
        stance, _ = env._reward_grasp_base_height()
        torch.testing.assert_close(stance * .5, torch.tensor([0., 0., 0., .5]))
        env._reward_pick_up()
        self.assertEqual(env.pick_counter.tolist(), [0, 0, 0, 1])
        self.assertFalse(env.episode_success.any().item())

    def test_base_direction_measures_horizontal_alignment(self):
        env = make_environment([.08] * 5, [0., 90., 180., 90., 0.])
        env._robot_root_states[:, :2] = 0
        objects = torch.tensor([[1., 0., 2.], [1., 0., -3.], [1., 0., .5],
                                [0., 1., 4.], [0., 0., 20.]])
        original = objects.clone()
        reward, _ = env._reward_base_dir(objects)
        torch.testing.assert_close(reward, torch.tensor([1., 0., -1., 1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(objects, original)
        self.assertTrue(torch.isfinite(reward).all().item())

    def test_yaw_penalty_has_no_terminal_or_failure_side_effects(self):
        env = make_environment([.08] * 4, [0., 59., 65., 80.])
        env.reset_buf[:] = torch.tensor([1, 0, 0, 0])
        env._failure_now = torch.tensor([True, False, False, False])
        resets, failures = env.reset_buf.clone(), env._failure_now.clone()
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        expected = torch.tensor([0., 0., -torch.tanh(torch.deg2rad(torch.tensor(65.))),
                                 -torch.tanh(torch.deg2rad(torch.tensor(80.)))])
        torch.testing.assert_close(reward, expected)
        torch.testing.assert_close(env.reset_buf, resets)
        torch.testing.assert_close(env._failure_now, failures)

    def test_yaw_penalty_and_failure_have_separate_thresholds(self):
        env = make_environment([.08] * 4, [59., 61., 69., 71.])
        env.check_termination()
        self.assertEqual(env._failure_now.tolist(), [False, False, False, True])
        self.assertEqual(env.reset_buf.tolist(), [0, 0, 0, 1])
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        self.assertEqual((reward < 0).tolist(), [False, True, True, True])

    def test_final_hold_yaw_failure_is_not_a_success_or_a_positive_sgs_label(self):
        for evaluate in (False, True):
            with self.subTest(evaluate=evaluate):
                env = make_environment([.21] * 4, [0., 65., 80., -80.])
                env.eval = evaluate
                env.global_step_counter = 0 if not evaluate else 20
                env.pick_counter.fill_(14)
                env.check_termination()
                self.assertEqual(env._failure_now.tolist(), [False, False, True, True])
                reward, _ = env._reward_pick_up()
                self.assertEqual(reward.tolist(), [1., 1., 0., 0.])
                self.assertEqual(env.episode_success.tolist(), [True, True, False, False])
                self.assertEqual(env.episode_onetime_success.tolist(), [True, True, False, False])
                # The later yaw reward must neither create nor erase outcomes.
                env._reward_limit_yaw_rotation_penalty()
                self.assertEqual(env.episode_success.tolist(), [True, True, False, False])
                env._sgs_pending_done.copy_(env.reset_buf.bool())
                env._sgs_natural_reset = True
                with patch.object(env.sgs_sampler, "sample", return_value=torch.arange(4)):
                    env._reset_sgs_envs(torch.arange(4))
                torch.testing.assert_close(env.sgs_sampler.success_rates()[:4],
                                           torch.tensor([1., 1., 0., 0.]))
                self.assertEqual(env.sgs_sampler.count[:4].tolist(), [1, 1, 1, 1])

    def test_disabled_yaw_term_preserves_previous_no_yaw_termination_behavior(self):
        for scales in ({"limit_yaw_rotation_penalty": 0.}, {}):
            with self.subTest(scales=scales):
                env = make_environment([.21, .21], [80., -80.])
                env.reward_scales = scales
                env.pick_counter.fill_(14)
                env.check_termination()
                self.assertFalse(env.reset_buf.any().item())
                self.assertFalse(env._failure_now.any().item())
                reward, _ = env._reward_pick_up()
                self.assertEqual(reward.tolist(), [1., 1.])

    def test_yaw_difference_wraps_across_pi(self):
        env = make_environment([.08, .08, .08], [-179., 179., 10.])
        env.initial_yaw_z = torch.deg2rad(torch.tensor([179., -179., 350.]))
        torch.testing.assert_close(torch.rad2deg(env._yaw_deviation()),
                                   torch.tensor([2., 2., 20.]), atol=3e-5, rtol=0)
        env.check_termination()
        self.assertFalse(env._failure_now.any().item())
        self.assertFalse(env.reset_buf.any().item())
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        self.assertEqual(reward.tolist(), [0., 0., 0.])


if __name__ == "__main__":
    unittest.main()
