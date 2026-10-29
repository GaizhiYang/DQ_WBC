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
    _prepare_reward_function = source_method("b1z1_base.py", "B1Z1Base", "_prepare_reward_function")
    compute_reward = source_method("b1z1_base.py", "B1Z1Base", "compute_reward")
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
    env.cfg["reward"] = {"base_height_target": .55, "only_positive_rewards": False}
    env.actions[:, 6] = -1.
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
    def test_stance_reward_requires_valid_grasp_and_new_hold_progress(self):
        env = make_environment([.08, .18, .199, .21])
        env.check_termination()
        self.assertEqual(env.lifted_now.tolist(), [False, False, False, True])
        self.assertEqual(env.lifted_object.tolist(), [False, False, False, True])
        stance, _ = env._reward_grasp_base_height()
        torch.testing.assert_close(stance * .5, torch.tensor([0., 0., 0., .5]))
        env._reward_pick_up()
        self.assertEqual(env.pick_counter.tolist(), [0, 0, 0, 1])
        self.assertFalse(env.episode_success.any().item())

    def test_base_direction_preserves_baseline_zero_reward(self):
        env = make_environment([.08] * 5, [0., 90., 180., 90., 0.])
        env._robot_root_states[:, :2] = 0
        objects = torch.tensor([[1., 0., 2.], [1., 0., -3.], [1., 0., .5],
                                [0., 1., 4.], [0., 0., 20.]])
        original = objects.clone()
        reward, _ = env._reward_base_dir(objects)
        # Deliberately retain 633_test_pro's vertical object-direction vector,
        # which is orthogonal to the robot's horizontal yaw-only heading.
        torch.testing.assert_close(reward, torch.zeros(5), atol=1e-6, rtol=0)
        torch.testing.assert_close(objects, original)
        self.assertTrue(torch.isfinite(reward).all().item())

    def test_yaw_penalty_has_no_terminal_or_failure_side_effects(self):
        env = make_environment([.08] * 4, [0., 59., 65., 80.])
        env.reset_buf[:] = torch.tensor([1, 0, 0, 0])
        env._terminate_buf[:] = torch.tensor([1, 0, 0, 0])
        resets, failures = env.reset_buf.clone(), env._terminate_buf.clone()
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        expected = torch.tensor([0., 0., -torch.tanh(torch.deg2rad(torch.tensor(65.))),
                                 -torch.tanh(torch.deg2rad(torch.tensor(80.)))])
        torch.testing.assert_close(reward, expected)
        torch.testing.assert_close(env.reset_buf, resets)
        torch.testing.assert_close(env._terminate_buf, failures)

    def test_yaw_penalty_and_failure_have_separate_thresholds(self):
        env = make_environment([.08] * 4, [59., 61., 69., 71.])
        env.check_termination()
        self.assertEqual(env._terminate_buf.bool().tolist(), [False, False, False, True])
        self.assertEqual(env.reset_buf.tolist(), [0, 0, 0, 1])
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        self.assertEqual((reward < 0).tolist(), [False, True, True, True])

    def test_final_hold_yaw_failure_is_not_a_success_or_a_positive_sgs_label(self):
        for evaluate in (False, True):
            with self.subTest(evaluate=evaluate):
                env = make_environment([.21] * 4, [0., 65., 80., -80.])
                env.eval = evaluate
                env.global_step_counter = 0 if not evaluate else 20
                env.pick_counter.fill_(env.hold_steps - 1)
                env.check_termination()
                self.assertEqual(env._terminate_buf.bool().tolist(), [False, False, True, True])
                reward, _ = env._reward_pick_up()
                self.assertEqual(reward.tolist(), [1., 1., 0., 0.])
                self.assertEqual(env.episode_success.tolist(), [True, True, False, False])
                self.assertEqual(env.success_onetime_counter.tolist(), [1, 1, 0, 0])
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

    def test_yaw_failure_is_independent_of_reward_weight(self):
        for scales in ({"limit_yaw_rotation_penalty": 0.}, {}):
            with self.subTest(scales=scales):
                env = make_environment([.21, .21], [80., -80.])
                env.reward_scales = scales
                env.pick_counter.fill_(env.hold_steps - 1)
                env.check_termination()
                self.assertTrue(env.reset_buf.bool().all().item())
                self.assertTrue(env._terminate_buf.bool().all().item())
                reward, _ = env._reward_pick_up()
                self.assertEqual(reward.tolist(), [0., 0.])
                self.assertFalse(env.pick_success_recorded.any().item())

    def test_yaw_difference_preserves_baseline_absolute_subtraction(self):
        env = make_environment([.08, .08, .08], [-179., 179., 10.])
        env.initial_yaw_z = torch.deg2rad(torch.tensor([179., -179., 350.]))
        torch.testing.assert_close(torch.rad2deg(env._yaw_deviation()),
                                   torch.tensor([358., 358., 340.]), atol=5e-5, rtol=0)
        env.check_termination()
        self.assertTrue(env._terminate_buf.bool().all().item())
        self.assertTrue(env.reset_buf.bool().all().item())
        reward, _ = env._reward_limit_yaw_rotation_penalty()
        self.assertTrue((reward < 0).all().item())

    def test_repeated_nine_step_holds_pay_each_progress_milestone_only_once(self):
        for sgs_enabled in (False, True):
            with self.subTest(sgs_enabled=sgs_enabled):
                env = make_environment([.21])
                env.sgs_enabled = sgs_enabled
                total = 0.
                for step in range(100):
                    # The 0.240 m center interrupts the grasp; the next nine
                    # steps must not repay milestones already earned before it.
                    height = .255 if step % 10 < 9 else .240
                    env._cube_root_states[:, 2] = env.table_heights + height
                    env._rigid_body_state[:, 1, :3] = env._cube_root_states[:, :3]
                    env._rigid_body_state[:, 1, 0] -= .03
                    env.update_roboinfo()
                    env.curr_ee_goal_cart[:] = quat_rotate_inverse(
                        env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
                    env.progress_buf.fill_(step + 1)
                    env.check_termination()
                    total += .5 * env._reward_grasp_base_height()[0].item()
                    self.assertEqual(env._reward_pick_up()[0].item(), 0.)
                    self.assertFalse(env.pick_success_recorded.item())
                    if step < 99:
                        self.assertFalse(env.reset_buf.item())
                self.assertEqual(total, 4.5)
                self.assertEqual(env.best_pick_counter.item(), 9)
                self.assertTrue(env.timeout_buf.item())

    def test_zero_pick_up_weight_still_completes_in_both_sgs_modes(self):
        for sgs_enabled in (False, True):
            with self.subTest(sgs_enabled=sgs_enabled):
                env = make_environment([.21])
                env.sgs_enabled = sgs_enabled
                self.assertEqual(env.hold_steps, 10)
                env.reward_scales = {"pick_up": 0.}
                env.rew_buf = torch.zeros(1)
                env._prepare_reward_function()
                self.assertNotIn("pick_up", env.reward_names)
                for step in range(10):
                    env.progress_buf += 1
                    env.check_termination()
                    env.compute_reward()
                    self.assertEqual(env.rew_buf.item(), 0.)
                    self.assertEqual(env.pick_success_recorded.item(), step == 9)
                self.assertEqual(env.success_counter.item(), 1)
                self.assertTrue(env.reset_buf.item())

    def test_sgs_switch_preserves_hold_reward_and_failure_trace(self):
        traces = []
        for sgs_enabled in (False, True):
            env = make_environment([.21, .21])
            env.sgs_enabled = sgs_enabled
            trace = []
            for step in range(10):
                env.progress_buf += 1
                # Opening one gripper on step six restarts just that hold.
                env.actions[1, 6] = 1. if step == 5 else -1.
                env.check_termination()
                before = (env.pick_counter.clone(), env.success_counter.clone(),
                          env.pick_success_recorded.clone(), env.reset_buf.clone())
                reward, _ = env._reward_pick_up()
                env._reward_pick_up()
                after = (env.pick_counter, env.success_counter,
                         env.pick_success_recorded, env.reset_buf)
                for first, second in zip(before, after):
                    torch.testing.assert_close(first, second)
                trace.append((reward.tolist(), env._reward_grasp_base_height()[0].tolist(),
                              env.pick_counter.tolist(), env.reset_buf.tolist(),
                              env.success_counter.tolist()))
            self.assertEqual(env.success_counter.tolist(), [1, 0])
            self.assertEqual(env.pick_counter.tolist(), [10, 4])
            traces.append(trace)
        self.assertEqual(traces[0], traces[1])


if __name__ == "__main__":
    unittest.main()
