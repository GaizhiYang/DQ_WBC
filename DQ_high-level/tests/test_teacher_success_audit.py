"""Additional boundary checks for the current teacher's success definition.

The existing SGS/reward tests cover hold duration, reset attribution and yaw/IK
failure priority. These checks cover physical pose boundaries, other base
failures and the separation between predicted observations and success labels.
"""

import unittest
from types import SimpleNamespace

import torch

from test_sgs_env import quat_from_euler_xyz, quat_rotate_inverse
from test_teacher_rewards import make_environment


class TeacherSuccessAuditTests(unittest.TestCase):
    def test_goal_requires_twenty_cm_above_rest_height_and_close_end_effector(self):
        env = make_environment([.05, .10, .199, .201, .21])
        # Near-Goal's 5--10 cm initial lift cannot itself satisfy the goal.
        # The final environment is high enough but outside the 20 cm proximity.
        env._rigid_body_state[-1, 1, 0] = env._cube_root_states[-1, 0] - .201
        env.curr_ee_goal_cart[:] = quat_rotate_inverse(
            env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
        env.pick_counter.fill_(env.hold_steps - 1)
        env.check_termination()
        reward, _ = env._reward_pick_up()
        self.assertEqual(env.lifted_object.tolist(), [False, False, False, True, False])
        self.assertEqual(env.episode_success.tolist(), [False, False, False, True, False])
        self.assertEqual(reward.tolist(), [0., 0., 0., 1., 0.])

    def test_roll_pitch_and_base_collapse_override_final_hold_success(self):
        env = make_environment([.21] * 4)
        roll = torch.tensor([.81, 0., 0., 0.])
        pitch = torch.tensor([0., -.81, 0., 0.])
        env._robot_root_states[:, 3:7] = quat_from_euler_xyz(roll, pitch, torch.zeros(4))
        env._robot_root_states[2, 2] = .09
        env._update_base_yaw_quat()
        env.update_roboinfo()
        # Match IK targets to isolate base failure from arm tracking failure.
        env.curr_ee_goal_cart[:] = quat_rotate_inverse(
            env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
        env.pick_counter.fill_(env.hold_steps - 1)
        env.check_termination()
        reward, _ = env._reward_pick_up()
        self.assertEqual(env._terminate_buf.bool().tolist(), [True, True, True, False])
        self.assertEqual(env.episode_success.tolist(), [False, False, False, True])
        self.assertEqual(reward.tolist(), [0., 0., 0., 1.])

    def test_timeout_requires_full_hold_even_if_goal_geometry_is_satisfied(self):
        env = make_environment([.21, .21])
        env.pick_counter[:] = torch.tensor([env.hold_steps - 2, env.hold_steps - 1])
        env.progress_buf.fill_(env.max_episode_length)
        env.check_termination()
        env._reward_pick_up()
        self.assertEqual(env.timeout_buf.tolist(), [True, False])
        self.assertTrue(env.reset_buf.bool().all().item())
        self.assertFalse(env._terminate_buf.any().item())
        # A completed ten-step hold at the deadline is success, not timeout.
        self.assertEqual(env.episode_success.tolist(), [False, True])

    def test_success_uses_simulator_pose_when_ekf_future_pose_disagrees(self):
        env = make_environment([.08, .21])
        predicted = env._cube_root_states[:, None, :7].repeat(1, 2, 1)
        predicted[0, :, 2] += 1.
        predicted[1, :, 2] -= 1.
        env.object_pose_predictor = SimpleNamespace(
            future_poses=predicted, valid=torch.ones(2, dtype=torch.bool))
        current_pose = env._cube_root_states.clone()
        env.pick_counter.fill_(env.hold_steps - 1)
        env.check_termination()
        env._reward_pick_up()
        self.assertEqual(env.episode_success.tolist(), [False, True])
        torch.testing.assert_close(env._cube_root_states, current_pose)
        torch.testing.assert_close(env.object_pose_predictor.future_poses, predicted)

    def test_final_hold_requires_a_negative_close_command(self):
        env = make_environment([.21, .21, .21])
        env.actions[:, 6] = torch.tensor([-1., 0., 1.])
        env.pick_counter.fill_(env.hold_steps - 1)
        env.check_termination()
        reward, _ = env._reward_pick_up()
        self.assertEqual(env.lifted_object.tolist(), [True, False, False])
        self.assertEqual(env.pick_counter.tolist(), [10, 0, 0])
        self.assertEqual(env.pick_success_recorded.tolist(), [True, False, False])
        self.assertEqual(reward.tolist(), [1., 0., 0.])

    def test_exact_lift_threshold_is_inclusive_but_proximity_is_strict(self):
        env = make_environment([.20, .20, .20])
        target = env.table_heights + env.init_height + env.lifted_success_threshold
        env._cube_root_states[:, 2] = target
        env._cube_root_states[0, 2] = torch.nextafter(target[0], torch.tensor(-torch.inf))
        env._cube_root_states[:, 0] = 0.
        env._rigid_body_state[:, 1, :3] = env._cube_root_states[:, :3]
        env._rigid_body_state[:, 1, 0] = torch.tensor([.03, .03, .2])
        env.update_roboinfo()
        env.curr_ee_goal_cart[:] = quat_rotate_inverse(
            env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
        env.pick_counter.fill_(env.hold_steps - 1)
        env.check_termination()
        self.assertEqual(env.lifted_object.tolist(), [False, True, False])
        self.assertEqual(env.pick_success_recorded.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
