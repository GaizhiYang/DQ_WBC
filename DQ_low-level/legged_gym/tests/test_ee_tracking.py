"""Kinematic/command regressions; no simulator instance or GPU is needed.

Run in the Isaac Gym environment from DQ_low-level:
    python -m unittest discover -s legged_gym/tests -p test_ee_tracking.py
"""

import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401 -- must precede torch
from isaacgym.torch_utils import quat_from_euler_xyz
import numpy as np
from scipy.spatial.transform import Rotation
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from legged_gym.envs.manip_loco.d1_piper_l_config import D1PiperLRoughCfg
from legged_gym.envs.manip_loco.manip_loco import ManipLoco
from legged_gym.envs.rewards.maniploco_rewards import ManipLoco_rewards


def command_env(cfg):
    """Only the state needed by the production goal-update methods."""
    env = ManipLoco.__new__(ManipLoco)
    env.cfg = cfg
    env.device = "cpu"
    env.num_envs = 2
    env.root_states = torch.zeros(2, 13)
    env.base_yaw_quat = quat_from_euler_xyz(torch.zeros(2), torch.zeros(2), torch.tensor([0., .7]))
    center = cfg.goal_ee.sphere_center
    env.ee_goal_center_offset = torch.tensor([center.x_offset, center.y_offset, center.z_invariant_offset]).repeat(2, 1)
    for name in ("curr_ee_goal_cart", "curr_ee_goal_cart_world", "ee_goal_cart",
                 "ee_goal_orn_euler", "ee_goal_orn_delta_rpy", "ee_start_orn_delta_rpy",
                 "curr_ee_orn_delta_rpy", "ee_start_sphere", "ee_goal_sphere"):
        setattr(env, name, torch.zeros(2, 3))
    env.ee_goal_orn_quat = torch.zeros(2, 4)
    env.init_start_ee_sphere = torch.tensor(cfg.goal_ee.ranges.init_pos_start).view(1, 3)
    env.init_end_ee_sphere = torch.tensor(cfg.goal_ee.ranges.init_pos_end).view(1, 3)
    env.curr_ee_goal_sphere = env.init_start_ee_sphere.repeat(2, 1)
    offset = torch.tensor(getattr(cfg.goal_ee, "tool_orientation_offset_rpy", [0., 0., 0.])).repeat(2, 1)
    env.ee_tool_orientation_offset = quat_from_euler_xyz(offset[:, 0], offset[:, 1], offset[:, 2])
    env.goal_timer = torch.zeros(2)
    env.traj_timesteps = torch.full((2,), 4.)
    env.traj_total_timesteps = torch.full((2,), 8.)
    env.stop_update_goal = False
    return env


class EndEffectorTrackingTests(unittest.TestCase):
    def test_piper_tool_axes_and_b1_command_compatibility(self):
        for cfg in (D1PiperLRoughCfg(), B1Z1RoughCfg()):
            env = command_env(cfg)
            env._update_ee_goal_pose()
            pitch = cfg.goal_ee.arm_induced_pitch - cfg.goal_ee.ranges.init_pos_start[1]
            nominal = Rotation.from_euler("xyz", [[math.pi / 2, pitch, 0.], [math.pi / 2, pitch, .7]])
            actual = Rotation.from_quat(env.ee_goal_orn_quat.numpy()).as_matrix()
            if isinstance(cfg, D1PiperLRoughCfg):
                np.testing.assert_allclose(actual[:, :, 2], nominal.as_matrix()[:, :, 0], atol=2e-6)
                np.testing.assert_allclose(actual[:, :, 1], -nominal.as_matrix()[:, :, 2], atol=2e-6)
            else:
                np.testing.assert_allclose(actual, nominal.as_matrix(), atol=2e-6)

    def test_piper_default_pose_matches_target_and_is_nonsingular(self):
        cfg = D1PiperLRoughCfg()
        path = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        joints = {j.find("child").get("link"): j for j in ET.parse(path).getroot().findall("joint")}
        chain, link = [], cfg.asset.gripper_name
        while link in joints:
            joint = joints[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        transform = np.eye(4)
        transform[:3, 3] = cfg.init_state.pos
        axes, origins = [], []
        for joint in reversed(chain):
            origin = joint.find("origin")
            local = np.eye(4)
            local[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            local[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")).as_matrix()
            transform = transform @ local
            if joint.get("type") == "fixed":
                continue
            q = cfg.init_state.default_joint_angles[joint.get("name")]
            limit = joint.find("limit")
            self.assertGreater(q - float(limit.get("lower")), .05)
            self.assertGreater(float(limit.get("upper")) - q, .05)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            axes.append(transform[:3, :3] @ axis)
            origins.append(transform[:3, 3].copy())
            transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(q * axis).as_matrix()
        env = command_env(cfg)
        env._update_ee_goal_pose()
        np.testing.assert_allclose(transform[:3, 3], env.curr_ee_goal_cart_world[0].numpy(), atol=2e-6)
        expected_rotation = Rotation.from_quat(env.ee_goal_orn_quat[0].numpy()).as_matrix()
        np.testing.assert_allclose(transform[:3, :3], expected_rotation, atol=2e-6)
        jacobian = np.concatenate((np.cross(axes, transform[:3, 3] - origins).T, np.array(axes).T))
        self.assertGreater(np.linalg.svd(jacobian, compute_uv=False)[-1], .05)

    def test_reset_updates_only_selected_environment_before_ik(self):
        env = command_env(D1PiperLRoughCfg())
        env.curr_ee_goal_sphere[:] = env.init_end_ee_sphere
        env._update_ee_goal_pose()
        unchanged_position = env.curr_ee_goal_cart_world[1].clone()
        unchanged_quat = env.ee_goal_orn_quat[1].clone()
        env._resample_ee_goal(torch.tensor([0]), is_init=True)
        expected = command_env(D1PiperLRoughCfg())
        expected._update_ee_goal_pose()
        torch.testing.assert_close(env.curr_ee_goal_cart_world[0], expected.curr_ee_goal_cart_world[0])
        torch.testing.assert_close(env.ee_goal_orn_quat[0], expected.ee_goal_orn_quat[0])
        torch.testing.assert_close(env.curr_ee_goal_cart_world[1], unchanged_position)
        torch.testing.assert_close(env.ee_goal_orn_quat[1], unchanged_quat)

    def test_orientation_offsets_follow_trajectory_time(self):
        env = command_env(D1PiperLRoughCfg())
        env.ee_start_sphere[:] = env.init_start_ee_sphere
        env.ee_goal_sphere[:] = env.init_start_ee_sphere
        env.ee_goal_orn_delta_rpy[:] = torch.tensor([.1, -.1, .15])
        env._update_curr_ee_goal()
        torch.testing.assert_close(env.curr_ee_orn_delta_rpy, torch.zeros(2, 3))
        env.goal_timer[:] = 2
        env._update_curr_ee_goal()
        torch.testing.assert_close(env.curr_ee_orn_delta_rpy, env.ee_goal_orn_delta_rpy / 2)

    def test_orientation_reward_tracks_live_quaternion_and_ignores_sign(self):
        env = SimpleNamespace(cfg=D1PiperLRoughCfg(), ee_orn=torch.tensor([[0., 0., 0., 1.]]))
        rewards = ManipLoco_rewards(env)
        env.ee_goal_orn_quat = -env.ee_orn
        rew, error = rewards._reward_tracking_ee_orn()
        torch.testing.assert_close(rew, torch.ones(1))
        torch.testing.assert_close(error, torch.zeros(1))
        env.ee_goal_orn_quat = torch.tensor([[0., 0., 1., 0.]])
        rew, error = rewards._reward_tracking_ee_orn()
        torch.testing.assert_close(error, torch.tensor([math.pi]))
        torch.testing.assert_close(rew, torch.tensor([math.exp(-math.pi)]))


if __name__ == "__main__":
    unittest.main()
