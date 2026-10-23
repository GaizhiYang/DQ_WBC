"""Offline, physics-validated reset-bank construction for B1Z1PickMulti.

This module never runs during training reset.  Isaac Gym must be imported before
importing this module in an executable (Isaac Gym's PyTorch import requirement).
"""

from contextlib import contextmanager
from copy import deepcopy
import sys

import numpy as np
import torch

from utils.sgs_build_config import (
    load_build_config, validate_build_config,
    SINGLE_VALUE_POSITION_TOLERANCE_M, SINGLE_VALUE_HEADING_TOLERANCE_DEG,
)
from utils.sgs_grounding import (
    FOOT_RADIUS, FOOT_NAMES, APPROX_GROUND_LIMITS,
    foot_indices, grounding_state, approx_grounded_mask,
)


@contextmanager
def _environment_arguments():
    """The existing environment reads the teacher CLI during construction."""
    previous = sys.argv
    sys.argv = [previous[0], "--timesteps", "120000"]
    try:
        yield
    finally:
        sys.argv = previous


def create_builder_environment(num_envs=32, device="cuda:0", seed=43,
                               config_path="data/cfg/DQ_teacher.yaml"):
    """Construct the real teacher environment without PPO or a training run."""
    from envs.b1z1_pickmulti import B1Z1PickMulti
    from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
    from train_multistate_DQ_teacher import get_predict_point
    from skrl.utils import set_seed
    from utils.config import load_cfg

    set_seed(seed)
    cfg = deepcopy(load_cfg(config_path))
    if cfg["env"].get("D1_bench_task_level") != "Level02":
        raise ValueError("The SGS builder requires the original Level02 table motion.")
    if cfg["env"]["asset"]["assetFileRobot"] != "b1z1-col/urdf/b1z1.urdf":
        raise ValueError("The offline IK chain supports b1z1-col/urdf/b1z1.urdf only.")
    if cfg["env"].get("floatingBase", False):
        raise ValueError("The builder requires the walking low-level controller.")
    objects = cfg["env"]["asset"]["asset_multi"]
    if "sugar_box" not in objects:
        raise ValueError("The initial builder supports the configured sugar_box.")
    cfg["env"]["asset"]["asset_multi"] = {"sugar_box": objects["sugar_box"]}
    cfg["env"].update(
        numEnvs=int(num_envs), wandb=False, enableDebugVis=False,
        cameraMode="full", useTanh=False, noVelocityObs=True,
        smallValueSetZero=True, near_goal_stop=False, obj_move_prob=0.0,
        sgs={"mode": "disabled"}, successMode="hold",
    )
    cfg["sensor"]["enableCamera"] = False
    camera, grasps, initial = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=int(num_envs), intervel=23, delta_height=0.1,
        object_indices=[objects["sugar_box"]["dict_idx"]],
    )
    with _environment_arguments():
        env = B1Z1PickMulti(
            cfg=cfg, rl_device=device, sim_device=device,
            graphics_device_id=-1, headless=True, use_roboinfo=True,
            observe_gait_commands=True, rand_control=False,
            camera_6p_tensor=camera, grasp_cv_tensor=grasps,
            cube_init_tensor=initial, cfg_terrain=B1Z1RoughCfg(),
        )
    env.ee_goal_global_orn_quat = torch.zeros((num_envs, 4), device=env.device)
    env.ee_goal_global_orn_quat[:, 3] = 1
    env.reset()
    return env


class Z1Kinematics:
    """Batched geometric IK for the exact six-joint Z1 chain in b1z1.urdf."""

    def __init__(self, device, dtype=torch.float32):
        self.device, self.dtype = device, dtype
        self.origins = torch.tensor([
            [0, 0, .0585], [0, 0, .045], [-.35, 0, 0],
            [.218, 0, .057], [.07, 0, 0], [.0492, 0, 0],
        ], device=device, dtype=dtype)
        self.axes = torch.tensor([
            [0, 0, 1], [0, 1, 0], [0, 1, 0],
            [0, 1, 0], [0, 0, 1], [1, 0, 0],
        ], device=device, dtype=dtype)
        self.tip = torch.tensor([.186, 0, 0], device=device, dtype=dtype)

    @staticmethod
    def rotation(axis, angles):
        batch = len(angles)
        eye = torch.eye(3, device=angles.device, dtype=angles.dtype).expand(batch, -1, -1)
        skew = torch.zeros((3, 3), device=angles.device, dtype=angles.dtype)
        skew[0, 1], skew[0, 2] = -axis[2], axis[1]
        skew[1, 0], skew[1, 2] = axis[2], -axis[0]
        skew[2, 0], skew[2, 1] = -axis[1], axis[0]
        return (eye + torch.sin(angles)[:, None, None] * skew
                + (1 - torch.cos(angles))[:, None, None] * (skew @ skew))

    def forward(self, joints):
        batch = len(joints)
        rotation = torch.eye(3, device=joints.device, dtype=joints.dtype).repeat(batch, 1, 1)
        position = torch.zeros((batch, 3), device=joints.device, dtype=joints.dtype)
        joint_positions, joint_axes = [], []
        for index in range(6):
            position = position + (rotation @ self.origins[index]).squeeze(-1)
            joint_positions.append(position)
            joint_axes.append((rotation @ self.axes[index]).squeeze(-1))
            rotation = rotation @ self.rotation(self.axes[index], joints[:, index])
        position = position + rotation @ self.tip
        axes = torch.stack(joint_axes, dim=2)
        origins = torch.stack(joint_positions, dim=2)
        linear = torch.cross(axes, position[:, :, None] - origins, dim=1)
        return position, rotation, torch.cat((linear, axes), dim=1)

    def solve(self, position, rotation, initial, lower, upper, iterations=160):
        """Return joint values and final position/orientation residuals."""
        joints = initial.clone()
        eye = torch.eye(6, device=joints.device, dtype=joints.dtype)
        for _ in range(iterations):
            current_position, current_rotation, jacobian = self.forward(joints)
            orientation_error = .5 * torch.cross(
                current_rotation, rotation, dim=1).sum(dim=2)
            error = torch.cat((position - current_position, orientation_error), dim=1)
            delta = jacobian.transpose(1, 2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(1, 2) + .0025 * eye,
                error[:, :, None],
            )
            joints = torch.maximum(torch.minimum(
                joints + delta.squeeze(-1).clamp(-.15, .15), upper), lower)
        current_position, current_rotation, _ = self.forward(joints)
        rotation_cos = ((current_rotation.transpose(1, 2) @ rotation).diagonal(
            dim1=1, dim2=2).sum(dim=1) - 1) * .5
        return joints, (position-current_position).norm(dim=1), rotation_cos.clamp(-1, 1).acos()


def quaternion_matrix(quaternion):
    """xyzw quaternion to a rotation matrix, without SciPy/GPU transfers."""
    from isaacgym.torch_utils import quat_rotate
    basis = torch.eye(3, device=quaternion.device, dtype=quaternion.dtype)
    return torch.stack([
        quat_rotate(quaternion, basis[index].expand(len(quaternion), -1))
        for index in range(3)
    ], dim=2)


def synchronize_targets(env, world_position, world_quaternion, gripper_target):
    from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate, euler_from_quat
    env.update_roboinfo()
    env.curr_ee_goal_cart[:] = quat_rotate_inverse(
        env._robot_root_states[:, 3:7], world_position - env.arm_base)
    local_quaternion = quat_mul(
        quat_conjugate(env._robot_root_states[:, 3:7]), world_quaternion)
    angles = torch.stack(euler_from_quat(local_quaternion), dim=1)
    env.curr_ee_goal_orn_rpy[:] = torch.remainder(angles + np.pi, 2*np.pi) - np.pi
    env.ee_goal_global_cart = world_position.clone()
    env.ee_goal_global_orn_quat = world_quaternion.clone()
    env.gripper_dof_pos[:] = gripper_target


def snapshot(env):
    """Copy only reset state, never simulator handles or opaque contact caches."""
    return {
        "robot_root": env._robot_root_states.clone(),
        "object_root": env._cube_root_states.clone(),
        "table_root": env._table_root_states.clone(),
        "dof_pos": env._dof_pos.clone(),
        "dof_vel": env._dof_vel.clone(),
        "ee_goal_cart": env.curr_ee_goal_cart.clone(),
        "ee_goal_orn_rpy": env.curr_ee_goal_orn_rpy.clone(),
        "gripper_target": env.gripper_dof_pos.clone(),
        "table_vy": env.vel_line_forward_y_level02.clone(),
        "rotation_enabled": env.lin_circle_select.clone(),
        "table_yaw_rate": env.vel_angular_yaw.clone(),
        "table_change_interval": env.rand_change_step.to(torch.float32).clone(),
        "table_height": env.table_heights.clone(),
        "rigid_body_state": env._rigid_body_state.view(env.num_envs, -1, 13).clone(),
        "ee_jacobian": env.ee_j_eef.clone(),
        # Evidence from this simulated frame; never restored into live contacts.
        "initial_foot_contact_force": grounding_state(env)[1].clone(),
    }


def restore_snapshot(env, state):
    """Replay the training reset contract without advancing the simulator."""
    from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul, quat_rotate_inverse
    from envs.b1z1_base import B1Z1Base
    ids = torch.arange(env.num_envs, device=env.device)
    B1Z1Base._reset_env_tensors(env, ids)
    for name in ("actions", "gait_indices", "clock_inputs", "gait_wait_timer",
                 "is_walking", "low_obs_history_buf", "action_data_history_buf",
                 "states_history_buf", "timeout_buf", "foot_contacts_from_sensor",
                 "force_sensor_tensor", "_contact_forces"):
        value = getattr(env, name, None)
        if value is not None:
            value[:] = 0
    env.rand_change_step[:] = state.get("table_change_interval", 50)
    env.surpass_dist_flag[:] = True
    env.lifted_object[:] = False
    env.lifted_now[:] = False
    env.one_epoch_step[:] = 0
    env._robot_root_states[:] = state["robot_root"]
    env._table_root_states[:] = state["table_root"]
    env._cube_root_states[:] = state["object_root"]
    env._dof_pos[:] = state["dof_pos"]
    env._dof_vel[:] = state["dof_vel"]
    env._last_dof_vel[:] = state["dof_vel"]
    env.curr_ee_goal_cart[:] = state["ee_goal_cart"]
    env.curr_ee_goal_orn_rpy[:] = state["ee_goal_orn_rpy"]
    env.gripper_dof_pos[:] = state["gripper_target"]
    env.gripper_dof_vel[:] = 0
    env.table_heights[:] = state["table_height"]
    env.table_commands[:] = 0
    env.table_commands[:, 4] = state["table_height"] - env.table_dimz / 2
    env.vel_line_forward_y_level02[:] = state["table_vy"]
    env.lin_circle_select[:] = state["rotation_enabled"]
    env.vel_angular_yaw[:] = state["table_yaw_rate"]
    env._sgs_pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    env._sgs_has_pending_reset = True
    env._rigid_body_state.view(env.num_envs, -1, 13)[:] = state["rigid_body_state"]
    env.ee_j_eef[:] = state["ee_jacobian"]
    env._update_base_yaw_quat()
    env.update_roboinfo()
    env.last_ee_pos[:] = quat_rotate_inverse(
        env._robot_root_states[:, 3:7], env.ee_pos - env.arm_base)
    env.ee_goal_global_cart = (
        quat_apply(env._robot_root_states[:, 3:7], env.curr_ee_goal_cart) + env.arm_base)
    local = quat_from_euler_xyz(*env.curr_ee_goal_orn_rpy.unbind(-1))
    env.ee_goal_global_orn_quat = quat_mul(env._robot_root_states[:, 3:7], local)
    env.curr_dist[:] = (env.ee_pos - env._cube_root_states[:, :3]).norm(dim=1)
    env.closest_dist[:] = env.curr_dist
    env.curr_height[:] = env._cube_root_states[:, 2] - env.table_heights - env.init_height
    env.highest_object[:] = env.curr_height
    env._compute_observations()


def restored_physics_tick(env, state):
    """Commit a cold reset with exactly the first real controller/physics tick."""
    from isaacgym import gymtorch
    from envs.b1z1_base import reindex_all
    restore_snapshot(env, state)
    with torch.no_grad():
        env._compute_low_level_observations()
        env.last_low_actions[:] = reindex_all(
            env.low_level_policy(env.low_obs_buf.detach(), hist_encoding=True))
        env.floating_table_control()
        targets = env.get_all_pos_targets(env.ee_goal_global_cart, env.ee_goal_global_orn_quat)
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env._refresh_sim_tensors()
        env.update_roboinfo()
        env._compute_observations()


class PhysicsBankBuilder:
    """Direct reset geometry with a single physics tick, without task rollouts."""

    FAMILY_NAMES = ("full_task", "pregrasp", "at_grasp")
    NON_GRASP_CONTACT_LIMIT_N = 5.
    GRASP_CONTACT_LIMIT_N = 250.
    AT_GRASP_POSITION_TOLERANCE_M = .005
    AT_GRASP_ANGLE_TOLERANCE_RAD = .10

    def __init__(self, env, per_family=512, seed=43, reset_config=None):
        self.env = env
        self.per_family, self.seed = int(per_family), int(seed)
        if self.per_family < 1:
            raise ValueError("Family size must be positive")
        self.reset_config = validate_build_config(
            load_build_config() if reset_config is None else reset_config)
        self.kinematics = Z1Kinematics(env.device)
        self.parts, self.counts, self.audit = [[], [], []], [0, 0, 0], []

    def _sample_range(self, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(self.env.num_envs, device=self.env.device)

    @staticmethod
    def _in_range(values, bounds, singleton_tolerance=SINGLE_VALUE_POSITION_TOLERANCE_M):
        low, high = bounds
        tolerance = singleton_tolerance if low == high else 0.
        return (values >= low - tolerance - 1e-6) & (values <= high + tolerance + 1e-6)

    def _base_in_range(self, state, base):
        root, obj = state["robot_root"], state["object_root"]
        delta = obj[:, :2] - root[:, :2]
        x, y, z, w = root[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        heading = yaw - torch.atan2(delta[:, 1], delta[:, 0])
        heading = torch.remainder(heading + np.pi, 2 * np.pi) - np.pi
        low, high = base["heading_offset_range_deg"]
        # The -180/+180 seam is the same physical orientation. Fixed headings
        # use a circular tolerance, rather than a linear interval at that seam.
        offset = torch.remainder(torch.rad2deg(heading) - low, 360.)
        angular_margin = SINGLE_VALUE_HEADING_TOLERANCE_DEG if low == high else 1e-4
        angle_valid = (offset <= high - low + angular_margin) | (offset >= 360. - angular_margin)
        return (self._in_range(delta[:, 0], base["x_range_m"])
                & self._in_range(delta[:, 1], base["y_range_m"])
                & angle_valid)

    def _place_robot(self, base):
        """Move a grounded, ungrasped robot and its world-frame caches together."""
        from isaacgym.torch_utils import euler_from_quat, quat_from_euler_xyz, quat_mul
        env = self.env
        root = env._robot_root_states
        old_position = root[:, :3].clone()
        dx = self._sample_range(base["x_range_m"])
        dy = self._sample_range(base["y_range_m"])
        yaw = torch.atan2(dy, dx) + torch.deg2rad(self._sample_range(base["heading_offset_range_deg"]))
        old_yaw = euler_from_quat(root[:, 3:7])[2]
        zero = torch.zeros_like(yaw)
        rotation_quat = quat_from_euler_xyz(zero, zero, yaw - old_yaw)
        rotation = quaternion_matrix(rotation_quat)
        root[:, 0] = env._cube_root_states[:, 0] - dx
        root[:, 1] = env._cube_root_states[:, 1] - dy
        root[:, 3:7] = quat_mul(rotation_quat, root[:, 3:7])
        for section in (slice(7, 10), slice(10, 13)):
            root[:, section] = (rotation @ root[:, section, None]).squeeze(-1)
        bodies = env._rigid_body_state.view(env.num_envs, -1, 13)[:, :len(env.body_names)]
        bodies[:, :, :3] = (root[:, None, :3] + torch.einsum(
            "nij,nkj->nki", rotation, bodies[:, :, :3] - old_position[:, None, :]))
        body_rotations = rotation_quat[:, None, :].expand(-1, bodies.shape[1], -1)
        bodies[:, :, 3:7] = quat_mul(body_rotations.reshape(-1, 4),
                                    bodies[:, :, 3:7].reshape(-1, 4)).view(env.num_envs, -1, 4)
        for section in (slice(7, 10), slice(10, 13)):
            bodies[:, :, section] = torch.einsum("nij,nkj->nki", rotation, bodies[:, :, section])
        for section in (slice(0, 3), slice(3, 6)):
            env.ee_j_eef[:, section, :] = rotation @ env.ee_j_eef[:, section, :]
        env._sgs_pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._sgs_has_pending_reset = True
        env._update_base_yaw_quat()
        env.update_roboinfo()

    def _opening_to_joint(self, opening):
        """0 = closed upper limit, 1 = open lower limit, as in set_gripper."""
        return (self.env.dof_limits_upper[-1] + opening *
                (self.env.dof_limits_lower[-1] - self.env.dof_limits_upper[-1]))

    def _joint_to_opening(self, joint):
        return ((joint - self.env.dof_limits_upper[-1]) /
                (self.env.dof_limits_lower[-1] - self.env.dof_limits_upper[-1]))

    def _ground_robot(self):
        """Align the actual foot collision geometry to the flat ground plane."""
        env = self.env
        bottom, _ = grounding_state(env)
        delta_z = .001 - bottom.amin(dim=1)
        env._robot_root_states[:, 2] += delta_z
        env._rigid_body_pos[:, :len(env.body_names), 2] += delta_z[:, None]
        env._robot_root_states[:, 7:] = 0
        env._dof_vel[:] = 0
        env._rigid_body_state.view(env.num_envs, -1, 13)[:, :len(env.body_names), 7:] = 0
        env._sgs_pending_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        env._sgs_has_pending_reset = True
        env.update_roboinfo()

    def _pose_tick(self, gripper):
        """Upload roots and joints once; obtain real FK/contact data in 5 ms."""
        from isaacgym import gymtorch
        env = self.env
        env._dof_vel[:] = 0
        env._robot_root_states[:, 7:] = 0
        env._cube_root_states[:, 7:] = 0
        env._table_root_states[:, 7:] = 0
        env.gripper_dof_pos[:] = gripper
        robots = env._robot_actor_ids.to(torch.int32)
        actors = torch.cat((robots, env._cube_actor_ids.to(torch.int32),
                            env._table_actor_ids.to(torch.int32)))
        env.gym.set_dof_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env._dof_state),
                                           gymtorch.unwrap_tensor(robots), len(robots))
        env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env._root_states),
                                                  gymtorch.unwrap_tensor(actors), len(actors))
        env._sgs_pending_reset.zero_()
        env._sgs_has_pending_reset = False
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env._dof_pos.contiguous()))
        env.last_low_actions.zero_()
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env._refresh_sim_tensors()
        env.update_roboinfo()
        synchronize_targets(env, env.ee_pos.clone(), env.ee_orn.clone(), gripper)

    def _basic_valid(self):
        """Basic pose/joint/ground/contact checks; no stable-stance requirement."""
        env = self.env
        roots = torch.stack((env._robot_root_states, env._cube_root_states, env._table_root_states), dim=1)
        allowed = set(FOOT_NAMES) | {"gripperStator", "gripperMover", "ee_gripper_link", "link06"}
        prohibited = [i for name, i in env.body_names_to_idx.items() if name not in allowed]
        fingers = [i for name, i in env.body_names_to_idx.items()
                   if name in allowed and name not in FOOT_NAMES]
        forces = env._contact_forces
        body_collision = forces[:, prohibited].norm(dim=-1).amax(dim=1) > self.NON_GRASP_CONTACT_LIMIT_N
        grasp_collision = forces[:, fingers].norm(dim=-1).amax(dim=1) > self.GRASP_CONTACT_LIMIT_N
        height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
        return (torch.isfinite(roots).all(dim=(1, 2))
                & torch.isfinite(env._dof_pos).all(dim=1)
                & torch.isfinite(env._dof_vel).all(dim=1)
                & torch.isfinite(forces).all(dim=(1, 2))
                & (env._dof_pos >= env.dof_limits_lower - .002).all(dim=1)
                & (env._dof_pos <= env.dof_limits_upper + .002).all(dim=1)
                & approx_grounded_mask(grounding_state(env)[0])
                & ~body_collision & ~grasp_collision
                & (height.abs() < .02))

    def _state(self, candidate=None):
        state = snapshot(self.env)
        count, device = self.env.num_envs, self.env.device
        state["grasp_candidate_index"] = (torch.full((count,), -1., device=device)
                                            if candidate is None else candidate.to(torch.float32))
        state["gripper_opening"] = self._joint_to_opening(state["dof_pos"][:, -1])
        return state

    def _full(self):
        env = self.env
        env.reset_idx()
        self._place_robot(self.reset_config["full_task"]["base"])
        self._ground_robot()
        env._dof_pos[:, 12:18] = env._initial_dof_pos[:, 12:18]
        opened = self._opening_to_joint(torch.ones(env.num_envs, device=env.device))[:, None]
        env._dof_pos[:, -1:] = opened
        self._pose_tick(opened)
        state = self._state()
        valid = (self._basic_valid() & self._base_in_range(state, self.reset_config["full_task"]["base"])
                 & ((env._dof_pos[:, 12:18] - env._initial_dof_pos[:, 12:18]).abs().amax(dim=1) < .03))
        return state, valid

    def _grasp_pose(self, initial, family):
        from isaacgym.torch_utils import euler_from_quat, quat_from_euler_xyz, quat_rotate_inverse
        env = self.env
        restore_snapshot(env, initial)
        cfg = self.reset_config[family]
        self._place_robot(cfg["base"])
        self._ground_robot()
        ids = torch.arange(env.num_envs, device=env.device)
        pose = torch.cat((env._cube_root_states[:, :3],
                          torch.stack(euler_from_quat(env._cube_root_states[:, 3:7]), dim=1)), dim=1)
        points, angles = env.get_grasp_world(pose, ids)
        candidate = torch.randint(points.shape[1], (env.num_envs,), device=env.device)
        point = points[ids, candidate]
        quaternion = quat_from_euler_xyz(*angles[ids, candidate].unbind(-1))
        rotation = quaternion_matrix(quaternion)
        if family == "pregrasp":
            distance = self._sample_range(cfg["ee_approach_distance_range_m"])
            opening = torch.ones(env.num_envs, device=env.device)
        else:
            distance = torch.zeros(env.num_envs, device=env.device)
            opening = self._sample_range(cfg["gripper_opening_range"])
        target = point - distance[:, None] * rotation[:, :, 0]
        base_rotation = quaternion_matrix(env._robot_root_states[:, 3:7])
        joints, position_error, angle_error = self.kinematics.solve(
            quat_rotate_inverse(env._robot_root_states[:, 3:7], target - env.arm_base),
            base_rotation.transpose(1, 2) @ rotation, env._dof_pos[:, 12:18],
            env.dof_limits_lower[12:18], env.dof_limits_upper[12:18])
        gripper = self._opening_to_joint(opening)[:, None]
        env._dof_pos[:, 12:18] = joints
        env._dof_pos[:, -1:] = gripper
        self._pose_tick(gripper)
        state = self._state(candidate)
        # Verify the chosen candidate in the object's actual pose after the tick.
        pose = torch.cat((env._cube_root_states[:, :3],
                          torch.stack(euler_from_quat(env._cube_root_states[:, 3:7]), dim=1)), dim=1)
        points, angles = env.get_grasp_world(pose, ids)
        point = points[ids, candidate]
        quaternion = quat_from_euler_xyz(*angles[ids, candidate].unbind(-1))
        actual_distance = (env.ee_pos - point).norm(dim=1)
        angular_error = 2 * (env.ee_orn * quaternion).sum(dim=1).abs().clamp(max=1).acos()
        valid = (self._basic_valid() & self._base_in_range(state, cfg["base"])
                 & (position_error < .003) & (angle_error < .05)
                 & (angular_error < self.AT_GRASP_ANGLE_TOLERANCE_RAD))
        if family == "pregrasp":
            valid &= self._in_range(actual_distance, cfg["ee_approach_distance_range_m"])
        else:
            valid &= actual_distance <= self.AT_GRASP_POSITION_TOLERANCE_M
            # Fixed-aperture settings allow a small normalized response tolerance.
            valid &= self._in_range(state["gripper_opening"], cfg["gripper_opening_range"], .005)
        return state, valid

    def _collect(self, family, state, valid):
        chosen = valid.nonzero(as_tuple=False).flatten()[:self.per_family - self.counts[family]]
        if not len(chosen):
            return 0
        rows = {key: value[chosen].detach().cpu() for key, value in state.items()}
        rows["family_id"] = torch.full((len(chosen),), family, dtype=torch.long)
        self.parts[family].append(rows)
        self.counts[family] += len(chosen)
        return len(chosen)

    def build(self, max_batches=512):
        import time
        from utils.sgs import ConfigurationBank
        env, started = self.env, time.monotonic()
        for batch in range(int(max_batches)):
            if min(self.counts) >= self.per_family:
                break
            batch_started = time.monotonic()
            initial, full_valid = self._full()
            accepted = [0, 0, 0]
            if self.counts[0] < self.per_family:
                accepted[0] = self._collect(0, initial, full_valid)
            for family in (1, 2):
                if self.counts[family] < self.per_family:
                    state, valid = self._grasp_pose(initial, self.FAMILY_NAMES[family])
                    accepted[family] = self._collect(family, state, valid)
            record = {"batch": batch + 1, "batch_size": env.num_envs,
                      "accepted_this_batch": accepted, "accepted_totals": self.counts.copy(),
                      "batch_seconds": round(time.monotonic() - batch_started, 3),
                      "elapsed_seconds": round(time.monotonic() - started, 3)}
            self.audit.append(record)
            print("SGS_BUILD", record, flush=True)
        if min(self.counts) < self.per_family:
            raise RuntimeError(f"Geometry sampling exhausted {max_batches} batches: accepted "
                               f"{self.counts}, required {self.per_family} per family. No bank saved.")
        tensors = {key: torch.cat([part[key] for family in self.parts for part in family])
                   for key in self.parts[0][0]}
        metadata = {
            "version": "b1z1-sugar-sgs-v5", "object_name": env.obj_list[0],
            "robot_asset": env.cfg["env"]["asset"]["assetFileRobot"],
            "num_dofs": env.num_dofs, "num_bodies": env._rigid_body_pos.shape[1],
            "hold_steps": env.hold_steps, "lifted_threshold": env.lifted_success_threshold,
            "seed": self.seed, "source_num_envs": env.num_envs, "table_level": "Level02",
            "coordinate_convention": "existing_reset_tensor_coordinates_no_env_origin_offset",
            "family_names": list(self.FAMILY_NAMES),
            "construction": {
                "reset_config": deepcopy(self.reset_config),
                "sampling": "uniform geometry targets followed by IK and basic physics filtering",
                "base_displacement": "object world XY minus robot root world XY",
                "heading_offset": "robot yaw minus bearing to object, degrees",
                "singleton_position_tolerance_m": SINGLE_VALUE_POSITION_TOLERANCE_M,
                "singleton_heading_tolerance_deg": SINGLE_VALUE_HEADING_TOLERANCE_DEG,
                "at_grasp_position_tolerance_m": self.AT_GRASP_POSITION_TOLERANCE_M,
                "at_grasp_angle_tolerance_rad": self.AT_GRASP_ANGLE_TOLERANCE_RAD,
                "gripper_opening": "0=closed upper joint limit, 1=open lower joint limit",
            },
            "physical_validation": {
                "passed": True, "validation_profile": "geometry_v1", "feet_near_ground": True,
                "collision_checked": True, "object_attachment": False,
                "grounding": {"foot_body_indices": foot_indices(env), "foot_radius_m": FOOT_RADIUS,
                              "limits": APPROX_GROUND_LIMITS.copy()},
                "collision_check": {"method": "joint limits, one PhysX tick, non-grasp contact forces and object support",
                                    "non_grasp_contact_limit_n": self.NON_GRASP_CONTACT_LIMIT_N,
                                    "grasp_contact_limit_n": self.GRASP_CONTACT_LIMIT_N},
                "physics_ticks_per_pose_check": 1, "stable_standing_required": False,
                "scripted_grasp_required": False, "cross_slot_rollout_required": False,
                "simulator": "Isaac Gym GPU PhysX", "accepted_per_family": self.counts.copy(),
                "batches": self.audit,
            },
        }
        return ConfigurationBank(tensors, metadata)
