"""Offline, physics-validated reset-bank construction for B1Z1PickMulti.

This module never runs during training reset.  Isaac Gym must be imported before
importing this module in an executable (Isaac Gym's PyTorch import requirement).
"""

from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
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
    grasp_contact_body_indices,
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


def pregrasp_local_offset(distance, lateral, vertical):
    """Keep the requested Euclidean distance while varying candidate-local y/z."""
    axial_squared = distance.square() - lateral.square() - vertical.square()
    if bool((axial_squared <= 0).any()):
        raise ValueError("pregrasp lateral/vertical offsets must fit inside the requested distance")
    return torch.stack((-axial_squared.sqrt(), lateral, vertical), dim=-1)


def quaternion_angles_deg(quaternion):
    from isaacgym.torch_utils import euler_from_quat
    # The task's nominal object quaternion uses rounded 0.707 components.
    # Euler extraction assumes unit quaternions; normalize relative rotations
    # so yaw acceptance near a configured boundary measures the actual angle.
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    angles = torch.stack(euler_from_quat(quaternion), dim=-1)
    return torch.rad2deg(torch.remainder(angles + np.pi, 2 * np.pi) - np.pi)


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
    """Diverse reset geometry with a bounded physics window, without task rollouts."""

    FAMILY_NAMES = ("full_task", "pregrasp", "at_grasp", "near_goal")
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
            load_build_config(lifted_threshold=env.lifted_success_threshold) if reset_config is None else reset_config,
            lifted_threshold=env.lifted_success_threshold)
        self.family_names = self.FAMILY_NAMES[:4 if "near_goal" in self.reset_config else 3]
        self.kinematics = Z1Kinematics(env.device)
        self.parts = [[] for _ in self.family_names]
        self.counts, self.audit = [0] * len(self.family_names), []
        from utils.sgs_build_audit import BuildAudit
        self.sampling_audit = BuildAudit(self.family_names)
        if "near_goal" in self.reset_config:
            # Exact source collision vertices for checking that the held object
            # clears the support plane. The builder currently supports sugar_box.
            asset = env.cfg["env"]["asset"]
            mesh = Path(asset["assetRoot"]) / asset["assetFileObj"] / "sugar_box/collision.obj"
            vertices = [list(map(float, line.split()[1:4])) for line in mesh.read_text().splitlines()
                        if line.startswith("v ")]
            self.object_vertices = torch.tensor(vertices, device=env.device) * asset["asset_multi"]["sugar_box"].get("scale", 1.)

    def _sample_range(self, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(self.env.num_envs, device=self.env.device)

    @staticmethod
    def _in_range(values, bounds, singleton_tolerance=SINGLE_VALUE_POSITION_TOLERANCE_M):
        low, high = bounds
        tolerance = singleton_tolerance if low == high else 0.
        return (values >= low - tolerance - 1e-6) & (values <= high + tolerance + 1e-6)

    @staticmethod
    def _angle_in_range(values, bounds):
        low, high = bounds
        offset = torch.remainder(values - low, 360.)
        margin = SINGLE_VALUE_HEADING_TOLERANCE_DEG if low == high else 1e-4
        return (offset <= high - low + margin) | (offset >= 360. - margin)

    def _base_in_range(self, state, base):
        root, obj = state["robot_root"], state["object_root"]
        delta = obj[:, :2] - root[:, :2]
        x, y, z, w = root[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        heading = yaw - torch.atan2(delta[:, 1], delta[:, 0])
        heading = torch.remainder(heading + np.pi, 2 * np.pi) - np.pi
        # The -180/+180 seam is the same physical orientation. Fixed headings
        # use a circular tolerance, rather than a linear interval at that seam.
        angle_valid = self._angle_in_range(torch.rad2deg(heading), base["heading_offset_range_deg"])
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

    def _physics_window(self, gripper):
        """Upload once, hold sampled joints, and check every continuous step.

        Robot/object velocities are cleared only before the window. Repeatedly
        resetting them would hide impulses. The floating support table is held
        at its nominal pose, consistent with training's per-tick height control.
        """
        from isaacgym import gymtorch
        env = self.env
        env._dof_vel[:] = 0
        env._robot_root_states[:, 7:] = 0
        env._cube_root_states[:, 7:] = 0
        env._table_root_states[:, 7:] = 0
        env._table_root_states[:, 2] = env.table_heights - env.table_dimz / 2
        table_target = env._table_root_states.clone()
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
        targets = env._dof_pos.clone().contiguous()
        object_position = env._cube_root_states[:, :3].clone()
        peak_force = torch.zeros(env.num_envs, device=env.device)
        displacement = torch.zeros_like(peak_force)
        checks = None
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.last_low_actions.zero_()
        tables = env._table_actor_ids.to(torch.int32)
        for step in range(self.reset_config["validation"]["physics_steps"]):
            if step:
                # This heavy floating table has gravity enabled. Training also
                # controls it every tick; letting it free-fall here would make
                # the saved support height inconsistent with the reset command.
                # Only the table is commanded; robot/object states stay live.
                env._table_root_states[:] = table_target
                env.gym.set_actor_root_state_tensor_indexed(
                    env.sim, gymtorch.unwrap_tensor(env._root_states),
                    gymtorch.unwrap_tensor(tables), len(tables))
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env._refresh_sim_tensors()
            current = self._frame_checks()
            checks = current if checks is None else {
                key: checks[key] & current[key] for key in checks}
            peak_force = torch.maximum(peak_force, env._contact_forces.norm(dim=-1).amax(dim=1))
            displacement = torch.maximum(
                displacement, (env._cube_root_states[:, :3] - object_position).norm(dim=1))
        env.update_roboinfo()
        synchronize_targets(env, env.ee_pos.clone(), env.ee_orn.clone(), gripper)
        validation = self.reset_config["validation"]
        checks["peak_contact_force"] = peak_force <= validation["max_contact_force_n"]
        checks["object_displacement"] = displacement <= validation["max_object_displacement_m"]
        checks["feet_near_ground"] = approx_grounded_mask(grounding_state(env)[0])
        height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
        checks["object_on_table"] = height.abs() < .02
        evidence = {"validation_peak_contact_force_n": peak_force,
                    "validation_max_object_displacement_m": displacement}
        return checks, evidence

    def _frame_checks(self):
        """Checks accumulated across the window; final grounding is separate."""
        env = self.env
        roots = torch.stack((env._robot_root_states, env._cube_root_states, env._table_root_states), dim=1)
        allowed = set(FOOT_NAMES) | {"gripperStator", "gripperMover", "ee_gripper_link", "link06"}
        prohibited = [i for name, i in env.body_names_to_idx.items() if name not in allowed]
        fingers = [i for name, i in env.body_names_to_idx.items()
                   if name in allowed and name not in FOOT_NAMES]
        forces = env._contact_forces
        return {
            "finite_state": (torch.isfinite(roots).all(dim=(1, 2))
                & torch.isfinite(env._dof_pos).all(dim=1)
                & torch.isfinite(env._dof_vel).all(dim=1)
                & torch.isfinite(forces).all(dim=(1, 2))),
            "joint_limits": ((env._dof_pos >= env.dof_limits_lower - .002).all(dim=1)
                             & (env._dof_pos <= env.dof_limits_upper + .002).all(dim=1)),
            "non_grasp_collision": (forces[:, prohibited].norm(dim=-1)
                                    <= self.NON_GRASP_CONTACT_LIMIT_N).all(dim=1),
            "grasp_collision": (forces[:, fingers].norm(dim=-1)
                                <= self.GRASP_CONTACT_LIMIT_N).all(dim=1),
        }

    def _state(self, evidence, candidate=None):
        from isaacgym.torch_utils import quat_mul, quat_conjugate
        state = snapshot(self.env)
        count, device = self.env.num_envs, self.env.device
        state["grasp_candidate_index"] = (torch.full((count,), -1., device=device)
                                            if candidate is None else candidate.to(torch.float32))
        state["gripper_opening"] = self._joint_to_opening(state["dof_pos"][:, -1])
        state["lift_height"] = state["object_root"][:, 2] - state["table_height"] - self.env.init_height
        state["object_yaw_offset_deg"] = quaternion_angles_deg(quat_mul(
            state["object_root"][:, 3:7], quat_conjugate(self.env.init_quat)))[:, 2]
        state["sampled_object_yaw_offset_deg"] = self.sampled_object_yaw.clone()
        for key in ("candidate_local_offset", "ee_rotation_offset_deg",
                    "sampled_candidate_local_offset", "sampled_ee_rotation_offset_deg"):
            state[key] = torch.zeros((count, 3), device=device)
        state.update(evidence)
        for key in ("near_goal_relative_slip_m", "near_goal_relative_rotation_deg",
                    "near_goal_contact_fraction", "near_goal_object_speed_m_s",
                    "near_goal_restore_verified", "sampled_lift_height"):
            state[key] = torch.zeros(count, device=device)
        return state

    def _full(self):
        from isaacgym.torch_utils import quat_mul, quat_from_euler_xyz
        env = self.env
        env.reset_idx()
        self.sampled_object_yaw = self._sample_range(self.reset_config["object"]["yaw_offset_range_deg"])
        zero = torch.zeros_like(self.sampled_object_yaw)
        env._cube_root_states[:, 3:7] = quat_mul(
            quat_from_euler_xyz(zero, zero, torch.deg2rad(self.sampled_object_yaw)), env.init_quat)
        self._place_robot(self.reset_config["full_task"]["base"])
        self._ground_robot()
        env._dof_pos[:, 12:18] = env._initial_dof_pos[:, 12:18]
        opened = self._opening_to_joint(torch.ones(env.num_envs, device=env.device))[:, None]
        env._dof_pos[:, -1:] = opened
        checks, evidence = self._physics_window(opened)
        state = self._state(evidence)
        checks["base_range"] = self._base_in_range(state, self.reset_config["full_task"]["base"])
        checks["object_yaw_range"] = self._angle_in_range(
            state["object_yaw_offset_deg"], self.reset_config["object"]["yaw_offset_range_deg"])
        checks["default_arm_pose"] = (
            env._dof_pos[:, 12:18] - env._initial_dof_pos[:, 12:18]).abs().amax(dim=1) < .03
        return state, checks

    def _grasp_pose(self, initial, family):
        from isaacgym.torch_utils import (
            euler_from_quat, quat_from_euler_xyz, quat_rotate_inverse, quat_rotate,
            quat_mul, quat_conjugate,
        )
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
        if family == "pregrasp":
            distance = self._sample_range(cfg["ee_approach_distance_range_m"])
            lateral = self._sample_range(cfg["ee_lateral_offset_range_m"])
            vertical = self._sample_range(cfg["ee_vertical_offset_range_m"])
            sample_geometry = lateral.square() + vertical.square() < distance.square()
            # Invalid combinations are counted as rejected proposals. Use finite
            # IK placeholders for those rows without shrinking the user's ranges.
            local_offset = pregrasp_local_offset(
                distance, torch.where(sample_geometry, lateral, 0.),
                torch.where(sample_geometry, vertical, 0.))
            rotation_offsets = torch.stack([self._sample_range(
                cfg["ee_rotation_offset_range_deg"][axis]) for axis in ("roll", "pitch", "yaw")], dim=1)
            opening = torch.ones(env.num_envs, device=env.device)
        else:
            local_offset = torch.zeros((env.num_envs, 3), device=env.device)
            rotation_offsets = torch.zeros_like(local_offset)
            opening = (torch.ones(env.num_envs, device=env.device) if family == "near_goal"
                       else self._sample_range(cfg["gripper_opening_range"]))
        target = point + quat_rotate(quaternion, local_offset)
        target_quaternion = quat_mul(quaternion, quat_from_euler_xyz(*torch.deg2rad(rotation_offsets).unbind(-1)))
        rotation = quaternion_matrix(target_quaternion)
        base_rotation = quaternion_matrix(env._robot_root_states[:, 3:7])
        joints, position_error, angle_error = self.kinematics.solve(
            quat_rotate_inverse(env._robot_root_states[:, 3:7], target - env.arm_base),
            base_rotation.transpose(1, 2) @ rotation, env._dof_pos[:, 12:18],
            env.dof_limits_lower[12:18], env.dof_limits_upper[12:18])
        gripper = self._opening_to_joint(opening)[:, None]
        env._dof_pos[:, 12:18] = joints
        env._dof_pos[:, -1:] = gripper
        window_checks, evidence = self._physics_window(gripper)
        state = self._state(evidence, candidate)
        # Verify the chosen candidate in the object's actual final pose.
        pose = torch.cat((env._cube_root_states[:, :3],
                          torch.stack(euler_from_quat(env._cube_root_states[:, 3:7]), dim=1)), dim=1)
        points, angles = env.get_grasp_world(pose, ids)
        point = points[ids, candidate]
        quaternion = quat_from_euler_xyz(*angles[ids, candidate].unbind(-1))
        actual_offset = quat_rotate_inverse(quaternion, env.ee_pos - point)
        actual_distance = actual_offset.norm(dim=1)
        actual_rotation = quaternion_angles_deg(quat_mul(quat_conjugate(quaternion), env.ee_orn))
        state["candidate_local_offset"] = actual_offset
        state["ee_rotation_offset_deg"] = actual_rotation
        state["sampled_candidate_local_offset"] = local_offset
        state["sampled_ee_rotation_offset_deg"] = rotation_offsets
        angular_error = 2 * (env.ee_orn * quaternion).sum(dim=1).abs().clamp(max=1).acos()
        checks = {**({"sample_geometry": sample_geometry} if family == "pregrasp" else {}),
                  "ik_position": position_error < .003, "ik_orientation": angle_error < .05,
                  **window_checks, "base_range": self._base_in_range(state, cfg["base"]),
                  "object_yaw_range": self._angle_in_range(
                      state["object_yaw_offset_deg"], self.reset_config["object"]["yaw_offset_range_deg"])}
        if family == "pregrasp":
            checks["ee_distance"] = self._in_range(actual_distance, cfg["ee_approach_distance_range_m"])
            checks["ee_approach_side"] = actual_offset[:, 0] < 0
            checks["ee_lateral_offset"] = self._in_range(actual_offset[:, 1], cfg["ee_lateral_offset_range_m"])
            checks["ee_vertical_offset"] = self._in_range(actual_offset[:, 2], cfg["ee_vertical_offset_range_m"])
            for index, axis in enumerate(("roll", "pitch", "yaw")):
                checks["ee_rotation_" + axis] = self._angle_in_range(
                    actual_rotation[:, index], cfg["ee_rotation_offset_range_deg"][axis])
        else:
            checks["ee_distance"] = actual_distance <= self.AT_GRASP_POSITION_TOLERANCE_M
            checks["ee_orientation"] = angular_error < self.AT_GRASP_ANGLE_TOLERANCE_RAD
            # Fixed-aperture settings allow a small normalized response tolerance.
            checks["gripper_opening"] = self._in_range(
                state["gripper_opening"], cfg.get("gripper_opening_range", [1., 1.]), .005)
        return state, checks

    def _grasp_relative_pose(self):
        from isaacgym.torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate
        env = self.env
        return (quat_rotate_inverse(env.ee_orn, env._cube_root_states[:, :3] - env.ee_pos),
                quat_mul(quat_conjugate(env.ee_orn), env._cube_root_states[:, 3:7]))

    def _grasp_geometry(self):
        """Contact evidence plus object center between the two finger surfaces."""
        from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse
        env = self.env
        cfg = self.reset_config["near_goal"]
        indices = grasp_contact_body_indices(env)
        contacts = (env._contact_forces[:, indices].norm(dim=-1)
                    >= cfg["min_finger_contact_force_n"]).all(dim=1)
        relative, _ = self._grasp_relative_pose()
        mover = env._rigid_body_state.view(env.num_envs, -1, 13)[:, indices[1]]
        tip = torch.tensor([.085, 0., 0.], device=env.device).expand(env.num_envs, -1)
        moving_tip = quat_rotate_inverse(env.ee_orn, mover[:, :3] + quat_rotate(mover[:, 3:7], tip) - env.ee_pos)
        between = ((relative[:, 0].abs() < .08) & (relative[:, 1].abs() < .08)
                   & (relative[:, 2] >= -.035) & (relative[:, 2] <= moving_tip[:, 2] + .015))
        rotation = quaternion_matrix(env._cube_root_states[:, 3:7])
        lowest_z = (rotation[:, 2, :] @ self.object_vertices.T).amin(dim=1) + env._cube_root_states[:, 2]
        clear = lowest_z - env.table_heights > .005
        return contacts, between, clear

    def _near_goal_step(self, world_position, world_quaternion, closed, step,
                        table_target=None):
        """Advance live robot/object physics; only the support table is controlled."""
        from isaacgym import gymtorch
        from envs.b1z1_base import reindex_all
        env = self.env
        env.gripper_dof_pos[:] = closed
        if table_target is not None:
            # The task's incremental Jacobian controller has a gravity tracking
            # offset. Accumulate a bounded setpoint correction during scripted
            # construction; retain this actual command in the saved reset state.
            self._near_command_offset += 6. * env.sim_params.dt * (world_position - env.ee_pos)
            self._near_command_offset.clamp_(-.15, .15)
            world_position = world_position + self._near_command_offset
        self._near_command_position = world_position.clone()
        ticks = self._near_elapsed_ticks
        env.progress_buf[:] = ticks // (env.control_freq_inv * env.control_freq_low)
        if ticks % env.control_freq_inv == 0:
            synchronize_targets(env, world_position, world_quaternion, closed)
            env._compute_low_level_observations()
            with torch.no_grad():
                env.last_low_actions[:] = reindex_all(env.low_level_policy(env.low_obs_buf.detach(), hist_encoding=True))
        if table_target is None:
            # Cold replay uses the actual low-level policy and Level02 controller.
            env.floating_table_control()
        else:
            env._table_root_states[:] = table_target
            tables = env._table_actor_ids.to(torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env._root_states), gymtorch.unwrap_tensor(tables), len(tables))
        targets = env.get_all_pos_targets(world_position, world_quaternion)
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
        env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
        env._refresh_sim_tensors()
        env.update_roboinfo()
        self._near_elapsed_ticks += 1

    def _near_goal_hold(self, world_position, world_quaternion, closed, table_target=None):
        env, cfg = self.env, self.reset_config["near_goal"]
        steps = math.ceil(cfg["hold_time_s"] / env.sim_params.dt)
        position0, quaternion0 = self._grasp_relative_pose()
        object0 = env._cube_root_states[:, :3].clone()
        slip = torch.zeros(env.num_envs, device=env.device)
        rotation_error = torch.zeros_like(slip)
        contact_frames = torch.zeros_like(slip)
        peak_force, displacement = torch.zeros_like(slip), torch.zeros_like(slip)
        checks = None
        for step in range(steps):
            self._near_goal_step(world_position, world_quaternion, closed, step, table_target)
            current = self._frame_checks()
            contacts, between, clear = self._grasp_geometry()
            height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
            current.update(held_region=between, object_clear_of_table=clear,
                           lift_height=self._in_range(height, cfg["object_lift_height_range_m"]),
                           not_task_success=(height > 0) & (height < env.lifted_success_threshold))
            checks = current if checks is None else {key: checks[key] & current[key] for key in checks}
            position, quaternion = self._grasp_relative_pose()
            slip = torch.maximum(slip, (position - position0).norm(dim=1))
            cosine = ((quaternion * quaternion0).sum(dim=1).abs() /
                      (quaternion.norm(dim=1) * quaternion0.norm(dim=1)).clamp_min(1e-12))
            rotation_error = torch.maximum(rotation_error, torch.rad2deg(2 * cosine.clamp(max=1).acos()))
            contact_frames += contacts.float()
            peak_force = torch.maximum(peak_force, env._contact_forces.norm(dim=-1).amax(dim=1))
            displacement = torch.maximum(displacement, (env._cube_root_states[:, :3] - object0).norm(dim=1))
        speed = env._cube_root_states[:, 7:10].norm(dim=1)
        contact_fraction = torch.where(contact_frames == steps, torch.ones_like(contact_frames), contact_frames / steps)
        checks.update(relative_slip=slip <= cfg["max_relative_slip_m"],
                      relative_rotation=rotation_error <= cfg["max_relative_rotation_deg"],
                      finger_contact=contact_fraction >= cfg["min_contact_fraction"],
                      final_finger_contact=contacts,
                      object_speed=speed <= cfg["max_object_speed_m_s"],
                      feet_near_ground=approx_grounded_mask(grounding_state(env)[0]),
                      peak_contact_force=peak_force <= self.reset_config["validation"]["max_contact_force_n"])
        evidence = {"near_goal_relative_slip_m": slip, "near_goal_relative_rotation_deg": rotation_error,
                    "near_goal_contact_fraction": contact_fraction, "near_goal_object_speed_m_s": speed,
                    "validation_peak_contact_force_n": peak_force,
                    "validation_max_object_displacement_m": displacement}
        return checks, evidence

    def _near_goal(self, initial):
        """Close and lift a real object, then verify a cold buffered reset."""
        env, cfg = self.env, self.reset_config["near_goal"]
        prepared, initial_checks = self._grasp_pose(initial, "near_goal")
        checks = {"start_" + key: value for key, value in initial_checks.items()}
        closed = self._opening_to_joint(torch.zeros(env.num_envs, device=env.device))[:, None]
        world_position, world_quaternion = env.ee_pos.clone(), env.ee_orn.clone()
        table = env._table_root_states.clone()
        table[:, 2] = env.table_heights - env.table_dimz / 2
        table[:, 7:] = 0
        self._near_elapsed_ticks = 0
        self._near_command_offset = torch.zeros_like(world_position)
        motion_peak = prepared["validation_peak_contact_force_n"].clone()
        # No object root/DOF writes after the initial open-gripper placement.
        for step in range(math.ceil(cfg["close_time_s"] / env.sim_params.dt)):
            self._near_goal_step(world_position, world_quaternion, closed, step, table)
            motion_peak = torch.maximum(motion_peak, env._contact_forces.norm(dim=-1).amax(dim=1))
            for key, value in self._frame_checks().items():
                name = "motion_" + key
                checks[name] = checks.get(name, torch.ones_like(value)) & value
        target_height = self._sample_range(cfg["object_lift_height_range_m"])
        height_now = env._cube_root_states[:, 2] - env.table_heights - env.init_height
        rise = target_height - height_now
        start = world_position.clone()
        lift_steps = math.ceil(cfg["lift_time_s"] / env.sim_params.dt)
        for step in range(lift_steps):
            fraction = (step + 1) / lift_steps
            # Smooth start/stop to avoid testing an artificial lift impulse.
            world_position = start.clone()
            world_position[:, 2] += rise * (fraction * fraction * (3 - 2 * fraction))
            self._near_goal_step(world_position, world_quaternion, closed, step, table)
            motion_peak = torch.maximum(motion_peak, env._contact_forces.norm(dim=-1).amax(dim=1))
            for key, value in self._frame_checks().items():
                checks["motion_" + key] &= value
        hold_checks, evidence = self._near_goal_hold(world_position, world_quaternion, closed, table)
        checks["motion_peak_contact_force"] = motion_peak <= self.reset_config["validation"]["max_contact_force_n"]
        evidence["validation_peak_contact_force_n"] = torch.maximum(evidence["validation_peak_contact_force_n"], motion_peak)
        checks.update({"hold_" + key: value for key, value in hold_checks.items()})
        # Save actual physical state with a closed command, not a forced closed
        # joint position. Collision-constrained fingers may remain partly open.
        synchronize_targets(env, self._near_command_position.clone(), world_quaternion, closed)
        state = self._state(evidence, prepared["grasp_candidate_index"])
        state.update(evidence)
        state["sampled_lift_height"] = target_height
        from isaacgym.torch_utils import euler_from_quat, quat_from_euler_xyz, quat_rotate_inverse, quat_mul, quat_conjugate
        ids = torch.arange(env.num_envs, device=env.device)
        pose = torch.cat((env._cube_root_states[:, :3],
                          torch.stack(euler_from_quat(env._cube_root_states[:, 3:7]), dim=1)), dim=1)
        points, angles = env.get_grasp_world(pose, ids)
        candidate = state["grasp_candidate_index"].long()
        candidate_quat = quat_from_euler_xyz(*angles[ids, candidate].unbind(-1))
        state["candidate_local_offset"] = quat_rotate_inverse(candidate_quat, env.ee_pos - points[ids, candidate])
        state["ee_rotation_offset_deg"] = quaternion_angles_deg(quat_mul(quat_conjugate(candidate_quat), env.ee_orn))
        checks["final_base_range"] = self._base_in_range(state, cfg["base"])
        checks["final_object_yaw"] = self._angle_in_range(state["object_yaw_offset_deg"],
                                                         self.reset_config["object"]["yaw_offset_range_deg"])
        # Contacts and low-level history are deliberately reconstructed, exactly
        # as in training; opaque PhysX solver caches are never stored in a bank.
        restore_snapshot(env, state)
        self._near_elapsed_ticks = 0
        replay_checks, replay_evidence = self._near_goal_hold(
            env.ee_goal_global_cart.clone(), env.ee_goal_global_orn_quat.clone(), closed)
        checks.update({"restore_" + key: value for key, value in replay_checks.items()})
        state["near_goal_restore_verified"] = torch.stack(list(replay_checks.values()), dim=1).all(dim=1).float()
        for key in ("near_goal_relative_slip_m", "near_goal_relative_rotation_deg",
                    "near_goal_object_speed_m_s", "validation_peak_contact_force_n",
                    "validation_max_object_displacement_m"):
            state[key] = torch.maximum(state[key], replay_evidence[key])
        state["near_goal_contact_fraction"] = torch.minimum(
            state["near_goal_contact_fraction"], replay_evidence["near_goal_contact_fraction"])
        return state, checks

    def _collect(self, family, state, checks):
        valid = torch.stack(list(checks.values()), dim=1).all(dim=1)
        chosen = valid.nonzero(as_tuple=False).flatten()[:self.per_family - self.counts[family]]
        self.sampling_audit.record(family, state["grasp_candidate_index"], checks, chosen)
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
            initial, full_checks = self._full()
            accepted = [0] * len(self.family_names)
            if self.counts[0] < self.per_family:
                accepted[0] = self._collect(0, initial, full_checks)
            for family in range(1, len(self.family_names)):
                if self.counts[family] < self.per_family:
                    state, checks = (self._near_goal(initial) if self.family_names[family] == "near_goal"
                                     else self._grasp_pose(initial, self.family_names[family]))
                    accepted[family] = self._collect(family, state, checks)
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
            "version": "b1z1-sugar-sgs-v9", "object_name": env.obj_list[0],
            "object_init_height_m": float(env.init_height[0].item()),
            "robot_asset": env.cfg["env"]["asset"]["assetFileRobot"],
            "num_dofs": env.num_dofs, "num_bodies": env._rigid_body_pos.shape[1],
            "hold_steps": env.hold_steps, "lifted_threshold": env.lifted_success_threshold,
            "seed": self.seed, "source_num_envs": env.num_envs, "table_level": "Level02",
            "coordinate_convention": "existing_reset_tensor_coordinates_no_env_origin_offset",
            "family_names": list(self.family_names),
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
                "pregrasp_position": "candidate local [-sqrt(d*d-y*y-z*z), y, z]; d is Euclidean distance",
                "pregrasp_orientation": "candidate quaternion * local roll-pitch-yaw delta quaternion",
                "object_yaw": "world-z rotation * normal-reset object quaternion; offline construction only",
            },
            "physical_validation": {
                "passed": True, "validation_profile": ("geometry_grasp_v2" if "near_goal" in self.family_names else "geometry_v1"),
                "feet_near_ground": True,
                "collision_checked": True, "object_attachment": False,
                "near_goal_grasp_verified": "near_goal" in self.family_names,
                "grounding": {"foot_body_indices": foot_indices(env), "foot_radius_m": FOOT_RADIUS,
                              "limits": APPROX_GROUND_LIMITS.copy()},
                "collision_check": {"method": "every-frame joint limits/contact forces, peak object displacement, final support",
                                    "non_grasp_contact_limit_n": self.NON_GRASP_CONTACT_LIMIT_N,
                                    "grasp_contact_limit_n": self.GRASP_CONTACT_LIMIT_N,
                                    "peak_contact_limit_n": self.reset_config["validation"]["max_contact_force_n"]},
                "max_object_displacement_m": self.reset_config["validation"]["max_object_displacement_m"],
                "physics_ticks_per_pose_check": self.reset_config["validation"]["physics_steps"],
                "physics_dt_seconds": env.cfg["sim"]["dt"],
                "table_control": "nominal support pose held each tick; Level02 motion resumes at training reset",
                "stable_standing_required": False,
                "scripted_grasp_required": "near_goal" in self.family_names,
                "validation_by_family": {name: ("physical close/lift/hold and cold-reset grip verification"
                                                if name == "near_goal" else "short geometric physics window")
                                         for name in self.family_names},
                "near_goal_contact_check": "both finger net contact forces, object between finger surfaces, collision mesh above table",
                "cross_slot_rollout_required": False,
                "simulator": "Isaac Gym GPU PhysX", "accepted_per_family": self.counts.copy(),
                "batches": self.audit,
            },
            "sampling_audit": self.sampling_audit.summary(tensors),
        }
        return ConfigurationBank(tensors, metadata)
