"""GPU task adapter for the SGS/EKF teacher world model.

The learned part predicts physical state and the frozen low-level controller's
memory. Known high-level command rules, reward history, and causal EKF history
are advanced analytically. No Isaac Gym import is needed by imagined rollouts.
The analytic-head/state-integration separation follows SuperMimic-Reproduced;
the DQ state layout and task equations below are specific to this repository.
"""

import math
import hashlib
from pathlib import Path
from collections import OrderedDict

import torch
from torch.nn import functional as F

from modules.pose_prediction import (
    adjoint_batch, poses_to_transforms, quaternion_to_matrix,
    rotation_and_jacobian, se3_exp_batch,
)
from utils.pick_success import advance_pick_hold, lifted_grasp_mask
from utils.gripper_control import GRIPPER_ACTION_MODE, CLOSING_OPENING_THRESHOLD, closing_command


def quat_normalize(q):
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def quat_conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def quat_mul(a, b):
    return torch.cat((a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
                      + torch.cross(a[..., :3], b[..., :3], dim=-1),
                      a[..., 3:] * b[..., 3:]
                      - (a[..., :3] * b[..., :3]).sum(-1, keepdim=True)), -1)


def quat_apply(q, v):
    vector, value = torch.broadcast_tensors(q[..., :3], v)
    cross = torch.cross(vector, value, dim=-1)
    return value + 2 * (q[..., 3:] * cross
                        + torch.cross(vector, cross, dim=-1))


def quat_inverse_apply(q, v):
    return quat_apply(quat_conjugate(q), v)


def safe_atan2(y, x):
    # Euler angles are undefined at a pole; choose zero there without allowing
    # an inactive atan2(0,0) branch to poison an otherwise valid BPTT gradient.
    degenerate = x.square()+y.square() < 1e-16
    return torch.atan2(torch.where(degenerate, torch.zeros_like(y), y),
                       torch.where(degenerate, torch.ones_like(x), x))


def euler(q):
    x, y, z, w = quat_normalize(q).unbind(-1)
    # Avoid infinite derivatives at Euler singularities. Normal teacher states
    # retain the existing observation convention to float32 precision.
    return torch.stack((safe_atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                        torch.asin((2*(w*y-z*x)).clamp(-1+1e-7, 1-1e-7)),
                        safe_atan2(2*(w*z+x*y), 1-2*(y*y+z*z))), -1)


def from_euler(rpy):
    r, p, y = (rpy / 2).unbind(-1)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                        cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy), -1)


def from_rotvec(vector):
    angle = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    factor = .5 * torch.sinc(angle / (2*math.pi))
    return torch.cat((vector * factor, torch.cos(angle / 2)), -1)


def matrix_euler_xyz(matrix):
    """PredictPoint.decompose_pose_to_xyz_rpy's XYZ convention, batched."""
    pitch = torch.asin((-matrix[..., 2, 0]).clamp(-1+1e-7, 1-1e-7))
    locked = matrix[..., 2, 0].abs() >= 1.
    roll = torch.where(locked, safe_atan2(-matrix[..., 0, 1], matrix[..., 0, 2]),
                       safe_atan2(matrix[..., 2, 1], matrix[..., 2, 2]))
    yaw = torch.where(locked, torch.zeros_like(pitch),
                      safe_atan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    return torch.stack((roll, pitch, yaw), -1)


def euler_matrix(rpy):
    """Same Rz @ Ry @ Rx evaluation order as PredictPoint._rpy_to_matrix."""
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr, cp, sp, cy, sy = roll.cos(), roll.sin(), pitch.cos(), pitch.sin(), yaw.cos(), yaw.sin()
    zero, one = torch.zeros_like(roll), torch.ones_like(roll)
    rz = torch.stack((cy,-sy,zero,sy,cy,zero,zero,zero,one),-1).reshape(*rpy.shape[:-1],3,3)
    ry = torch.stack((cp,zero,sp,zero,one,zero,-sp,zero,cp),-1).reshape(*rpy.shape[:-1],3,3)
    rx = torch.stack((one,zero,zero,zero,cr,-sr,zero,sr,cr),-1).reshape(*rpy.shape[:-1],3,3)
    return rz @ ry @ rx


def matrix_quaternion(matrix):
    """Stable xyzw conversion; inactive sqrt branches have finite gradients."""
    shape = matrix.shape[:-2]
    r = matrix.reshape(-1, 3, 3)
    a, b, c = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    squared = torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), -1)
    roots = squared.clamp_min(1e-12).sqrt()
    candidates = torch.stack((
        torch.stack((squared[:, 0], r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), -1),
        torch.stack((r[:, 2, 1]-r[:, 1, 2], squared[:, 1], r[:, 0, 1]+r[:, 1, 0], r[:, 0, 2]+r[:, 2, 0]), -1),
        torch.stack((r[:, 0, 2]-r[:, 2, 0], r[:, 0, 1]+r[:, 1, 0], squared[:, 2], r[:, 1, 2]+r[:, 2, 1]), -1),
        torch.stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0], r[:, 1, 2]+r[:, 2, 1], squared[:, 3]), -1),
    ), 1) / (2 * roots.unsqueeze(-1))
    selected = candidates.gather(1, roots.argmax(-1)[:, None, None].expand(-1, 1, 4)).squeeze(1)
    selected = quat_normalize(selected)
    selected = torch.where(selected[:, :1] < 0, -selected, selected)
    return torch.cat((selected[:, 1:], selected[:, :1]), -1).reshape(*shape, 4)


def se3_log(transform):
    q = matrix_quaternion(transform[:, :3, :3])
    sine = torch.linalg.vector_norm(q[:, :3], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[:, 3:])
    scale = torch.where(sine < 1e-8, torch.full_like(sine, 2.), angle / sine.clamp_min(1e-12))
    w = q[:, :3] * scale
    _, jacobian = rotation_and_jacobian(w)
    v = torch.linalg.solve(jacobian, transform[:, :3, 3:4]).squeeze(-1)
    return torch.cat((v, w), -1)


def causal_ekf(poses, ages, count, horizons, q_scale=1., r_scale=.01):
    """Functional version of RollingSE3Predictor.predict, with BPTT enabled.

    ``ages`` are timestamps relative to the newest sample. They avoid float32
    precision loss after long simulator runs. Unready histories are replaced
    before the batched solve and return the current pose with valid=False.
    """
    batch, history, _ = poses.shape
    ready = count.reshape(-1) >= history
    poses = torch.where(ready[:, None, None], poses, poses[:, -1:, :])
    times = torch.where(ready[:, None], ages, torch.zeros_like(ages))
    origin = poses[:, :1, :3]
    local = torch.cat((poses[..., :3]-origin, poses[..., 3:]), -1)
    measured = poses_to_transforms(local.reshape(-1, 7)).reshape(batch, history, 4, 4)
    transform = measured[:, 0]
    twist = poses.new_zeros(batch, 6)
    eye12 = torch.eye(12, device=poses.device, dtype=poses.dtype)
    eye6 = torch.eye(6, device=poses.device, dtype=poses.dtype)
    covariance = eye12.expand(batch, -1, -1)
    process = torch.diag(poses.new_tensor([1e-4]*9+[5e-4]*3)) * q_scale
    measurement_noise = eye6 * (1e-3 * r_scale)
    zero6 = poses.new_zeros(batch, 6, 6)
    bottom = poses.new_tensor([0., 0., 0., 1.]).expand(batch, 1, 4)
    for i in range(1, history):
        dt = times[:, i] - times[:, i-1]
        increment = se3_exp_batch(twist * dt[:, None])
        transform = transform @ increment
        transition = torch.cat((torch.cat((adjoint_batch(increment), eye6[None]*dt[:, None, None]), -1),
                                torch.cat((zero6, eye6.expand(batch, -1, -1)), -1)), -2)
        covariance = transition @ covariance @ transition.transpose(-1, -2) + process
        rotation = transform[:, :3, :3].transpose(-1, -2)
        inverse = torch.cat((torch.cat((rotation, -rotation @ transform[:, :3, 3:4]), -1), bottom), -2)
        innovation = se3_log(inverse @ measured[:, i])
        innovation_covariance = covariance[:, :6, :6] + measurement_noise
        gain = torch.linalg.solve(innovation_covariance.transpose(-1, -2), covariance[:, :, :6].transpose(-1, -2)).transpose(-1, -2)
        dx = (gain @ innovation.unsqueeze(-1)).squeeze(-1)
        covariance = covariance - gain @ covariance[:, :6, :]
        transform = transform @ se3_exp_batch(dx[:, :6])
        twist = twist + dx[:, 6:]
    horizon = poses.new_tensor(horizons)
    future = transform[:, None] @ se3_exp_batch((twist[:, None] * horizon[None, :, None]).reshape(-1, 6)).reshape(batch, len(horizons), 4, 4)
    prediction = torch.cat((future[..., :3, 3]+origin, matrix_quaternion(future[..., :3, :3])), -1)
    finite = torch.isfinite(prediction).all(-1).all(-1) & torch.isfinite(covariance).all(-1).all(-1)
    valid = ready & finite
    return torch.where(valid[:, None, None], prediction, poses[:, -1:, :]), valid


class TeacherWorldModelAdapter:
    """Full-state adapter for the existing nonvisual B1Z1PickMulti teacher.

    State XY is shifted by a fixed per-environment origin; world height and
    yaw are retained. ``context[:, 0]`` indexes immutable GPU lookup tables and
    is removed from the learned conditioning vector.
    """

    action_dim = 9
    termination_feature_dim = 8

    def __init__(self, env):
        self.env = env
        self.device = torch.device(env.device)
        self.dtype = env._robot_root_states.dtype
        unsupported = [name for name in ("tsil_enabled", "enable_camera", "floating_base", "pitch_control", "pred_success", "no_feature", "near_goal_stop", "obj_move_prob") if getattr(env, name, False)]
        if env.cfg["env"].get("lastCommands", False):
            unsupported.append("lastCommands")
        if not env.use_roboinfo or env.num_gripper_dof != 1 or env.num_actions != 9:
            unsupported.append("teacher requires roboinfo, one gripper DOF and nine actions")
        if env.cfg["env"].get("gripperActionMode") != GRIPPER_ACTION_MODE:
            unsupported.append("teacher requires continuous_opening_v1 gripper actions")
        if env.table_level not in ("Level00", "Level01", "Level02"):
            unsupported.append("table levels other than Level00/01/02")
        if getattr(env, "dr_randomizations", {}).get("observations") or getattr(env, "dr_randomizations", {}).get("actions"):
            unsupported.append("additional observation/action randomization")
        if unsupported:
            raise ValueError("World-model adapter does not support: " + ", ".join(unsupported))
        self.ndof = env._dof_pos.shape[-1]
        self.low_shape = tuple(env.low_obs_history_buf.shape[1:])
        self.history = int(getattr(env.object_pose_predictor, "history", 5))
        self.horizons = tuple(env.pose_prediction_horizons) if env.num_future_pose_obs else ()
        cfg = env.cfg["env"].get("posePrediction", {})
        self.q_scale, self.r_scale = cfg.get("qScale", 1.), cfg.get("rScale", .01)
        self.use_tanh = env.cfg["env"].get("useTanh", False)
        self.small_commands = env.cfg["env"].get("smallValueSetZero", False)
        self.forward_limit = .45 if env.commands_curriculum else .4
        self.stop_pick = env.stop_pick
        self.max_steps, self.hold_steps = env.max_episode_length, env.hold_steps
        self.lift_threshold = float(env.lifted_success_threshold)
        self.base_distance = float(env.base_object_distace_threshold)
        self.base_height_target = float(env.cfg["reward"]["base_height_target"])
        self.clip_actions, self.clip_obs = float(env.clip_actions), float(env.clip_obs)
        self.low_dt = float(env.dt)
        self.velocity_obs = env.num_velocity_obs != 0
        self.reward_scales = dict(env.reward_scales)
        allowed_rewards = {"approaching", "lifting", "pick_up", "acc_penalty", "command_reward", "action_rate", "ee_orn", "base_dir", "rad_penalty", "base_ang_pen", "base_approaching", "grasp_base_height", "gripper_rate", "limit_yaw_rotation_penalty", "base_height", "action_penalty"}
        unknown = set(self.reward_scales) - allowed_rewards
        if unknown:
            raise ValueError("World-model reward adapter has no exact equation for: " + ", ".join(sorted(unknown)))
        self.only_positive = env.cfg["reward"]["only_positive_rewards"]
        widths = OrderedDict((name, width) for name, width in (
            ("robot", 13), ("object", 13), ("table", 13), ("ee", 13),
            ("dof_pos", self.ndof), ("dof_vel", self.ndof), ("last_dof_vel", self.ndof),
            ("low_history", math.prod(self.low_shape)), ("low_actions", 18),
            ("gait", 1), ("clock", 4), ("gait_wait", 1), ("walking", 1), ("foot_contacts", 4)))
        self.learned_names = tuple(widths)
        self.slices = OrderedDict()
        cursor = 0
        for name, width in widths.items():
            self.slices[name] = slice(cursor, cursor+width)
            cursor += width
        self.physical_dim = cursor
        # Each rigid body uses a 3-d rotation-vector delta, not four independent
        # quaternion coordinates. Other learned fields use scaled residuals.
        self.delta_dim = self.physical_dim - 4
        for name, width in (("goal_pos", 3), ("goal_rpy", 3), ("commands", 3), ("prev_action", 9),
                            ("ekf_poses", self.history*7), ("ekf_ages", self.history), ("ekf_count", 1),
                            ("closest", 1), ("highest", 1), ("hold", 1), ("best_hold", 1),
                            ("completed", 1), ("progress", 1), ("table_height", 1)):
            self.slices[name] = slice(cursor, cursor+width)
            cursor += width
        self.state_slices = self.slices
        self.state_dim = self.feature_dim = cursor
        self.context_dim = 26  # slot, initial yaw, rest height, reward gate, motor18, cmd scale, table vx/vy/yaw
        self.feature_table = env.feature_obs.detach().clone()
        self.grasp_table = env.T_grasp_in_obj.detach().clone()
        self.grasp_correction = (env.Rx_rot @ env.Rz_rot @ env.Rx_rot).detach().clone()
        self.origin = env._initial_robot_root_states[:, :3].detach().clone()
        self.origin[:, 2] = 0
        self.reindex = torch.tensor([3,4,5,0,1,2,9,10,11,6,7,8] + list(range(12, self.ndof)), device=self.device)
        self.arm_offset = torch.tensor([.3, 0., .09], device=self.device, dtype=self.dtype)
        body_props = getattr(env, "wm_body_context", None)
        shape_props = getattr(env, "wm_shape_context", None)
        if body_props is None or shape_props is None:
            raise ValueError("Environment is missing creation-time wm_body_context/wm_shape_context; recreate the environment with the world-model hooks")
        self.physical_context = torch.cat((torch.as_tensor(body_props, device=self.device, dtype=self.dtype),
                                           torch.as_tensor(shape_props, device=self.device, dtype=self.dtype)), -1)
        self.physical_context_scale = torch.ones(self.physical_context.shape[-1], device=self.device, dtype=self.dtype)*2.
        self.physical_context_scale[:8] = self.physical_context.new_tensor([20., .1, .1, .1, 400., 40., 5., 2.5])
        self.object_count = len(env.obj_list)
        self.context_feature_dim = self.context_dim-1+self.physical_context.shape[-1]+self.object_count
        self.input_scale = torch.ones(self.state_dim, device=self.device, dtype=self.dtype)
        self.error_scale = torch.ones(self.physical_dim, device=self.device, dtype=self.dtype)
        for name in ("robot", "object", "table", "ee"):
            sl = self.slices[name]
            self.input_scale[sl.start+10:sl.stop] = 3.
            self.error_scale[sl] = self.error_scale.new_tensor([.05]*3+[.1]*4+[.5]*3+[1.]*3)
        for name in ("dof_vel", "last_dof_vel"):
            self.input_scale[self.slices[name]] = 5.
            self.error_scale[self.slices[name]] = 2.
        self.error_scale[self.slices["dof_pos"]] = .1
        self.input_scale[self.slices["progress"]] = self.max_steps
        for name in ("hold", "best_hold"):
            self.input_scale[self.slices[name]] = self.hold_steps
        self.input_scale[self.slices["gait_wait"]] = 35.
        # pre_physics_step currently sets action_delay=0 in both branches:
        # the frozen low controller reads current commands, low_actions and
        # low_history, never the previous raw high-level action. Observation
        # and reward histories remain in state but cannot create spurious
        # physical gradients through the dynamics input.
        self.dynamics_fields = self.learned_names + ("goal_pos", "goal_rpy", "commands", "progress", "table_height")
        self.feature_mask = torch.zeros(self.state_dim, device=self.device, dtype=self.dtype)
        for name in self.dynamics_fields:
            self.feature_mask[self.slices[name]] = 1.

    def get(self, state, name):
        return state[..., self.slices[name]]

    def _pack(self, fields):
        return torch.cat([fields[name] for name in self.slices], -1)

    def _ids(self, context):
        return context[:, 0].long()

    @torch.no_grad()
    def capture(self, env_ids=None):
        env = self.env
        ids = torch.arange(env.num_envs, device=self.device) if env_ids is None else env_ids.to(self.device)
        origin = self.origin[ids]
        fields = {}
        rigid = {"robot": env._robot_root_states, "object": env._cube_root_states, "table": env._table_root_states,
                 "ee": torch.cat((env._rigid_body_pos[:, env.gripper_idx], env._rigid_body_rot[:, env.gripper_idx],
                                   env._rigid_body_vel[:, env.gripper_idx], env._rigid_body_ang_vel[:, env.gripper_idx]), -1)}
        for name, value in rigid.items():
            value = value[ids].clone()
            value[:, :3] -= origin
            fields[name] = value
        for name, attribute in (("dof_pos", "_dof_pos"), ("dof_vel", "_dof_vel"), ("last_dof_vel", "_last_dof_vel"),
                                ("low_history", "low_obs_history_buf"), ("low_actions", "last_low_actions"),
                                ("gait", "gait_indices"), ("clock", "clock_inputs"), ("gait_wait", "gait_wait_timer"),
                                ("walking", "is_walking"), ("foot_contacts", "foot_contacts_from_sensor"),
                                ("goal_pos", "curr_ee_goal_cart"), ("goal_rpy", "curr_ee_goal_orn_rpy"),
                                ("commands", "commands"), ("closest", "closest_dist"), ("highest", "highest_object"),
                                ("hold", "pick_counter"), ("best_hold", "best_pick_counter"), ("completed", "pick_success_recorded"),
                                ("progress", "progress_buf"), ("table_height", "table_heights")):
            fields[name] = getattr(env, attribute)[ids].to(self.dtype).reshape(len(ids), -1)
        fields["prev_action"] = env.action_history_buf[ids, -1]
        predictor = env.object_pose_predictor
        if predictor is not None:
            poses = predictor.poses[ids].clone()
            poses[..., :3] -= origin[:, None]
            ages = predictor.times[ids]-predictor.times[ids, -1:]
            fields["ekf_ages"] = torch.nan_to_num(ages, nan=0., posinf=0., neginf=0.).to(self.dtype)
            fields["ekf_poses"] = poses.flatten(1)
            fields["ekf_count"] = predictor.count[ids, None].to(self.dtype)
        else:
            fields["ekf_poses"] = fields["object"][:, None, :7].expand(-1, self.history, -1).flatten(1)
            fields["ekf_ages"] = fields["object"].new_zeros(len(ids), self.history)
            fields["ekf_count"] = fields["object"].new_zeros(len(ids), 1)
        command_scale = env.command_scale[ids, None] if env.rand_cmd_scale else fields["object"].new_ones(len(ids), 1)
        table_velocity = fields["object"].new_zeros(len(ids), 3)
        if env.table_level == "Level01":
            table_velocity[:, 1] = env.vel_line_forward_y[ids]
        elif env.table_level == "Level02":
            table_velocity[:, 1] = env.vel_line_forward_y_level02[ids]
            table_velocity[:, 2] = env.vel_angular_yaw[ids] * env.lin_circle_select[ids]
        # Production stores the common initial heading as a one-element
        # tensor; tests/custom tasks may provide one heading per environment.
        initial_yaw = torch.as_tensor(env.initial_yaw_z, device=self.device, dtype=self.dtype).reshape(-1).expand(env.num_envs)
        # A context belongs to the action about to execute; post_physics_step
        # increments the global counter before evaluating that action's reward.
        context = torch.cat((ids[:, None].to(self.dtype), initial_yaw[ids, None], env.init_height[ids, None],
                             fields["object"].new_full((len(ids), 1), float(env.global_step_counter+1 >= env.train_reward_strict)),
                             env.motor_strength[ids], command_scale, table_velocity), -1)
        return self._pack(fields).detach().clone(), context.detach().clone()

    def features(self, state):
        return state / self.input_scale * self.feature_mask

    def normalize_context(self, context):
        ids = self._ids(context)
        objects = F.one_hot(ids % self.object_count, self.object_count).to(context.dtype)
        return torch.cat((context[:, 1:], self.physical_context[ids]/self.physical_context_scale, objects), -1)

    def dynamics_action(self, state, action):
        """Encode the command actually applied to the closed-loop controller.

        Gripper opening is continuous and differentiable, matching the real
        joint-target interpolation; history/reward retain that same opening.
        """
        clipped, goal, orientation, command = self._control(state, action)
        return torch.cat((goal-self.get(state, "goal_pos"),
                          orientation-self.get(state, "goal_rpy"), clipped[:, 6:7],
                          command[:, 0:1], command[:, 2:3]), -1)

    def metadata(self):
        """Stable checkpoint/probe contract, independent of simulator slots.

        Hash source asset files, rather than randomly sampled physics tensors
        or GPU-derived transforms. Thus a smaller fresh probe environment and
        CPU/CUDA construction produce the same contract. This is a setup-time
        operation; asset bytes never enter the training hot loop.
        """
        env = self.env
        task_cfg = env.cfg.get("env", {})
        sim_cfg = env.cfg.get("sim", {})
        asset_cfg = task_cfg.get("asset", {})
        project_root = Path(__file__).resolve().parents[1]

        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else project_root / path

        def file_hash(path):
            if not path.is_file():
                return None
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024*1024), b""):
                    digest.update(block)
            return digest.hexdigest()

        def tree_hash(path):
            if not path.is_dir():
                return None
            digest = hashlib.sha256()
            # Include collision geometry, URDF, features, visual geometry and
            # textures. No sampled/transformed runtime tensors are hashed.
            for item in sorted(value for value in path.rglob("*") if value.is_file()):
                digest.update(item.relative_to(path).as_posix().encode("utf8"))
                digest.update(b"\0")
                digest.update(file_hash(item).encode("ascii"))
            return digest.hexdigest()

        asset_root = resolve(asset_cfg.get("assetRoot", "data/asset"))
        robot_file = asset_cfg.get("assetFileRobot")
        robot_path = asset_root / robot_file if robot_file else None
        objects = []
        transform_root = project_root / "PredictPiontTransform/transform_info"
        for name in env.obj_list:
            description = asset_cfg.get("asset_multi", {}).get(name, {})
            object_root = asset_root / asset_cfg.get("assetFileObj", "obj_set") / name
            index = description.get("dict_idx")
            grasp_file = (transform_root / "contact_grasp_info_mul" / ("predictions_image_%02d.json" % index)) if index is not None else None
            objects.append({"name": name, "description": description,
                            "asset_sha256": tree_hash(object_root),
                            "grasp_sha256": file_hash(grasp_file) if grasp_file is not None else None})
        low_policy = task_cfg.get("low_policy_path")
        finite = lambda value: value if math.isfinite(float(value)) else ("Infinity" if value > 0 else "-Infinity")
        metadata = {
            "version": 3,
            "schema": "dq-sgs-ekf-full-state-v3-continuous-gripper",
            "state_dim": self.state_dim,
            "feature_dim": self.feature_dim,
            "physical_dim": self.physical_dim,
            "delta_dim": self.delta_dim,
            "action_dim": self.action_dim,
            "context_dim": self.context_dim,
            "context_feature_dim": self.context_feature_dim,
            "termination_feature_dim": self.termination_feature_dim,
            "slices": [[name, value.start, value.stop] for name, value in self.slices.items()],
            "learned_fields": list(self.learned_names),
            "dynamics_input_fields": list(self.dynamics_fields),
            "ndof": self.ndof,
            "low_shape": list(self.low_shape),
            "history": self.history,
            "horizons": list(self.horizons),
            "ekf_q_scale": float(self.q_scale),
            "ekf_r_scale": float(self.r_scale),
            "velocity_obs": self.velocity_obs,
            "feature_width": self.feature_table.shape[-1],
            "grasp_candidates": self.grasp_table.shape[1],
            "reward_scales": dict(self.reward_scales),
            "loss_contract": {
                "main": "equal aggregate weight per learned physical/memory field",
                "main_scales": {"position_m": .05, "rotation_rad": .1, "linear_velocity_m_s": .5,
                                "angular_velocity_rad_s": 1., "joint_position_rad": .1, "joint_velocity_rad_s": 2.,
                                "low_controller_memory": 1.},
                "relative_auxiliary": "mean of five normalized group mean-square errors in the EE frame",
                "relative_groups": ["object_ee_position", "object_ee_orientation", "object_ee_linear_velocity",
                                    "object_ee_angular_velocity", "object_height_above_table"],
                "relative_scales": [.05, .1, .5, 1., .05],
                "relative_orientation": "sign-invariant quaternion chord, scaled by 2 (local radians)",
            },
            "only_positive_rewards": self.only_positive,
            "base_height_target": self.base_height_target,
            "hold_steps": self.hold_steps,
            "max_episode_steps": self.max_steps,
            "lift_threshold": self.lift_threshold,
            "base_distance": self.base_distance,
            "reward_curriculum_step": float(env.train_reward_strict),
            "clip_actions": finite(self.clip_actions),
            "clip_observations": finite(self.clip_obs),
            "control": {
                "gripper_action_mode": GRIPPER_ACTION_MODE,
                "gripper_target": "q_closed + opening * (q_open - q_closed)",
                "gripper_closing_threshold": CLOSING_OPENING_THRESHOLD,
                "low_dt": self.low_dt,
                "control_cycles": int(env.control_freq_low_init),
                "rand_control": bool(env.rand_control),
                "use_tanh": self.use_tanh,
                "small_commands": self.small_commands,
                "forward_limit": self.forward_limit,
                "stop_pick": self.stop_pick,
                "rand_command_scale": bool(env.rand_cmd_scale),
                "mask_arm": bool(getattr(env, "mask_arm", False)),
                "observe_gait_commands": bool(getattr(env, "observe_gait_commands", self.low_shape[-1] > 76)),
                "table_level": env.table_level,
                "low_policy_sha256": file_hash(resolve(low_policy)) if low_policy else None,
            },
            "physics": {
                "sim": {key: sim_cfg[key] for key in ("dt", "substeps", "up_axis", "gravity") if key in sim_cfg},
                "physx": {key: sim_cfg.get("physx", {})[key] for key in
                          ("solver_type", "num_position_iterations", "num_velocity_iterations", "contact_offset",
                           "rest_offset", "bounce_threshold_velocity", "max_depenetration_velocity")
                          if key in sim_cfg.get("physx", {})},
                "terrain": task_cfg.get("terrainType", "plane"),
                "rigid_body_mass_addition": [0., 15.],
                "rigid_body_com_offset": [-.1, .1],
                "robot_shape_friction": [.5, 3.],
                "motor_strength": [.7, 1.3],
                "arm_kp_kd": [400., 40.],
                "gripper_kp": [2., 5.],
                "gripper_kd": 2.5,
                "command_scale": [.5, 1.] if env.rand_cmd_scale else [1., 1.],
                "asset_control": asset_cfg.get("control", {}),
            },
            "assets": {
                "robot_urdf_sha256": file_hash(robot_path) if robot_path else None,
                "robot_asset_sha256": tree_hash(robot_path.parent.parent) if robot_path else None,
                "objects": objects,
                "grasp_reference_sha256": file_hash(transform_root/"30all_nomove_cube_root_states5.pt"),
            },
        }

        def plain(value):
            if isinstance(value, dict):
                return {str(key): plain(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [plain(item) for item in value]
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, float):
                return finite(value)
            if isinstance(value, (str, int, bool)) or value is None:
                return value
            if hasattr(value, "item"):
                return plain(value.item())
            raise TypeError("Non-JSON task metadata value: " + type(value).__name__)
        return plain(metadata)

    def _geometry(self, state):
        robot, obj, ee = (self.get(state, name) for name in ("robot", "object", "ee"))
        base = quat_normalize(robot[:, 3:7])
        angles = euler(base)
        heading = from_euler(torch.cat((torch.zeros_like(angles[:, :2]), angles[:, 2:]), -1))
        arm = robot[:, :3] + quat_apply(base, self.arm_offset.expand(len(state), -1))
        return robot, obj, ee, base, angles, heading, arm

    def _object_observation(self, pose, arm, heading):
        relative = quat_inverse_apply(heading, pose[..., :3]-arm)
        return torch.cat((relative[..., :2], pose[..., 2:3], euler(quat_mul(quat_conjugate(heading), pose[..., 3:7]))), -1)

    def observation(self, state, context):
        robot, obj, ee, base, angles, heading, arm = self._geometry(state)
        object_obs = self._object_observation(obj[:, :7], arm, heading)
        ee_position = quat_inverse_apply(base, ee[:, :3]-arm)
        ee_orientation = euler(quat_mul(quat_conjugate(base), ee[:, 3:7]))
        q = self.get(state, "dof_pos")[:, self.reindex]
        qd = self.get(state, "dof_vel")[:, :-1][:, self.reindex[:-1]] * .05
        blocks = [object_obs, ee_position, ee_orientation, q, qd, self.get(state, "commands"),
                  self.get(state, "goal_pos"), self.get(state, "goal_rpy")]
        if self.velocity_obs:
            velocity = quat_inverse_apply(heading, robot[:, 7:10])
            relative_velocity = quat_inverse_apply(heading, obj[:, 7:10])-velocity
            blocks.extend((velocity, relative_velocity[:, :2]))
        if self.horizons:
            poses = self.get(state, "ekf_poses").reshape(-1, self.history, 7)
            future, valid = causal_ekf(poses, self.get(state, "ekf_ages"), self.get(state, "ekf_count"), self.horizons, self.q_scale, self.r_scale)
            future_obs = self._object_observation(future, arm[:, None], heading[:, None])
            future_obs = torch.where(valid[:, None, None], future_obs, object_obs[:, None])
            blocks.extend((future_obs.flatten(1), valid[:, None].to(state.dtype)))
        ids = self._ids(context)
        rotation = euler_matrix(euler(obj[:, 3:7]))
        bottom = state.new_tensor([0.,0.,0.,1.]).expand(len(state),1,4)
        object_transform = torch.cat((torch.cat((rotation,obj[:,:3,None]),-1),bottom),-2)
        transform = object_transform[:, None] @ self.grasp_table[ids]
        position = quat_inverse_apply(base[:, None], transform[..., :3, 3]-arm[:, None])
        position = torch.cat((position[..., :2], position[..., 2:3].clamp(-.6, .6)), -1)
        world_rotation = transform[..., :3, :3] @ self.grasp_correction
        # Preserve the current get_grasp_world XYZ->quaternion conversion
        # before expressing each candidate in the full robot base frame.
        grasp_quaternion = from_euler(matrix_euler_xyz(world_rotation))
        candidate_q = quat_normalize(quat_mul(quat_conjugate(base[:, None]), grasp_quaternion))
        local_rotation = quaternion_to_matrix(candidate_q.reshape(-1, 4)).reshape(len(state), -1, 3, 3)
        # Existing candidate layout is [yaw,pitch,roll], unlike object/EE RPY.
        candidate_euler = euler(candidate_q)[..., [2, 1, 0]]
        r31 = local_rotation[..., 2, 0]
        candidate_euler = torch.stack((torch.where(r31.abs() > .9999, torch.zeros_like(r31), candidate_euler[..., 0]),
                                       candidate_euler[..., 1], torch.where(r31.abs() > .9999,
                                       safe_atan2(-local_rotation[..., 1, 0], local_rotation[..., 0, 0]), candidate_euler[..., 2])), -1)
        blocks.extend((position.flatten(1), candidate_euler.flatten(1), self.get(state, "prev_action")))
        return torch.cat((self.feature_table[ids], *blocks), -1).clamp(-self.clip_obs, self.clip_obs)

    observe = observation

    def _control(self, state, action):
        action = action.clamp(-self.clip_actions, self.clip_actions)
        action = torch.cat((action[:, :6], action[:, 6:7].clamp(0., 1.), action[:, 7:]), -1)
        dp = action[:, :3] * .02 if self.use_tanh else action[:, :3].clamp(-.02, .02)
        dr = action[:, 3:6] * .06 if self.use_tanh else action[:, 3:6].clamp(-.06, .06)
        goal = self.get(state, "goal_pos") + dp
        goal = torch.stack((goal[:, 0].clamp(0., .7), goal[:, 1].clamp(-.7, .7), goal[:, 2].clamp(-.6, .6)), -1)
        rpy = self.get(state, "goal_rpy") + dr
        forward = action[:, 7] * self.forward_limit if self.use_tanh else action[:, 7].clamp(-self.forward_limit, self.forward_limit)
        yaw = action[:, 8] * .6 if self.use_tanh else action[:, 8].clamp(-.6, .6)
        command = torch.stack((forward, torch.zeros_like(forward), yaw), -1)
        if self.small_commands:
            command = command * ((forward.abs() > .15) | (yaw.abs() > .35))[:, None]
        if self.stop_pick:
            command = torch.where(closing_command(action[:, 6:7]), torch.zeros_like(command), command)
        return action, goal, rpy, command

    def task_outcome(self, previous, state, action, context):
        """The existing hard task rules, including failure-before-success."""
        robot, obj, ee, base, angles, heading, arm = self._geometry(state)
        relative_ee = quat_inverse_apply(base, ee[:, :3]-arm)
        ik_error = (self.get(state, "goal_pos")[:, 2]-relative_ee[:, 2]).abs()
        yaw_deviation = (angles[:, 2]-context[:, 1]).abs()
        failed = ((angles[:, :2].abs() > .8).any(-1) | (robot[:, 2] < .1)
                  | (ik_error > .2) | (obj[:, 2]+.015 < self.get(state, "table_height")[:, 0])
                  | (yaw_deviation > math.radians(70)))
        distance = torch.linalg.vector_norm(obj[:, :3]-ee[:, :3], dim=-1)
        lifted = lifted_grasp_mask(obj[:, 2], self.get(state, "table_height")[:, 0], context[:, 2],
                                   distance, action[:, 6], self.lift_threshold)
        count, best, hold_progress, success = advance_pick_hold(
            lifted, failed, self.get(previous, "hold")[:, 0], self.get(previous, "best_hold")[:, 0],
            self.get(previous, "completed")[:, 0] > .5, self.hold_steps)
        completed = (self.get(previous, "completed")[:, 0] > .5) | success
        timeout = (self.get(state, "progress")[:, 0] >= self.max_steps) & ~completed & ~failed
        return {"failed": failed, "success": success, "completed": completed, "timeout": timeout,
                "lifted": lifted, "hold": count, "best_hold": best, "hold_progress": hold_progress,
                "distance": distance, "height": obj[:, 2]-self.get(state, "table_height")[:, 0]-context[:, 2]}

    def integrate(self, state, delta, action, context, dt):
        """Learned semi-implicit physical step + exact high-level memory update.

        All rigid-body angular velocities are world-frame, hence quaternion
        increments multiply on the left. Low-level history is a learned state,
        not repeatedly replaced with real future history during imagination.
        """
        fields = {name: self.get(state, name) for name in self.slices}
        cursor = 0
        for name in ("robot", "object", "table", "ee"):
            old = fields[name]
            dv, dw = delta[:, cursor:cursor+3]*.3, delta[:, cursor+3:cursor+6]*.5
            dp, dr = delta[:, cursor+6:cursor+9]*.01, delta[:, cursor+9:cursor+12]*.02
            velocity, angular = old[:, 7:10]+dv, old[:, 10:13]+dw
            rotation = quat_normalize(quat_mul(from_rotvec(angular*dt+dr), quat_normalize(old[:, 3:7])))
            fields[name] = torch.cat((old[:, :3]+dt*velocity+dp, rotation, velocity, angular), -1)
            cursor += 12
        dof_delta = delta[:, cursor:cursor+self.ndof] * .02
        cursor += self.ndof
        velocity_delta = delta[:, cursor:cursor+self.ndof] * 2.
        cursor += self.ndof
        fields["dof_vel"] = fields["dof_vel"] + velocity_delta
        fields["dof_pos"] = fields["dof_pos"] + dt*fields["dof_vel"] + dof_delta
        for name in self.learned_names[6:]:
            width = self.slices[name].stop-self.slices[name].start
            scale = 2. if name == "last_dof_vel" else .1
            fields[name] = fields[name] + delta[:, cursor:cursor+width] * scale
            cursor += width
        action, goal, rpy, command = self._control(state, action)
        fields.update(goal_pos=goal, goal_rpy=rpy, commands=command, prev_action=action,
                      progress=fields["progress"]+1)
        poses = fields["ekf_poses"].reshape(-1, self.history, 7)
        first = fields["ekf_count"][:, 0] < 1
        new_pose = fields["object"][:, :7]
        poses = torch.cat((poses[:, 1:], new_pose[:, None]), 1)
        poses = torch.where(first[:, None, None], new_pose[:, None], poses)
        fields["ekf_poses"] = poses.flatten(1)
        ages = torch.cat((fields["ekf_ages"][:, 1:]-dt, torch.zeros_like(dt)), -1)
        fields["ekf_ages"] = torch.where(first[:, None], torch.zeros_like(ages), ages)
        fields["ekf_count"] = (fields["ekf_count"]+1).clamp_max(self.history)
        candidate = self._pack(fields)
        outcome = self.task_outcome(state, candidate, action, context)
        fields["hold"], fields["best_hold"] = outcome["hold"][:, None], outcome["best_hold"][:, None]
        fields["completed"] = outcome["completed"][:, None].to(state.dtype)
        fields["closest"] = torch.where(fields["closest"] < 0, outcome["distance"][:, None],
                                         torch.minimum(fields["closest"], outcome["distance"][:, None]))
        fields["highest"] = torch.where(fields["highest"] < 0, outcome["height"][:, None],
                                         torch.maximum(fields["highest"], outcome["height"][:, None]))
        return self._pack(fields)

    def error_vector(self, predicted, target):
        errors = []
        # Equal aggregate weight per physical field prevents 760 history
        # entries from hiding centimetre-scale contact prediction errors.
        total = self.physical_dim
        for name in self.learned_names:
            a, b = self.get(predicted, name), self.get(target, name)
            if name in ("robot", "object", "table", "ee"):
                rotation = quat_mul(quat_conjugate(quat_normalize(b[:, 3:7])), quat_normalize(a[:, 3:7]))
                rotation = torch.where(rotation[:, 3:] < 0, -rotation, rotation)
                rotation_error = 2*rotation[:, :3]/.1
                error = torch.cat(((a[:, :3]-b[:, :3])/.05, rotation_error,
                                   (a[:, 7:10]-b[:, 7:10])/.5, a[:, 10:13]-b[:, 10:13]), -1)
            else:
                error = (a-b)/self.error_scale[self.slices[name]]
            errors.append(error * math.sqrt(total/(len(self.learned_names)*error.shape[-1])))
        return torch.cat(errors, -1)

    def auxiliary_loss(self, predicted, target):
        """Task-focused relational supervision, additional to balanced state loss.

        Relative pose and motion are derived from the same predicted physical
        state used by the Actor, rather than independently regressed heads.
        Quaternion chord distance is invariant to q/-q and behaves like a
        rotation-angle error locally. Each of the five groups has equal weight.
        """
        def relative(state):
            obj, ee = self.get(state, "object"), self.get(state, "ee")
            inverse = quat_conjugate(quat_normalize(ee[:, 3:7]))
            position = quat_apply(inverse, obj[:, :3]-ee[:, :3])
            orientation = quat_normalize(quat_mul(inverse, quat_normalize(obj[:, 3:7])))
            linear_velocity = quat_apply(inverse, obj[:, 7:10]-ee[:, 7:10])
            angular_velocity = quat_apply(inverse, obj[:, 10:13]-ee[:, 10:13])
            height = obj[:, 2:3]-self.get(state, "table_height")
            return position, orientation, linear_velocity, angular_velocity, height

        pred_position, pred_rotation, pred_linear, pred_angular, pred_height = relative(predicted)
        true_position, true_rotation, true_linear, true_angular, true_height = relative(target)
        true_rotation = torch.where((pred_rotation*true_rotation).sum(-1, keepdim=True) < 0,
                                    -true_rotation, true_rotation)
        errors = ((pred_position-true_position)/.05,
                  2*(pred_rotation-true_rotation)/.1,
                  (pred_linear-true_linear)/.5,
                  pred_angular-true_angular,
                  (pred_height-true_height)/.05)
        return torch.stack([error.square().mean(-1) for error in errors], -1).mean(-1)

    def reward_terms(self, previous, state, action, context):
        """Exact default reward equations; no surrogate success reward."""
        action = action.clamp(-self.clip_actions, self.clip_actions)
        outcome = self.task_outcome(previous, state, action, context)
        robot, obj, ee, base, angles, heading, arm = self._geometry(state)
        distance, height = outcome["distance"], outcome["height"]
        closest = self.get(previous, "closest")[:, 0]
        highest = self.get(previous, "highest")[:, 0]
        closest = torch.where(closest < 0, distance, closest)
        highest = torch.where(highest < 0, height, highest)
        base_distance = torch.linalg.vector_norm(obj[:, :2]-arm[:, :2], dim=-1)
        object_direction = obj[:, :3]-ee[:, :3]
        ee_x = quat_apply(quat_normalize(ee[:, 3:7]), state.new_tensor([1., 0., 0.]).expand(len(state), -1))
        orientation = F.cosine_similarity(ee_x, object_direction, dim=-1)
        orientation = torch.where(distance >= .01, orientation, torch.zeros_like(orientation))
        base_height = torch.exp(-(robot[:, 2]-self.base_height_target).abs())
        yaw_deviation = (angles[:, 2]-context[:, 1]).abs()
        acc = torch.linalg.vector_norm(self.get(state, "dof_vel")[:, -7:-1]-self.get(state, "last_dof_vel")[:, -7:-1], dim=-1)/self.low_dt
        commands = self.get(state, "commands")
        terms = {
            "approaching": torch.tanh(10*(closest-distance).clamp(0., 10.)) * ~outcome["lifted"],
            "lifting": torch.tanh(10*(height-highest).clamp(0., 10.)) * ~outcome["lifted"],
            "pick_up": outcome["success"].to(state.dtype),
            "acc_penalty": 1-torch.exp(-acc),
            "command_reward": torch.where(base_distance < .2, torch.exp(-commands[:, 0].abs()), torch.zeros_like(distance))*context[:, 3],
            "action_rate": torch.linalg.vector_norm(action[:, 7:9]-self.get(previous, "prev_action")[:, 7:9], dim=-1),
            "gripper_rate": (action[:, 6]-self.get(previous, "prev_action")[:, 6]).abs()*context[:, 3],
            "action_penalty": 1-torch.exp(-10*torch.linalg.vector_norm(action, dim=-1)),
            "ee_orn": orientation,
            # Existing code zeroes XY of the object direction. Its dot product
            # with the horizontal heading is zero; preserve this baseline.
            "base_dir": torch.zeros_like(distance),
            "base_approaching": torch.tanh(-10*(base_distance-self.base_distance).abs())+1,
            "base_height": base_height,
            "grasp_base_height": base_height*outcome["hold_progress"],
            "rad_penalty": torch.exp(-(torch.linalg.vector_norm(self.get(state, "goal_pos"), dim=-1)-.9).abs()/.15),
            "base_ang_pen": torch.linalg.vector_norm(robot[:, 10:13], dim=-1),
            "limit_yaw_rotation_penalty": torch.where(yaw_deviation > math.radians(60), -torch.tanh(yaw_deviation), torch.zeros_like(distance)),
        }
        return terms

    def reward(self, previous, state, action, context):
        terms = self.reward_terms(previous, state, action, context)
        result = sum(terms[name]*scale for name, scale in self.reward_scales.items())
        return result.clamp_min(0.) if self.only_positive else result

    def termination_features(self, state, context):
        robot, obj, ee, base, angles, heading, arm = self._geometry(state)
        relative_ee = quat_inverse_apply(base, ee[:, :3]-arm)
        return torch.stack((angles[:, 0]/.8, angles[:, 1]/.8, robot[:, 2]/.1,
                            (self.get(state, "goal_pos")[:, 2]-relative_ee[:, 2]).abs()/.2,
                            (obj[:, 2]+.015-self.get(state, "table_height")[:, 0])/.05,
                            (angles[:, 2]-context[:, 1]).abs()/math.radians(70),
                            self.get(state, "hold")[:, 0]/self.hold_steps,
                            self.get(state, "progress")[:, 0]/self.max_steps), -1)

    def timeouts(self, state, context=None):
        return self.get(state, "progress")[:, 0] >= self.max_steps

    def sample_dt(self, batch_size):
        maximum = self.env.control_freq_low_init
        if self.env.rand_control:
            cycles = torch.randint(maximum-1, maximum+1, (batch_size, 1), device=self.device)
            return cycles.to(self.dtype)*self.low_dt
        return torch.full((batch_size, 1), maximum*self.low_dt, device=self.device, dtype=self.dtype)
