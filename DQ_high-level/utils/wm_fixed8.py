"""Fixed-eight hierarchical teacher adapter, with 64 learned physical states.

The serialized rollout state also carries deterministic controller memory.
Its large history buffer is never a dynamics-network input or prediction target.
The legacy adapter supplies the already-audited teacher observation and reward
equations; this adapter replaces its physical integration and controller loop.
"""

import math
from pathlib import Path

import torch
from torch.nn import functional as F

from utils.wm_env import (TeacherWorldModelAdapter, quat_normalize, quat_mul,
                          quat_conjugate, quat_apply, quat_inverse_apply,
                          from_rotvec, from_euler, euler)
from utils.world_model.kinematics import B1Z1Kinematics
from utils.world_model.representation import rotation_6d, rotation_vector


class Fixed8TeacherWorldModelAdapter(TeacherWorldModelAdapter):
    low_steps = 8
    low_action_dim = 20  # actual leg command12, held world EE pose7, gripper target1
    low_action_feature_dim = 22  # EE orientation is represented by rot6d
    low_feature_dim = 81
    high_dt = 0.16
    termination_feature_dim = 0
    hard_absorbing = True

    def __init__(self, env):
        super().__init__(env)
        if self.ndof != 19 or self.low_shape != (10, 71):
            raise ValueError("Fixed-eight teacher requires 19 DOFs and 10 x 71 low-level history")
        if not math.isclose(self.low_dt, .02, abs_tol=1e-8):
            raise ValueError("Fixed-eight world model requires 20 ms low-level steps")
        if env.control_freq_low_init != 8 or env.rand_control:
            raise ValueError("Real sampling must use fixed 8 low-level steps before constructing the model")
        for name, width in (("held_ee_target", 7), ("held_gripper_target", 1)):
            self.slices[name] = slice(self.state_dim, self.state_dim + width)
            self.state_dim += width
        # Only these four fields are learned; all other serialized fields are
        # caches, nominal environment conditions, or derived geometry.
        self.learned_names = ("robot", "object", "dof_pos", "dof_vel")
        self.physical_dim, self.delta_dim = 64, 62
        self.feature_dim = self.low_feature_dim
        # Initial world heading and reward curriculum are task bookkeeping,
        # not physical conditions of a canonical 20 ms transition.
        self.context_feature_dim -= 2
        # Physical/context inputs plus the deterministic controller/task memory
        # needed by a value function on imagined states (not an actor input).
        self.critic_feature_dim = 896 + 10 * self.history + self.context_feature_dim
        self.dynamics_fields = self.learned_names + ("table",)
        self.initial_q = env._initial_dof_pos.detach().clone()
        self.mask_arm = bool(env.mask_arm)
        self.feet_reindex = torch.tensor([1, 0, 3, 2], device=self.device)
        self.q_closed = float(env.dof_limits_upper[-1])
        self.q_open = float(env.dof_limits_lower[-1])
        self.table_thickness = float(env.table_dimz)
        root = Path(__file__).resolve().parents[1]
        assets = env.cfg["env"]["asset"]
        asset_root = Path(assets.get("assetRoot", "data/asset"))
        if not asset_root.is_absolute():
            asset_root = root / asset_root
        # Isaac root tensor pose is the link origin, but its linear velocity
        # is at the compiled rigid body's COM. Body 0 includes the collapsed
        # arm mount; randomized body 1 (trunk) is a different rigid body.
        root_com = env.gym.get_actor_rigid_body_properties(
            env.envs[0], env.robot_handles[0])[0].com
        self.kinematics = B1Z1Kinematics(
            asset_root / assets["assetFileRobot"], dof_names=env.dof_names,
            base_com=(root_com.x, root_com.y, root_com.z), device=self.device)
        owner = getattr(env.low_level_policy, "__self__", None)
        if owner is None or not hasattr(owner, "actor"):
            raise ValueError("Expected the pretrained ActorCritic.act_inference low-level policy")
        self.low_actor = owner.actor
        self.low_actor.requires_grad_(False)
        self.low_actor.eval()

    def _pack(self, fields):
        # capture() inherited from the legacy adapter does not know the two
        # additional held-target slots. Fill them only until capture adds them.
        reference = fields["robot"]
        return torch.cat([fields[name] if name in fields else reference.new_zeros(
            *reference.shape[:-1], sl.stop-sl.start) for name, sl in self.slices.items()], -1)

    def replace(self, state, **changes):
        return self._pack({name: changes.get(name, state[..., sl])
                           for name, sl in self.slices.items()})

    def fk(self, state):
        return self.kinematics(super().get(state, "robot"),
                               super().get(state, "dof_pos"),
                               super().get(state, "dof_vel"))

    def get(self, state, name):
        if name == "ee" and hasattr(self, "kinematics"):
            fk = self.fk(state)
            return torch.cat([fk[key] for key in ("ee_pos", "ee_quat", "ee_vel", "ee_omega")], -1)
        return super().get(state, name)

    @torch.no_grad()
    def capture(self, env_ids=None):
        state, context = super().capture(env_ids)
        ids = self._ids(context)
        env = self.env
        if hasattr(env, "ee_goal_global_cart"):
            pose = torch.cat((env.ee_goal_global_cart[ids] - self.origin[ids],
                              env.ee_goal_global_orn_quat[ids]), -1)
        else:
            _, _, _, base, _, _, arm = self._geometry(state)
            pose = torch.cat((arm + quat_apply(base, self.get(state, "goal_pos")),
                              quat_mul(base, from_euler(self.get(state, "goal_rpy")))), -1)
        opening = self.get(state, "prev_action")[:, 6:7].clamp(0., 1.)
        grip = self.q_closed + opening * (self.q_open - self.q_closed)
        state = self.replace(state, held_ee_target=pose, held_gripper_target=grip)
        return state.detach(), context

    @torch.no_grad()
    def capture_low_action(self, env_ids):
        env, ids = self.env, env_ids
        return torch.cat((env.last_low_actions[ids, :12],
                          env.ee_goal_global_cart[ids] - self.origin[ids],
                          env.ee_goal_global_orn_quat[ids],
                          env.gripper_dof_pos[ids].reshape(len(ids), 1)), -1).detach().clone()

    def _heading(self, state):
        root = self.get(state, "robot")
        angles = euler(root[:, 3:7])
        heading = from_euler(torch.cat((torch.zeros_like(angles[:, :2]), angles[:, 2:]), -1))
        return root, heading

    def low_features(self, state):
        root, heading = self._heading(state)
        inverse = quat_conjugate(heading)
        blocks = [root[:, 2:3], rotation_6d(quat_mul(inverse, quat_normalize(root[:, 3:7]))),
                  quat_apply(inverse, root[:, 7:10]),
                  quat_apply(inverse, root[:, 10:13]) / 3.,
                  self.get(state, "dof_pos"), self.get(state, "dof_vel") / 5.]
        for name in ("object", "table"):
            body = self.get(state, name)
            blocks.extend((quat_apply(inverse, body[:, :3] - root[:, :3]),
                           rotation_6d(quat_mul(inverse, quat_normalize(body[:, 3:7]))),
                           quat_apply(inverse, body[:, 7:10]),
                           quat_apply(inverse, body[:, 10:13]) / 3.))
        return torch.cat(blocks, -1)

    features = low_features

    def policy_observation(self, state, context):
        """Match real high-level sensor noise; keep model labels/critic clean."""
        observation = self.observation(state, context)
        noise = getattr(self.env, "world_model_observation_noise", None)
        return noise(observation) if noise is not None else observation

    def normalize_context(self, context):
        ids = self._ids(context)
        objects = F.one_hot(ids % self.object_count, self.object_count).to(context.dtype)
        conditions = torch.cat((context[:, 2:3], context[:, 4:]), -1)
        return torch.cat((conditions, self.physical_context[ids] / self.physical_context_scale,
                          objects), -1)

    def low_action_features(self, state, action):
        root, heading = self._heading(state)
        return torch.cat((action[:, :12],
                          quat_inverse_apply(heading, action[:, 12:15] - root[:, :3]),
                          rotation_6d(quat_mul(quat_conjugate(heading), quat_normalize(action[:, 15:19]))),
                          action[:, 19:20]), -1)

    def critic_features(self, state, context):
        """Privileged, fixed-scale value input retaining differentiable memory.

        Unlike the physical MLP, the value depends on low-policy history,
        high-level targets, the causal EKF and task progress. No simulator EE
        cache or environment slot number is treated as a predictive feature.
        Initial joint offsets are included because low-policy observations
        subtract them. Raw yaw deviation matches the task's actual failure rule.
        """
        root, heading = self._heading(state)
        inverse = quat_conjugate(heading)
        ids = self._ids(context)
        history_scale = state.new_ones(71)
        history_scale[2:5] = 3.  # body angular velocity
        history_scale[23:41] = .25  # recorded qd has already been multiplied by .05
        history_scale[57:60] = state.new_tensor([self.forward_limit, .6, .6])
        history_scale[60:63] = .7
        history_scale[63:66] = math.pi
        history_scale[66] = 2.  # gait phase can enter the stopping interval [1,2)
        low_history = self.get(state, "low_history").reshape(-1, 10, 71) / history_scale
        held = self.get(state, "held_ee_target")
        held_pose = torch.cat((quat_apply(inverse, held[:, :3] - root[:, :3]) / .7,
                               rotation_6d(quat_mul(inverse, quat_normalize(held[:, 3:7])))), -1)
        poses = self.get(state, "ekf_poses").reshape(-1, self.history, 7)
        pose_memory = torch.cat((quat_apply(inverse[:, None], poses[..., :3] - root[:, None, :3]),
                                 rotation_6d(quat_mul(inverse[:, None], quat_normalize(poses[..., 3:7])))), -1)
        duration = max(self.high_dt * self.history, self.high_dt)
        hold_scale = max(self.hold_steps, 1)
        grip_scale = max(abs(self.q_open), abs(self.q_closed), 1e-6)
        blocks = [self.low_features(state), self.normalize_context(context),
                  (euler(root[:, 3:7])[:, 2:3] - context[:, 1:2]) / math.pi,
                  context[:, 3:4], self.get(state, "last_dof_vel") / 5.,
                  low_history.flatten(1), self.get(state, "low_actions"),
                  self.get(state, "gait") / 2., self.get(state, "clock"),
                  self.get(state, "gait_wait") / 35., self.get(state, "walking"),
                  self.get(state, "foot_contacts"), self.get(state, "goal_pos") / .7,
                  self.get(state, "goal_rpy") / math.pi,
                  self.get(state, "commands") / state.new_tensor([self.forward_limit, .6, .6]),
                  self.get(state, "prev_action"), held_pose,
                  self.get(state, "held_gripper_target") / grip_scale,
                  pose_memory.flatten(1), self.get(state, "ekf_ages") / duration,
                  self.get(state, "ekf_count") / max(self.history, 1),
                  self.get(state, "closest"), self.get(state, "highest"),
                  self.get(state, "hold") / hold_scale, self.get(state, "best_hold") / hold_scale,
                  self.get(state, "completed"), self.get(state, "progress") / max(self.max_steps, 1),
                  self.get(state, "table_height"), self.initial_q[ids]]
        return torch.cat(blocks, -1)

    def prepare_high(self, state, action):
        action, goal, rpy, commands = self._control(state, action)
        root = self.get(state, "robot")
        base = quat_normalize(root[:, 3:7])
        arm = root[:, :3] + quat_apply(base, self.arm_offset.expand(len(state), -1))
        target = torch.cat((arm + quat_apply(base, goal),
                            quat_mul(base, from_euler(rpy))), -1)
        grip = self.q_closed + action[:, 6:7] * (self.q_open - self.q_closed)
        return self.replace(state, goal_pos=goal, goal_rpy=rpy, commands=commands,
                            prev_action=action, held_ee_target=target, held_gripper_target=grip)

    def low_observation(self, state, context):
        """Return (781-D input, advanced gait/history state), without NN inference."""
        commands = self.get(state, "commands")
        walking = ((commands[:, 0].abs() > .15) | (commands[:, 1].abs() > .35)
                   | (commands[:, 2].abs() > .35))
        was_walking = self.get(state, "walking")[:, 0].bool()
        wait = self.get(state, "gait_wait")[:, 0]
        phase = torch.remainder(self.get(state, "gait")[:, 0] + self.low_dt * 2., 1.)
        sudden = (wait > 0) | (~walking & was_walking)
        phase = phase + sudden.to(state.dtype)
        overdue = wait >= 35
        phase = torch.where(overdue, torch.zeros_like(phase), phase)
        wait = torch.where(overdue, torch.zeros_like(wait), wait)
        stopped = (wait == 0) & ~walking & ~was_walking
        phase = torch.where(stopped, torch.zeros_like(phase), phase)
        clock = torch.sin(2 * math.pi * (phase[:, None] + state.new_tensor([.5, 0., 0., .5])))
        root = self.get(state, "robot")
        q = (self.get(state, "dof_pos") - self.initial_q[self._ids(context)])[:, self.reindex[:-1]]
        qd = self.get(state, "dof_vel")[:, self.reindex[:-1]] * .05
        goal = self.get(state, "goal_pos")
        if self.mask_arm:
            q = torch.cat((q[:, :12], torch.zeros_like(q[:, 12:])), -1)
            qd = torch.cat((qd[:, :12], torch.zeros_like(qd[:, 12:])), -1)
            goal = state.new_tensor([.46, 0., .31]).expand(len(state), -1)
        # context[22] is the per-environment forward command scale.
        low_commands = torch.cat((commands[:, :1] * context[:, 22:23], commands[:, 1:]), -1)
        current = torch.cat((euler(root[:, 3:7])[:, :2],
                             quat_inverse_apply(quat_normalize(root[:, 3:7]), root[:, 10:13]),
                             q, qd, self.get(state, "low_actions")[:, self.reindex[:12]],
                             self.get(state, "foot_contacts")[:, self.feet_reindex],
                             low_commands, goal, torch.zeros_like(goal), phase[:, None], clock), -1)
        history = self.get(state, "low_history").reshape(-1, 10, 71)
        history = torch.where((self.get(state, "progress") < 1)[:, :, None],
                              current[:, None].expand(-1, 10, -1), history)
        observation = torch.cat((current, history.flatten(1)), -1)
        history = torch.cat((history[:, 1:], current[:, None]), 1)
        advanced = self.replace(state, gait=phase[:, None], clock=clock, gait_wait=wait[:, None],
                                walking=walking[:, None].to(state.dtype), low_history=history.flatten(1))
        return observation, advanced

    def low_control(self, state, context):
        observation, state = self.low_observation(state, context)
        # Weights are frozen, input Jacobians deliberately remain enabled.
        outputs = self.low_actor(observation, hist_encoding=True)
        actions = outputs[:, self.reindex[:-1]]
        state = self.replace(state, low_actions=actions)
        command = torch.cat((actions[:, :12], self.get(state, "held_ee_target"),
                              self.get(state, "held_gripper_target")), -1)
        return state, command

    def nominal_table_step(self, state, context):
        table = self.get(state, "table")
        position = table[:, :3]
        yaw = euler(table[:, 3:7])[:, 2]
        level = self.env.table_level
        local_velocity = torch.cat((context[:, 23:25], torch.zeros_like(context[:, :1])), -1)
        angular = torch.cat((torch.zeros_like(context[:, :2]), context[:, 25:26]), -1)
        if level == "Level01":
            yaw = torch.zeros_like(yaw)
        # The actual controller overwrites surface velocity every 5 ms. Track
        # this known nominal motion; never read real future table tensors.
        for _ in range(4):
            rotation = from_euler(torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), -1))
            velocity = quat_apply(rotation, local_velocity)
            position = position + velocity * .005
            yaw = yaw + angular[:, 2] * .005
        position = torch.cat((position[:, :2], self.get(state, "table_height") - self.table_thickness / 2), -1)
        rotation = from_euler(torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), yaw), -1))
        return torch.cat((position, rotation, velocity, angular), -1)

    def integrate_low(self, state, delta, action, context, dt):
        changes = {}
        _, heading = self._heading(state)
        for i, name in enumerate(("robot", "object")):
            old = self.get(state, name)
            d = delta[:, i*12:(i+1)*12]
            dv = quat_apply(heading, d[:, :3] * .3)
            dw = quat_apply(heading, d[:, 3:6] * .5)
            dp = quat_apply(heading, d[:, 6:9] * .01)
            dr = quat_apply(heading, d[:, 9:12] * .02)
            velocity, angular = old[:, 7:10] + dv, old[:, 10:13] + dw
            rotation = quat_normalize(quat_mul(from_rotvec(angular * dt + dr),
                                                 quat_normalize(old[:, 3:7])))
            temporary = torch.cat((old[:, :3], rotation, velocity, angular), -1)
            position_velocity = (self.kinematics.root_origin_velocity(temporary)
                                 if name == "robot" else velocity)
            changes[name] = torch.cat((old[:, :3] + position_velocity * dt + dp,
                                       rotation, velocity, angular), -1)
        qd = self.get(state, "dof_vel") + delta[:, 43:62] * 2.
        changes.update(dof_vel=qd, dof_pos=self.get(state, "dof_pos") + qd * dt + delta[:, 24:43] * .02,
                       table=self.nominal_table_step(state, context))
        # Real post_physics_step refreshes once more without advancing physics;
        # its acceleration reward compares the same endpoint velocities.
        changes["last_dof_vel"] = qd
        return self.replace(state, **changes)

    def with_contacts(self, state, contacts):
        return self.replace(state, foot_contacts=contacts)

    def finish_high(self, previous, state, action, context):
        fields = {name: state[:, sl] for name, sl in self.slices.items()}
        fields["progress"] = self.get(previous, "progress") + 1
        poses = self.get(previous, "ekf_poses").reshape(-1, self.history, 7)
        pose = self.get(state, "object")[:, :7]
        first = self.get(previous, "ekf_count") < 1
        fields["ekf_poses"] = torch.where(first[:, :, None], pose[:, None],
            torch.cat((poses[:, 1:], pose[:, None]), 1)).flatten(1)
        ages = torch.cat((self.get(previous, "ekf_ages")[:, 1:] - self.high_dt,
                          torch.zeros_like(first, dtype=state.dtype)), -1)
        fields["ekf_ages"] = torch.where(first, torch.zeros_like(ages), ages)
        fields["ekf_count"] = (self.get(previous, "ekf_count") + 1).clamp_max(self.history)
        candidate = self._pack(fields)
        executed = self.get(state, "prev_action")
        outcome = self.task_outcome(previous, candidate, executed, context)
        fields.update(hold=outcome["hold"][:, None], best_hold=outcome["best_hold"][:, None],
                      completed=outcome["completed"][:, None].to(state.dtype))
        for name, value in (("closest", outcome["distance"]), ("highest", outcome["height"])):
            old = self.get(previous, name)
            combine = torch.minimum if name == "closest" else torch.maximum
            fields[name] = torch.where(old < 0, value[:, None], combine(old, value[:, None]))
        return self._pack(fields)

    def error_vector(self, predicted, target):
        groups = []
        for name in ("robot", "object"):
            a, b = self.get(predicted, name), self.get(target, name)
            dq = quat_mul(quat_conjugate(quat_normalize(b[:, 3:7])), quat_normalize(a[:, 3:7]))
            groups.append(torch.cat(((a[:, :3]-b[:, :3])/.05, rotation_vector(dq)/.1,
                                      (a[:, 7:10]-b[:, 7:10])/.5, a[:, 10:13]-b[:, 10:13]), -1))
        for sl in (slice(0, 12), slice(12, 18), slice(18, 19)):
            groups.append(torch.cat(((self.get(predicted, "dof_pos")[:, sl]-self.get(target, "dof_pos")[:, sl])/.1,
                                      (self.get(predicted, "dof_vel")[:, sl]-self.get(target, "dof_vel")[:, sl])/2.), -1))
        return torch.cat([g * math.sqrt(62 / (len(groups)*g.shape[-1])) for g in groups], -1)

    def sample_dt(self, batch_size):
        return torch.full((batch_size, 1), self.high_dt, device=self.device, dtype=self.dtype)

    def metadata(self):
        result = super().metadata()
        result.update(version=5, schema="dq-sgs-ekf-fixed8-physical64-rot6d-v5", low_action_dim=20,
                      low_action_feature_dim=self.low_action_feature_dim,
                      low_feature_dim=self.low_feature_dim, critic_feature_dim=self.critic_feature_dim,
                      critic_features="canonical_physics_context_and_explicit_memory_fixed_scales_v1",
                      orientation_features="heading_rot6d_first_two_columns_row_major",
                      low_steps=8, high_dt=.16, contact_outputs=4,
                      table_model="known_nominal_four_5ms_steps_no_learned_table",
                      low_controller_memory="explicit_10x71_shift_and_frozen_policy",
                      end_effector="differentiable_urdf_fk",
                      termination="separate_hard_success_failure_timeout_absorption")
        result["loss_contract"]["main"] = "equal group weights: root, object, legs, arm, gripper; 62 residuals"
        result["loss_contract"]["main_rotation"] = "shortest quaternion logarithm (rotation vector in radians)"
        result["loss_contract"]["contact"] = "four sensor labels, BCE; hard binary low-policy inputs"
        result["loss_contract"]["main_scales"].pop("low_controller_memory", None)
        return result
