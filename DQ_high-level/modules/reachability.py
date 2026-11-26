"""Learned time-to-reach T(s, g) and the interception margin M (main line A).

T(s, g) is the time, in seconds, the robot needs to bring its end effector
(EE) to the position g, starting from the robot state s. s uses only
proprioception and controller state, all available on hardware (see
REACH_STATE_NAMES). g is expressed in the query frame: origin at the arm base,
x along the base heading, z up (world vertical), frozen at the query time.

Labels are hindsight upper bounds of the minimum time: the pair
(s_t, EE position at t+k) took k steps. A low expectile (tau < 0.5) fits the
lower envelope of the observed times. A conservative physical lower bound is
added as a floor, so T >= floor for any parameters.

M(s_t) = max over grasp candidates c and horizons h <= remaining time of
[h - T(s_t, g_c(t+h))]. M >= 0 means some candidate can be reached no later
than the object gets there. This module imports no Isaac Gym.
"""

import copy
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.pose_prediction import quaternion_to_matrix

# Proprioceptive inputs of T, in feature order. Widths are fixed except the
# gripper, whose width is the number of gripper DOFs (1 on B1Z1).
REACH_STATE_LAYOUT = (
    ("ee_pos_query_frame", 3),     # EE position relative to the arm base, query frame
    ("ee_rot_query_frame_6d", 6),  # EE rotation, first two matrix columns, query frame
    ("projected_gravity", 3),      # gravity direction in the base frame (IMU)
    ("base_ang_vel", 3),           # base angular velocity in the base frame (IMU)
    ("arm_dof_pos", 6),
    ("arm_dof_vel_x0.1", 6),
    ("gripper_dof_pos", None),
    ("ee_goal_cart", 3),           # the EE target integrator, arm-base frame
    ("ee_goal_rpy_sin", 3),
    ("ee_goal_rpy_cos", 3),
    ("base_commands_vx_wz", 2),    # commands sent to the low-level policy
)
REACH_STATE_NAMES = tuple(name for name, _ in REACH_STATE_LAYOUT)
ARM_BASE_OFFSET = (0.3, 0.0, 0.09)  # arm base in the base frame (b1z1 URDF)
REACH_MODEL_KINDS = ("analytic", "mlp", "quasimetric")
REACH_MODEL_FORMAT = 1
# Actor-visible margin block (env.margin.tokens): these scalars, then one token per horizon.
MARGIN_SCALAR_NAMES = ("margin_ekf_s", "best_horizon_s")
# Per token: best candidate at that horizon in the query frame, its margin, validity.
MARGIN_TOKEN_NAMES = ("pos_x", "pos_y", "pos_z", "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3",
                      "rot6d_4", "rot6d_5", "horizon_s", "margin_s", "valid")
MARGIN_TOKEN_DIM = len(MARGIN_TOKEN_NAMES)


def reach_state_dim(num_gripper_dofs=1):
    return sum(num_gripper_dofs if width is None else width for _, width in REACH_STATE_LAYOUT)


def yaw_from_quat(quat):
    """Heading of xyzw quaternions, [...] radians."""
    x, y, z, w = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_rotate_inverse(yaw, vectors):
    """Rotate world vectors [..., 3] into a frame with heading yaw [...]."""
    cos, sin = torch.cos(yaw)[..., None], torch.sin(yaw)[..., None]
    x, y, z = vectors[..., 0:1], vectors[..., 1:2], vectors[..., 2:3]
    return torch.cat((cos * x + sin * y, -sin * x + cos * y, z), dim=-1)


def quat_to_matrix(quat):
    """xyzw quaternions of any batch shape to rotation matrices [..., 3, 3]."""
    shape = quat.shape[:-1]
    return quaternion_to_matrix(quat.reshape(-1, 4)).reshape(*shape, 3, 3)


def matrix_to_6d(rotation):
    """First two columns, [..., 6]; continuous, unlike Euler angles."""
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def yaw_matrix(yaw):
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack((cos, -sin, zero, sin, cos, zero, zero, zero, one), dim=-1).reshape(
        *yaw.shape, 3, 3)


def arm_base_position(base_pos, base_quat):
    offset = base_pos.new_tensor(ARM_BASE_OFFSET).expand(base_pos.shape)
    return base_pos + (quat_to_matrix(base_quat) @ offset.unsqueeze(-1)).squeeze(-1)


def query_frame(base_pos, base_quat):
    """(origin [N, 3], yaw [N]) of the yaw-aligned arm-base frame."""
    return arm_base_position(base_pos, base_quat), yaw_from_quat(base_quat)


def to_query_frame(points, origin, yaw):
    """World points [N, ..., 3] into the frame (origin [N, 3], yaw [N])."""
    extra = points.dim() - 2
    origin = origin.reshape(origin.shape[0], *([1] * extra), 3)
    yaw = yaw.reshape(yaw.shape[0], *([1] * extra))
    return yaw_rotate_inverse(yaw, points - origin)


def reach_state_features(base_pos, base_quat, base_ang_vel, ee_pos, ee_quat, arm_dof_pos,
                         arm_dof_vel, gripper_dof_pos, ee_goal_cart, ee_goal_rpy, base_commands):
    """[N, reach_state_dim] proprioceptive state of T; world inputs, xyzw quaternions.

    base_ang_vel is the world-frame root angular velocity (rotated into the base
    frame here, as an IMU measures it). base_commands holds (vx, wz).
    """
    origin, yaw = query_frame(base_pos, base_quat)
    base_rotation = quat_to_matrix(base_quat)
    inverse = base_rotation.transpose(-1, -2)
    ee_query = yaw_rotate_inverse(yaw, ee_pos - origin)
    ee_rotation = yaw_matrix(yaw).transpose(-1, -2) @ quat_to_matrix(ee_quat)
    gravity = (inverse @ base_pos.new_tensor([0., 0., -1.]).expand(base_pos.shape).unsqueeze(-1)).squeeze(-1)
    angular = (inverse @ base_ang_vel.unsqueeze(-1)).squeeze(-1)
    return torch.cat((ee_query, matrix_to_6d(ee_rotation), gravity, angular, arm_dof_pos,
                      0.1 * arm_dof_vel, gripper_dof_pos, ee_goal_cart, torch.sin(ee_goal_rpy),
                      torch.cos(ee_goal_rpy), base_commands), dim=-1)


@dataclass
class ReachLimits:
    """Command limits of the high-level action interface (b1z1_base.py)."""
    base_speed: float = 0.45       # m/s, |vx| command limit (clip_commands)
    yaw_rate: float = 0.6          # rad/s, |wz| command limit (pre_physics_step)
    arm_speed: float = 0.25        # m/s, 0.02 m EE-goal increment per axis per ~0.14 s step (reach_limits_for)
    arm_reach: float = 0.9         # m, generous planar EE reach from the arm base
    nominal_reach: float = 0.6     # m, analytic estimate: comfortable grasp distance
    max_time: float = 8.0          # s, upper clamp of every estimate

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"reach limit {name} must be positive and finite")
        if self.nominal_reach > self.arm_reach:
            raise ValueError("nominal_reach must not exceed arm_reach")
        return self


# The default arm_speed belongs to the 0.02 m per-axis EE-goal increment; the
# command-limited EE speed scales with the increment (env.eeDeltaPositionLimit).
LEGACY_EE_DELTA_POSITION_M = 0.02


def reach_limits_for(ee_delta_position_m):
    """ReachLimits whose arm_speed matches this per-step EE-goal increment."""
    return ReachLimits(arm_speed=ReachLimits.arm_speed * ee_delta_position_m
                       / LEGACY_EE_DELTA_POSITION_M).validate()


def physical_lower_bound(ee_query, goals, limits):
    """A floor of the minimum time, [N, G] seconds; ee_query [N, 3], goals [N, G, 3].

    The arm base moves at most at base_speed + yaw_rate * 0.3 m; the EE adds the
    arm's own speed and the rotation of the reach sphere. The planar goal must
    come within arm_reach of the arm base before the EE can reach it.
    """
    offset = ARM_BASE_OFFSET[0]
    arm_base_speed = limits.base_speed + limits.yaw_rate * offset
    ee_speed = arm_base_speed + limits.yaw_rate * limits.arm_reach + limits.arm_speed
    direct = (goals - ee_query[:, None, :]).norm(dim=-1) / ee_speed
    planar = goals[..., :2].norm(dim=-1)
    walk = (planar - limits.arm_reach).clamp_min(0.) / arm_base_speed
    return torch.maximum(direct, walk).clamp(max=limits.max_time)


def analytic_time_estimate(ee_query, goals, limits):
    """DYNAMIT-style distance over maximum speed (baseline B3), [N, G] seconds.

    Turn toward the goal, walk until it is nominal_reach away, then move the
    arm. A heuristic estimate, not a bound: it ignores acceleration limits.
    """
    planar = goals[..., :2].norm(dim=-1)
    needs_walk = planar > limits.nominal_reach
    bearing = torch.atan2(goals[..., 1], goals[..., 0]).abs()
    turn = torch.where(needs_walk, bearing / limits.yaw_rate, torch.zeros_like(bearing))
    walk = (planar - limits.nominal_reach).clamp_min(0.) / limits.base_speed
    arm_distance = (goals - ee_query[:, None, :]).norm(dim=-1).clamp(max=limits.nominal_reach)
    return (turn + walk + arm_distance / limits.arm_speed).clamp(max=limits.max_time)


GOAL_FEATURES = 11


def goal_features(ee_query, goals, floor):
    """[N, G, 11] per-goal inputs in meters/seconds/unit circle."""
    delta = goals - ee_query[:, None, :]
    planar = goals[..., :2].norm(dim=-1, keepdim=True)
    bearing = torch.atan2(goals[..., 1:2], goals[..., 0:1])
    return torch.cat((goals, delta, delta.norm(dim=-1, keepdim=True), planar,
                      torch.sin(bearing), torch.cos(bearing), floor.unsqueeze(-1)), dim=-1)


def _mlp(sizes, final_activation=False):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or final_activation:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class TimeToReach(nn.Module):
    """T(s, g) = floor(s, g) + softplus(residual), clamped at limits.max_time.

    kind:
      analytic     no parameters: analytic_time_estimate (baseline B3)
      mlp          state encoder phi(s) once per environment; each goal adds a
                   linear projection and a small head (cheap for 480 goals/env)
      quasimetric  MRN distance d(phi(s), psi(g)) = max_k relu(psi_k - phi_k)
                   + ||phi' - psi'|| (Liu et al. 2022), as in QRL
    State features are standardized with statistics fit from training data.
    """

    def __init__(self, state_dim, kind="mlp", hidden=256, latent=64, limits=None):
        super().__init__()
        if kind not in REACH_MODEL_KINDS:
            raise ValueError(f"reach model kind must be one of {REACH_MODEL_KINDS}")
        if type(state_dim) is not int or state_dim < 4:
            raise ValueError("state_dim must be an integer >= 4")
        if kind == "quasimetric" and latent % 2:
            raise ValueError("quasimetric latent size must be even")
        self.kind = kind
        self.state_dim = state_dim
        self.hidden = hidden
        self.latent = latent
        self.limits = (limits or ReachLimits()).validate()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_var", torch.ones(state_dim))
        self.register_buffer("state_count", torch.zeros((), dtype=torch.float64))
        if kind == "analytic":
            return
        self.state_encoder = _mlp([state_dim, hidden, hidden, latent])
        if kind == "mlp":
            self.goal_projection = nn.Linear(GOAL_FEATURES, latent)
            self.head = nn.Sequential(nn.ELU(), nn.Linear(latent, latent), nn.ELU(), nn.Linear(latent, 1))
        else:
            self.goal_encoder = _mlp([GOAL_FEATURES, hidden // 2, latent])
            self.log_scale = nn.Parameter(torch.zeros(()))
        # Start near the floor: a small positive residual.
        last = self.head[-1] if kind == "mlp" else None
        if last is not None:
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, -2.0)

    @property
    def learned(self):
        return self.kind != "analytic"

    @torch.no_grad()
    def update_normalizer(self, states):
        """Chan et al. parallel mean/variance update over a batch of states."""
        states = states.reshape(-1, self.state_dim).to(self.state_mean.dtype)
        count = states.shape[0]
        if count == 0:
            return
        batch_mean = states.mean(0)
        batch_var = states.var(0, unbiased=False)
        total = self.state_count + count
        delta = batch_mean - self.state_mean
        ratio = (count / total).to(self.state_mean.dtype)
        self.state_mean.add_(delta * ratio)
        old = (self.state_count / total).to(self.state_mean.dtype)
        self.state_var.copy_(old * self.state_var + ratio * batch_var + old * ratio * delta.square())
        self.state_count.copy_(total)

    def encode_state(self, states):
        if not self.learned:
            return None
        normalized = (states - self.state_mean) / self.state_var.clamp_min(1e-6).sqrt()
        return self.state_encoder(normalized.clamp(-10., 10.))

    def forward(self, states, goals, encoded=None):
        """states [N, S], goals [N, G, 3] in the query frame -> [N, G] seconds."""
        if states.dim() != 2 or states.shape[-1] != self.state_dim:
            raise ValueError("states must have shape [N, state_dim]")
        if goals.dim() != 3 or goals.shape[0] != states.shape[0] or goals.shape[-1] != 3:
            raise ValueError("goals must have shape [N, G, 3]")
        ee_query = states[:, :3]
        floor = physical_lower_bound(ee_query, goals, self.limits)
        if not self.learned:
            return torch.maximum(analytic_time_estimate(ee_query, goals, self.limits), floor)
        phi = self.encode_state(states) if encoded is None else encoded
        features = goal_features(ee_query, goals, floor)
        if self.kind == "mlp":
            residual = self.head(phi[:, None, :] + self.goal_projection(features)).squeeze(-1)
            extra = F.softplus(residual)
        else:
            psi = self.goal_encoder(features)
            half = self.latent // 2
            x = phi[:, None, :].expand_as(psi)
            asymmetric = (psi[..., :half] - x[..., :half]).clamp_min(0.).amax(-1)
            symmetric = (psi[..., half:] - x[..., half:]).norm(dim=-1)
            extra = self.log_scale.exp() * (asymmetric + symmetric)
        return (floor + extra).clamp(max=self.limits.max_time)

    @torch.no_grad()
    def evaluate(self, states, goals, chunk=1024):
        """Inference in environment chunks, bounding intermediate memory."""
        outputs = []
        for start in range(0, states.shape[0], chunk):
            outputs.append(self(states[start:start + chunk], goals[start:start + chunk]))
        if not outputs:
            return goals.new_zeros(goals.shape[:2])
        return torch.cat(outputs, dim=0)

    def config(self):
        return {"kind": self.kind, "state_dim": self.state_dim, "hidden": self.hidden,
                "latent": self.latent, "limits": asdict(self.limits)}


def build_reach_model(config):
    return TimeToReach(config["state_dim"], kind=config["kind"], hidden=config.get("hidden", 256),
                       latent=config.get("latent", 64), limits=ReachLimits(**config.get("limits", {})))


def save_reach_model(path, model, metadata=None):
    """A self-describing file: architecture, weights, normalizer and provenance."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": REACH_MODEL_FORMAT, "config": model.config(),
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "state_names": list(REACH_STATE_NAMES), "metadata": dict(metadata or {})}, path)


def load_reach_model(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != REACH_MODEL_FORMAT:
        raise ValueError(f"{path} is not a reach model file (format {REACH_MODEL_FORMAT})")
    if payload.get("state_names") != list(REACH_STATE_NAMES):
        raise ValueError("reach model state features do not match this code's REACH_STATE_LAYOUT")
    model = build_reach_model(payload["config"])
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(), payload.get("metadata", {})


def expectile_loss(prediction, target, tau):
    """Asymmetric squared loss: tau < 0.5 fits a lower envelope of the targets."""
    error = target - prediction
    weight = torch.where(error < 0, 1.0 - tau, tau)
    return (weight * error.square()).mean()


class HindsightBuffer:
    """Ring buffer of the last `window` post-step states of every environment.

    A pair (start, end) is valid when both entries belong to the same episode;
    its goal is the EE position at the end expressed in the start's query frame
    and its label is the elapsed time between them.
    """

    def __init__(self, num_envs, window, state_dim, device):
        if type(window) is not int or window < 2:
            raise ValueError("hindsight window must be an integer >= 2")
        self.num_envs, self.window, self.state_dim = num_envs, window, state_dim
        self.device = torch.device(device)
        self.state = torch.zeros((window, num_envs, state_dim), device=device)
        self.ee = torch.zeros((window, num_envs, 3), device=device)
        self.origin = torch.zeros((window, num_envs, 3), device=device)
        self.yaw = torch.zeros((window, num_envs), device=device)
        self.time = torch.zeros((window, num_envs), device=device)
        self.episode = torch.full((window, num_envs), -1, dtype=torch.long, device=device)
        self.head = 0
        self.filled = 0

    @torch.no_grad()
    def push(self, state, ee_world, origin, yaw, episode_time, episode_id):
        slot = self.head
        self.state[slot] = state
        self.ee[slot] = ee_world
        self.origin[slot] = origin
        self.yaw[slot] = yaw
        self.time[slot] = episode_time
        self.episode[slot] = episode_id
        self.head = (slot + 1) % self.window
        self.filled = min(self.filled + 1, self.window)

    def _slot(self, age):
        return (self.head - 1 - age) % self.window

    @torch.no_grad()
    def sample(self, batch, max_steps, env_ids, generator=None):
        """Up to `batch` valid pairs from env_ids; invalid draws are dropped.

        Returns (states [B, S], goals [B, 3], labels [B] seconds, steps [B]).
        """
        empty = (self.state.new_zeros((0, self.state_dim)), self.ee.new_zeros((0, 3)),
                 self.time.new_zeros(0), torch.zeros(0, dtype=torch.long, device=self.device))
        if self.filled < 1 or env_ids.numel() == 0 or batch <= 0:
            return empty
        max_steps = min(int(max_steps), self.filled - 1)
        draw = lambda high: torch.randint(0, high, (batch,), device=self.device, generator=generator)
        envs = env_ids[draw(env_ids.numel())]
        start_age = draw(self.filled)
        steps = draw(max_steps + 1)
        end_age = start_age - steps
        ok = end_age >= 0
        start_slot = self._slot(start_age)
        end_slot = self._slot(end_age.clamp_min(0))
        start_episode = self.episode[start_slot, envs]
        ok &= (start_episode >= 0) & (start_episode == self.episode[end_slot, envs])
        start_slot, end_slot, envs, steps = start_slot[ok], end_slot[ok], envs[ok], steps[ok]
        if envs.numel() == 0:
            return empty
        goals = to_query_frame(self.ee[end_slot, envs][:, None, :], self.origin[start_slot, envs],
                               self.yaw[start_slot, envs])[:, 0]
        labels = (self.time[end_slot, envs] - self.time[start_slot, envs]).clamp_min(0.)
        return self.state[start_slot, envs], goals, labels, steps


class ReachLearner:
    """Trains T from the hindsight buffer, outside the PPO graph.

    `model` takes gradient steps with its own optimizer; `target` is an EMA
    copy that the environment queries for rewards, tokens and the curriculum.
    Environments with id % validation_every == 0 never enter training, so
    `validate` reports held-out error (gate D2).
    """

    def __init__(self, model, num_envs, device, window=48, max_steps=32, batch_size=8192,
                 learning_rate=3e-4, expectile=0.1, ema_decay=0.99, validation_every=10,
                 normalizer_updates=200, seed=0):
        if not model.learned:
            raise ValueError("the analytic reach model has nothing to learn")
        if not 0 < expectile < 1 or not 0 <= ema_decay < 1:
            raise ValueError("expectile must be in (0, 1) and ema_decay in [0, 1)")
        if type(max_steps) is not int or not 1 <= max_steps < window:
            raise ValueError("max_steps must be an integer in [1, window)")
        self.device = torch.device(device)
        self.model = model.to(self.device).train()
        self.target = copy.deepcopy(self.model).eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = HindsightBuffer(num_envs, window, model.state_dim, self.device)
        self.max_steps, self.batch_size = max_steps, batch_size
        self.expectile, self.ema_decay = expectile, ema_decay
        self.normalizer_updates = normalizer_updates
        env_ids = torch.arange(num_envs, device=self.device)
        validation = env_ids % validation_every == 0 if validation_every else torch.zeros_like(env_ids, dtype=torch.bool)
        if bool(validation.all()):
            validation[:] = False
        self.train_envs, self.validation_envs = env_ids[~validation], env_ids[validation]
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.updates = 0

    def observe(self, state, ee_world, origin, yaw, episode_time, episode_id):
        self.buffer.push(state, ee_world, origin, yaw, episode_time, episode_id)

    def update(self, num_updates):
        """Gradient steps; returns mean metrics (floats) or {} without data."""
        totals = {}
        steps = 0
        with torch.enable_grad():
            for _ in range(num_updates):
                states, goals, labels, _ = self.buffer.sample(
                    self.batch_size, self.max_steps, self.train_envs, self.generator)
                if labels.numel() < 2:
                    break
                if self.updates < self.normalizer_updates:
                    self.model.update_normalizer(states)
                labels = labels.clamp(max=self.model.limits.max_time)
                prediction = self.model(states, goals[:, None, :])[:, 0]
                loss = expectile_loss(prediction, labels, self.expectile)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self._update_target()
                self.updates += 1
                steps += 1
                for name, value in (("loss", loss), ("label_mean", labels.mean()),
                                    ("prediction_mean", prediction.detach().mean())):
                    totals[name] = totals.get(name, 0.) + value.detach()
        if not steps:
            return {}
        return {name: (value / steps).item() for name, value in totals.items()}

    @torch.no_grad()
    def _update_target(self):
        decay = self.ema_decay
        for target, source in zip(self.target.parameters(), self.model.parameters()):
            target.lerp_(source.detach(), 1.0 - decay)
        for target, source in zip(self.target.buffers(), self.model.buffers()):
            target.copy_(source)

    @torch.no_grad()
    def validate(self, batch_size=None, model=None):
        """Held-out quality of T versus the analytic estimate and the floor (gate D2)."""
        model = self.target if model is None else model
        states, goals, labels, steps = self.buffer.sample(
            batch_size or self.batch_size, self.max_steps, self.validation_envs, self.generator)
        if labels.numel() == 0:
            return {}
        return reach_quality(model, states, goals, labels)

    def state_dict(self):
        return {"model": self.model.state_dict(), "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(), "updates": self.updates,
                "config": self.model.config()}

    def load_state_dict(self, state):
        if state.get("config") != self.model.config():
            raise ValueError("reach learner checkpoint architecture does not match")
        self.model.load_state_dict(state["model"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state["updates"])

    def eval(self):
        """skrl's agent.load() calls eval() on loaded modules; the target stays in eval."""
        return self


@torch.no_grad()
def reach_quality(model, states, goals, labels, bins=(0.5, 1.0, 2.0, 3.0)):
    """Error statistics of T against hindsight labels, which are upper bounds.

    A good lower envelope has labels >= prediction for most pairs (small
    `above_label_fraction`) and a small mean gap. MAE by label bin shows where
    T fails. The floor and analytic estimate are reported on the same pairs.
    """
    ee_query = states[:, :3]
    goals = goals[:, None, :]
    prediction = model(states, goals)[:, 0]
    floor = physical_lower_bound(ee_query, goals, model.limits)[:, 0]
    analytic = torch.maximum(analytic_time_estimate(ee_query, goals, model.limits)[:, 0], floor)
    labels = labels.clamp(max=model.limits.max_time)
    result = {"pairs": float(labels.numel()), "label_mean": labels.mean().item()}
    for name, value in (("learned", prediction), ("analytic", analytic), ("floor", floor)):
        error = value - labels
        result[name + "_mae"] = error.abs().mean().item()
        result[name + "_bias"] = error.mean().item()
        result[name + "_above_label_fraction"] = (error > 0.15).float().mean().item()
        if labels.numel() > 2 and labels.std() > 0 and value.std() > 0:
            stacked = torch.stack((value, labels))
            result[name + "_correlation"] = torch.corrcoef(stacked)[0, 1].item()
    edges = (0.0,) + tuple(bins) + (float("inf"),)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (labels >= low) & (labels < high)
        if mask.any():
            key = f"learned_mae_label_{low:g}_{high:g}s"
            result[key] = (prediction[mask] - labels[mask]).abs().mean().item()
    return result


def interception_margin(times, horizons, remaining):
    """Margins from times [N, H, C] = T(s, g_c(t + h_k)), horizons [H], remaining [N].

    Returns a dict:
      best        [N, H]  best candidate margin h - T at each horizon, ignoring remaining
      per_horizon [N, H]  the same, -inf where h > remaining
      candidate   [N, H]  its index
      margin      [N]     M = max over valid horizons (h = 0 is always valid)
      horizon     [N]     index of the maximizing horizon
    """
    # Callers validate once that horizons[0] == 0 (margin_horizons); no device sync here.
    if times.dim() != 3 or times.shape[1] != horizons.numel():
        raise ValueError("times must have shape [N, H, C] matching the horizons")
    margins = horizons.view(1, -1, 1) - times
    best, candidate = margins.max(dim=-1)
    valid = horizons.view(1, -1) <= remaining.view(-1, 1) + 1e-6
    valid[:, 0] = True
    per_horizon = torch.where(valid, best, torch.full_like(best, -float("inf")))
    margin, horizon = per_horizon.max(dim=-1)
    return {"best": best, "per_horizon": per_horizon, "candidate": candidate, "margin": margin,
            "horizon": horizon, "valid": valid}


def intercept_time(margins, horizons, max_time):
    """Earliest interception time tau* [N] from per-horizon margins [N, H] = h_k - T(s, g(t + h_k)).

    tau* is where m(h) = h - T first reaches 0, interpolated linearly between
    horizons; for a static goal it equals T(s, g), and it is 0 at the goal.
    Without a crossing up to the last horizon h_L it continues as h_L - m_L
    (T to the object's position at h_L), so tau* is continuous at that boundary.
    Unlike the margin M, which a slow object pushes to the last horizon and the
    clip saturates near the goal, tau* keeps decreasing until the grasp.
    Clamped to [0, max_time]. horizons [H] start at 0 (margin_horizons).
    """
    if margins.dim() != 2 or margins.shape[1] != horizons.numel():
        raise ValueError("margins must have shape [N, H] matching the horizons")
    reached = margins >= 0
    first = reached.to(margins.dtype).argmax(dim=-1)  # first reached horizon (0 if none)
    previous = (first - 1).clamp_min(0)
    m1 = margins.gather(1, first[:, None]).squeeze(1)
    m0 = margins.gather(1, previous[:, None]).squeeze(1)
    h1, h0 = horizons[first], horizons[previous]
    # first > 0 implies m0 < 0 <= m1, so the crossing fraction lies in (0, 1].
    fraction = (-m0 / (m1 - m0).clamp_min(1e-6)).clamp(0., 1.)
    crossing = torch.where(first > 0, h0 + (h1 - h0) * fraction, torch.zeros_like(m1))
    beyond = horizons[-1] - margins[:, -1]
    return torch.where(reached.any(dim=-1), crossing, beyond).clamp(0., max_time)


def planar_carrier_future(body_pos, body_quat, carrier_pos, carrier_yaw, body_velocity,
                          yaw_rate, horizons):
    """Future world poses of a body rigidly attached to a planar carrier.

    The carrier (the moving table) keeps a constant body-frame planar velocity
    body_velocity [N, 2] and yaw rate [N] (DQ-Bench Level00-02 exactly, Level03/04
    until the next command switch). horizons [H] seconds. Returns positions
    [N, H, 3] and xyzw quaternions [N, H, 4] (privileged: true current motion).
    """
    h = horizons.view(1, -1)
    w = yaw_rate.view(-1, 1)
    angle = w * h
    # sin(wh)/w and (1 - cos(wh))/w without division by zero.
    a = h * torch.sinc(angle / math.pi)
    b = 0.5 * w * h.square() * torch.sinc(angle / (2 * math.pi)).square()
    vx, vy = body_velocity[:, 0:1], body_velocity[:, 1:2]
    local = torch.stack((a * vx - b * vy, b * vx + a * vy), dim=-1)  # [N, H, 2]
    cos, sin = torch.cos(carrier_yaw).view(-1, 1), torch.sin(carrier_yaw).view(-1, 1)
    shift = torch.stack((cos * local[..., 0] - sin * local[..., 1],
                         sin * local[..., 0] + cos * local[..., 1]), dim=-1)
    lever = body_pos[:, :2] - carrier_pos[:, :2]
    cos_d, sin_d = torch.cos(angle), torch.sin(angle)
    rotated = torch.stack((cos_d * lever[:, None, 0] - sin_d * lever[:, None, 1],
                           sin_d * lever[:, None, 0] + cos_d * lever[:, None, 1]), dim=-1)
    xy = carrier_pos[:, None, :2] + shift + rotated
    z = body_pos[:, None, 2:3].expand(-1, h.shape[1], 1)
    half = 0.5 * angle
    turn = torch.stack((torch.zeros_like(half), torch.zeros_like(half), torch.sin(half), torch.cos(half)), dim=-1)
    quat = quat_multiply(turn, body_quat[:, None, :].expand_as(turn))
    return torch.cat((xy, z), dim=-1), quat


def quat_multiply(a, b):
    """Hamilton product of xyzw quaternions (a applied after b)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz), dim=-1)


def margin_horizons(start, step, count):
    """Horizons [count] seconds: start, start + step, ...; the first must be 0."""
    if (isinstance(count, bool) or type(count) is not int or count < 1
            or not math.isfinite(float(step)) or step <= 0 or start != 0):
        raise ValueError("margin horizons need start 0, a positive step and an integer count >= 1")
    return [round(start + i * step, 6) for i in range(count)]


def grasp_positions_world(object_pos, object_quat, grasp_in_object):
    """Candidate grasp positions [N, H, C, 3].

    object_pos [N, H, 3], object_quat [N, H, 4]; grasp_in_object [N, C, 4, 4]
    holds each candidate's transform in the object frame (PredictPoint).
    """
    rotation = quat_to_matrix(object_quat)  # [N, H, 3, 3]
    offsets = grasp_in_object[:, None, :, :3, 3].unsqueeze(-1)  # [N, 1, C, 3, 1]
    return object_pos[:, :, None, :] + (rotation[:, :, None] @ offsets).squeeze(-1)


def selected_grasp_rotations(object_quat, grasp_in_object, candidate, correction=None):
    """World rotations [N, H, 3, 3] of one chosen candidate per horizon.

    candidate [N, H] indexes grasp_in_object's C axis. `correction` [3, 3] is
    the fixed gripper-convention change PredictPoint applies (Rx Rz Rx).
    """
    rotation = quat_to_matrix(object_quat)
    index = candidate[..., None, None].expand(-1, -1, 3, 3)
    local = grasp_in_object[:, :, :3, :3].gather(1, index)
    result = rotation @ local
    return result if correction is None else result @ correction
