"""Configuration-indexed Temporal Self-Imitation Learning memory.

The fastest-first/return-fallback selection and zero-tail discounted return
recurrence are adapted from Yinsen Jia and Boyuan Chen's TSIL (Apache-2.0),
``core/training/algo/tsil/{memory,sampling}.py`` at commit 137e3ab.
See ``third_party/tsil/LICENSE`` and ``third_party/tsil/NOTICE.md``.

DQ adaptations: persistent SGS configuration IDs, complete asynchronous
episodes, exact causal observations, current command-reward relabeling, and
the paper's all-configuration positive-gap sampling (the upstream main path
uses a different sampler). This module does not import Isaac Gym.
"""

import math
from dataclasses import dataclass

import torch


GEOMETRY_DESCRIPTOR_SCHEMA = "dq_full_task_planar_v2"
GEOMETRY_DESCRIPTOR_NAMES = (
    "object_base_yaw_x", "object_base_yaw_y",
    "object_heading_base_yaw_cos", "object_heading_base_yaw_sin", "table_height",
)
LOOKUP_NEAREST = 0
LOOKUP_NO_SUCCESS = 1
LOOKUP_OUT_OF_SUPPORT = 2
LOOKUP_INVALID = 3
LOOKUP_CONFIGURATION = 4
LOOKUP_STATUS_NAMES = {
    LOOKUP_NEAREST: "nearest", LOOKUP_NO_SUCCESS: "no_success",
    LOOKUP_OUT_OF_SUPPORT: "out_of_support", LOOKUP_INVALID: "invalid",
    LOOKUP_CONFIGURATION: "configuration",
}


def _planar_heading(columns):
    length = torch.linalg.vector_norm(columns, dim=-1, keepdim=True)
    # An object with a vertical local-x axis has no heading under this schema;
    # never silently interpret its zero vector as an ordinary horizontal pose.
    return torch.where(length > 1e-6, columns / length.clamp_min(1e-6),
                       torch.full_like(columns, float("nan")))


def geometry_descriptors(robot_root, object_root, table_height, dof_pos=None):
    """Observable full-task configuration, independent of reset settling.

    Roots use xyzw quaternions. The SGS full-task builder varies base position
    and heading, object world-z rotation, and table height. Its default arm and
    grounded legs are initialization mechanisms, not sampled task coordinates.
    Ordinary reset observes them before settling, so height, roll/pitch and
    joints must not define the lookup support. Keep object heading: unlike the
    small settling tilt, the builder's +/-15 degree yaw is a task variable.

    Columns are object xy in robot's yaw frame, the unit planar projection of
    object local-x in that frame, and table height. This schema is invariant to
    world xy translation/yaw, excludes hidden velocity and EKF predictions,
    and applies to objects whose local-x axis has a horizontal projection.
    ``dof_pos`` is accepted for callers of the legacy 32-dimensional schema.
    """
    robot = torch.as_tensor(robot_root)
    obj = torch.as_tensor(object_root, device=robot.device, dtype=robot.dtype)
    q = robot[..., 3:7]
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = yaw.cos(), yaw.sin()
    delta = obj[..., :3] - robot[..., :3]
    relative = torch.stack((c * delta[..., 0] + s * delta[..., 1],
                            -s * delta[..., 0] + c * delta[..., 1]), -1)
    oq = obj[..., 3:7]
    oq = oq / torch.linalg.vector_norm(oq, dim=-1, keepdim=True).clamp_min(1e-8)
    ox, oy, oz, ow = oq.unbind(-1)
    # Rotation-matrix local-x avoids quaternion sign ambiguity and yaw wrapping.
    axis_x, axis_y = 1 - 2 * (oy * oy + oz * oz), 2 * (ox * oy + ow * oz)
    heading = _planar_heading(torch.stack((c * axis_x + s * axis_y,
                                           -s * axis_x + c * axis_y), -1))
    height = torch.as_tensor(table_height, device=robot.device, dtype=robot.dtype).reshape(-1, 1)
    return torch.cat((relative, heading, height), -1).float()


configuration_descriptors = geometry_descriptors


class TemporalState:
    """Targets and geometric evaluation references belonging to a model."""

    STATE_VERSION = 2

    def __init__(self, num_configs, t_ref, device="cpu", bank_fingerprint="",
                 family_ids=None, descriptors=None, descriptor_names=None,
                 support_margin=0.05, constant_tolerance=1e-4,
                 descriptor_schema="generic_v1"):
        if int(num_configs) < 0 or not math.isfinite(float(t_ref)) or t_ref <= 0:
            raise ValueError("TSIL requires nonnegative configuration count and positive t_ref")
        self.device = torch.device(device)
        self.num_configs = int(num_configs)
        self.t_ref = float(t_ref)
        self.bank_fingerprint = str(bank_fingerprint)
        self.targets = torch.full((self.num_configs,), self.t_ref, device=self.device)
        self.seen_success = torch.zeros(self.num_configs, dtype=torch.bool, device=self.device)
        self.pending_min = torch.full_like(self.targets, float("inf"))
        self.family_ids = (torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
                           if family_ids is None else torch.as_tensor(
                               family_ids, device=self.device, dtype=torch.long).clone())
        self.descriptors = (torch.empty((self.num_configs, 0), device=self.device)
                            if descriptors is None else torch.as_tensor(
                                descriptors, device=self.device, dtype=torch.float32).clone())
        if self.family_ids.shape != (self.num_configs,) or self.descriptors.shape[0] != self.num_configs:
            raise ValueError("TSIL configuration metadata shape mismatch")
        if self.descriptors.ndim != 2 or not bool(torch.isfinite(self.descriptors).all()):
            raise ValueError("TSIL descriptors must be finite matrices")
        self.descriptor_schema = str(descriptor_schema)
        self.descriptor_names = self._validate_schema(
            self.descriptor_schema, self.descriptors.shape[1], descriptor_names)
        self.support_margin = float(support_margin)
        self.constant_tolerance = float(constant_tolerance)
        if (not math.isfinite(self.support_margin) or self.support_margin < 0
                or not math.isfinite(self.constant_tolerance) or self.constant_tolerance < 0):
            raise ValueError("TSIL descriptor support tolerances must be finite and nonnegative")
        self._refresh_support()

    @staticmethod
    def _validate_schema(schema, width, names):
        if schema == GEOMETRY_DESCRIPTOR_SCHEMA:
            expected = list(GEOMETRY_DESCRIPTOR_NAMES)
            if width != len(expected) or (names is not None and list(names) != expected):
                raise ValueError("TSIL planar descriptor schema/names mismatch")
            return expected
        if schema != "generic_v1":
            raise ValueError(f"Unsupported TSIL descriptor schema: {schema}")
        result = list(names) if names else [f"feature_{i}" for i in range(width)]
        if len(result) != width or len(set(result)) != width:
            raise ValueError("TSIL generic descriptor names do not match descriptor width")
        return result

    @property
    def D_table(self):
        return self.targets

    def _refresh_support(self):
        full = self.descriptors[self.family_ids == 0]
        if full.shape[0]:
            self.descriptor_min = full.amin(0)
            self.descriptor_max = full.amax(0)
        else:
            self.descriptor_min = torch.zeros(self.descriptors.shape[1], device=self.device)
            self.descriptor_max = self.descriptor_min.clone()

    @torch.no_grad()
    def observe_success(self, config_ids, times):
        ids = torch.as_tensor(config_ids, device=self.device, dtype=torch.long).reshape(-1)
        values = torch.as_tensor(times, device=self.device, dtype=torch.float32).reshape(-1)
        if ids.numel() != values.numel():
            raise ValueError("TSIL success IDs and times differ in length")
        if not ids.numel():
            return
        if bool(((ids < 0) | (ids >= self.num_configs)).any()):
            raise ValueError("TSIL success configuration ID out of range")
        if not bool((torch.isfinite(values) & (values > 0)).all()):
            raise ValueError("TSIL success times must be finite and positive")
        # Repeated configuration IDs in one vector must reduce, not overwrite.
        self.pending_min.scatter_reduce_(0, ids, values, reduce="amin", include_self=True)

    @torch.no_grad()
    def commit_targets(self):
        improved = self.pending_min < self.targets
        self.seen_success |= torch.isfinite(self.pending_min)
        self.targets.copy_(torch.minimum(self.targets, self.pending_min))
        self.pending_min.fill_(float("inf"))
        return int(improved.sum().item())

    @torch.no_grad()
    def lookup_targets(self, descriptors, neighbors=5, return_status=False):
        queries = torch.as_tensor(descriptors, dtype=torch.float32, device=self.device)
        if queries.ndim != 2 or queries.shape[1] != self.descriptors.shape[1]:
            raise ValueError("TSIL evaluation descriptor schema does not match checkpoint")
        if int(neighbors) < 1:
            raise ValueError("TSIL lookup requires at least one neighbor")
        result = torch.full((queries.shape[0],), self.t_ref, device=self.device)
        finite = torch.isfinite(queries).all(-1)
        status = torch.full((queries.shape[0],), LOOKUP_NO_SUCCESS,
                            dtype=torch.long, device=self.device)
        status[~finite] = LOOKUP_INVALID
        def finish():
            return (result, status) if return_status else result
        available = self.seen_success & (self.family_ids == 0)
        if not bool(available.any()) or self.descriptors.shape[1] == 0:
            return finish()
        ranges = self.descriptor_max - self.descriptor_min
        varying = ranges > self.constant_tolerance
        tolerance = torch.where(varying, ranges * self.support_margin,
                                torch.full_like(ranges, self.constant_tolerance))
        supported = (finite
                     & (queries >= self.descriptor_min - tolerance).all(-1)
                     & (queries <= self.descriptor_max + tolerance).all(-1))
        status[finite] = LOOKUP_OUT_OF_SUPPORT
        if not bool(supported.any()):
            return finish()
        reference = self.descriptors[available]
        q = queries[supported]
        if bool(varying.any()):
            distance = torch.cdist(q[:, varying] / ranges[varying],
                                   reference[:, varying] / ranges[varying])
        else:
            distance = torch.zeros((len(q), len(reference)), device=self.device)
        closest = distance.topk(min(int(neighbors), len(reference)), largest=False).indices
        result[supported] = self.targets[available][closest].mean(-1)
        status[supported] = LOOKUP_NEAREST
        return finish()

    lookup_deadlines = lookup_targets

    def state_dict(self):
        return {"version": self.STATE_VERSION, "t_ref": self.t_ref,
                "bank_fingerprint": self.bank_fingerprint,
                "targets": self.targets.detach().cpu().clone(),
                "seen_success": self.seen_success.detach().cpu().clone(),
                "pending_min": self.pending_min.detach().cpu().clone(),
                "family_ids": self.family_ids.detach().cpu().clone(),
                "descriptors": self.descriptors.detach().cpu().clone(),
                "descriptor_schema": self.descriptor_schema,
                "descriptor_names": list(self.descriptor_names),
                "descriptor_min": self.descriptor_min.detach().cpu().clone(),
                "descriptor_max": self.descriptor_max.detach().cpu().clone(),
                "support_margin": self.support_margin,
                "constant_tolerance": self.constant_tolerance}

    def load_state_dict(self, state):
        required = ("version", "t_ref", "bank_fingerprint", "targets", "seen_success",
                    "family_ids", "descriptors", "descriptor_min", "descriptor_max")
        if any(name not in state for name in required) or state["version"] not in (1, self.STATE_VERSION):
            raise ValueError("Missing or incompatible TSIL checkpoint state")
        if state["version"] == self.STATE_VERSION and (
                "descriptor_schema" not in state or "descriptor_names" not in state):
            raise ValueError("TSIL checkpoint is missing its descriptor schema")
        if not math.isclose(float(state["t_ref"]), self.t_ref, rel_tol=1e-6):
            raise ValueError("TSIL checkpoint time reference does not match environment")
        if self.num_configs and state["bank_fingerprint"] != self.bank_fingerprint:
            raise ValueError("TSIL checkpoint bank fingerprint does not match training bank")
        targets = torch.as_tensor(state["targets"], dtype=torch.float32, device=self.device)
        if targets.ndim != 1 or not bool((torch.isfinite(targets) & (targets > 0) & (targets <= self.t_ref)).all()):
            raise ValueError("TSIL checkpoint contains invalid targets")
        if self.num_configs and self.num_configs != targets.numel():
            raise ValueError("TSIL checkpoint configuration count does not match training bank")
        family_ids = torch.as_tensor(state["family_ids"], dtype=torch.long, device=self.device)
        seen_success = torch.as_tensor(state["seen_success"], dtype=torch.bool, device=self.device)
        descriptors = torch.as_tensor(state["descriptors"], dtype=torch.float32, device=self.device)
        if family_ids.shape != targets.shape or seen_success.shape != targets.shape:
            raise ValueError("TSIL checkpoint configuration metadata shape mismatch")
        if descriptors.ndim != 2 or descriptors.shape[0] != targets.numel():
            raise ValueError("TSIL checkpoint descriptor shape mismatch")
        pending_min = torch.as_tensor(state.get("pending_min", torch.full_like(targets, float("inf"))),
                                      dtype=torch.float32, device=self.device)
        if (pending_min.shape != targets.shape
                or not bool(((pending_min > 0) & ~torch.isnan(pending_min)).all())):
            raise ValueError("TSIL checkpoint pending target state is invalid")
        if not bool(torch.isfinite(descriptors).all()):
            raise ValueError("TSIL checkpoint descriptors must be finite")

        # Validate the original checkpoint scales before any schema migration.
        full = descriptors[family_ids == 0]
        lower = full.amin(0) if len(full) else torch.zeros(descriptors.shape[1], device=self.device)
        upper = full.amax(0) if len(full) else lower.clone()
        for name, actual in (("descriptor_min", lower), ("descriptor_max", upper)):
            expected = torch.as_tensor(state[name], dtype=torch.float32, device=self.device)
            if expected.shape != actual.shape or not torch.allclose(expected, actual):
                raise ValueError("TSIL checkpoint descriptor scale is inconsistent")

        names = state.get("descriptor_names")
        schema = state.get("descriptor_schema", "generic_v1")
        if state["version"] == 1 and descriptors.shape[1] == 32 and not names:
            # The original DQ checkpoint stored xyz, two rotation columns,
            # gravity xy, table/base height, and 19 joints, with no column names.
            # All new coordinates are retained exactly in that representation.
            descriptors = torch.cat((descriptors[:, :2], _planar_heading(descriptors[:, 3:5]),
                                     descriptors[:, 11:12]), dim=-1)
            schema, names = GEOMETRY_DESCRIPTOR_SCHEMA, GEOMETRY_DESCRIPTOR_NAMES
        candidate = TemporalState(
            targets.numel(), self.t_ref, self.device, state["bank_fingerprint"],
            family_ids, descriptors, names,
            state.get("support_margin", 0.05), state.get("constant_tolerance", 1e-4),
            descriptor_schema=schema)
        if self.num_configs and (candidate.descriptor_schema != self.descriptor_schema
                                 or candidate.descriptor_names != self.descriptor_names):
            raise ValueError("TSIL checkpoint descriptor schema does not match training bank")
        candidate.targets.copy_(targets)
        candidate.seen_success.copy_(seen_success)
        candidate.pending_min.copy_(pending_min)
        # Only replace state after all integrity/schema checks succeed.
        self.__dict__.update(candidate.__dict__)


@dataclass
class Trajectory:
    uid: tuple
    config_id: int
    observations: torch.Tensor
    actions: torch.Tensor
    reward_fixed: torch.Tensor
    reward_command_raw: torch.Tensor
    collected_rewards: torch.Tensor
    success: bool
    elapsed: float
    return_score: float
    prefix: object = None

    def __len__(self):
        return self.actions.shape[0]

    def full_observations(self):
        if self.prefix is None:
            return self.observations.clone()
        return torch.cat((self.prefix.expand(len(self), -1), self.observations), -1)


def discounted_returns(rewards, gamma):
    """Upstream TSIL recurrence, with the DQ finite-task zero terminal tail."""
    if rewards.device.type == "cpu":
        # Archives are on CPU. Thousands of tiny Tensor dispatcher calls per
        # update dwarf this recurrence's arithmetic; Python doubles retain the
        # same recurrence with one final cast to the stored reward dtype.
        values = rewards.tolist()
        result = [0.0] * len(values)
        running = 0.0
        for index in range(len(values) - 1, -1, -1):
            running = values[index] + float(gamma) * running
            result[index] = running
        return rewards.new_tensor(result)
    returns = torch.empty_like(rewards)
    running = rewards.new_zeros(())
    for index in range(rewards.numel() - 1, -1, -1):
        running = rewards[index] + float(gamma) * running
        returns[index] = running
    return returns


class ReplaySelection:
    """One immutable sampled transition set, reused by every PPO epoch.

    Only the nonconstant observation suffix is materialized. The shared exact
    static object feature is reconstructed for each accelerator microbatch.
    """

    def __init__(self, observations, actions, returns, efficiency, config_id,
                 success, elapsed, uids, prefix, stats):
        self.observations, self.actions, self.returns = observations, actions, returns
        self.efficiency, self.config_id, self.success = efficiency, config_id, success
        self.elapsed, self.uids, self.prefix, self.stats = elapsed, uids, prefix, stats

    def __len__(self):
        return self.actions.shape[0]

    def get_batch(self, indices, device="cpu"):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        suffix = self.observations[indices].to(device)
        states = (suffix if self.prefix is None else
                  torch.cat((self.prefix.to(device).expand(len(indices), -1), suffix), -1))
        return {"states": states, "actions": self.actions[indices].to(device),
                "returns": self.returns[indices].to(device),
                "efficiency": self.efficiency[indices].to(device),
                "config_id": self.config_id[indices].to(device),
                "success": self.success[indices].to(device), "elapsed": self.elapsed[indices].to(device)}


class TemporalReplay:
    """Complete episode recorder and per-configuration union of two Top-K lists.

    Active episodes use a bounded CPU ring and archives are allocated only for
    admitted episodes. The 1024 static feature is stored once, *only after exact
    equality has been verified*. No EKF channel is reconstructed or quantized.
    """

    def __init__(self, state, num_envs, observation_size, action_size, gamma=0.99,
                 topk=5, time_indices=(-191, -190), max_episode_steps=100,
                 static_prefix_size=1024, success_reward_scale=50.0):
        self.state, self.num_envs = state, int(num_envs)
        self.observation_size, self.action_size = int(observation_size), int(action_size)
        self.gamma, self.topk = float(gamma), int(topk)
        self.time_indices = tuple(int(i) % self.observation_size for i in time_indices)
        self.max_episode_steps = int(max_episode_steps)
        self.static_prefix_size = min(int(static_prefix_size), self.observation_size)
        self.success_reward_scale = float(success_reward_scale)
        if not 0 <= self.gamma <= 1 or self.topk < 1 or self.max_episode_steps < 1:
            raise ValueError("Invalid TSIL replay configuration")
        if not math.isfinite(self.success_reward_scale) or self.success_reward_scale <= 0:
            raise ValueError("TSIL success reward scale must be finite and positive")
        self.fastest, self.high_return = {}, {}
        self._prefix = None
        self._allocated = False
        self._episode_ids = torch.full((self.num_envs,), -1, dtype=torch.long)
        self._config_ids = self._episode_ids.clone()
        self._lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool)
        self._scores = torch.zeros(self.num_envs, dtype=torch.float64)
        self.completed_episodes = 0
        self.aborted_episodes = 0

    def _allocate(self, states):
        p = self.static_prefix_size
        if p and torch.equal(states[:, :p], states[:1, :p].expand(self.num_envs, -1)):
            self._prefix = states[0, :p].detach().to("cpu", torch.float32).clone()
        else:
            self.static_prefix_size = p = 0
        shape = (self.num_envs, self.max_episode_steps)
        self._obs = torch.empty((*shape, self.observation_size - p), dtype=torch.float32)
        self._actions = torch.empty((*shape, self.action_size), dtype=torch.float32)
        self._fixed = torch.empty(shape)
        self._command = torch.empty(shape)
        self._raw = torch.empty(shape)
        self._allocated = True

    @staticmethod
    def _cpu_vector(value, dtype):
        return torch.as_tensor(value).detach().to(device="cpu", dtype=dtype).reshape(-1)

    @torch.no_grad()
    def record(self, states, actions, raw_rewards, terminated, tsil_info):
        if not self._allocated:
            self._allocate(states)
        if self._prefix is not None and not torch.equal(
                states[:, :self.static_prefix_size],
                self._prefix.to(states.device).expand(self.num_envs, -1)):
            raise ValueError("TSIL static feature changed: disable prefix compression for noisy/dynamic features")
        ids = self._cpu_vector(tsil_info["config_id"], torch.long)
        episodes = self._cpu_vector(tsil_info["episode_id"], torch.long)
        done = self._cpu_vector(terminated, torch.bool)
        natural = self._cpu_vector(tsil_info.get("natural_done", terminated), torch.bool)
        success = self._cpu_vector(tsil_info["success"], torch.bool)
        elapsed = self._cpu_vector(tsil_info["elapsed"], torch.float32)
        fixed = self._cpu_vector(tsil_info.get("reward_fixed", tsil_info.get("r_fixed")), torch.float32)
        command = self._cpu_vector(tsil_info.get("reward_command_raw", tsil_info.get("command_raw")), torch.float32)
        raw = self._cpu_vector(raw_rewards, torch.float32)
        starts = episodes != self._episode_ids
        self.aborted_episodes += int((starts & self._valid & (self._lengths > 0)).sum())
        self._lengths[starts] = 0
        self._scores[starts] = 0
        self._episode_ids[starts] = episodes[starts]
        self._config_ids[starts] = ids[starts]
        start_time = states[:, self.time_indices[0]].detach().cpu()
        self._valid[starts] = ((start_time[starts].abs() <= 1e-6)
                               & (ids[starts] >= 0) & (ids[starts] < self.state.num_configs))
        if bool((self._valid & (ids != self._config_ids)).any()):
            raise ValueError("TSIL configuration changed without an episode reset")
        active = self._valid.nonzero(as_tuple=False).flatten()
        lengths = self._lengths[active]
        if bool((lengths >= self.max_episode_steps).any()):
            raise ValueError("TSIL episode exceeded the configured finite task budget")
        # Explicit CPU copies own their storage before env.reset mutates buffers.
        self._obs[active, lengths] = states[active.to(states.device), self.static_prefix_size:].detach().cpu()
        self._actions[active, lengths] = actions[active.to(actions.device)].detach().cpu()
        self._fixed[active, lengths] = fixed[active]
        self._command[active, lengths] = command[active]
        self._raw[active, lengths] = raw[active]
        self._scores[active] += (self.gamma ** lengths.double()) * raw[active].double()
        self._lengths[active] += 1
        ended = done & self._valid
        successful = ended & natural & success
        self.state.observe_success(ids[successful], elapsed[successful])
        for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
            if natural[env_id]:
                self.completed_episodes += 1
                config = int(ids[env_id])
                duration, score = float(elapsed[env_id]), float(self._scores[env_id])
                succeeded = bool(success[env_id])
                if self._should_admit(config, succeeded, duration, score):
                    length = int(self._lengths[env_id])
                    trajectory = Trajectory(
                        (env_id, int(episodes[env_id])), config,
                        self._obs[env_id, :length].clone(), self._actions[env_id, :length].clone(),
                        self._fixed[env_id, :length].clone(), self._command[env_id, :length].clone(),
                        self._raw[env_id, :length].clone(), succeeded, duration, score, self._prefix)
                    self.insert(trajectory)
            else:
                self.aborted_episodes += 1
            self._valid[env_id] = False
            self._lengths[env_id] = 0

    def abort(self, env_ids=None):
        ids = torch.arange(self.num_envs) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long)
        self.aborted_episodes += int((self._valid[ids] & (self._lengths[ids] > 0)).sum())
        self._valid[ids] = False
        self._lengths[ids] = 0

    def _should_admit(self, config, success, elapsed, score):
        high = self.high_return.get(config, [])
        fast = self.fastest.get(config, [])
        return (len(high) < self.topk or score > high[-1].return_score
                or (success and (len(fast) < self.topk or elapsed < fast[-1].elapsed)))

    def insert(self, trajectory):
        """Adapted upstream Top-K ranking; both lists reference one owned object."""
        c = trajectory.config_id
        if c < 0 or c >= self.state.num_configs or not len(trajectory):
            raise ValueError("Invalid archived TSIL episode")
        if not math.isfinite(trajectory.return_score) or not math.isfinite(trajectory.elapsed) or trajectory.elapsed <= 0:
            raise ValueError("Invalid archived TSIL episode score or duration")
        if any(t.uid == trajectory.uid for t in self.high_return.get(c, []) + self.fastest.get(c, [])):
            return False
        high = self.high_return.get(c, []) + [trajectory]
        self.high_return[c] = sorted(high, key=lambda t: (-t.return_score, t.uid))[:self.topk]
        if trajectory.success:
            fast = self.fastest.get(c, []) + [trajectory]
            self.fastest[c] = sorted(fast, key=lambda t: (t.elapsed, t.uid))[:self.topk]
        return any(t.uid == trajectory.uid for t in self.high_return[c] + self.fastest.get(c, []))

    def selected(self, config_id):
        """Upstream fastest-first then return-ranked, deduplicated same-config fill."""
        selected, seen = [], set()
        for trajectory in self.fastest.get(config_id, []) + self.high_return.get(config_id, []):
            if trajectory.uid in seen:
                continue
            selected.append(trajectory)
            seen.add(trajectory.uid)
            if len(selected) == self.topk:
                break
        return selected

    def iter_selected(self):
        for config_id in sorted(set(self.fastest) | set(self.high_return)):
            yield from self.selected(config_id)

    @property
    def stats(self):
        archived = {t.uid for values in self.fastest.values() for t in values}
        archived.update(t.uid for values in self.high_return.values() for t in values)
        return {"archive_configurations": len(set(self.fastest) | set(self.high_return)),
                "archive_trajectories": len(archived),
                "archive_success_configurations": len(self.fastest),
                "completed_episodes": self.completed_episodes, "aborted_episodes": self.aborted_episodes}

    def relabel(self, trajectory, target, reward_scale=1.0, command_gate=1.0):
        if not math.isfinite(float(reward_scale)) or reward_scale <= 0:
            raise ValueError("TSIL reward scale must be finite and positive")
        states = trajectory.full_observations()
        states[:, self.time_indices[1]] = float(target) / self.state.t_ref
        rewards = trajectory.reward_fixed + float(command_gate) * trajectory.reward_command_raw
        rewards = rewards.clone()
        efficiency = 1.0 + (min(float(target) / trajectory.elapsed, 1.0) if trajectory.success else 0.0)
        if trajectory.success:
            rewards[-1] += self.success_reward_scale * efficiency
        returns = discounted_returns(rewards / float(reward_scale), self.gamma)
        return states, returns, efficiency

    @torch.no_grad()
    def prepare(self, screen_gap_fn, reward_scale=1.0, command_gate=1.0,
                transitions_per_trajectory=32, screen_batch_size=4096):
        """Paper sampler: every selected trajectory, once, positive-gap weighted.

        ``screen_gap_fn(raw_states, scaled_returns[:, None])`` returns gaps in
        the *same* normalized value space as the eventual loss. It must keep
        observation/value scalers read-only. Callbacks may move data to CUDA.
        """
        if transitions_per_trajectory < 1 or screen_batch_size < 1:
            raise ValueError("TSIL sampling and screening batch sizes must be positive")
        targets = self.state.targets.detach().cpu().clone()
        rows = []
        stats = dict(self.stats, replay_trajectories=0, replay_dataset_steps=0,
                     replay_positive_available=0, replay_steps=0)
        pending, pending_size = [], 0

        def screen(items):
            if not items:
                return
            observations = torch.cat([item[1] for item in items])
            returns = torch.cat([item[2] for item in items]).reshape(-1, 1)
            # A trajectory remains a sampling unit even if its critic evaluation
            # requires several accelerator microbatches. Do not truncate it.
            gap_chunks = []
            for begin in range(0, len(observations), screen_batch_size):
                stop = begin + screen_batch_size
                gap_chunks.append(screen_gap_fn(observations[begin:stop], returns[begin:stop])
                                  .detach().float().cpu().reshape(-1))
            gaps = torch.cat(gap_chunks)
            if gaps.numel() != len(observations) or not bool(torch.isfinite(gaps).all()):
                raise ValueError("TSIL screening returned invalid value gaps")
            offset = 0
            for trajectory, obs, ret, weight in items:
                gap = gaps[offset:offset + len(trajectory)].clamp_min(0)
                offset += len(trajectory)
                positive = (gap > 0).nonzero(as_tuple=False).flatten()
                stats["replay_positive_available"] += len(positive)
                if not len(positive):
                    continue
                count = min(int(transitions_per_trajectory), len(positive))
                choices = positive[torch.multinomial(gap[positive], count, replacement=False)]
                rows.append((trajectory, obs[choices, self.static_prefix_size:].clone(),
                             ret[choices].clone(), choices, weight))

        for trajectory in self.iter_selected():
            if pending_size + len(trajectory) > screen_batch_size:
                screen(pending)
                pending, pending_size = [], 0
            obs, ret, efficiency = self.relabel(trajectory, targets[trajectory.config_id], reward_scale, command_gate)
            pending.append((trajectory, obs, ret, efficiency))
            pending_size += len(trajectory)
            stats["replay_trajectories"] += 1
            stats["replay_dataset_steps"] += len(trajectory)
        screen(pending)
        count = sum(len(row[3]) for row in rows)
        suffix = torch.empty((count, self.observation_size - self.static_prefix_size))
        actions = torch.empty((count, self.action_size))
        returns, weights = torch.empty((count, 1)), torch.empty((count, 1))
        configs = torch.empty(count, dtype=torch.long)
        success = torch.empty(count, dtype=torch.bool)
        elapsed = torch.empty(count)
        uids = torch.empty((count, 3), dtype=torch.long)
        start = 0
        for trajectory, obs, ret, choices, weight in rows:
            end = start + len(choices)
            suffix[start:end] = obs
            actions[start:end] = trajectory.actions[choices]
            returns[start:end, 0] = ret
            weights[start:end] = weight
            configs[start:end] = trajectory.config_id
            success[start:end] = trajectory.success
            elapsed[start:end] = trajectory.elapsed
            uids[start:end, 0] = trajectory.uid[0]
            uids[start:end, 1] = trajectory.uid[1]
            uids[start:end, 2] = choices
            start = end
        stats["replay_steps"] = count
        stats["replay_success_fraction"] = float(success.float().mean()) if count else 0.0
        return ReplaySelection(suffix, actions, returns, weights, configs, success, elapsed,
                               uids, self._prefix, stats)
