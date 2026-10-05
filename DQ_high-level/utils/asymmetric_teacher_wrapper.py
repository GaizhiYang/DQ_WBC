"""Fresh M0/M1/M2 observations and training-only perception rewards."""
from copy import deepcopy
import gym
import numpy as np
import torch
from .teacher_vision_wrapper import TeacherVisionWrapper
from .asymmetric_camera import pose_matrix, sensor_packet, visible_surface_points


class AsymmetricTeacherWrapper(TeacherVisionWrapper):
    """Pack privileged1276, images62208, optional task5, optional belief16.

    Filtering runs once per observation, outside PPO minibatch computation.
    The finite task deadline retains the original terminal/no-bootstrap rule.
    """
    def __init__(self, env, settings=None):
        super().__init__(env)
        self.settings = deepcopy(settings or {"mode": "m0"})
        mode = self.settings.get("mode", "m2")
        if mode not in ("m0", "m1", "m2"):
            raise ValueError("mode must be m0, m1 or m2")
        self.m1, self.m2 = mode != "m0", mode == "m2"
        size = 63484 + 5 * self.m1 + 16 * self.m2
        self._teacher_vision_observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self._force_reset = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_perception = None
        self.last_perception_metrics = {}
        if self.m2:
            from .asymmetric_perception import AsymmetricPerception, PerceptionConfig
            self.perception = AsymmetricPerception(
                self.num_envs, 54, 96, self.device,
                PerceptionConfig(**self.settings.get("perception", {})),
                seed=int(self.settings.get("seed", 43)))
            self._history = torch.zeros(self.num_envs, 3, 4, 54, 96, device=self.device)
            self._history_initialized = torch.zeros_like(self._force_reset)
            self._belief = torch.zeros(self.num_envs, 16, device=self.device)
            self._now = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
            self._previous_camera = torch.eye(4, device=self.device).repeat(self.num_envs, 2, 1, 1)
            self._previous_stamp = torch.full((self.num_envs, 2), float("nan"), device=self.device, dtype=torch.float64)
            self._reference_local = torch.zeros(self.num_envs, 3, device=self.device)
            self._reference_valid = torch.zeros_like(self._force_reset)

    def get_runtime_rng_state(self):
        return self.perception.get_rng_state() if self.m2 else None

    def set_runtime_rng_state(self, state):
        if self.m2:
            self.perception.set_rng_state(state)

    def seed_runtime(self, seed):
        if self.m2:
            self.perception.generator.manual_seed(int(seed))

    def reset_runtime(self, env_ids=None):
        """Reset only requested episodes, never advance other camera histories."""
        ids = slice(None) if env_ids is None else env_ids
        self._force_reset[ids] = True
        if self.m2:
            self.perception.reset(env_ids)
            self._history[ids] = 0
            self._history_initialized[ids] = False
            self._belief[ids] = 0
            self._now[ids] = 0
            self._previous_stamp[ids] = float("nan")
            self._reference_valid[ids] = False
            self._reference_local[ids] = 0

    def _task_state(self):
        env = self._env
        remaining = (1 - env.progress_buf.float() / float(env.max_episode_length)).clamp(0, 1)
        distance = torch.where(env.closest_dist >= 0, env.closest_dist, env.curr_dist).clamp(0, 10) / 2
        height = torch.where(env.highest_object >= 0, env.highest_object, env.curr_height).clamp(0, 2)
        holding = (env.pick_counter.float() / max(1, int(env.hold_steps))).clamp(0, 1)
        return torch.stack((remaining, distance, height, holding, env.lifted_object.float()), dim=-1)

    @torch.no_grad()
    def _update_perception(self, ids):
        if not len(ids):
            return
        packet = sensor_packet(self._env)
        now = self._now.clone()
        now[ids] = packet["now"][ids].to(now)
        dt = packet["timestamps"] - self._previous_stamp
        rotation = packet["T_world_camera"][..., :3, :3] @ self._previous_camera[..., :3, :3].transpose(-1, -2)
        angle = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1).acos()
        speed = torch.where(torch.isfinite(dt) & (dt > 1e-9), angle / dt.clamp_min(1e-9), torch.zeros_like(angle))
        angular_velocity = torch.zeros(self.num_envs, 2, 3, device=self.device)
        # The corruption model uses angular-speed magnitude, not its axis.
        angular_velocity[..., 2] = speed.float()
        corrupt = self.settings.get("perception_corruption", True)
        if getattr(self._env, "eval", False):
            corrupt = self.settings.get("perception_corruption_eval", corrupt)
        result = self.perception.step(
            packet["images"], packet["T_world_camera"], packet["intrinsics"],
            packet["timestamps"], packet["T_world_base"], now=now,
            camera_angular_velocity=angular_velocity, corrupt=bool(corrupt))
        frames = result["images"]
        current = torch.stack((frames[:, 0, 0], frames[:, 1, 0], frames[:, 0, 1] / 3., frames[:, 1, 1] / 3.), dim=1)
        fresh = ids[~self._history_initialized[ids]]
        continuing = ids[self._history_initialized[ids]]
        self._history[fresh] = current[fresh, None].expand(-1, 3, -1, -1, -1)
        if len(continuing):
            previous = self._history[continuing].clone()
            self._history[continuing] = torch.cat((previous[:, 1:], current[continuing, None]), dim=1)
        self._history_initialized[ids] = True
        self._belief[ids] = result["belief"][ids]
        self._now[ids] = now[ids]
        self._previous_camera[ids] = packet["T_world_camera"][ids]
        self._previous_stamp[ids] = packet["timestamps"][ids].to(self._previous_stamp)
        self.last_perception = result
        # GT transforms this clean surface reference only in the reward path.
        if hasattr(self._env, "_m2_reward_capture_object_states"):
            points, visible, counts = visible_surface_points(
                packet["images"], packet["T_world_camera"], packet["intrinsics"], self.perception.config)
            weights = counts.float() * visible
            point = (points * weights[..., None]).sum(1) / weights.sum(1).clamp_min(1)[:, None]
            capture = self._env._m2_reward_capture_object_states
            capture_pose = pose_matrix(capture[:, :3], capture[:, 3:7])
            local = (capture_pose[:, :3, :3].transpose(-1, -2) @ (point - capture[:, :3])[..., None]).squeeze(-1)
            good = ids[visible[ids].any(-1)]
            self._reference_local[good] = local[good]
            self._reference_valid[good] = True

    def _pack(self, observations, ids):
        packed = TeacherVisionWrapper._pack_observation(self, observations)
        if self.m2:
            self._update_perception(ids)
            packed = torch.cat((packed[:, :1276], self._history.flatten(1).to(packed)), dim=-1)
        additions = [packed]
        if self.m1:
            additions.append(self._task_state().to(packed))
        if self.m2:
            additions.append(self._belief.to(packed))
        self._force_reset[ids] = False
        return torch.cat(additions, dim=-1)

    @torch.no_grad()
    def _perception_reward(self):
        if not self.m2 or self.last_perception is None:
            return torch.zeros(self.num_envs, device=self.device)
        options = self.settings.get("perception_reward", {})
        weight = float(options.get("weight", 0.02))
        if not hasattr(self._env, "_cube_root_states") or not hasattr(self._env, "_m2_reward_capture_object_states"):
            raise RuntimeError("Perception reward requires training-only object pose/velocity and capture reference")
        gt = self._env._cube_root_states
        pose = pose_matrix(gt[:, :3], gt[:, 3:7])
        offset = (pose[:, :3, :3] @ self._reference_local[..., None]).squeeze(-1)
        reference = gt[:, :3] + offset
        velocity = gt[:, 7:10] + torch.cross(gt[:, 10:13], offset, dim=-1)
        horizon = float(options.get("prediction_horizon_s", 0.15))
        target = reference + horizon * velocity
        predicted = self.last_perception["world_position"] + horizon * self.last_perception["world_velocity"]
        error = (predicted - target).norm(dim=-1)
        valid = self._reference_valid & self.last_perception["initialized"] & torch.isfinite(error)
        sigma = float(options.get("sigma_m", 0.1))
        bonus = torch.where(valid, weight * torch.exp(-0.5 * (error / sigma).square()), torch.zeros_like(error))
        self.last_perception_metrics = {
            "prediction_error_m": float(error[valid].mean()) if valid.any() else None,
            "initialized_fraction": float(self.last_perception["initialized"].float().mean()),
            "visible_fraction": float(self.last_perception["valid"].float().mean()),
            "camera_age_s": float(self._belief[:, 11:13].mean()),
            "prediction_age_s": float(self._belief[:, 14].mean()),
            "rejected_stale_measurements": float(self.last_perception["rejected_stale_measurements"].float().mean()),
            "reward_mean": float(bonus.mean())}
        return bonus

    def step(self, actions):
        observations, reward, terminated, info = self._env.step(actions)
        ids = torch.arange(self.num_envs, device=self.device)
        packed = self._pack(observations, ids)
        reward = reward.clone().reshape(-1, 1)
        if self.m2:
            bonus = self._perception_reward()
            if not getattr(self._env, "eval", False):
                reward = reward + bonus.to(reward).reshape(-1, 1)
            info = dict(info)
            info["perception"] = dict(self.last_perception_metrics)
        terminal = terminated.reshape(-1, 1)
        return packed, reward, terminal, torch.zeros_like(terminal), info

    def reset(self):
        pending = getattr(self._env, "reset_buf", torch.zeros_like(self._force_reset)).bool()
        ids = (pending | self._force_reset).nonzero(as_tuple=False).flatten()
        if len(ids):
            self.reset_runtime(ids)
        observations = self._env.reset()
        return self._pack(observations, ids), {}
