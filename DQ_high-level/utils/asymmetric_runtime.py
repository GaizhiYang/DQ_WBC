"""Sensor-only M2 deployment frontend; no simulator or target truth dependency."""
from collections.abc import Mapping

import torch

from .asymmetric_perception import AsymmetricPerception, PerceptionConfig
from .asymmetric_teacher_preprocessor import load_deployment


class M2DeploymentRuntime:
    """Build image history and belief from real sensor packets, then act.

    Call once per control tick with positive optical-Z depth in metres and
    capture-time calibration/poses. Camera order is base, wrist; modality
    order is mask, depth. Timestamps and ``now`` share a monotonic clock.
    Real sensor latency is represented by timestamps: synthetic corruption
    is disabled here. Missing cameras should supply zero images and their
    last capture timestamp. Reset explicitly at episode/target boundaries.
    """

    def __init__(self, source, num_envs=1, device="cpu"):
        payload = source if isinstance(source, Mapping) else torch.load(
            source, map_location="cpu", weights_only=True)
        self.policy = load_deployment(payload, device)
        if not self.policy.m2:
            raise ValueError("M2DeploymentRuntime requires an M2 deployment actor")
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.perception = AsymmetricPerception(
            self.num_envs, 54, 96, self.device,
            PerceptionConfig(**payload["metadata"].get("perception_config", {})))
        self.history = torch.zeros(self.num_envs, 3, 4, 54, 96, device=self.device)
        self.history_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_now = torch.full((self.num_envs,), -float("inf"), dtype=torch.float64, device=self.device)
        self.last_perception = None

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.perception.reset(env_ids)
        self.history[ids] = 0
        self.history_initialized[ids] = False
        self.last_now[ids] = -float("inf")

    @torch.no_grad()
    def step(self, images, T_world_camera, intrinsics, timestamps,
             T_world_base, now, raw_proprio):
        """Return deterministic [batch,9] means; actuator limits remain external."""
        now = torch.as_tensor(now, device=self.device, dtype=torch.float64)
        if now.ndim == 0:
            now = now.expand(self.num_envs)
        if now.shape != (self.num_envs,):
            raise ValueError("now must have one timestamp per environment")
        if (now < self.last_now).any():
            raise ValueError("Clock moved backwards; reset the affected environment first")
        result = self.perception.step(
            images, T_world_camera, intrinsics, timestamps, T_world_base,
            now=now, corrupt=False)
        frames = result["images"]
        current = torch.stack((frames[:, 0, 0], frames[:, 1, 0],
                               frames[:, 0, 1] / 3., frames[:, 1, 1] / 3.), dim=1)
        changed = now > self.last_now
        fresh = changed & ~self.history_initialized
        continuing = changed & self.history_initialized
        self.history[fresh] = current[fresh, None].expand(-1, 3, -1, -1, -1)
        previous = self.history[continuing].clone()
        self.history[continuing] = torch.cat((previous[:, 1:], current[continuing, None]), dim=1)
        self.history_initialized[changed] = True
        self.last_now[changed] = now[changed]
        self.last_perception = result
        proprio = torch.as_tensor(raw_proprio, device=self.device, dtype=self.history.dtype)
        return self.policy(self.history.flatten(1), proprio, result["belief"])
