"""Batched, causal SE(3) EKF forecasts for teacher observations.

The equations match modules/kalman_filter.py (the Paparazzo reference).
Each forecast refits a fresh filter on a rolling window of measured world poses,
matching the offline benchmark. No simulator velocities or future states enter.
"""

import math
from typing import Tuple

import torch
from torch import Tensor


@torch.jit.script
def skew(v: Tensor) -> Tensor:
    out = torch.zeros((v.size(0), 3, 3), dtype=v.dtype, device=v.device)
    out[:, 0, 1], out[:, 0, 2] = -v[:, 2], v[:, 1]
    out[:, 1, 0], out[:, 1, 2] = v[:, 2], -v[:, 0]
    out[:, 2, 0], out[:, 2, 1] = -v[:, 1], v[:, 0]
    return out


@torch.jit.script
def quaternion_to_matrix(q: Tensor) -> Tensor:
    """Unit quaternions in Isaac Gym's xyzw order, [N, 4]."""
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
        2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
        2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y),
    ), dim=-1).reshape(-1, 3, 3)


@torch.jit.script
def matrix_to_quaternion(r: Tensor) -> Tensor:
    """Stable near both zero and pi rotation; returns normalized xyzw."""
    a, b, c = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    squared = torch.stack((1+a+b+c, 1+a-b-c, 1-a+b-c, 1-a-b+c), dim=-1).clamp_min(0.)
    roots = torch.sqrt(squared)
    candidates = torch.stack((
        torch.stack((squared[:, 0], r[:, 2, 1]-r[:, 1, 2], r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1]), dim=-1),
        torch.stack((r[:, 2, 1]-r[:, 1, 2], squared[:, 1], r[:, 0, 1]+r[:, 1, 0], r[:, 0, 2]+r[:, 2, 0]), dim=-1),
        torch.stack((r[:, 0, 2]-r[:, 2, 0], r[:, 0, 1]+r[:, 1, 0], squared[:, 2], r[:, 1, 2]+r[:, 2, 1]), dim=-1),
        torch.stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0], r[:, 1, 2]+r[:, 2, 1], squared[:, 3]), dim=-1),
    ), dim=1) / (2 * roots.clamp_min(1e-8).unsqueeze(-1))
    index = roots.argmax(dim=-1).view(-1, 1, 1).expand(-1, 1, 4)
    q = candidates.gather(1, index).squeeze(1)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[:, :1] < 0, -q, q)
    return torch.cat((q[:, 1:], q[:, :1]), dim=-1)


@torch.jit.script
def rotation_and_jacobian(w: Tensor) -> Tuple[Tensor, Tensor]:
    theta2 = (w*w).sum(-1).view(-1, 1, 1)
    theta = torch.sqrt(theta2.clamp_min(1e-16))
    small = theta2 < 1e-6
    # Taylor expansions avoid catastrophic cancellation for float32 rotations.
    a = torch.where(small, 1-theta2/6+theta2*theta2/120, torch.sin(theta)/theta)
    b = torch.where(small, .5-theta2/24+theta2*theta2/720, (1-torch.cos(theta))/theta2.clamp_min(1e-16))
    c = torch.where(small, 1/6-theta2/120+theta2*theta2/5040, (theta-torch.sin(theta))/(theta2*theta).clamp_min(1e-24))
    hat = skew(w)
    hat2 = hat @ hat
    eye = torch.eye(3, dtype=w.dtype, device=w.device).unsqueeze(0)
    return eye + a*hat + b*hat2, eye + b*hat + c*hat2


@torch.jit.script
def se3_exp_batch(xi: Tensor) -> Tensor:
    rotation, jacobian = rotation_and_jacobian(xi[:, 3:])
    out = torch.zeros((xi.size(0), 4, 4), dtype=xi.dtype, device=xi.device)
    out[:, :3, :3] = rotation
    out[:, :3, 3] = (jacobian @ xi[:, :3].unsqueeze(-1)).squeeze(-1)
    out[:, 3, 3] = 1
    return out


@torch.jit.script
def se3_log_batch(transform: Tensor) -> Tensor:
    q = matrix_to_quaternion(transform[:, :3, :3])
    sine = torch.linalg.vector_norm(q[:, :3], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sine, q[:, 3:])
    scale = torch.where(sine < 1e-8, torch.full_like(sine, 2.), angle/sine.clamp_min(1e-12))
    w = q[:, :3] * scale
    _, jacobian = rotation_and_jacobian(w)
    v, _ = torch.linalg.solve_ex(jacobian, transform[:, :3, 3:4])
    return torch.cat((v.squeeze(-1), w), dim=-1)


@torch.jit.script
def adjoint_batch(transform: Tensor) -> Tensor:
    out = torch.zeros((transform.size(0), 6, 6), dtype=transform.dtype, device=transform.device)
    out[:, :3, :3] = transform[:, :3, :3]
    out[:, 3:, 3:] = transform[:, :3, :3]
    out[:, :3, 3:] = skew(transform[:, :3, 3]) @ transform[:, :3, :3]
    return out


@torch.jit.script
def poses_to_transforms(poses: Tensor) -> Tensor:
    out = torch.zeros((poses.size(0), 4, 4), dtype=poses.dtype, device=poses.device)
    out[:, :3, :3] = quaternion_to_matrix(poses[:, 3:7])
    out[:, :3, 3] = poses[:, :3]
    out[:, 3, 3] = 1
    return out


class RollingSE3Predictor:
    """Independent history per environment; outputs [N, horizons, 7] world poses."""

    def __init__(self, num_envs, device, history=5, horizons=(0.30, 0.60),
                 q_scale=1.0, r_scale=0.01, dtype=torch.float32):
        if history < 2 or int(history) != history:
            raise ValueError("EKF history must be an integer >= 2")
        if not horizons or any(not math.isfinite(t) or t <= 0 for t in horizons):
            raise ValueError("Forecast horizons must be positive, finite seconds")
        if any(a >= b for a, b in zip(horizons, horizons[1:])):
            raise ValueError("Forecast horizons must be strictly increasing")
        if any(not math.isfinite(s) or s <= 0 for s in (q_scale, r_scale)):
            raise ValueError("EKF Q/R scales must be positive and finite")
        self.history = int(history)
        self.horizons = tuple(horizons)
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.dtype = dtype
        self.poses = torch.zeros((num_envs, history, 7), dtype=dtype, device=device)
        self.poses[..., 6] = 1
        # Double timestamps retain sub-second intervals during long training runs.
        self.times = torch.full((num_envs, history), -float("inf"), dtype=torch.float64, device=device)
        self.count = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.env_ids = torch.arange(num_envs, device=device)
        self.future_poses = self.poses[:, -1:, :].repeat(1, len(horizons), 1)
        self.valid = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.eye12 = torch.eye(12, dtype=dtype, device=device)
        self.q = torch.diag(torch.tensor([1e-4]*9 + [5e-4]*3, dtype=dtype, device=device)) * q_scale
        self.r = torch.eye(6, dtype=dtype, device=device) * (1e-3 * r_scale)
        self.horizon_tensor = torch.tensor(horizons, dtype=dtype, device=device)

    @torch.no_grad()
    def reset(self, env_ids):
        """Call before the first observation of the new episode."""
        self.count[env_ids] = 0
        self.times[env_ids] = -float("inf")
        self.valid[env_ids] = False
        self.poses[env_ids] = 0
        self.poses[env_ids, :, 6] = 1
        self.future_poses[env_ids] = 0
        self.future_poses[env_ids, :, 6] = 1

    @torch.no_grad()
    def push(self, world_poses, timestamp, env_ids=None):
        """Record once per physical sample. Repeated timestamps are ignored.

        world_poses corresponds to env_ids (or all environments when omitted).
        Invalid measurements invalidate only the affected environment's history.
        """
        ids = self.env_ids if env_ids is None else env_ids
        poses = world_poses.detach().to(device=self.device, dtype=self.dtype)
        now = torch.as_tensor(timestamp, dtype=torch.float64, device=self.device).expand(ids.numel())
        norm = torch.linalg.vector_norm(poses[:, 3:7], dim=-1)
        good = torch.isfinite(poses).all(-1) & torch.isfinite(now) & (norm > 1e-8)
        bad_ids = ids[~good]
        self.reset(bad_ids)
        accept = good & (now > self.times[ids, -1])
        ids, poses, now = ids[accept], poses[accept].clone(), now[accept]
        if ids.numel() == 0:
            return
        poses[:, 3:7] /= torch.linalg.vector_norm(poses[:, 3:7], dim=-1, keepdim=True)
        first = self.count[ids] == 0
        previous_poses = torch.roll(self.poses[ids], -1, dims=1)
        previous_times = torch.roll(self.times[ids], -1, dims=1)
        self.poses[ids] = torch.where(first[:, None, None], poses[:, None, :], previous_poses)
        self.times[ids] = torch.where(first[:, None], now[:, None], previous_times)
        self.poses[ids, -1] = poses
        self.times[ids, -1] = now
        self.count[ids] = (self.count[ids] + 1).clamp_max(self.history)
        self.future_poses[ids] = poses[:, None, :]
        self.valid[ids] = False

    @torch.no_grad()
    def predict(self):
        """Refit on the window, then extrapolate without measurement updates."""
        ready = self.count >= self.history
        ids = self.env_ids[ready]
        if ids.numel() == 0:
            return self.future_poses, self.valid
        poses = self.poses[ids].clone()
        # Translation invariance permits a local origin for numerical accuracy
        # in large grids of Isaac Gym environments. Return world coordinates.
        origin = poses[:, :1, :3].clone()
        poses[..., :3] -= origin
        measured = poses_to_transforms(poses.reshape(-1, 7)).reshape(-1, self.history, 4, 4)
        transform = measured[:, 0].clone()
        twist = torch.zeros((ids.numel(), 6), device=self.device, dtype=self.dtype)
        covariance = self.eye12.unsqueeze(0).repeat(ids.numel(), 1, 1)
        times = self.times[ids]
        for i in range(1, self.history):
            dt = (times[:, i] - times[:, i-1]).to(self.dtype)
            increment = se3_exp_batch(twist * dt[:, None])
            transform = transform @ increment
            transition = self.eye12.unsqueeze(0).repeat(ids.numel(), 1, 1)
            transition[:, :6, :6] = adjoint_batch(increment)
            transition[:, :6, 6:] = torch.eye(6, dtype=self.dtype, device=self.device)[None] * dt[:, None, None]
            covariance = transition @ covariance @ transition.transpose(-1, -2) + self.q
            # Inverse of a rigid transform, without a general 4x4 inversion.
            inverse = torch.zeros_like(transform)
            inverse[:, :3, :3] = transform[:, :3, :3].transpose(-1, -2)
            inverse[:, :3, 3:4] = -inverse[:, :3, :3] @ transform[:, :3, 3:4]
            inverse[:, 3, 3] = 1
            innovation = se3_log_batch(inverse @ measured[:, i])
            s = covariance[:, :6, :6] + self.r
            gain_t, _ = torch.linalg.solve_ex(s.transpose(-1, -2), covariance[:, :, :6].transpose(-1, -2))
            gain = gain_t.transpose(-1, -2)
            dx = (gain @ innovation.unsqueeze(-1)).squeeze(-1)
            covariance = covariance - gain @ covariance[:, :6, :]
            transform = transform @ se3_exp_batch(dx[:, :6])
            twist = twist + dx[:, 6:]
        future_twist = twist[:, None, :] * self.horizon_tensor[None, :, None]
        future = transform[:, None] @ se3_exp_batch(future_twist.reshape(-1, 6)).reshape(-1, len(self.horizons), 4, 4)
        prediction = torch.cat((future[..., :3, 3] + origin,
                                matrix_to_quaternion(future[..., :3, :3].reshape(-1, 3, 3)).reshape(-1, len(self.horizons), 4)), dim=-1)
        finite = torch.isfinite(prediction).all(-1).all(-1) & torch.isfinite(covariance).all(-1).all(-1)
        self.future_poses[ids] = torch.where(finite[:, None, None], prediction, self.poses[ids, -1:, :])
        self.valid[ids] = finite
        return self.future_poses, self.valid
