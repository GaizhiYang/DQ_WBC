"""Sensor-only M2 perception: camera corruption and a delayed-measurement KF.

Depth is positive optical-Z in metres. Camera transforms map OpenCV optical
coordinates (+X right, +Y down, +Z forward) into the world. No object pose,
object identity, segmentation truth beyond the supplied sensor mask, or other
privileged simulation state is accepted by this module.

The measurement is the mean of the *visible masked surface*. It is not the
object centre. View-dependent surface bias and shared camera errors are not
eliminated by this constant-velocity, single-reference-point approximation.

A fixed-lag replay filter handles the two synthetic camera delays. Received
measurements are replayed in capture-time order from an anchored posterior;
replaying is a reconstruction, never an extra update of the current posterior.
Measurements older than that finite window are explicitly rejected.
"""

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import torch


BELIEF_CONTRACT = {
    "version": 1,
    "dim": 16,
    "reference": "visible_surface_reference_point_not_object_center",
    "frame": "current_robot_base",
    "velocity": "absolute_world_velocity_rotated_to_current_base_not_relative_velocity",
    "fusion_mode": "world_constant_velocity_KF_with_fixed_lag_capture_time_replay",
    "normalization": "SI_units_with_fixed_configured_clipping_no_running_statistics",
    "fields": [
        {"name": "position", "slice": [0, 3], "unit": "m"},
        {"name": "absolute_velocity", "slice": [3, 6], "unit": "m/s"},
        {"name": "position_std", "slice": [6, 9], "unit": "m"},
        {"name": "camera_visible", "slice": [9, 11], "unit": "boolean"},
        {"name": "camera_observation_age", "slice": [11, 13], "unit": "s"},
        {"name": "initialized", "slice": [13, 14], "unit": "boolean"},
        {"name": "prediction_age", "slice": [14, 15], "unit": "s"},
        {"name": "valid_depth_fraction", "slice": [15, 16], "unit": "fraction"},
    ],
    "missing": "predict_only; nonfresh_or_dropped_deliveries_are_zero_images",
    "uninitialized": "position_velocity_std_zero; age_since_first_step_if_no_observation",
    "camera_order": ["base", "wrist"],
}


@dataclass(frozen=True)
class PerceptionConfig:
    # Synthetic corruption is a hypothesis until fitted to real sensor data.
    camera_delay_frames: tuple = (1, 3)
    history_extra_frames: int = 2
    min_depth_m: float = 0.05
    max_depth_m: float = 3.0
    min_valid_pixels: int = 3
    mask_threshold: float = 0.5
    measurement_std_m: float = 0.04
    initial_velocity_std_mps: float = 1.0
    # White acceleration spectral density in m^2/s^3, not per-step variance.
    process_accel_variance: float = 0.5
    depth_noise_std_m: float = 0.003
    depth_noise_distance_scale: float = 0.003
    depth_noise_angular_scale: float = 0.002
    pixel_dropout_prob: float = 0.02
    frame_dropout_prob: float = 0.01
    frame_dropout_distance_scale: float = 0.01
    frame_dropout_angular_scale: float = 0.03
    small_target_dropout_scale: float = 0.05
    burst_length_min: int = 2
    burst_length_max: int = 5
    max_dropout_probability: float = 0.95
    belief_position_clip_m: float = 10.0
    belief_velocity_clip_mps: float = 10.0
    belief_std_clip_m: float = 10.0
    belief_age_clip_s: float = 10.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            values = value if name == "camera_delay_frames" else (value,)
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
                raise ValueError(name + " must contain finite numbers")
        delays = tuple(self.camera_delay_frames)
        if len(delays) != 2 or any(int(x) != x or x < 0 for x in delays):
            raise ValueError("camera_delay_frames must contain two nonnegative integers")
        object.__setattr__(self, "camera_delay_frames", tuple(int(x) for x in delays))
        if self.history_extra_frames < 2 or int(self.history_extra_frames) != self.history_extra_frames:
            raise ValueError("history_extra_frames must be an integer >= 2")
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("Expected 0 < min_depth_m < max_depth_m")
        if self.min_valid_pixels < 1 or self.burst_length_min < 1 or self.burst_length_max < self.burst_length_min:
            raise ValueError("Invalid pixel count or burst lengths")
        if any(int(x) != x for x in (self.min_valid_pixels, self.burst_length_min, self.burst_length_max)):
            raise ValueError("Pixel count and burst lengths must be integers")
        for name in ("measurement_std_m", "initial_velocity_std_mps", "belief_position_clip_m",
                     "belief_velocity_clip_mps", "belief_std_clip_m", "belief_age_clip_s"):
            if getattr(self, name) <= 0:
                raise ValueError(name + " must be positive")
        for name in ("process_accel_variance", "depth_noise_std_m", "depth_noise_distance_scale",
                     "depth_noise_angular_scale", "frame_dropout_distance_scale",
                     "frame_dropout_angular_scale", "small_target_dropout_scale"):
            if getattr(self, name) < 0:
                raise ValueError(name + " must be nonnegative")
        for name in ("pixel_dropout_prob", "frame_dropout_prob", "max_dropout_probability", "mask_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(name + " must be between zero and one")

    def to_dict(self):
        result = asdict(self)
        result["camera_delay_frames"] = list(self.camera_delay_frames)
        return result


class AsymmetricPerception:
    """Batched two-camera estimator with independent per-environment clocks.

    ``step`` advances only environments whose ``now`` increased. Repeated
    camera timestamps do not become new measurements, even when ``now`` moves.
    For partial resets call ``reset(ids)``; the next sample for those ids is a
    fresh first frame, while other environments may keep their previous now.
    Clocks are float64; floating-point timestamps supplied by the caller must
    retain sufficient precision (do not pre-cast epoch seconds to float32).

    ``corrupt=False`` is the deployment path: no synthetic noise, burst loss or
    delay. Actual masks, invalid depths and timestamp latency are still used.
    Changing corruption mode requires resetting all active environments first.
    """

    def __init__(self, num_envs, height, width, device="cpu", config=None, seed=None):
        self.config = (PerceptionConfig(**dict(config)) if isinstance(config, Mapping)
                       else config if isinstance(config, PerceptionConfig) else PerceptionConfig())
        if config is not None and not isinstance(config, (Mapping, PerceptionConfig)):
            raise TypeError("config must be a mapping or PerceptionConfig")
        self.num_envs, self.height, self.width = int(num_envs), int(height), int(width)
        if min(self.num_envs, self.height, self.width) <= 0:
            raise ValueError("num_envs, height and width must be positive")
        self.device = torch.device(device)
        self.history_length = max(self.config.camera_delay_frames) + int(self.config.history_extra_frames)
        # Default to the experiment's global RNG, which the trainer already
        # snapshots around evaluation. Explicit seeds opt into a local stream.
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(seed))
        b, n, h, w = self.num_envs, self.history_length, self.height, self.width
        self._eye6 = torch.eye(6, device=self.device)
        self._eye3 = torch.eye(3, device=self.device)
        rows, cols = torch.meshgrid(torch.arange(h, device=self.device),
                                    torch.arange(w, device=self.device), indexing="ij")
        self._rows, self._cols = rows.float(), cols.float()
        self._images = torch.zeros(b, n, 2, 2, h, w, device=self.device)
        self._points = torch.zeros(b, n, 2, 3, device=self.device)
        self._variances = torch.ones(b, n, 2, 3, device=self.device)
        self._timestamps = torch.full((b, n, 2), -torch.inf, device=self.device, dtype=torch.float64)
        self._tick_times = torch.full((b, n), -torch.inf, device=self.device, dtype=torch.float64)
        self._fresh = torch.zeros(b, n, 2, dtype=torch.bool, device=self.device)
        self._valid = torch.zeros_like(self._fresh)
        self._received = torch.zeros_like(self._fresh)
        self._anchor_x = torch.zeros(b, 6, device=self.device)
        self._anchor_p = torch.zeros(b, 6, 6, device=self.device)
        self._anchor_time = torch.full((b,), -torch.inf, device=self.device, dtype=torch.float64)
        self._anchor_initialized = torch.zeros(b, dtype=torch.bool, device=self.device)
        self._state = self._anchor_x.clone()
        self._covariance = self._anchor_p.clone()
        self._initialized = self._anchor_initialized.clone()
        self._last_now = torch.full((b,), -torch.inf, device=self.device, dtype=torch.float64)
        self._birth_time = torch.zeros(b, device=self.device, dtype=torch.float64)
        self._last_input_time = torch.full((b, 2), -torch.inf, device=self.device, dtype=torch.float64)
        self._last_visible_time = self._last_input_time.clone()
        self._last_update_time = self._last_now.clone()
        self._history_count = torch.zeros(b, dtype=torch.long, device=self.device)
        self._burst_remaining = torch.zeros(b, 2, dtype=torch.long, device=self.device)
        self._delivered_images = torch.zeros(b, 2, 2, h, w, device=self.device)
        self._delivered_points = torch.zeros(b, 2, 3, device=self.device)
        self._delivered_valid = torch.zeros(b, 2, dtype=torch.bool, device=self.device)
        self._delivered_times = self._last_input_time.clone()
        self._rejected_stale = torch.zeros(b, dtype=torch.long, device=self.device)
        self._mode = None

    def get_rng_state(self):
        if self.generator is not None:
            return self.generator.get_state().clone()
        if self.device.type == "cuda":
            return torch.cuda.get_rng_state(self.device)
        return torch.random.get_rng_state()

    def set_rng_state(self, state):
        if self.generator is not None:
            self.generator.set_state(state)
        elif self.device.type == "cuda":
            torch.cuda.set_rng_state(state, self.device)
        else:
            torch.random.set_rng_state(state)

    def reset(self, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None
               else torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten())
        if ids.numel() and ((ids < 0).any() or (ids >= self.num_envs).any()):
            raise ValueError("reset environment index out of range")
        for name in ("_images", "_points", "_fresh", "_valid", "_received", "_anchor_x",
                     "_anchor_p", "_anchor_initialized", "_state", "_covariance", "_initialized",
                     "_birth_time", "_history_count", "_burst_remaining", "_delivered_images",
                     "_delivered_points", "_delivered_valid", "_rejected_stale"):
            getattr(self, name)[ids] = 0
        self._variances[ids] = 1
        for name in ("_timestamps", "_tick_times", "_anchor_time", "_last_now", "_last_input_time",
                     "_last_visible_time", "_last_update_time", "_delivered_times"):
            getattr(self, name)[ids] = -torch.inf
        if not torch.isfinite(self._last_now).any():
            self._mode = None

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _clock(self, value, cameras=False):
        value = torch.as_tensor(value, device=self.device, dtype=torch.float64)
        if cameras:
            if value.ndim == 0:
                value = value.expand(self.num_envs, 2)
            elif value.shape == (self.num_envs,):
                value = value[:, None].expand(-1, 2)
            if value.shape != (self.num_envs, 2):
                raise ValueError("timestamps must be scalar, [B], or [B,2]")
        else:
            if value.ndim == 0:
                value = value.expand(self.num_envs)
            if value.shape != (self.num_envs,):
                raise ValueError("now must be scalar or [B]")
        if not torch.isfinite(value).all():
            raise ValueError("Sensor clocks must be finite")
        return value

    def _measurement(self, images, camera_pose, intrinsics):
        cfg = self.config
        mask = images[:, :, 0] > cfg.mask_threshold
        depth = images[:, :, 1]
        # The sensor clips far-range pixels at max_depth_m. That saturation
        # is not a measured surface location and must not initialize the KF.
        valid = mask & torch.isfinite(depth) & (depth >= cfg.min_depth_m) & (depth < cfg.max_depth_m)
        count = valid.sum(dim=(-2, -1))
        safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
        k = intrinsics
        x = (self._cols - k[:, :, 0, 2, None, None]) * safe_depth / k[:, :, 0, 0, None, None]
        y = (self._rows - k[:, :, 1, 2, None, None]) * safe_depth / k[:, :, 1, 1, None, None]
        camera_point = torch.stack((x.sum((-2, -1)), y.sum((-2, -1)), safe_depth.sum((-2, -1))), dim=-1)
        camera_point = camera_point / count.clamp_min(1)[..., None]
        world_point = (camera_pose[:, :, :3, :3] @ camera_point[..., None]).squeeze(-1) + camera_pose[:, :, :3, 3]
        return world_point, count >= cfg.min_valid_pixels, valid, count, mask.sum((-2, -1))

    def _corrupt(self, images, fresh, angular_speed):
        cfg = self.config
        clean_mask = torch.isfinite(images[:, :, 0]) & (images[:, :, 0] > cfg.mask_threshold)
        depth = images[:, :, 1]
        good = clean_mask & torch.isfinite(depth) & (depth >= cfg.min_depth_m) & (depth < cfg.max_depth_m)
        count = good.sum((-2, -1))
        distance = torch.where(good, depth, torch.zeros_like(depth)).sum((-2, -1)) / count.clamp_min(1)
        sigma = cfg.depth_noise_std_m + cfg.depth_noise_distance_scale * distance + cfg.depth_noise_angular_scale * angular_speed
        probability = (cfg.frame_dropout_prob + cfg.frame_dropout_distance_scale * distance +
                       cfg.frame_dropout_angular_scale * angular_speed +
                       cfg.small_target_dropout_scale / count.clamp_min(1).float().sqrt()).clamp(0, cfg.max_dropout_probability)
        random_start = torch.rand(probability.shape, device=self.device, generator=self.generator) < probability
        old_burst = self._burst_remaining > 0
        start = fresh & ~old_burst & random_start
        duration = torch.randint(cfg.burst_length_min, cfg.burst_length_max + 1, probability.shape,
                                 device=self.device, generator=self.generator)
        remaining = torch.where(start, duration, self._burst_remaining)
        drop = fresh & (remaining > 0)
        self._burst_remaining = torch.where(fresh, (remaining - 1).clamp_min(0), self._burst_remaining)
        pixel_drop = torch.rand(depth.shape, device=self.device, generator=self.generator) < cfg.pixel_dropout_prob
        noisy_depth = depth + sigma[..., None, None] * torch.randn(depth.shape, device=self.device, generator=self.generator)
        keep = good & ~pixel_drop & ~drop[..., None, None] & (noisy_depth >= cfg.min_depth_m) & (noisy_depth < cfg.max_depth_m)
        # Depth holes leave the detected mask intact. The valid-depth fraction
        # therefore reports reliability rather than becoming identically one.
        reported_mask = clean_mask & ~drop[..., None, None]
        result = torch.stack((reported_mask.float(), torch.where(keep, noisy_depth, torch.zeros_like(depth))), dim=2)
        return result, sigma

    def _predict(self, x, p, dt):
        # Continuous white-acceleration Q has the semigroup property: splitting
        # prediction at a newly received measurement does not change the model.
        dt = dt.to(x.dtype)  # Subtract float64 clocks before state arithmetic.
        f = self._eye6.expand(self.num_envs, -1, -1).clone()
        f[:, :3, 3:] = self._eye3 * dt[:, None, None]
        q = torch.zeros_like(p)
        q[:, :3, :3] = self._eye3 * (dt.pow(3) / 3)[:, None, None]
        q[:, :3, 3:] = self._eye3 * (dt.square() / 2)[:, None, None]
        q[:, 3:, :3] = q[:, :3, 3:]
        q[:, 3:, 3:] = self._eye3 * dt[:, None, None]
        return ((f @ x[..., None]).squeeze(-1),
                f @ p @ f.transpose(-1, -2) + self.config.process_accel_variance * q)

    def _replay(self, end_time):
        x, p = self._anchor_x.clone(), self._anchor_p.clone()
        initialized = self._anchor_initialized.clone()
        clock = self._anchor_time.clone()
        times = self._timestamps.reshape(self.num_envs, -1)
        points = self._points.reshape(self.num_envs, -1, 3)
        variances = self._variances.reshape(self.num_envs, -1, 3)
        usable = (self._received.reshape(self.num_envs, -1) &
                  (times > self._anchor_time[:, None]) & (times <= end_time[:, None]))
        order = torch.where(usable, times, torch.full_like(times, torch.inf)).argsort(dim=1)
        batch = torch.arange(self.num_envs, device=self.device)
        for index in range(times.shape[1]):
            slot = order[:, index]
            active = usable[batch, slot]
            time = times[batch, slot]
            dt = torch.where(active & initialized, time - clock, torch.zeros_like(time)).clamp_min(0)
            x, p = self._predict(x, p, dt)
            z = points[batch, slot]
            r = torch.diag_embed(variances[batch, slot].clamp_min(1e-8))
            # Innovation covariance is positive definite even for inactive
            # rows, so batched solve remains well defined before initialization.
            gain = torch.linalg.solve(p[:, :3, :3] + r, p[:, :3, :]).transpose(-1, -2)
            corrected_x = x + (gain @ (z - x[:, :3])[..., None]).squeeze(-1)
            ikh = self._eye6.expand_as(p).clone()
            ikh[:, :, :3] -= gain
            corrected_p = ikh @ p @ ikh.transpose(-1, -2) + gain @ r @ gain.transpose(-1, -2)
            existing = active & initialized
            first = active & ~initialized
            x = torch.where(existing[:, None], corrected_x, x)
            p = torch.where(existing[:, None, None], corrected_p, p)
            initial_x = torch.cat((z, torch.zeros_like(z)), dim=-1)
            initial_p = torch.zeros_like(p)
            initial_p[:, :3, :3] = r
            initial_p[:, 3:, 3:] = self._eye3 * self.config.initial_velocity_std_mps ** 2
            x = torch.where(first[:, None], initial_x, x)
            p = torch.where(first[:, None, None], initial_p, p)
            initialized |= active
            clock = torch.where(active, time, clock)
        dt = torch.where(initialized, end_time - clock, torch.zeros_like(end_time)).clamp_min(0)
        x, p = self._predict(x, p, dt)
        p = 0.5 * (p + p.transpose(-1, -2))
        return x, p, initialized

    def _advance_history(self, advance):
        full = advance & (self._history_count >= self.history_length)
        if full.any():
            cutoff = torch.where(full, self._tick_times[:, 0], self._last_now)
            x, p, initialized = self._replay(cutoff)
            self._anchor_x[full], self._anchor_p[full] = x[full], p[full]
            self._anchor_initialized[full] = initialized[full]
            self._anchor_time[full] = cutoff[full]
        for name in ("_images", "_points", "_variances", "_timestamps", "_tick_times",
                     "_fresh", "_valid", "_received"):
            array = getattr(self, name)
            array[advance, :-1] = array[advance, 1:].clone()
            array[advance, -1] = -torch.inf if name in ("_timestamps", "_tick_times") else 0
        self._history_count[advance] = (self._history_count[advance] + 1).clamp_max(self.history_length)

    def _output(self, now, base_pose):
        cfg = self.config
        rotation = base_pose[:, :3, :3].transpose(-1, -2)
        position = (rotation @ (self._state[:, :3] - base_pose[:, :3, 3])[..., None]).squeeze(-1)
        velocity = (rotation @ self._state[:, 3:, None]).squeeze(-1)
        local_covariance = rotation @ self._covariance[:, :3, :3] @ rotation.transpose(-1, -2)
        std = local_covariance.diagonal(dim1=-2, dim2=-1).clamp_min(0).sqrt()
        position = torch.where(self._initialized[:, None], position, torch.zeros_like(position))
        velocity = torch.where(self._initialized[:, None], velocity, torch.zeros_like(velocity))
        std = torch.where(self._initialized[:, None], std, torch.zeros_like(std))
        seen_time = torch.where(torch.isfinite(self._last_visible_time), self._last_visible_time, self._birth_time[:, None])
        update_time = torch.where(torch.isfinite(self._last_update_time), self._last_update_time, self._birth_time)
        camera_age = (now[:, None] - seen_time).clamp(0, cfg.belief_age_clip_s)
        prediction_age = (now - update_time).clamp(0, cfg.belief_age_clip_s)
        mask = self._delivered_images[:, :, 0] > cfg.mask_threshold
        depth = self._delivered_images[:, :, 1]
        valid_depth = mask & torch.isfinite(depth) & (depth >= cfg.min_depth_m) & (depth < cfg.max_depth_m)
        fraction = valid_depth.sum((1, 2, 3)).float() / mask.sum((1, 2, 3)).clamp_min(1)
        belief = torch.cat((position.clamp(-cfg.belief_position_clip_m, cfg.belief_position_clip_m),
                            velocity.clamp(-cfg.belief_velocity_clip_mps, cfg.belief_velocity_clip_mps),
                            std.clamp_max(cfg.belief_std_clip_m), self._delivered_valid.float(), camera_age.float(),
                            self._initialized[:, None].float(), prediction_age[:, None].float(), fraction[:, None]), dim=-1)
        return {
            "images": self._delivered_images.clone(), "belief": belief,
            "world_position": self._state[:, :3].clone(), "world_velocity": self._state[:, 3:].clone(),
            "position_variance": self._covariance[:, :3, :3].diagonal(dim1=-2, dim2=-1).clone(),
            "covariance": self._covariance.clone(), "initialized": self._initialized.clone(),
            "measurement_world": self._delivered_points.clone(), "valid": self._delivered_valid.clone(),
            "capture_timestamps": self._delivered_times.clone(),
            "rejected_stale_measurements": self._rejected_stale.clone(),
        }

    @torch.no_grad()
    def step(self, images, T_world_camera, intrinsics, timestamps, T_world_base, now,
             camera_angular_velocity=None, corrupt=True):
        images = self._tensor(images)
        camera_pose, base_pose = self._tensor(T_world_camera), self._tensor(T_world_base)
        k = self._tensor(intrinsics)
        if k.shape == (2, 3, 3):
            k = k.unsqueeze(0).expand(self.num_envs, -1, -1, -1)
        if images.shape != (self.num_envs, 2, 2, self.height, self.width):
            raise ValueError("images must have shape [B,2 cameras,2 modalities,H,W]")
        if camera_pose.shape != (self.num_envs, 2, 4, 4) or base_pose.shape != (self.num_envs, 4, 4):
            raise ValueError("Expected T_world_camera[B,2,4,4] and T_world_base[B,4,4]")
        if k.shape != (self.num_envs, 2, 3, 3) or (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any():
            raise ValueError("intrinsics must be [B,2,3,3] or [2,3,3] with positive focal lengths")
        if not torch.isfinite(camera_pose).all() or not torch.isfinite(base_pose).all() or not torch.isfinite(k).all():
            raise ValueError("Camera/base geometry must be finite")
        now, timestamps = self._clock(now), self._clock(timestamps, cameras=True)
        if (now < self._last_now).any():
            raise ValueError("Environment time moved backwards; reset those environments first")
        advance = now > self._last_now
        # IsaacGym renders every camera during a partial reset. Callers may
        # hold the clocks of untouched environments: their new packet rows
        # must be ignored, including the newer capture timestamps.
        if ((timestamps > now[:, None] + 1e-6) & advance[:, None]).any():
            raise ValueError("A camera capture timestamp cannot be later than now")
        if not advance.any():
            return self._output(now, base_pose)
        if self._mode is not None and self._mode != bool(corrupt):
            raise ValueError("Reset active environments before changing corruption mode")
        self._mode = bool(corrupt)
        first = advance & ~torch.isfinite(self._last_now)
        self._birth_time[first] = now[first]
        fresh = advance[:, None] & (timestamps > self._last_input_time)
        if camera_angular_velocity is None:
            angular_speed = torch.zeros(self.num_envs, 2, device=self.device)
        else:
            omega = self._tensor(camera_angular_velocity)
            if omega.shape != (self.num_envs, 2, 3) or not torch.isfinite(omega).all():
                raise ValueError("camera_angular_velocity must be finite [B,2,3] rad/s")
            angular_speed = omega.norm(dim=-1)
        self._advance_history(advance)
        if corrupt:
            captured, sigma = self._corrupt(images, fresh, angular_speed)
        else:
            captured = images.clone()
            # Keep the mask when only depth is invalid so reliability remains
            # observable; absent masks carry zero target depth.
            valid_mask = torch.isfinite(captured[:, :, 0]) & (captured[:, :, 0] > self.config.mask_threshold)
            captured[:, :, 0] = valid_mask.float()
            valid_depth = (torch.isfinite(captured[:, :, 1]) &
                           (captured[:, :, 1] >= self.config.min_depth_m) &
                           (captured[:, :, 1] < self.config.max_depth_m))
            captured[:, :, 1] = torch.where(valid_mask & valid_depth, captured[:, :, 1],
                                            torch.zeros_like(captured[:, :, 1]))
            sigma = torch.zeros(self.num_envs, 2, device=self.device)
        points, valid, _, _, _ = self._measurement(captured, camera_pose, k)
        self._images[advance, -1] = captured[advance]
        self._points[advance, -1] = points[advance]
        self._variances[advance, -1] = (self.config.measurement_std_m ** 2 + sigma[advance].square())[..., None].expand(-1, -1, 3)
        self._timestamps[advance, -1] = timestamps[advance]
        self._tick_times[advance, -1] = now[advance]
        self._fresh[advance, -1] = fresh[advance]
        self._valid[advance, -1] = valid[advance]
        self._last_input_time = torch.where(fresh, timestamps, self._last_input_time)
        self._delivered_images[advance] = 0
        self._delivered_points[advance] = 0
        self._delivered_valid[advance] = False
        self._delivered_times[advance] = -torch.inf
        for camera in range(2):
            delay = self.config.camera_delay_frames[camera] if corrupt else 0
            slot = self.history_length - 1 - delay
            delivered = advance & self._fresh[:, slot, camera]
            self._delivered_images[delivered, camera] = self._images[delivered, slot, camera]
            self._delivered_times[delivered, camera] = self._timestamps[delivered, slot, camera]
            reliable = delivered & self._valid[:, slot, camera]
            self._delivered_valid[reliable, camera] = True
            self._delivered_points[reliable, camera] = self._points[reliable, slot, camera]
            capture_time = self._timestamps[:, slot, camera]
            self._last_visible_time[:, camera] = torch.where(reliable, torch.maximum(capture_time, self._last_visible_time[:, camera]), self._last_visible_time[:, camera])
            accepted = reliable & (capture_time > self._anchor_time)
            self._rejected_stale += (reliable & ~accepted).long()
            self._received[accepted, slot, camera] = True
            self._last_update_time = torch.where(accepted, torch.maximum(capture_time, self._last_update_time), self._last_update_time)
        x, p, initialized = self._replay(now)
        self._state[advance], self._covariance[advance] = x[advance], p[advance]
        self._initialized[advance] = initialized[advance]
        self._last_now[advance] = now[advance]
        return self._output(now, base_pose)
