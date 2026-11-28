"""Smooth remote-control-car motion for the moving table, without simulator imports."""

import math

import torch


class RemoteCarDriver:
    """Persistent driving intentions with separate throttle and steering clocks.

    Goals are normalized controls; input_speed/input_curvature are physical
    quantities after input smoothing. DRIVE lasts for a sampled intention,
    BRAKE waits for the filtered car to nearly stop, and PAUSE then waits its
    full sampled duration before starting the pending direction. This is a
    procedural input model, not a model calibrated against human driving.
    """

    DRIVE, BRAKE, PAUSE = 0, 1, 2
    event_names = ("command_changed", "throttle_changed", "steering_changed",
                   "turn_changed", "intent_changed")
    float_state_names = (
        "direction", "pending_direction", "phase_time_left", "throttle_time_left",
        "steering_time_left", "turn_time_left", "cruise_throttle", "throttle_goal",
        "steering_reference", "steering_goal", "input_speed", "input_curvature",
    )
    state_names = float_state_names + ("initialized", "phase") + event_names

    def __init__(self, num_envs, device, settings):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.max_speed = float(settings["maxSpeed"])
        self.min_speed = float(settings.get("minSpeed", 0.0))
        self.minimum_throttle = max(.12, self.min_speed / self.max_speed)
        self.curvature_limit = float(settings["maxYawRate"]) / self.max_speed
        self.drive_duration = tuple(settings["driveDuration"])
        self.throttle_update_time = tuple(settings["throttleUpdateTime"])
        self.steering_update_time = tuple(settings["steeringUpdateTime"])
        self.turn_hold_time = tuple(settings["turnHoldTime"])
        self.stop_hold_time = tuple(settings["stopHoldTime"])
        self.stop_probability = float(settings["stopProbability"])
        self.reverse_probability = float(settings["reverseProbability"])
        self.straight_probability = float(settings["straightProbability"])
        self.input_response_time = float(settings["inputResponseTime"])
        self.throttle_tracking_gain = float(settings["throttleTrackingGain"])
        self.throttle_variation = float(settings["throttleVariation"])
        self.cruise_throttle_range = tuple(
            max(self.min_speed / self.max_speed, float(value)) for value in settings["cruiseThrottleRange"])
        self.steering_tracking_gain = float(settings["steeringTrackingGain"])
        self.steering_variation = float(settings["steeringVariation"])
        self.turn_magnitude_range = tuple(settings["turnMagnitudeRange"])
        for name in self.float_state_names:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device,
                                           dtype=torch.float32))
        for name in ("initialized",) + self.event_names:
            setattr(self, name, torch.zeros(self.num_envs, device=self.device, dtype=torch.bool))
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._input_weight_cache = {}

    def reset(self, env_ids=None):
        """Forget only selected intentions and input history; initialize next step."""
        indices = slice(None) if env_ids is None else env_ids
        for name in self.state_names:
            getattr(self, name)[indices] = 0

    def _uniform(self, env_ids, bounds):
        low, high = bounds
        return low + (high - low) * torch.rand(env_ids.numel(), device=self.device)

    def _renew_intent(self, env_ids):
        if env_ids.numel() == 0:
            return
        self.phase_time_left[env_ids] = self._uniform(env_ids, self.drive_duration)
        large_throttle = torch.rand(env_ids.numel(), device=self.device) < .1
        self.cruise_throttle[env_ids] = torch.where(
            large_throttle, self._uniform(env_ids, (max(.85, self.minimum_throttle), 1.)),
            self._uniform(env_ids, self.cruise_throttle_range))
        self.intent_changed[env_ids] = True

    def _start_drive(self, env_ids, direction):
        if env_ids.numel() == 0:
            return
        self.initialized[env_ids] = True
        self.phase[env_ids] = self.DRIVE
        self.direction[env_ids] = direction
        self.pending_direction[env_ids] = direction
        self._renew_intent(env_ids)
        # Start within the driving band. The input and motion filters retain
        # their histories, including when this follows a stop or a reversal.
        self.throttle_goal[env_ids] = max(.25, self.minimum_throttle)
        self.steering_goal[env_ids] = 0.
        self.throttle_time_left[env_ids] = 0.
        self.steering_time_left[env_ids] = 0.
        self.turn_time_left[env_ids] = 0.

    def _choose_turn(self, env_ids):
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        straight = torch.rand(count, device=self.device) < self.straight_probability
        large_turn = torch.rand(count, device=self.device) < .1
        magnitude = torch.where(large_turn, self._uniform(env_ids, (.85, 1.)),
                                self._uniform(env_ids, self.turn_magnitude_range))
        sign = torch.where(torch.rand(count, device=self.device) < .5, -1., 1.)
        self.steering_reference[env_ids] = torch.where(straight, 0., sign * magnitude)
        self.turn_time_left[env_ids] = self._uniform(env_ids, self.turn_hold_time)
        self.turn_changed[env_ids] = True

    def step(self, dt, speed, acceleration):
        """Advance intentions using the car's previous filtered speed/acceleration."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.:
            raise ValueError("RemoteCarDriver dt must be finite and positive")
        for name in self.event_names:
            getattr(self, name).zero_()

        new_ids = (~self.initialized).nonzero(as_tuple=False).flatten()
        direction = torch.where(torch.rand(new_ids.numel(), device=self.device) < .5, -1., 1.)
        self._start_drive(new_ids, direction)
        phase_before = self.phase.clone()

        # Only long-lived driving intentions can request a stop or reversal.
        # A single draw makes the two outcomes mutually exclusive.
        expired_ids = ((self.phase == self.DRIVE) & (self.phase_time_left <= 0.)).nonzero(
            as_tuple=False).flatten()
        choice = torch.rand(expired_ids.numel(), device=self.device)
        stop = choice < self.stop_probability
        reverse = ((choice >= self.stop_probability)
                   & (choice < self.stop_probability + self.reverse_probability))
        self._renew_intent(expired_ids[~(stop | reverse)])
        braking_ids = expired_ids[stop | reverse]
        self.pending_direction[braking_ids] = self.direction[braking_ids]
        reverse_ids = expired_ids[reverse]
        self.pending_direction[reverse_ids] = -self.direction[reverse_ids]
        self.phase[braking_ids] = self.BRAKE
        self.phase_time_left[braking_ids] = 0.
        self.throttle_goal[braking_ids] = 0.
        self.steering_goal[braking_ids] = 0.
        self.steering_reference[braking_ids] = 0.
        self.intent_changed[braking_ids] = True
        self.throttle_changed[braking_ids] = True
        self.steering_changed[braking_ids] = True
        self.turn_changed[braking_ids] = True

        near_stop = ((speed.abs() < .015) & (acceleration.abs() < .05)
                     & (self.input_speed.abs() < .015))
        # Inspect the phase at entry: a single tick cannot skip BRAKE or PAUSE.
        pausing_ids = ((phase_before == self.BRAKE) & near_stop).nonzero(
            as_tuple=False).flatten()
        self.phase[pausing_ids] = self.PAUSE
        self.phase_time_left[pausing_ids] = self._uniform(pausing_ids, self.stop_hold_time)
        self.command_changed[pausing_ids] = True
        restarting_ids = ((phase_before == self.PAUSE) & (self.phase_time_left <= 0.)
                          & near_stop).nonzero(as_tuple=False).flatten()
        self._start_drive(restarting_ids, self.pending_direction[restarting_ids])

        driving = self.phase == self.DRIVE
        turn_ids = (driving & (self.turn_time_left <= 0.)).nonzero(as_tuple=False).flatten()
        self._choose_turn(turn_ids)
        throttle_ids = (driving & (self.throttle_time_left <= 0.)).nonzero(
            as_tuple=False).flatten()
        old_throttle = self.throttle_goal[throttle_ids]
        throttle_proposal = (
            old_throttle + self.throttle_tracking_gain * (self.cruise_throttle[throttle_ids] - old_throttle)
            + self._uniform(throttle_ids, (-self.throttle_variation, self.throttle_variation)))
        # Enforce the driving-speed floor in the raw targets, never by clamping
        # filtered physical speed: startup must still pass smoothly through zero.
        # Reflect proposals at the pedal bounds. Clipping larger corrections
        # creates a point mass at full throttle and unnaturally long plateaus.
        pedal_min, pedal_max = self.minimum_throttle, 1.
        span = pedal_max - pedal_min
        folded = torch.remainder(throttle_proposal - pedal_min, 2. * span)
        self.throttle_goal[throttle_ids] = pedal_min + torch.minimum(folded, 2. * span - folded)
        self.throttle_time_left[throttle_ids] = self._uniform(throttle_ids, self.throttle_update_time)
        self.throttle_changed[throttle_ids] = True

        steering_ids = (driving & (self.steering_time_left <= 0.)).nonzero(
            as_tuple=False).flatten()
        reference = self.steering_reference[steering_ids]
        noise = self._uniform(steering_ids, (-self.steering_variation, self.steering_variation))
        noise = torch.where(reference == 0., 0., noise)
        self.steering_goal[steering_ids] = (
            (1. - self.steering_tracking_gain) * self.steering_goal[steering_ids]
            + self.steering_tracking_gain * reference + noise).clamp(-1., 1.)
        self.steering_time_left[steering_ids] = self._uniform(steering_ids, self.steering_update_time)
        self.steering_changed[steering_ids] = True
        self.command_changed |= (self.intent_changed | self.throttle_changed
                                 | self.steering_changed | self.turn_changed)

        if dt not in self._input_weight_cache:
            self._input_weight_cache[dt] = -math.expm1(-dt / self.input_response_time)
        weight = self._input_weight_cache[dt]
        speed_goal = self.direction * self.throttle_goal * self.max_speed
        curvature_goal = self.steering_goal * self.curvature_limit
        self.input_speed.add_(weight * (speed_goal - self.input_speed)).clamp_(
            -self.max_speed, self.max_speed)
        self.input_curvature.add_(weight * (curvature_goal - self.input_curvature)).clamp_(
            -self.curvature_limit, self.curvature_limit)

        self.phase_time_left[self.phase != self.BRAKE] -= dt
        self.throttle_time_left[driving] -= dt
        self.steering_time_left[driving] -= dt
        self.turn_time_left[driving] -= dt
        return self.input_speed, self.input_curvature


class RemoteCarMotion:
    """Batch signed speed and steering curvature through two-stage low-pass filters.

    The exact continuous-time response has continuous velocity and acceleration
    (C1 speed and curvature), even when random targets change abruptly. For a
    bounded target of magnitude U, the two-stage filter satisfies
    ``abs(output_derivative) <= 2 * U / (e * tau)`` from rest. The chosen time
    constants bound each local-axis acceleration and yaw acceleration separately;
    world-frame acceleration also includes the effect of the changing heading.

    The existing speed/driver state represents local Y. With translationAxes=xy,
    x_motion supplies a second independently sampled and filtered local speed;
    yaw remains speed*curvature using Y. Legacy Y-only settings have x_motion=None
    and retain their original random sequence. This is a kinematic carrier
    model; independent X/Y commands also permit lateral translation.
    """

    def __init__(self, num_envs, device, settings):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.translation_axes = settings.get("translationAxes", "y")
        if self.translation_axes not in ("y", "xy"):
            raise ValueError("translationAxes must be 'y' or 'xy'")
        self.max_speed = float(settings["maxSpeed"])
        self.max_yaw_rate = float(settings["maxYawRate"])
        self.min_speed = float(settings.get("minSpeed", 0.0))
        self.curvature_limit = self.max_yaw_rate / self.max_speed
        self.driver = RemoteCarDriver(self.num_envs, self.device, settings)
        self.driver_enabled = True

        # Half of the yaw-acceleration budget is reserved for each term in
        # d(v*kappa)/dt = dv/dt*kappa + v*dkappa/dt.
        yaw_tau = (4.0 * self.max_speed * self.curvature_limit
                   / (math.e * float(settings["yawAcceleration"])))
        self.speed_tau = max(
            2.0 * self.max_speed / (math.e * float(settings["linearAcceleration"])),
            yaw_tau,
        )
        self.steering_tau = yaw_tau

        for name in ("speed_stage", "speed", "curvature_stage", "curvature",
                     "target_speed", "target_curvature"):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device,
                                           dtype=torch.float32))
        self.command_changed = torch.zeros(self.num_envs, device=self.device,
                                           dtype=torch.bool)
        self._manual_pending = torch.zeros_like(self.command_changed)
        self._coefficient_cache = {}
        self.x_motion = (RemoteCarMotion(num_envs, device, dict(settings, translationAxes="y"))
                         if self.translation_axes == "xy" else None)

    def reset(self, env_ids=None):
        """Clear selected motion/input history without changing driver/manual mode."""
        indices = slice(None) if env_ids is None else env_ids
        for name in ("speed_stage", "speed", "curvature_stage", "curvature",
                     "target_speed", "target_curvature"):
            getattr(self, name)[indices] = 0.0
        self.command_changed[indices] = False
        self._manual_pending[indices] = False
        self.driver.reset(env_ids)
        if self.x_motion is not None:
            self.x_motion.reset(env_ids)

    def _manual_target(self, value, limit, name):
        result = torch.as_tensor(value, device=self.device, dtype=torch.float32)
        if result.ndim == 0 or (result.ndim == 1 and result.numel() == 1):
            result = result.expand(self.num_envs)
        elif result.ndim != 1 or result.numel() != self.num_envs:
            raise ValueError("{} must be scalar or have one value per environment".format(name))
        if not torch.isfinite(result).all().item() or (result.abs() > limit).any().item():
            raise ValueError("{} must be finite and within +/-{}".format(name, limit))
        # Calls may use views of our own targets; keep the two updates atomic
        # even if either argument aliases a target that is about to change.
        return result.clone()

    def set_manual_targets(self, speed, curvature, x_speed=0.):
        """Disable the driver and set bounded targets without altering filter history.

        Each argument accepts a scalar or one value per environment. Changed
        targets appear in command_changed on the next step, once only. This
        explicit mode supports boundary demonstrations and deterministic tests.
        speed is local Y; x_speed sets local X in xy mode and otherwise must
        be zero. Omitting x_speed holds X at zero in this manual mode.
        """
        speed = self._manual_target(speed, self.max_speed, "speed")
        curvature = self._manual_target(curvature, self.curvature_limit, "curvature")
        x_speed = self._manual_target(x_speed, self.max_speed if self.x_motion is not None else 0.,
                                      "x_speed")
        if self.x_motion is not None:
            self.x_motion.set_manual_targets(x_speed, 0.)
        self._manual_pending |= ((speed != self.target_speed) | (curvature != self.target_curvature))
        self.driver_enabled = False
        self.target_speed.copy_(speed)
        self.target_curvature.copy_(curvature)

    @staticmethod
    def _coefficients(dt, tau):
        z = dt / tau
        decay = math.exp(-z)
        stage_target_weight = -math.expm1(-z)
        stage_weight = z * decay
        # Each update is a convex combination. expm1 avoids loss of accuracy
        # for short physics steps; max handles only roundoff near z == 0.
        target_weight = max(0.0, stage_target_weight - stage_weight)
        return decay, stage_target_weight, stage_weight, target_weight

    @staticmethod
    def _advance(stage, output, target, coefficients, limit):
        decay, stage_target_weight, stage_weight, target_weight = coefficients
        # Update the output first so both equations use the OLD first stage.
        output.mul_(decay).add_(stage, alpha=stage_weight).add_(target, alpha=target_weight)
        stage.mul_(decay).add_(target, alpha=stage_target_weight)
        # Convexity already gives the bound analytically. Guard against a final
        # float32 rounding ulp exceeding a physical limit at steady state.
        output.clamp_(-limit, limit)
        stage.clamp_(-limit, limit)

    def step(self, dt):
        """Return (N, 3) local [vx, vy, yaw_rate] after one physics step."""
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("RemoteCarMotion dt must be finite and positive")
        if self.driver_enabled:
            acceleration = (self.speed_stage - self.speed) / self.speed_tau
            speed, curvature = self.driver.step(dt, self.speed, acceleration)
            self.target_speed.copy_(speed)
            self.target_curvature.copy_(curvature)
            self.command_changed.copy_(self.driver.command_changed)
        else:
            self.command_changed.copy_(self._manual_pending)
        self._manual_pending.zero_()

        if dt not in self._coefficient_cache:
            self._coefficient_cache[dt] = (
                self._coefficients(dt, self.speed_tau),
                self._coefficients(dt, self.steering_tau),
            )
        speed_coefficients, curvature_coefficients = self._coefficient_cache[dt]
        self._advance(self.speed_stage, self.speed, self.target_speed,
                      speed_coefficients, self.max_speed)
        self._advance(self.curvature_stage, self.curvature, self.target_curvature,
                      curvature_coefficients, self.curvature_limit)
        if self.x_motion is None:
            x_speed = torch.zeros_like(self.speed)
        else:
            self.x_motion.step(dt)
            x_speed = self.x_motion.speed
            self.command_changed |= self.x_motion.command_changed
        return torch.stack((x_speed, self.speed, self.speed * self.curvature), dim=-1)
