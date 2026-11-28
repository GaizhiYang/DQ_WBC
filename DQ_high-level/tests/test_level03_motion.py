"""Level03 local-XY motion, legacy local-Y replay and simulator-free hooks."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from test_sgs_env import TestEnvironment, source_method
from utils.config import level03_motion_config
from utils.table_motion import RemoteCarMotion


STATE_NAMES = ("speed_stage", "speed", "curvature_stage", "curvature",
               "target_speed", "target_curvature", "command_changed")


def make_motion(num_envs=3, **overrides):
    # Existing single-axis response tests explicitly exercise legacy local Y.
    # The XY tests below use the production defaults or request both axes.
    overrides.setdefault("translationAxes", "y")
    settings = level03_motion_config({"level03Motion": overrides})
    return RemoteCarMotion(num_envs, "cpu", settings)


def hold_targets(motion, speed, curvature):
    """Keep prescribed commands fixed while testing their physical response."""
    motion.set_manual_targets(speed, curvature)


def state_snapshot(motion):
    result = {name: getattr(motion, name).clone() for name in STATE_NAMES}
    result.update({"driver." + name: getattr(motion.driver, name).clone()
                   for name in motion.driver.state_names})
    if motion.x_motion is not None:
        result.update({"x_motion." + name: value
                       for name, value in state_snapshot(motion.x_motion).items()})
    return result


def derivatives(motion):
    linear_accel = (motion.speed_stage - motion.speed) / motion.speed_tau
    curvature_rate = (motion.curvature_stage - motion.curvature) / motion.steering_tau
    yaw_accel = linear_accel * motion.curvature + motion.speed * curvature_rate
    return linear_accel, yaw_accel


class MotionEnvironment:
    compute_table_command_level_03 = source_method(
        "b1z1_base.py", "B1Z1Base", "compute_table_command_level_03")

    def __init__(self, num_envs=3, physics_dt=.005):
        self.num_envs = num_envs
        self.device = "cpu"
        self.sim_params = SimpleNamespace(dt=physics_dt)
        self.dt = .15  # Deliberately distinct from the physics update interval.
        self.level03_motion = make_motion(num_envs)
        self.table_commands = torch.zeros(num_envs, 6)
        self.table_commands[:, 4] = .525


class RemoteCarMotionTests(unittest.TestCase):
    def test_new_state_is_at_rest_and_steering_cannot_move_or_spin_stationary_car(self):
        motion = make_motion()
        for name, value in state_snapshot(motion).items():
            self.assertEqual(torch.count_nonzero(value).item(), 0, name)
        curvature = motion.curvature_limit
        hold_targets(motion, [0., 0., 0.], [curvature, -curvature, curvature / 2])

        for _ in range(100):
            commands = motion.step(.02)
            torch.testing.assert_close(commands, torch.zeros(3, 3), atol=0, rtol=0)

        self.assertTrue((motion.curvature.abs() > .1).all().item())

    def test_forward_and_reverse_share_steering_and_turn_in_opposite_directions(self):
        motion = make_motion()
        speed, curvature = motion.max_speed, motion.curvature_limit
        hold_targets(motion, [speed, -speed, 0.], [curvature] * 3)
        motion.speed.copy_(motion.target_speed)
        motion.speed_stage.copy_(motion.target_speed)
        motion.curvature.copy_(motion.target_curvature)
        motion.curvature_stage.copy_(motion.target_curvature)

        commands = motion.step(.005)

        yaw = motion.max_yaw_rate
        torch.testing.assert_close(commands, torch.tensor([[0., speed, yaw], [0., -speed, -yaw], [0., 0., 0.]]))
        self.assertLessEqual(commands[:, 1].abs().max().item(), speed + 1e-7)
        self.assertLessEqual(commands[:, 2].abs().max().item(), yaw + 1e-7)

    def test_target_reversal_preserves_velocity_and_acceleration_continuity(self):
        motion = make_motion(num_envs=1)
        hold_targets(motion, [motion.max_speed], [motion.curvature_limit])
        for _ in range(30):
            previous_command = motion.step(.02)
        previous_accel = tuple(value.clone() for value in derivatives(motion))
        previous_state = {name: getattr(motion, name).clone()
                          for name in ("speed", "speed_stage", "curvature", "curvature_stage")}

        hold_targets(motion, [-motion.max_speed], [-motion.curvature_limit])

        for name, value in previous_state.items():
            torch.testing.assert_close(getattr(motion, name), value, atol=0, rtol=0)
        for actual, expected in zip(derivatives(motion), previous_accel):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        # Probe continuity over a vanishing interval; a finite interval can
        # legitimately change acceleration more with the faster response.
        next_command = motion.step(1e-7)
        torch.testing.assert_close(next_command, previous_command, atol=1e-6, rtol=0)
        for actual, expected in zip(derivatives(motion), previous_accel):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)
        # Deceleration cannot jump at the reversal: briefly the car is still
        # accelerating forward before smoothly decelerating into reverse.
        self.assertGreater(derivatives(motion)[0].item(), 0.)
        for _ in range(500):
            final_command = motion.step(.02)
        self.assertLess(final_command[0, 1].item(), -motion.max_speed + .001)
        self.assertGreater(final_command[0, 2].item(), motion.max_yaw_rate - .001)

    def test_constant_targets_have_same_response_at_different_physics_rates(self):
        slow = make_motion()
        fast = make_motion()
        for motion in (slow, fast):
            hold_targets(motion, [motion.max_speed, -motion.max_speed / 2, .001],
                         [motion.curvature_limit, -motion.curvature_limit / 2, 0.])

        for _ in range(10):
            slow_command = slow.step(.1)
        for _ in range(200):
            fast_command = fast.step(.005)

        torch.testing.assert_close(fast_command, slow_command, atol=2e-6, rtol=2e-5)
        for name in ("speed_stage", "speed", "curvature_stage", "curvature"):
            torch.testing.assert_close(getattr(fast, name), getattr(slow, name), atol=2e-6, rtol=2e-5)

    def test_step_response_converges_monotonically_without_overshoot(self):
        motion = make_motion()
        hold_targets(motion, [motion.max_speed, -motion.max_speed, .0001],
                     [motion.curvature_limit, -motion.curvature_limit, .0001])
        previous_speed = motion.speed.clone()
        previous_curvature = motion.curvature.clone()
        for _ in range(800):
            motion.step(.02)
            for actual, previous, target in (
                    (motion.speed, previous_speed, motion.target_speed),
                    (motion.curvature, previous_curvature, motion.target_curvature)):
                self.assertTrue((actual >= torch.minimum(previous, target) - 1e-7).all().item())
                self.assertTrue((actual <= torch.maximum(previous, target) + 1e-7).all().item())
            previous_speed = motion.speed.clone()
            previous_curvature = motion.curvature.clone()
        torch.testing.assert_close(motion.speed, motion.target_speed, atol=2e-6, rtol=0)
        # Cascaded float32 updates settle within a few micro-units of the
        # non-binary curvature limit (0.4 / 0.45).
        torch.testing.assert_close(motion.curvature, motion.target_curvature, atol=3e-6, rtol=0)

    def test_defaults_reach_new_speed_and_yaw_limits_with_fast_response(self):
        current = make_motion(num_envs=1)
        previous = make_motion(num_envs=1, minSpeed=.25, maxSpeed=.45, maxYawRate=.4,
                               linearAcceleration=.8, yawAcceleration=1.2)
        self.assertEqual(current.max_speed, .45)
        self.assertEqual(current.max_yaw_rate, .4)
        self.assertAlmostEqual(current.curvature_limit, .4 / .45)
        for motion in (current, previous):
            hold_targets(motion, [motion.max_speed], [motion.curvature_limit])

        dt = .005
        speed_response = [None, None]
        yaw_response = [None, None]
        for step in range(1200):
            elapsed = (step + 1) * dt
            for index, motion in enumerate((current, previous)):
                command = motion.step(dt)
                self.assertLessEqual(command[0, 1].item(), motion.max_speed + 1e-7)
                self.assertLessEqual(command[0, 2].item(), motion.max_yaw_rate + 1e-7)
                if speed_response[index] is None and command[0, 1].item() >= .9 * motion.max_speed:
                    speed_response[index] = elapsed
                if yaw_response[index] is None and command[0, 2].item() >= .9 * motion.max_yaw_rate:
                    yaw_response[index] = elapsed

        self.assertGreater(current.speed.item(), .449)
        self.assertGreater((current.speed * current.curvature).item(), .399)
        for measured_times in (speed_response, yaw_response):
            self.assertTrue(all(value is not None for value in measured_times))
            self.assertLess(measured_times[0], measured_times[1])

    def test_default_driver_keeps_direction_and_speed_band_without_stopping(self):
        motion = make_motion(num_envs=8)
        settings = level03_motion_config({})
        linear_budget = settings["linearAcceleration"]
        yaw_budget = settings["yawAcceleration"]
        dt = .02
        previous = torch.zeros(8, 3)
        previous_direction = motion.driver.direction.clone()
        previous_phase = motion.driver.phase.clone()
        previous_initialized = motion.driver.initialized.clone()
        throttle_only = steering_only = brake_steps = pause_steps = 0
        nearly_full_steps = torch.zeros(8, dtype=torch.long)
        longest_full_run = torch.zeros(8, dtype=torch.long)
        min_speed, max_speed = 0., 0.
        drive_start = torch.zeros(8)
        throttle_adjustments = []

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2026)
            for step in range(6000):
                previous_goal = motion.driver.input_speed.clone()
                previous_target_speed = motion.target_speed.clone()
                previous_target_curvature = motion.target_curvature.clone()
                command = motion.step(dt)
                driver = motion.driver
                self.assertTrue(torch.isfinite(command).all().item())
                self.assertTrue((command[:, 1].abs() <= motion.max_speed + 1e-7).all().item())
                self.assertTrue((command[:, 2].abs() <= motion.max_yaw_rate + 1e-7).all().item())
                torch.testing.assert_close(command[:, 0], torch.zeros(8), atol=0, rtol=0)
                torch.testing.assert_close(command[:, 2], motion.speed * motion.curvature)
                self.assertTrue(((command[:, 1] - previous[:, 1]).abs() <= linear_budget * dt + 1e-7).all().item())
                self.assertTrue(((command[:, 2] - previous[:, 2]).abs() <= yaw_budget * dt + 1e-7).all().item())
                linear_accel, yaw_accel = derivatives(motion)
                self.assertTrue((linear_accel.abs() <= linear_budget + 1e-6).all().item())
                self.assertTrue((yaw_accel.abs() <= yaw_budget + 1e-6).all().item())
                self.assertTrue((motion.target_speed.abs() <= motion.max_speed + 1e-7).all().item())
                self.assertTrue((motion.target_curvature.abs() <= motion.curvature_limit + 1e-7).all().item())
                self.assertTrue(((motion.target_speed - previous_target_speed).abs()
                                 <= 2 * motion.max_speed * dt / settings["inputResponseTime"] + 1e-7).all().item())
                self.assertTrue(((motion.target_curvature - previous_target_curvature).abs()
                                 <= 2 * motion.curvature_limit * dt / settings["inputResponseTime"] + 1e-7).all().item())
                driving = driver.phase == driver.DRIVE
                self.assertTrue(driving.all().item())
                raw_speed_goal = driver.direction * driver.throttle_goal * motion.max_speed
                self.assertTrue((raw_speed_goal.abs() >= settings["minSpeed"] - 1e-7).all().item())
                self.assertTrue((raw_speed_goal.abs() <= motion.max_speed + 1e-7).all().item())
                self.assertTrue((command[:, 1] * driver.direction > 0.).all().item())
                if step == 0:
                    # Starting from rest must cross the excluded low-speed band
                    # continuously instead of jumping to minSpeed.
                    self.assertTrue((command[:, 1].abs() < settings["minSpeed"]).all().item())
                if step * dt >= 6.:
                    # Conservative settling time even for a constant minimum
                    # target; afterwards every continuous DRIVE stays in band.
                    self.assertTrue((command[:, 1].abs() >= settings["minSpeed"] - 1e-5).all().item())
                kept_driving = driving & (previous_phase == driver.DRIVE) & previous_initialized
                torch.testing.assert_close(driver.direction[kept_driving], previous_direction[kept_driving])
                entered_drive = driving & (~previous_initialized | (previous_phase != driver.DRIVE))
                drive_start[entered_drive] = step * dt
                entered_brake = (driver.phase == driver.BRAKE) & (previous_phase == driver.DRIVE)
                self.assertTrue(((step * dt - drive_start[entered_brake]) >= 6. - 2 * dt).all().item())
                throttle_only += (driver.throttle_changed & ~driver.steering_changed).sum().item()
                steering_only += (driver.steering_changed & ~driver.throttle_changed).sum().item()
                brake_steps += (driver.phase == driver.BRAKE).sum().item()
                pause_steps += (driver.phase == driver.PAUSE).sum().item()
                small_adjustment = driver.throttle_changed & kept_driving & ~driver.intent_changed
                throttle_adjustments.extend((driver.input_speed - previous_goal)[small_adjustment].abs().tolist())
                nearly_full = ((command[:, 1].abs() > .98 * motion.max_speed)
                               & (motion.curvature.abs() > .98 * motion.curvature_limit))
                nearly_full_steps = torch.where(nearly_full, nearly_full_steps + 1, 0)
                longest_full_run = torch.maximum(longest_full_run, nearly_full_steps)
                min_speed = min(min_speed, command[:, 1].min().item())
                max_speed = max(max_speed, command[:, 1].max().item())
                previous = command.clone()
                previous_direction = driver.direction.clone()
                previous_phase = driver.phase.clone()
                previous_initialized = driver.initialized.clone()

        self.assertLess(min_speed, -.25)
        self.assertGreater(max_speed, .25)
        self.assertGreater(throttle_only, 100)
        self.assertGreater(steering_only, 100)
        self.assertEqual(brake_steps, 0)
        self.assertEqual(pause_steps, 0)
        self.assertLess(longest_full_run.max().item() * dt, 1.)
        self.assertGreater(len(throttle_adjustments), 100)
        self.assertLess(torch.tensor(throttle_adjustments).median().item(), .1)

    def test_direction_changes_only_after_braking_and_a_complete_near_stationary_pause(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse), torch.random.fork_rng(devices=[]):
                torch.manual_seed(27)
                motion = make_motion(num_envs=1, driveDuration=[2., 2.], stopHoldTime=[.6, .6],
                                     stopProbability=0. if reverse else 1.,
                                     reverseProbability=1. if reverse else 0.)
                driver = motion.driver
                dt = .01
                pause_started = None
                direction_before_pause = None
                complete_pauses = 0
                seen_moving = False
                for step in range(2400):
                    previous_phase = driver.phase.item()
                    previous_direction = driver.direction.item()
                    initialized = driver.initialized.item()
                    command = motion.step(dt)
                    phase = driver.phase.item()
                    seen_moving |= abs(command[0, 1].item()) > .05
                    if initialized and phase != previous_phase:
                        self.assertIn((previous_phase, phase), (
                            (driver.DRIVE, driver.BRAKE),
                            (driver.BRAKE, driver.PAUSE),
                            (driver.PAUSE, driver.DRIVE)))
                        if phase == driver.PAUSE:
                            self.assertLess(abs(motion.speed.item()), .015 + 1e-6)
                            self.assertLess(abs(derivatives(motion)[0].item()), .05 + 1e-6)
                            pause_started = step * dt
                            direction_before_pause = previous_direction
                        elif previous_phase == driver.PAUSE:
                            self.assertIsNotNone(pause_started)
                            self.assertGreaterEqual(step * dt - pause_started, .6 - dt - 1e-6)
                            expected = -direction_before_pause if reverse else direction_before_pause
                            self.assertEqual(driver.direction.item(), expected)
                            complete_pauses += 1
                    if phase == driver.PAUSE:
                        self.assertLess(abs(motion.speed.item()), .015 + 1e-6)
                        self.assertEqual(driver.direction.item(), direction_before_pause)
                self.assertTrue(seen_moving)
                self.assertGreaterEqual(complete_pauses, 2)

    def test_straight_probability_keeps_steering_centered_without_stopping_throttle(self):
        motion = make_motion(num_envs=8, stopProbability=0., reverseProbability=0., straightProbability=1.)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(11)
            for _ in range(500):
                command = motion.step(.02)
                torch.testing.assert_close(command[:, 2], torch.zeros(8), atol=0, rtol=0)
                torch.testing.assert_close(motion.target_curvature, torch.zeros(8), atol=0, rtol=0)
        self.assertTrue((command[:, 1].abs() > .01).all().item())
        self.assertTrue((motion.driver.phase == motion.driver.DRIVE).all().item())

    def test_high_throttle_corrections_stay_bounded_without_accumulating_at_full_pedal(self):
        for variation in (.18, 1.):
            with self.subTest(variation=variation), torch.random.fork_rng(devices=[]):
                torch.manual_seed(31)
                motion = make_motion(num_envs=64, stopProbability=0., reverseProbability=0.,
                                     throttleVariation=variation)
                motion.step(.005)
                driver = motion.driver
                driver.phase_time_left.fill_(1000.)
                driver.cruise_throttle.fill_(1.)
                driver.throttle_goal.fill_(1.)
                full_pedal_samples = 0
                for _ in range(100):
                    driver.throttle_time_left.zero_()
                    motion.step(.005)
                    goals = driver.throttle_goal
                    self.assertTrue(driver.throttle_changed.all().item())
                    self.assertTrue(torch.isfinite(goals).all().item())
                    min_pedal = max(.12, level03_motion_config({})["minSpeed"] / motion.max_speed)
                    self.assertTrue((goals >= min_pedal - 1e-7).all().item())
                    self.assertTrue((goals <= 1. + 1e-7).all().item())
                    full_pedal_samples += (goals >= 1. - 1e-7).sum().item()
                # A noisy high reference may approach the limit, but clipping
                # corrections would incorrectly pile a large fraction at 1.
                self.assertLess(full_pedal_samples / (100 * motion.num_envs), .01)

    def test_manual_boundary_targets_reach_yaw_limit_from_adverse_states_within_five_seconds(self):
        motion = make_motion(num_envs=8)
        speed_sign = torch.tensor([1., 1., -1., -1.] * 2)
        curvature_sign = torch.tensor([1., -1., 1., -1.] * 2)
        hold_targets(motion, speed_sign * motion.max_speed, curvature_sign * motion.curvature_limit)
        # First four begin with the opposite yaw rate. The remaining four must
        # reverse both throttle and steering, passing through zero in between.
        motion.speed.copy_(-motion.target_speed)
        motion.speed_stage.copy_(motion.speed)
        motion.curvature.copy_(motion.target_curvature)
        motion.curvature[4:].neg_()
        motion.curvature_stage.copy_(motion.curvature)
        previous_speed = motion.speed.clone()
        previous_yaw = motion.speed * motion.curvature
        desired_yaw_sign = speed_sign * curvature_sign
        self.assertTrue((previous_yaw[:4] * desired_yaw_sign[:4] < -.399).all().item())

        settings = level03_motion_config({})
        dt = .005
        for _ in range(1000):
            command = motion.step(dt)
            self.assertTrue((command[:, 1].abs() <= motion.max_speed + 1e-7).all().item())
            self.assertTrue((command[:, 2].abs() <= motion.max_yaw_rate + 1e-7).all().item())
            self.assertTrue(((command[:, 1] - previous_speed).abs()
                             <= settings["linearAcceleration"] * dt + 1e-7).all().item())
            self.assertTrue(((command[:, 2] - previous_yaw).abs()
                             <= settings["yawAcceleration"] * dt + 1e-7).all().item())
            previous_speed = command[:, 1].clone()
            previous_yaw = command[:, 2].clone()

        self.assertTrue((command[:, 2] * desired_yaw_sign > .399).all().item())

    def test_partial_reset_clears_only_selected_filter_and_scheduler_state(self):
        motion = make_motion(num_envs=4)
        for _ in range(40):
            motion.step(.02)
        snapshot = state_snapshot(motion)

        motion.reset(torch.tensor([0, 2]))

        for name, actual in state_snapshot(motion).items():
            self.assertEqual(torch.count_nonzero(actual[[0, 2]]).item(), 0, name)
            torch.testing.assert_close(actual[[1, 3]], snapshot[name][[1, 3]], atol=0, rtol=0)
        motion.step(.005)
        self.assertTrue(motion.command_changed[[0, 2]].all().item())
        self.assertTrue(motion.driver.initialized.all().item())


class XYMotionTests(unittest.TestCase):
    def test_default_xy_driver_stays_in_each_signed_band_and_changes_smoothly(self):
        settings = level03_motion_config({})
        motion = RemoteCarMotion(32, "cpu", settings)
        self.assertEqual(motion.translation_axes, "xy")
        self.assertEqual(motion.x_motion.translation_axes, "y")
        previous = torch.zeros(32, 3)
        initial_directions = None
        independent_throttle_updates = 0
        dt = .02
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(44)
            for step in range(1500):
                command = motion.step(dt)
                self.assertTrue(torch.isfinite(command).all().item())
                directions = torch.stack((motion.x_motion.driver.direction, motion.driver.direction), -1)
                if initial_directions is None:
                    initial_directions = directions.clone()
                    self.assertEqual(torch.unique(directions, dim=0).shape[0], 4)
                    self.assertTrue((command[:, :2].abs() < settings["minSpeed"]).all().item())
                torch.testing.assert_close(directions, initial_directions, atol=0, rtol=0)
                self.assertTrue((command[:, :2] * directions > 0.).all().item())
                self.assertTrue((command[:, :2].abs() <= .45 + 1e-7).all().item())
                self.assertTrue((command[:, 2].abs() <= .4 + 1e-7).all().item())
                self.assertTrue(((command[:, :2] - previous[:, :2]).abs()
                                 <= settings["linearAcceleration"] * dt + 1e-7).all().item())
                self.assertTrue(((command[:, 2] - previous[:, 2]).abs()
                                 <= settings["yawAcceleration"] * dt + 1e-7).all().item())
                for axis in (motion.x_motion, motion):
                    driver = axis.driver
                    self.assertTrue((driver.phase == driver.DRIVE).all().item())
                    raw_target = driver.throttle_goal * axis.max_speed
                    self.assertTrue((raw_target >= .25 - 1e-7).all().item())
                    self.assertTrue((raw_target <= .45 + 1e-7).all().item())
                    self.assertTrue((derivatives(axis)[0].abs()
                                     <= settings["linearAcceleration"] + 1e-6).all().item())
                self.assertTrue((derivatives(motion)[1].abs()
                                 <= settings["yawAcceleration"] + 1e-6).all().item())
                if step * dt >= 6.:
                    self.assertTrue((command[:, :2].abs() >= .25 - 1e-5).all().item())
                independent_throttle_updates += (motion.driver.throttle_changed
                                                ^ motion.x_motion.driver.throttle_changed).sum().item()
                previous = command.clone()
        self.assertGreater(independent_throttle_updates, 100)

    def test_xy_manual_targets_are_atomic_and_preserve_both_axes_continuity(self):
        motion = make_motion(2, translationAxes="xy")
        motion.set_manual_targets([.45, -.45], [.5, -.5], x_speed=[-.45, .45])
        for _ in range(60):
            previous = motion.step(.02)
        previous_state = state_snapshot(motion)
        previous_accels = tuple(derivatives(axis)[0].clone() for axis in (motion.x_motion, motion))
        for invalid in (.46, float("nan"), [0., 0., 0.]):
            with self.subTest(x_speed=invalid), self.assertRaisesRegex(ValueError, "x_speed"):
                motion.set_manual_targets(0., 0., x_speed=invalid)
            for name, value in state_snapshot(motion).items():
                torch.testing.assert_close(value, previous_state[name], atol=0, rtol=0)
        motion.set_manual_targets([-.45, .45], [-.5, .5], x_speed=[.45, -.45])
        for axis, expected in zip((motion.x_motion, motion), previous_accels):
            torch.testing.assert_close(derivatives(axis)[0], expected, atol=0, rtol=0)
        torch.testing.assert_close(motion.step(1e-7), previous, atol=2e-7, rtol=0)
        for _ in range(500):
            command = motion.step(.02)
        torch.testing.assert_close(command[:, :2], torch.tensor([[.45, -.45], [-.45, .45]]),
                                   atol=2e-6, rtol=0)
        motion.set_manual_targets(0., 0.)
        self.assertFalse(motion.driver_enabled)
        self.assertFalse(motion.x_motion.driver_enabled)
        torch.testing.assert_close(motion.x_motion.target_speed, torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "x_speed"):
            make_motion().set_manual_targets(0., 0., x_speed=.1)

    def test_xy_partial_reset_clears_both_filters_and_only_selected_drivers(self):
        motion = make_motion(4, translationAxes="xy")
        for _ in range(100):
            motion.step(.02)
        before = state_snapshot(motion)
        motion.reset(torch.tensor([0, 2]))
        for name, value in state_snapshot(motion).items():
            self.assertEqual(torch.count_nonzero(value[[0, 2]]).item(), 0, name)
            torch.testing.assert_close(value[[1, 3]], before[name][[1, 3]], atol=0, rtol=0)
        command = motion.step(.005)
        self.assertTrue(motion.command_changed[[0, 2]].all().item())
        self.assertTrue((command[[0, 2], :2].abs() < .25).all().item())
        self.assertTrue(motion.x_motion.driver.initialized.all().item())

    def test_missing_axis_key_matches_explicit_legacy_y_random_sequence(self):
        settings = level03_motion_config({"level03Motion": {"maxSpeed": .4, "maxYawRate": .3}})
        legacy = dict(settings)
        del legacy["translationAxes"]
        trajectories = []
        for configuration in (legacy, settings):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(44)
                motion = RemoteCarMotion(4, "cpu", configuration)
                self.assertIsNone(motion.x_motion)
                trajectories.append(torch.stack([motion.step(.02) for _ in range(100)]))
        torch.testing.assert_close(*trajectories, atol=0, rtol=0)


class Level03EnvironmentHookTests(unittest.TestCase):
    def test_production_compute_submits_both_local_axes_and_yaw(self):
        env = MotionEnvironment(num_envs=4)
        motion = make_motion(4, translationAxes="xy")
        env.level03_motion = motion
        motion.set_manual_targets([.45, .45, -.45, -.45], motion.curvature_limit,
                                  x_speed=[.45, -.45, .45, -.45])
        for axis in (motion, motion.x_motion):
            for name in ("speed", "speed_stage"):
                getattr(axis, name).copy_(axis.target_speed)
            for name in ("curvature", "curvature_stage"):
                getattr(axis, name).copy_(axis.target_curvature)
        env.compute_table_command_level_03()
        torch.testing.assert_close(env.table_commands[:, [0, 1, 5]], torch.tensor([
            [.45, .45, .4], [-.45, .45, .4], [.45, -.45, -.4], [-.45, -.45, -.4]]))
        torch.testing.assert_close(env.table_commands[:, 4], torch.full((4,), .525))

    def test_production_compute_drives_local_y_and_clears_stale_x_with_both_turn_signs(self):
        env = MotionEnvironment(num_envs=4)
        motion = env.level03_motion
        curvature = .4 / .45
        hold_targets(motion, [.45, .45, -.45, -.45], [curvature, -curvature, curvature, -curvature])
        # Begin at steady targets to check the physical axis/sign contract,
        # including a stale X command left by a prior command source.
        motion.speed.copy_(motion.target_speed)
        motion.speed_stage.copy_(motion.target_speed)
        motion.curvature.copy_(motion.target_curvature)
        motion.curvature_stage.copy_(motion.target_curvature)
        env.table_commands[:, 0] = .17

        env.compute_table_command_level_03()

        torch.testing.assert_close(env.table_commands[:, [0, 1, 5]], torch.tensor([
            [0., .45, .4], [0., .45, -.4], [0., -.45, -.4], [0., -.45, .4]]))
        torch.testing.assert_close(env.table_commands[:, 0], torch.zeros(4), atol=0, rtol=0)
        torch.testing.assert_close(env.table_commands[:, 4], torch.full((4,), .525))

    def test_production_compute_submits_car_commands_using_physics_time_step(self):
        env = MotionEnvironment(physics_dt=.01)
        expected_motion = make_motion()
        for motion in (env.level03_motion, expected_motion):
            hold_targets(motion, [motion.max_speed, -motion.max_speed, 0.],
                         [motion.curvature_limit, motion.curvature_limit / 2, motion.curvature_limit])
        env.table_commands[:, 2] = .05
        expected = expected_motion.step(.01)

        env.compute_table_command_level_03()

        torch.testing.assert_close(env.table_commands[:, [0, 1, 5]], expected)
        torch.testing.assert_close(env.table_commands[:, 2], torch.zeros(3), atol=0, rtol=0)
        torch.testing.assert_close(env.table_commands[:, 4], torch.full((3,), .525))
        torch.testing.assert_close(env.mask_step_change, env.level03_motion.command_changed)

    def test_production_compute_reports_real_random_command_refresh(self):
        env = MotionEnvironment()
        env.compute_table_command_level_03()
        driver = env.level03_motion.driver
        driver.throttle_time_left.copy_(torch.tensor([0., 1., 0.]))
        driver.steering_time_left.fill_(1.)
        driver.turn_time_left.fill_(2.)
        driver.phase_time_left.fill_(3.)

        env.compute_table_command_level_03()

        self.assertEqual(env.mask_step_change.tolist(), [True, False, True])
        torch.testing.assert_close(env.table_commands[:, 0], torch.zeros(3), atol=0, rtol=0)
        linear_budget = level03_motion_config({})["linearAcceleration"]
        self.assertTrue((env.table_commands[:, 1].abs() < linear_budget * env.sim_params.dt).all().item())

    def test_production_partial_reset_clears_motion_but_preserves_height_and_other_envs(self):
        env = TestEnvironment()
        env.table_level = "Level03"
        env.level03_motion = make_motion(4)
        for _ in range(40):
            env.level03_motion.step(.02)
        env.table_commands.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 100)
        before = env.table_commands.clone()
        state_before = state_snapshot(env.level03_motion)
        ids = torch.tensor([0, 2])

        env._reset_env_tensors(ids)

        torch.testing.assert_close(env.table_commands[ids, :2], torch.zeros(2, 2))
        torch.testing.assert_close(env.table_commands[ids, 5], torch.zeros(2))
        torch.testing.assert_close(env.table_commands[ids, 2:5], before[ids, 2:5])
        torch.testing.assert_close(env.table_commands[[1, 3]], before[[1, 3]])
        for name, actual in state_snapshot(env.level03_motion).items():
            self.assertEqual(torch.count_nonzero(actual[ids]).item(), 0, name)
            torch.testing.assert_close(actual[[1, 3]], state_before[name][[1, 3]])

    def test_level02_reset_keeps_original_motion_behavior(self):
        env = TestEnvironment()
        env.table_level = "Level02"
        env.level03_motion = SimpleNamespace(reset=Mock())
        env.table_commands.copy_(torch.arange(24, dtype=torch.float32).reshape(4, 6) / 100)
        before = env.table_commands.clone()

        env._reset_env_tensors(torch.tensor([0, 2]))

        torch.testing.assert_close(env.table_commands, before)
        env.level03_motion.reset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
