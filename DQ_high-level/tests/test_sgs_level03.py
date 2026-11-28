"""Level03 bank motion snapshots preserve smooth starts and isolate reset slots."""

import unittest

import torch

from utils.config import level03_motion_config
from utils.sgs_level03 import (MOTION_FLOAT_NAMES, prepare_rest_motion,
                               snapshot_motion, restore_motion, validate_motion_state)
from utils.table_motion import RemoteCarDriver, RemoteCarMotion


class Level03BankMotionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.settings = level03_motion_config({})

    def motion(self, count=8):
        return RemoteCarMotion(count, "cpu", self.settings)

    def test_sampled_rest_start_preserves_intentions_and_accelerates_smoothly(self):
        motion = self.motion(64)
        prepare_rest_motion(motion, .005)
        state = snapshot_motion(motion)
        validate_motion_state(state, self.settings)
        for axis in (motion, motion.x_motion):
            for name in MOTION_FLOAT_NAMES:
                self.assertEqual(torch.count_nonzero(getattr(axis, name)).item(), 0)
            self.assertEqual(torch.count_nonzero(axis.driver.input_speed).item(), 0)
            self.assertEqual(torch.count_nonzero(axis.driver.input_curvature).item(), 0)
            self.assertTrue(axis.driver.initialized.all().item())
            self.assertTrue((axis.driver.phase == RemoteCarDriver.DRIVE).all().item())
            self.assertEqual(set(axis.driver.direction.tolist()), {-1., 1.})
            self.assertTrue((axis.driver.throttle_goal > 0.).all().item())
        command = motion.step(.005)
        self.assertTrue((command[:, 0] * motion.x_motion.driver.direction > 0.).all().item())
        self.assertTrue((command[:, 1] * motion.driver.direction > 0.).all().item())
        self.assertLess(command[:, :2].abs().max().item(), self.settings["linearAcceleration"] * .005)
        self.assertLess(command[:, 2].abs().max().item(), self.settings["yawAcceleration"] * .005)

    def test_round_trip_restores_full_history_and_future_commands(self):
        original, restored = self.motion(), self.motion()
        prepare_rest_motion(original, .005)
        # Include running filter state, not just the all-zero construction start.
        for _ in range(120):
            original.step(.005)
        saved = snapshot_motion(original)
        validate_motion_state(saved, self.settings)
        restore_motion(restored, saved)
        for key, value in snapshot_motion(restored).items():
            torch.testing.assert_close(value, saved[key], atol=0, rtol=0)
        # Future corrections should match when given the same future RNG state;
        # production resets deliberately do not reset the global RNG.
        rng = torch.get_rng_state()
        expected = torch.stack([original.step(.005) for _ in range(400)])
        torch.set_rng_state(rng)
        actual = torch.stack([restored.step(.005) for _ in range(400)])
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_partial_restore_keeps_unselected_slots_and_copies_bank_rows(self):
        original, restored = self.motion(3), self.motion(8)
        prepare_rest_motion(original, .005)
        prepare_rest_motion(restored, .005)
        for _ in range(40):
            restored.step(.005)
        before = snapshot_motion(restored)
        saved = snapshot_motion(original)
        ids = torch.tensor([5, 1, 7])
        restore_motion(restored, saved, ids)
        after = snapshot_motion(restored)
        for key, value in after.items():
            torch.testing.assert_close(value[ids], saved[key], atol=0, rtol=0)
            torch.testing.assert_close(value[[0, 2, 3, 4, 6]], before[key][[0, 2, 3, 4, 6]], atol=0, rtol=0)
        original.speed.fill_(.2)
        self.assertTrue((saved["level03_motion_speed"] == 0).all().item())

    def test_restore_shape_or_dtype_error_does_not_partially_mutate(self):
        source, destination = self.motion(3), self.motion(4)
        prepare_rest_motion(source, .005)
        prepare_rest_motion(destination, .005)
        initial = snapshot_motion(destination)
        for key, value in (("level03_driver_phase", torch.zeros(3)),
                           ("level03_driver_direction", torch.ones(4)),
                           ("level03_x_driver_phase", torch.zeros(3)),
                           ("level03_x_driver_direction", torch.ones(4))):
            state = snapshot_motion(source)
            state[key] = value
            with self.assertRaises(ValueError):
                restore_motion(destination, state, torch.tensor([0, 1, 3]))
            for name, actual in snapshot_motion(destination).items():
                torch.testing.assert_close(actual, initial[name], atol=0, rtol=0)

    def test_validation_rejects_missing_unknown_and_corrupt_fields(self):
        motion = self.motion()
        prepare_rest_motion(motion, .005)
        state = snapshot_motion(motion)
        mutations = (
            ("level03_driver_input_speed", torch.full((8,), self.settings["maxSpeed"] + .001)),
            ("level03_motion_curvature", torch.full((8,), self.settings["maxYawRate"] / self.settings["maxSpeed"] + .001)),
            ("level03_motion_speed_stage", torch.full((8,), float("nan"))),
            ("level03_driver_phase", torch.full((8,), 3, dtype=torch.long)),
            ("level03_driver_phase", torch.zeros(8)),
            ("level03_driver_direction", torch.zeros(8)),
            ("level03_driver_throttle_time_left", torch.full((8,), .81)),
            ("level03_driver_throttle_goal", torch.full((8,), -0.1)),
            ("level03_motion_manual_pending", torch.ones(8, dtype=torch.bool)),
            ("level03_driver_typo", torch.zeros(8)),
            ("level03_x_motion_speed", torch.full((8,), .46)),
            ("level03_x_driver_throttle_goal", torch.full((8,), .1)),
            ("level03_x_driver_initialized", torch.zeros(8, dtype=torch.bool)),
        )
        for key, value in mutations:
            with self.subTest(field=key):
                invalid = dict(state, **{key: value})
                with self.assertRaises(ValueError):
                    validate_motion_state(invalid, self.settings)
        del state["level03_driver_input_curvature"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_motion_state(state, self.settings)

    def test_manual_mode_is_not_silently_saved_and_prepare_enables_driver(self):
        motion = self.motion()
        motion.set_manual_targets(.4, .75)
        with self.assertRaisesRegex(ValueError, "automatic driver"):
            snapshot_motion(motion)
        prepare_rest_motion(motion, .005)
        self.assertTrue(motion.driver_enabled)
        validate_motion_state(snapshot_motion(motion), self.settings)

    def test_xy_requires_x_state_and_legacy_y_does_not_accept_it(self):
        xy = self.motion()
        prepare_rest_motion(xy, .005)
        state = snapshot_motion(xy)
        y_settings = dict(self.settings, translationAxes="y")
        y = RemoteCarMotion(8, "cpu", y_settings)
        y_state = {name: value for name, value in state.items()
                   if not name.startswith("level03_x_")}
        validate_motion_state(y_state, y_settings)
        restore_motion(y, y_state)
        self.assertTrue(bool((y.step(.005)[:, 0] == 0).all()))
        for wrong_state, settings in ((y_state, self.settings), (state, y_settings)):
            with self.assertRaisesRegex(ValueError, "invalid Level03 SGS fields"):
                validate_motion_state(wrong_state, settings)
        for destination, wrong_state in ((xy, y_state), (y, state)):
            with self.assertRaisesRegex(ValueError, "translation axes"):
                restore_motion(destination, wrong_state)

    def test_x_manual_driver_cannot_be_saved_and_prepare_restores_both(self):
        motion = self.motion()
        motion.x_motion.set_manual_targets(.4, 0.)
        with self.assertRaisesRegex(ValueError, "automatic driver"):
            snapshot_motion(motion)
        prepare_rest_motion(motion, .005)
        self.assertTrue(motion.x_motion.driver_enabled)
        validate_motion_state(snapshot_motion(motion), self.settings)

    def test_running_driver_and_expired_timers_remain_valid(self):
        self.settings = level03_motion_config({"level03Motion": {
            "stopProbability": .15, "reverseProbability": .35}})
        motion = self.motion(16)
        observed = set()
        for step in range(2000):
            motion.step(.02)
            if step % 25 == 0:
                validate_motion_state(snapshot_motion(motion), self.settings)
            observed.update(motion.driver.phase.tolist())
        self.assertEqual(observed, {RemoteCarDriver.DRIVE, RemoteCarDriver.BRAKE, RemoteCarDriver.PAUSE})

    def test_driving_goals_obey_floor_without_clamping_rest_or_braking_states(self):
        self.settings = level03_motion_config({"level03Motion": {
            "minSpeed": .25, "maxSpeed": .45, "maxYawRate": .4}})
        motion = self.motion()
        # Uninitialized/rest input and both motion filter stages remain zero.
        validate_motion_state(snapshot_motion(motion), self.settings)
        prepare_rest_motion(motion, .005)
        state = snapshot_motion(motion)
        validate_motion_state(state, self.settings)
        for name in ("cruise_throttle", "throttle_goal"):
            invalid = {key: value.clone() for key, value in state.items()}
            invalid["level03_driver_" + name][0] = .2 / .45
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "below minSpeed"):
                validate_motion_state(invalid, self.settings)
        # Stop/reversal-enabled legacy configurations may deliberately brake
        # through the excluded cruise interval; those states must stay valid.
        for phase in (RemoteCarDriver.BRAKE, RemoteCarDriver.PAUSE):
            stopped = {key: value.clone() for key, value in state.items()}
            stopped["level03_driver_phase"].fill_(phase)
            stopped["level03_driver_throttle_goal"].zero_()
            stopped["level03_driver_cruise_throttle"].zero_()
            validate_motion_state(stopped, self.settings)


if __name__ == "__main__":
    unittest.main()
