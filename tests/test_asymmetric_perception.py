"""Sensor-only M2 geometry, missing-data and delayed Kalman filtering tests."""

from dataclasses import replace
import inspect
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "DQ_high-level"))
from utils.asymmetric_perception import AsymmetricPerception, BELIEF_CONTRACT, PerceptionConfig


def clean_config(**changes):
    cfg = PerceptionConfig(camera_delay_frames=(0, 0), min_valid_pixels=1,
                           depth_noise_std_m=0, depth_noise_distance_scale=0,
                           depth_noise_angular_scale=0, pixel_dropout_prob=0,
                           frame_dropout_prob=0, frame_dropout_distance_scale=0,
                           frame_dropout_angular_scale=0, small_target_dropout_scale=0)
    return replace(cfg, **changes)


def sensor_inputs(batch=1, size=9, depth=2.):
    images = torch.zeros(batch, 2, 2, size, size)
    images[:, :, 0, size // 2, size // 2] = 1
    images[:, :, 1, size // 2, size // 2] = depth
    cameras = torch.eye(4).repeat(batch, 2, 1, 1)
    base = torch.eye(4).repeat(batch, 1, 1)
    k = torch.tensor([[2., 0, size // 2], [0, 2., size // 2], [0, 0, 1.]]).repeat(2, 1, 1)
    return images, cameras, k, base


def offline_kf(observations, now, cfg):
    """Independent NumPy full-history reference, with no fixed-lag storage."""
    observations = sorted(observations, key=lambda item: item[0])
    first_time, first_point = observations[0]
    x = np.r_[first_point, np.zeros(3)]
    p = np.diag([cfg.measurement_std_m ** 2] * 3 + [cfg.initial_velocity_std_mps ** 2] * 3)
    clock = first_time
    h = np.c_[np.eye(3), np.zeros((3, 3))]
    r = np.eye(3) * cfg.measurement_std_m ** 2

    def predict(x, p, dt):
        f = np.eye(6)
        f[:3, 3:] = np.eye(3) * dt
        q = np.kron([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]], np.eye(3))
        return f @ x, f @ p @ f.T + cfg.process_accel_variance * q

    for time, point in observations[1:]:
        x, p = predict(x, p, time - clock)
        gain = p @ h.T @ np.linalg.inv(h @ p @ h.T + r)
        x += gain @ (point - h @ x)
        left = np.eye(6) - gain @ h
        p = left @ p @ left.T + gain @ r @ gain.T
        clock = time
    return predict(x, p, now - clock)


class AsymmetricPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def run_step(self, module, inputs, time, *, capture=None, corrupt=True, omega=None):
        images, cameras, k, base = inputs
        return module.step(images, cameras, k, time if capture is None else capture,
                           base, time, camera_angular_velocity=omega, corrupt=corrupt)

    def test_optical_geometry_and_base_reference_frame(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        inputs[1][:, :, 0, 3] = 1
        inputs[3][:, 0, 3] = .25
        out = self.run_step(module, inputs, 1.)
        torch.testing.assert_close(out["world_position"], torch.tensor([[1., 0., 2.]]))
        torch.testing.assert_close(out["belief"][:, :3], torch.tensor([[.75, 0., 2.]]))
        self.assertTrue(out["initialized"].item())
        self.assertEqual(out["belief"].shape, (1, 16))
        self.assertTrue(out["valid"].all())

    def test_camera_self_motion_does_not_create_target_velocity(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config(camera_delay_frames=(0, 2)))
        for step in range(8):
            images, cameras, k, base = sensor_inputs()
            offset = step % 3
            cameras[:, :, 0, 3] = offset
            images.zero_()
            images[:, :, 0, 4, 4 - offset] = 1
            images[:, :, 1, 4, 4 - offset] = 2
            out = self.run_step(module, (images, cameras, k, base), float(step))
            torch.testing.assert_close(out["world_position"], torch.tensor([[0., 0., 2.]]), atol=1e-6, rtol=0)
            torch.testing.assert_close(out["world_velocity"], torch.zeros(1, 3), atol=1e-6, rtol=0)

    def test_fixed_lag_replay_matches_full_history_with_fractional_sensor_latency(self):
        cfg = clean_config(camera_delay_frames=(0, 2), process_accel_variance=.15)
        module = AsymmetricPerception(1, 9, 9, config=cfg)
        all_observations = []
        capture_points = []
        for step in range(15):
            now = 1. + step * .2
            capture = now - .07  # actual camera latency, before synthetic delay
            point = np.array([.4 * capture, 0., 2.])
            capture_points.append((capture, point))
            inputs = sensor_inputs()
            inputs[1][:, :, 0, 3] = float(point[0])
            out = self.run_step(module, inputs, now, capture=capture)
            all_observations.append(capture_points[step])
            if step >= 2:
                all_observations.append(capture_points[step - 2])
            expected_x, expected_p = offline_kf(all_observations, now, cfg)
            np.testing.assert_allclose(out["world_position"][0].numpy(), expected_x[:3], atol=2e-6)
            np.testing.assert_allclose(out["world_velocity"][0].numpy(), expected_x[3:], atol=3e-6)
            np.testing.assert_allclose(out["covariance"][0].numpy(), expected_p, atol=2e-6)
            self.assertEqual(out["rejected_stale_measurements"].item(), 0)

    def test_large_absolute_timestamps_retain_subsecond_resolution(self):
        cfg = clean_config(camera_delay_frames=(0, 2))
        relative = AsymmetricPerception(1, 9, 9, config=cfg)
        absolute = AsymmetricPerception(1, 9, 9, config=cfg)
        epoch = 1_800_000_000.
        for step in range(12):
            now = .25 * step
            inputs = sensor_inputs()
            inputs[1][:, :, 0, 3] = .1 * step
            expected = self.run_step(relative, inputs, now, capture=now - .125)
            actual = self.run_step(absolute, inputs, epoch + now, capture=epoch + now - .125)
            torch.testing.assert_close(actual["belief"], expected["belief"], rtol=0, atol=2e-6)
            torch.testing.assert_close(actual["covariance"], expected["covariance"], rtol=0, atol=2e-6)
            self.assertEqual(actual["belief"].dtype, torch.float32)
            self.assertEqual(actual["capture_timestamps"].dtype, torch.float64)

    def test_delivered_pixels_and_capture_metadata_have_independent_delays(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config(camera_delay_frames=(1, 3)))
        for step in range(5):
            inputs = sensor_inputs(depth=1 + .1 * step)
            inputs[1][:, :, 0, 3] = step
            out = self.run_step(module, inputs, float(step))
            for camera, delay in enumerate((1, 3)):
                if step < delay:
                    self.assertFalse(out["valid"][0, camera])
                    self.assertEqual(out["images"][0, camera].sum(), 0)
                else:
                    self.assertEqual(out["capture_timestamps"][0, camera], step - delay)
                    self.assertAlmostEqual(float(out["images"][0, camera, 1, 4, 4]), 1 + .1 * (step - delay), places=5)
                    self.assertEqual(out["measurement_world"][0, camera, 0], step - delay)
                    self.assertEqual(out["belief"][0, 11 + camera], delay)

    def test_missing_data_predicts_without_reusing_last_measurement(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        out = self.run_step(module, inputs, 0.)
        previous_variance = out["position_variance"].clone()
        for time in (1., 2., 3., 4.):
            # A held sensor frame retains its capture timestamp. It must not
            # be passed to the filter again as a new measurement.
            out = self.run_step(module, inputs, time, capture=0.)
            self.assertEqual(out["images"].sum(), 0)
            self.assertFalse(out["valid"].any())
            self.assertTrue((out["position_variance"] > previous_variance).all())
            self.assertEqual(out["belief"][0, 14], time)
            previous_variance = out["position_variance"].clone()

    def test_no_new_environment_time_does_not_advance_any_state_or_rng(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config(camera_delay_frames=(0, 2)), seed=7)
        inputs = sensor_inputs()
        first = self.run_step(module, inputs, 1.)
        rng = module.get_rng_state()
        inputs[0][:, :, 1] = 3.
        second = self.run_step(module, inputs, 1.)
        torch.testing.assert_close(first["belief"], second["belief"], rtol=0, atol=0)
        torch.testing.assert_close(first["images"], second["images"], rtol=0, atol=0)
        self.assertTrue(torch.equal(module.get_rng_state(), rng))

    def test_partial_reset_does_not_change_other_environment(self):
        module = AsymmetricPerception(2, 9, 9, config=clean_config())
        inputs = sensor_inputs(batch=2)
        first = self.run_step(module, inputs, torch.tensor([2., 2.]))
        module.reset([0])
        inputs[1][0, :, 0, 3] = 2
        second = self.run_step(module, inputs, torch.tensor([0., 2.]))
        self.assertEqual(second["world_position"][0, 0], 2)
        torch.testing.assert_close(first["belief"][1], second["belief"][1], rtol=0, atol=0)
        torch.testing.assert_close(first["covariance"][1], second["covariance"][1], rtol=0, atol=0)
        module.reset([0])
        # A global render during a partial reset replaces both camera packets,
        # but the wrapper deliberately advances only the reset environment.
        third = self.run_step(module, inputs, torch.tensor([3., 2.]), capture=torch.tensor([3., 3.]))
        torch.testing.assert_close(first["belief"][1], third["belief"][1], rtol=0, atol=0)
        torch.testing.assert_close(first["covariance"][1], third["covariance"][1], rtol=0, atol=0)

    def test_saturated_depth_does_not_become_a_false_surface(self):
        for corrupt in (False, True):
            module = AsymmetricPerception(1, 9, 9, config=clean_config())
            out = self.run_step(module, sensor_inputs(depth=3.), 0., corrupt=corrupt)
            self.assertFalse(out["initialized"].item())
            self.assertEqual(out["belief"][0, 15], 0)
            self.assertEqual(out["images"][:, :, 1].sum(), 0)

    def test_depth_invalidity_retains_mask_and_has_meaningful_valid_fraction(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        inputs[0][:, :, 0, 4, 5] = 1
        inputs[0][:, :, 1, 4, 5] = float("nan")
        out = self.run_step(module, inputs, 0., corrupt=False)
        self.assertAlmostEqual(float(out["belief"][0, 15]), .5)
        self.assertTrue(torch.isfinite(out["belief"]).all())
        self.assertTrue(torch.isfinite(out["images"]).all())
        module.reset()
        inputs[0][:, :, 1] = float("nan")
        out = self.run_step(module, inputs, 0., corrupt=False)
        self.assertFalse(out["initialized"].item())
        torch.testing.assert_close(out["belief"][0, :9], torch.zeros(9))

    def test_action_dependent_dropout_and_contiguous_burst(self):
        cfg = clean_config(frame_dropout_angular_scale=1., max_dropout_probability=1.,
                           burst_length_min=3, burst_length_max=3)
        module = AsymmetricPerception(1, 9, 9, config=cfg, seed=42)
        inputs = sensor_inputs()
        omega = torch.ones(1, 2, 3) * 10
        out = self.run_step(module, inputs, 0., omega=omega)
        self.assertFalse(out["valid"].any())
        for time in (1., 2.):
            out = self.run_step(module, inputs, time, omega=torch.zeros_like(omega))
            self.assertFalse(out["valid"].any())
            self.assertEqual(out["images"].sum(), 0)
        out = self.run_step(module, inputs, 3., omega=torch.zeros_like(omega))
        self.assertTrue(out["valid"].all())

    def test_deployment_disables_synthetic_delay_and_noise(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config(camera_delay_frames=(2, 4),
                                      frame_dropout_prob=1., max_dropout_probability=1.))
        out = self.run_step(module, sensor_inputs(), 2., capture=1.9, corrupt=False)
        self.assertTrue(out["valid"].all())
        self.assertTrue(out["initialized"].item())
        self.assertAlmostEqual(float(out["belief"][0, 14]), .1, places=5)
        self.assertGreater(float(out["position_variance"][0, 0]), module.config.measurement_std_m ** 2)

    def test_outside_history_measurement_is_rejected_not_relabelled_now(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        # The wrist publishes nothing initially; a later old timestamp is new
        # to that stream but older than the replay window.
        inputs[0][:, 1] = 0
        for time in range(6):
            self.run_step(module, inputs, float(time), capture=torch.tensor([[float(time), -1.]]))
        inputs[0][:, 1] = sensor_inputs()[0][:, 1]
        inputs[1][:, 1, 0, 3] = 100
        out = self.run_step(module, inputs, 6., capture=torch.tensor([[6., 0.]]))
        self.assertEqual(out["rejected_stale_measurements"].item(), 1)
        self.assertAlmostEqual(float(out["world_position"][0, 0]), 0.)

    def test_contract_config_json_and_no_privileged_input(self):
        cfg = PerceptionConfig(**json.loads(json.dumps(PerceptionConfig().to_dict())))
        self.assertEqual(cfg, PerceptionConfig())
        json.dumps(BELIEF_CONTRACT)
        self.assertEqual(BELIEF_CONTRACT["dim"], 16)
        self.assertEqual(BELIEF_CONTRACT["fields"][-1]["slice"], [15, 16])
        names = inspect.signature(AsymmetricPerception.step).parameters
        self.assertFalse(any("object" in name or "ground_truth" in name for name in names))
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        inputs[1][:, :, 0, 3] = 100
        out = self.run_step(module, inputs, 0.)
        self.assertEqual(out["world_position"][0, 0], 100)
        self.assertEqual(out["belief"][0, 0], module.config.belief_position_clip_m)

    def test_rejects_time_reversal_future_capture_and_mode_mix(self):
        module = AsymmetricPerception(1, 9, 9, config=clean_config())
        inputs = sensor_inputs()
        self.run_step(module, inputs, 1.)
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.run_step(module, inputs, 0.)
        with self.assertRaisesRegex(ValueError, "later than now"):
            self.run_step(module, inputs, 2., capture=3.)
        with self.assertRaisesRegex(ValueError, "changing corruption"):
            self.run_step(module, inputs, 2., corrupt=False)
        module.reset()
        self.assertTrue(self.run_step(module, inputs, 0., corrupt=False)["initialized"].item())

    def test_config_rejects_nonfinite_numbers(self):
        for name in PerceptionConfig.__dataclass_fields__:
            for invalid in (float("nan"), float("inf"), -float("inf")):
                value = (invalid, 0) if name == "camera_delay_frames" else invalid
                with self.subTest(name=name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        PerceptionConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
