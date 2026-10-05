"""Sensor-only deployment matches the non-corrupted training camera path."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from modules.asymmetric_teacher import AsymmetricTeacherPolicy, PROPRIO_INDICES, packed_obs_dim
from modules.teacher_vision import PACKED_OBS_DIM, PRIVILEGED_OBS_DIM
from utils.asymmetric_runtime import M2DeploymentRuntime
from utils.asymmetric_teacher_preprocessor import export_deployment
from utils.asymmetric_teacher_wrapper import AsymmetricTeacherWrapper
from utils.teacher_vision_preprocessor import PrefixRunningStandardScaler
from test_asymmetric_teacher_wrapper import TaskCameraEnv, camera_settings


class M2DeploymentRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1701)
        cls.policy = AsymmetricTeacherPolicy((packed_obs_dim(True, True),), (9,), m1=True, m2=True)
        cls.policy.eval()
        cls.scaler = PrefixRunningStandardScaler(size=packed_obs_dim(True, True), device="cpu")
        with torch.no_grad():
            cls.policy.actor.visual_adapter.weight.normal_(std=.02)
            cls.policy.actor.belief_adapter.weight.normal_(std=.02)
            cls.scaler.running_mean.copy_(torch.randn(1276, dtype=torch.float64) * .1)
            cls.scaler.running_variance.copy_(torch.rand(1276, dtype=torch.float64) + .2)
        cls.settings = camera_settings(weight=0)
        cls.settings["perception_corruption"] = False
        cls.settings["perception"].update(measurement_std_m=.12, process_accel_variance=.2,
                                          initial_velocity_std_mps=.8)
        cls.payload = export_deployment(cls.policy, cls.scaler)
        cls.payload["metadata"]["perception_config"] = deepcopy(cls.settings["perception"])

    def setUp(self):
        self.raw = TaskCameraEnv()
        self.wrapper = AsymmetricTeacherWrapper(self.raw, self.settings)
        self.runtime = M2DeploymentRuntime(self.payload, num_envs=2)

    def assert_runtime_matches(self, packed):
        packet = self.raw.get_asymmetric_sensor_packet()
        proprio = packed[:, list(PROPRIO_INDICES)]
        actions = self.runtime.step(**packet, raw_proprio=proprio)
        with torch.no_grad():
            expected = self.policy.compute({"states": self.scaler(packed)}, "policy")[0]
        torch.testing.assert_close(actions, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(self.runtime.history.flatten(1),
                                   packed[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM], atol=0, rtol=0)
        torch.testing.assert_close(self.runtime.last_perception["belief"], packed[:, -16:], atol=0, rtol=0)
        return actions

    def test_multistep_sensor_path_matches_exported_actor_and_training_wrapper(self):
        self.assertEqual(self.runtime.perception.config.measurement_std_m, .12)
        packed, _ = self.wrapper.reset()
        previous = self.assert_runtime_matches(packed)
        for step in range(4):
            self.raw.sensor_images[:, 0, 1] = 1.1 + step * .03
            self.raw.sensor_images[:, 1, 1] = 1.6 + step * .02
            self.raw.camera_transforms[:, :, 0, 3] += .01
            self.raw.base_transform[:, 1, 3] += .005
            packed, *_ = self.wrapper.step(torch.zeros(2, 9))
            actions = self.assert_runtime_matches(packed)
            self.assertFalse(torch.equal(actions, previous))
            self.assertEqual(actions.shape, (2, 9))
            self.assertFalse(actions.requires_grad)
            previous = actions

    def test_repeated_clock_does_not_advance_history_or_reassimilate_frame(self):
        packed, _ = self.wrapper.reset()
        self.assert_runtime_matches(packed)
        packed, *_ = self.wrapper.step(torch.zeros(2, 9))
        first = self.assert_runtime_matches(packed)
        history = self.runtime.history.clone()
        covariance = self.runtime.last_perception["covariance"].clone()
        # A duplicate sensor call may carry a changed buffer, but the same
        # timestamps must not turn it into a second observation.
        self.raw.sensor_images[:, :, 1] += .2
        repeated, _ = self.wrapper.reset()
        second = self.assert_runtime_matches(repeated)
        torch.testing.assert_close(second, first, atol=0, rtol=0)
        torch.testing.assert_close(self.runtime.history, history, atol=0, rtol=0)
        torch.testing.assert_close(self.runtime.last_perception["covariance"], covariance, atol=0, rtol=0)

    def test_partial_reset_restarts_one_camera_filter_without_changing_other_robot(self):
        packed, _ = self.wrapper.reset()
        self.assert_runtime_matches(packed)
        for _ in range(3):
            packed, *_ = self.wrapper.step(torch.zeros(2, 9))
            self.assert_runtime_matches(packed)
        other_history = self.runtime.history[1].clone()
        other_covariance = self.runtime.last_perception["covariance"][1].clone()
        self.raw.reset_buf[0] = True
        self.raw.sensor_images[0, :, 1] = .8
        packed, _ = self.wrapper.reset()
        self.runtime.reset(torch.tensor([0]))
        self.assert_runtime_matches(packed)
        torch.testing.assert_close(self.runtime.history[1], other_history, atol=0, rtol=0)
        torch.testing.assert_close(self.runtime.last_perception["covariance"][1], other_covariance, atol=0, rtol=0)
        self.assertEqual(float(self.runtime.last_now[0]), 0.)
        self.assertGreater(float(self.runtime.last_now[1]), 0.)
        packed, *_ = self.wrapper.step(torch.zeros(2, 9))
        self.assert_runtime_matches(packed)

    def test_file_payload_load_and_full_reset_match_fresh_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.save(self.payload, path)
            loaded = M2DeploymentRuntime(path, num_envs=2)
        packed, _ = self.wrapper.reset()
        self.assert_runtime_matches(packed)
        packed, *_ = self.wrapper.step(torch.zeros(2, 9))
        self.assert_runtime_matches(packed)
        self.runtime.reset()
        self.raw.reset_idx()
        self.wrapper.reset_runtime()
        packed, _ = self.wrapper.reset()
        restarted = self.assert_runtime_matches(packed)
        expected = loaded.step(**self.raw.get_asymmetric_sensor_packet(),
                               raw_proprio=packed[:, list(PROPRIO_INDICES)])
        torch.testing.assert_close(restarted, expected, atol=0, rtol=0)
        torch.testing.assert_close(self.runtime.history, loaded.history, atol=0, rtol=0)

    def test_m0_payload_and_backward_clock_without_reset_are_rejected(self):
        m0 = AsymmetricTeacherPolicy((PACKED_OBS_DIM,), (9,))
        with self.assertRaisesRegex(ValueError, "requires an M2"):
            M2DeploymentRuntime(export_deployment(m0, self.scaler), num_envs=2)
        packed, _ = self.wrapper.reset()
        self.assert_runtime_matches(packed)
        packed, *_ = self.wrapper.step(torch.zeros(2, 9))
        self.assert_runtime_matches(packed)
        packet = self.raw.get_asymmetric_sensor_packet()
        packet["now"][0] = 0
        packet["timestamps"][0] = 0
        with self.assertRaisesRegex(ValueError, "reset"):
            self.runtime.step(**packet, raw_proprio=packed[:, list(PROPRIO_INDICES)])


if __name__ == "__main__":
    unittest.main()
