"""Exercise Level03 bank validation and the production SGS load/reset paths."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import test_sgs_env as env_test
from utils.config import level03_motion_config
from utils.sgs import ConfigurationBank
from utils.sgs_level03 import prepare_rest_motion, snapshot_motion
from utils.table_motion import RemoteCarMotion


def legacy_motion_settings():
    return level03_motion_config({"level03Motion": dict(
        minSpeed=0., maxSpeed=.4, maxYawRate=.3, stopProbability=.15,
        reverseProbability=.35, cruiseThrottleRange=[.25, .95])})


def level03_bank(settings=None, legacy=False):
    base = env_test.make_active_near_goal_bank()
    settings = deepcopy(level03_motion_config({}) if settings is None else settings)
    motion = RemoteCarMotion(base.num_configs, "cpu", settings)
    prepare_rest_motion(motion, .005)
    tensors = {name: value.clone() for name, value in base.tensors.items()}
    tensors.update(snapshot_motion(motion))
    tensors["table_vy"].zero_()
    tensors["table_yaw_rate"].zero_()
    tensors["rotation_enabled"].zero_()
    metadata = deepcopy(base.metadata)
    metadata.update(table_level="Level03", level03_motion_config=settings,
                    level03_reset_mode="rest_with_sampled_driver_intention")
    if legacy:
        del metadata["level03_motion_config"]["minSpeed"]
        del metadata["level03_motion_config"]["translationAxes"]
    return ConfigurationBank(tensors, metadata)


class Level03BankValidationTests(unittest.TestCase):
    def test_roundtrip_keeps_driver_flags_and_configuration_identity(self):
        bank = level03_bank()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level03.pt"
            bank.save(path)
            restored = ConfigurationBank.load(path, expected_metadata={
                "table_level": "Level03", "level03_motion_config": level03_motion_config({})})
        self.assertEqual(restored.fingerprint, bank.fingerprint)
        self.assertEqual(restored.tensors["level03_driver_phase"].dtype, torch.long)
        self.assertEqual(restored.tensors["level03_driver_initialized"].dtype, torch.bool)
        self.assertEqual(restored.tensors["level03_x_driver_phase"].dtype, torch.long)
        self.assertEqual(restored.tensors["level03_x_driver_initialized"].dtype, torch.bool)

    def test_level03_rejects_missing_motion_state_and_incomplete_configuration(self):
        bank = level03_bank()
        missing = dict(bank.tensors)
        del missing["level03_driver_direction"]
        with self.assertRaisesRegex(ValueError, "level03_driver_direction"):
            ConfigurationBank(missing, bank.metadata)
        metadata = deepcopy(bank.metadata)
        del metadata["level03_motion_config"]["maxSpeed"]
        with self.assertRaisesRegex(ValueError, "complete level03_motion_config"):
            ConfigurationBank(bank.tensors, metadata)
        metadata["level03_motion_config"]["maxSpeed"] = -1
        with self.assertRaisesRegex(ValueError, "maxSpeed"):
            ConfigurationBank(bank.tensors, metadata)

    def test_legacy_zero_floor_bank_roundtrip_does_not_rewrite_provenance(self):
        bank = level03_bank(legacy_motion_settings(), legacy=True)
        fingerprint = bank.fingerprint
        metadata = deepcopy(bank.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            bank.save(path)
            restored = ConfigurationBank.load(path)
        self.assertEqual(restored.metadata, metadata)
        self.assertNotIn("minSpeed", restored.metadata["level03_motion_config"])
        self.assertNotIn("translationAxes", restored.metadata["level03_motion_config"])
        self.assertEqual(restored.fingerprint, fingerprint)
        # Only the historically absent floor/axes are optional; all prior fields
        # remain required even if the latest defaults happen to fill them.
        del metadata["level03_motion_config"]["throttleVariation"]
        with self.assertRaisesRegex(ValueError, "complete level03_motion_config"):
            ConfigurationBank(bank.tensors, metadata)

    def test_rest_mode_rejects_velocity_or_uninitialized_intention(self):
        bank = level03_bank()
        for name in ("level03_motion_speed", "level03_driver_input_speed", "table_vy",
                     "level03_x_motion_speed", "level03_x_driver_input_speed"):
            tensors = {key: value.clone() for key, value in bank.tensors.items()}
            tensors[name][0] = .01
            with self.subTest(field=name), self.assertRaisesRegex(ValueError, "nonzero motion"):
                ConfigurationBank(tensors, bank.metadata)
        tensors = {key: value.clone() for key, value in bank.tensors.items()}
        tensors["table_root"][0, 8] = .1
        with self.assertRaisesRegex(ValueError, "planar velocity"):
            ConfigurationBank(tensors, bank.metadata)
        tensors = {key: value.clone() for key, value in bank.tensors.items()}
        tensors["level03_driver_phase"][0] = 1
        with self.assertRaisesRegex(ValueError, "driving intention"):
            ConfigurationBank(tensors, bank.metadata)

    def test_vertical_physics_drift_does_not_mean_planar_motion(self):
        bank = level03_bank()
        tensors = {key: value.clone() for key, value in bank.tensors.items()}
        tensors["table_root"][:, 9] = -.049
        ConfigurationBank(tensors, bank.metadata)


class Level03SGSRuntimeTests(unittest.TestCase):
    initialize_sgs = staticmethod(env_test.source_method(
        "b1z1_pickmulti.py", "B1Z1PickMulti", "_init_sgs"))

    def environment(self, path, mode="sgs"):
        env = env_test.SGSInitializationTests().environment(path, mode)
        env.table_level = "Level03"
        env.level03_motion_cfg = level03_motion_config({})
        env.cfg["env"]["sgs"]["bank_size"] = None
        return env

    def test_both_sampler_modes_load_level03_with_matching_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level03.pt"
            bank = level03_bank()
            bank.save(path)
            for mode in ("sgs", "uniform"):
                env = self.environment(path, mode)
                with self.subTest(mode=mode), redirect_stdout(io.StringIO()):
                    self.initialize_sgs(env)
                self.assertTrue(env.sgs_enabled)
                self.assertEqual(env.sgs_bank.fingerprint, bank.fingerprint)

    def test_level03_rejects_level02_bank_or_different_speed_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.pt"
            env_test.make_active_near_goal_bank().save(path)
            with self.assertRaisesRegex(ValueError, "table_level"):
                self.initialize_sgs(self.environment(path))
            level03_bank().save(path)
            env = self.environment(path)
            env.level03_motion_cfg["maxSpeed"] = .3
            with self.assertRaisesRegex(ValueError, "level03_motion_config.maxSpeed"):
                self.initialize_sgs(env)

    def test_legacy_bank_loads_only_with_matching_legacy_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            settings = legacy_motion_settings()
            bank = level03_bank(settings, legacy=True)
            bank.save(path)
            env = self.environment(path)
            env.level03_motion_cfg = settings
            with redirect_stdout(io.StringIO()):
                self.initialize_sgs(env)
            self.assertEqual(env.sgs_bank.fingerprint, bank.fingerprint)
            self.assertNotIn("minSpeed", env.sgs_bank.metadata["level03_motion_config"])
            for changed, value in (("minSpeed", .25), ("maxSpeed", .45), ("maxYawRate", .4),
                                   ("stopProbability", 0.), ("reverseProbability", 0.),
                                   ("translationAxes", "xy")):
                env = self.environment(path)
                env.level03_motion_cfg = dict(settings, **{changed: value})
                with self.subTest(parameter=changed), self.assertRaisesRegex(
                        ValueError, "level03_motion_config." + changed):
                    self.initialize_sgs(env)

    def test_level03_rejects_level02_only_margin_curricula(self):
        for mode in ("prior", "frontier"):
            env = self.environment("unused.pt")
            env.cfg["env"]["margin"] = {"curriculum": mode}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "margin prior/frontier"):
                self.initialize_sgs(env)

    def test_real_reset_restores_motion_without_touching_other_environments(self):
        bank = level03_bank()
        env = env_test.TestEnvironment(bank=bank)
        env.table_level = "Level03"
        settings = level03_motion_config({})
        env.level03_motion = RemoteCarMotion(env.num_envs, "cpu", settings)
        for _ in range(100):
            env.level03_motion.step(.005)
        env.initialize(ids=(1, 3, 0, 4))
        restored = snapshot_motion(env.level03_motion)
        for name, value in restored.items():
            torch.testing.assert_close(value, bank.tensors[name][[1, 3, 0, 4]])
        for _ in range(60):
            env.level03_motion.step(.005)
        previous = snapshot_motion(env.level03_motion)
        with patch.object(env.sgs_sampler, "sample", return_value=torch.tensor([5, 2])):
            env._reset_sgs_envs(torch.tensor([0, 2]))
        current = snapshot_motion(env.level03_motion)
        for name, value in current.items():
            torch.testing.assert_close(value[[0, 2]], bank.tensors[name][[5, 2]])
            torch.testing.assert_close(value[[1, 3]], previous[name][[1, 3]])
        first = env.level03_motion.step(.005)[[0, 2]]
        self.assertTrue(bool((first[:, 0].abs() > 0).all()))
        self.assertTrue(bool((first[:, :2].abs() <= settings["linearAcceleration"] * .005).all()))
        self.assertTrue(bool((first[:, 2].abs() <= settings["yawAcceleration"] * .005).all()))


if __name__ == "__main__":
    unittest.main()
