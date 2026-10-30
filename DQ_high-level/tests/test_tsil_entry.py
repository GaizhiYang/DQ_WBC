"""Temporal teacher configuration, portable checkpoints, and evaluation clocks."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import get_params, configure_teacher_experiment
from utils.tsil_checkpoint import (TemporalCheckpointMetadata, read_temporal_checkpoint,
                                   restore_temporal_arguments, reject_temporal_student_teacher,
                                   VALUE_ADAPTER_BUFFERS, CONFIGURATION_DESCRIPTOR_SCHEMA)


class TemporalEntryTests(unittest.TestCase):
    def test_complete_method_defaults_and_chunk_override(self):
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.touch()
            args = get_params(["--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                               "--sgs_mode", "sgs", "--sgs_bank", str(bank), "--tsil",
                               "--tsil_micro_batch_size", "17"])
            settings = configure_teacher_experiment({"env": {}}, args)["env"]["tsil"]
            self.assertTrue(settings["enabled"])
            self.assertEqual(settings["top_k"], 5)
            self.assertEqual(settings["samples_per_trajectory"], 32)
            self.assertEqual(settings["micro_batch_size"], 17)
            self.assertEqual(settings["coefficient"], .05)
            self.assertEqual(settings["value_coefficient"], .05)

    def test_baseline_default_and_identity_requirement(self):
        self.assertFalse(configure_teacher_experiment({"env": {}}, get_params([]))["env"]["tsil"]["enabled"])
        with self.assertRaisesRegex(ValueError, "configuration identities"):
            configure_teacher_experiment({"env": {}}, get_params(["--tsil", "--task", "B1Z1PickMulti"]))

    def test_portable_checkpoint_requires_all_temporal_components(self):
        args = get_params(["--roboinfo", "--rand_control", "--future_pose_obs", "--no_velocity_obs"])
        cfg = {"env": {"posePrediction": {"enabled": True}, "holdSteps": 10}, "sim": {"dt": .005}}
        metadata = TemporalCheckpointMetadata(cfg, args, 1286, 16.)
        payload = {name: {} for name in ("policy", "value", "state_preprocessor", "value_preprocessor",
                                        "tsil_state", "reward_normalizer")}
        payload["tsil_metadata"] = metadata.state_dict()
        payload["value_preprocessor"] = {name: torch.ones(1) for name in VALUE_ADAPTER_BUFFERS}
        payload["tsil_state"] = {"version": 2, "descriptor_schema": CONFIGURATION_DESCRIPTOR_SCHEMA}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            torch.save(payload, path)
            saved = read_temporal_checkpoint(path)
            restored = get_params(["--seed", "77"])
            restore_temporal_arguments(restored, saved)
            self.assertEqual(restored.seed, 77)
            self.assertTrue(restored.tsil and restored.rand_control and restored.no_velocity_obs)
            with self.assertRaisesRegex(ValueError, "student entrypoint"):
                reject_temporal_student_teacher(path)
            del payload["reward_normalizer"]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Incomplete TSIL"):
                read_temporal_checkpoint(path)
            torch.save({"policy": {}, "state_preprocessor": {"running_mean": torch.zeros(2)}}, path)
            self.assertIsNone(read_temporal_checkpoint(path))

    def test_new_checkpoint_rejects_missing_adapter_and_mixed_descriptor_schema(self):
        metadata = TemporalCheckpointMetadata({"env": {}}, get_params([]), 1286, 16.)
        payload = {name: {} for name in ("policy", "value", "state_preprocessor", "reward_normalizer")}
        payload.update(tsil_metadata=metadata.state_dict(),
                       value_preprocessor={name: torch.ones(1) for name in VALUE_ADAPTER_BUFFERS},
                       tsil_state={"version": 2, "descriptor_schema": CONFIGURATION_DESCRIPTOR_SCHEMA})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            for key in VALUE_ADAPTER_BUFFERS:
                broken = deepcopy(payload)
                del broken["value_preprocessor"][key]
                torch.save(broken, path)
                with self.assertRaisesRegex(ValueError, "prediction adapter"):
                    read_temporal_checkpoint(path)
            for field, wrong in (("version", 1), ("descriptor_schema", "generic_v1")):
                broken = deepcopy(payload)
                broken["tsil_state"][field] = wrong
                torch.save(broken, path)
                with self.assertRaisesRegex(ValueError, "descriptor state"):
                    read_temporal_checkpoint(path)

    def test_legacy_checkpoint_migration_persists_actor_only_restriction(self):
        metadata = TemporalCheckpointMetadata({"env": {}}, get_params([]), 1286, 16.)
        legacy = metadata.state_dict()
        legacy["version"] = 1
        del legacy["value_normalization"], legacy["configuration_descriptor"]
        metadata.load_state_dict(legacy)
        migrated = metadata.state_dict()
        self.assertEqual(migrated["version"], 2)
        self.assertEqual(migrated["migrated_from_version"], 1)
        self.assertFalse(migrated["critic_available"])
        metadata.load_state_dict(migrated)
        self.assertFalse(metadata.state_dict()["critic_available"])

    def test_checkpoint_rejects_geometry_or_time_incompatible_runtime(self):
        cfg = {"env": {"posePrediction": {"enabled": True, "horizons": [.6, .9]}, "holdSteps": 10}}
        model = TemporalCheckpointMetadata(cfg, get_params([]), 1286, 16.)
        saved = model.state_dict()
        model.load_state_dict(saved)
        broken = deepcopy(saved)
        broken["time_reference"] = 14.
        with self.assertRaisesRegex(ValueError, "time reference"):
            model.load_state_dict(broken)
        broken = deepcopy(saved)
        broken["config"]["env"]["posePrediction"]["horizons"] = [.2, .5]
        with self.assertRaisesRegex(ValueError, "posePrediction"):
            model.load_state_dict(broken)


if __name__ == "__main__":
    unittest.main()
