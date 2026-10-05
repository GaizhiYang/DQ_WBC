"""Resolve asymmetric experiment provenance and reject invalid resume CLI."""

from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
sys.path.insert(0, str(ROOT / "DQ_high-level"))

from train_multistate_DQ_asymmetric_teacher import configure_experiment, get_trainer
from utils.config import get_params, load_cfg


class AsymmetricTeacherEntryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "m0.yaml"
        self.config = load_cfg(str(ROOT / "DQ_high-level/data/cfg/DQ_asymmetric_teacher.yaml"))
        self.config["env"]["numEnvs"] = 2
        self.config["asymmetric_teacher"].pop("schedule", None)
        self.config["asymmetric_teacher"].update({
            "rollouts": 2, "minibatch_size": 2, "learning_epochs": 1,
            "checkpoint_interval": 0, "eval_interval": 0, "eval_steps": 2, "seed": 71,
            "total_steps": 16, "mode": "m2",
        })
        self.path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        self.options = {"roboinfo": True, "observe_gait_commands": True,
                        "rand_control": True, "near_goal_stop": False,
                        "obj_move_prob": 0.0, "pred_success": False}
        source_config = deepcopy(self.config)
        source_config.pop("asymmetric_teacher")
        source_config["teacher_vision"] = {"vision_mode": "images", "reward_schedule_steps": 90000}
        source_config["env"]["D1_bench_task_level"] = "Level03"
        source_config["env"]["maxEpisodeLength"] = 137
        self.source = {"teacher_vision_state": {
            "origin_step": 120000, "completed_steps": 48, "global_step": 120048,
            "experiment_config": source_config,
            "environment_options": deepcopy(self.options),
        }}
        self.saved = {"asymmetric_teacher_state": {
            "version": 2, "origin_step": 120048, "completed_steps": 6, "global_step": 120054,
            "best_eval_score": 0.5, "last_eval_step": 4,
            "experiment_config": deepcopy(self.config),
            "environment_options": deepcopy(self.options),
        }}

    def configure(self, flags, checkpoint, is_eval=False):
        with mock.patch.object(sys, "argv", ["m0"] + list(flags)):
            args = get_params()
            result = configure_experiment(args, checkpoint, self.path, is_eval=is_eval)
        return args, result

    def test_no_checkpoint_starts_random_training_and_curriculum_at_zero(self):
        for mode in ("m0", "m1", "m2"):
            with self.subTest(mode=mode):
                args, (cfg, origin, completed, global_step) = self.configure([
                    "--num_envs=128", "--object_name", "green_bowl", "--asymmetric_mode", mode,
                    "--roboinfo", "--observe_gait_commands",
                ], {})
                self.assertEqual((origin, completed, global_step), (0, 0, 0))
                self.assertEqual(cfg["env"]["globalStepCounter"], 0)
                self.assertEqual(cfg["env"]["numEnvs"], 128)
                self.assertEqual(set(cfg["env"]["asset"]["asset_multi"]), {"green_bowl"})
                self.assertEqual(cfg["asymmetric_teacher"]["initialization"], "scratch")
                self.assertEqual(cfg["asymmetric_teacher"]["normalization"], "rollout")
                self.assertEqual(cfg["sensor"]["recordCameraGeometry"], mode == "m2")
                self.assertEqual(args.timesteps, 16)
                self.assertFalse(args.teacher_init_checkpoint or args.checkpoint)

    def test_scratch_configuration_and_normalization_survive_resume(self):
        _, (cfg, _, _, _) = self.configure(["--roboinfo", "--observe_gait_commands"], {})
        saved = deepcopy(self.saved)
        saved["asymmetric_teacher_state"].update(
            experiment_config=cfg, origin_step=0, global_step=6)
        _, (restored, origin, completed, global_step) = self.configure(["--checkpoint", "agent_6.pt"], saved)
        self.assertEqual((origin, completed, global_step), (0, 6, 6))
        self.assertEqual(restored["asymmetric_teacher"], cfg["asymmetric_teacher"])
        _, (migrated, origin, completed, global_step) = self.configure(
            ["--teacher_init_checkpoint", "agent_6.pt"], saved)
        self.assertEqual((origin, completed, global_step), (6, 0, 6))
        self.assertEqual(migrated["asymmetric_teacher"]["initialization"], "teacher")
        self.assertEqual(migrated["asymmetric_teacher"]["normalization"], "rollout")

    def test_scratch_rejects_migration_only_options_and_evaluation_requires_a_model(self):
        for flags in (["--teacher_initial_step", "120000"], ["--asymmetric_allow_legacy"]):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "requires --teacher_init_checkpoint"):
                self.configure(flags, {})
        with self.assertRaisesRegex(ValueError, "Evaluation requires --checkpoint"):
            self.configure([], {}, is_eval=True)
        with mock.patch.object(sys, "argv", ["asymmetric"]):
            with self.assertRaisesRegex(ValueError, "Evaluation requires --checkpoint"):
                get_trainer(is_eval=True)

    def test_distributed_size_is_saved_and_resume_cannot_change_sample_budget(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            _, (cfg, _, _, _) = self.configure(["--roboinfo", "--observe_gait_commands"], {})
            self.assertEqual(cfg["asymmetric_teacher"]["world_size"], 2)
            self.assertEqual(cfg["env"]["numEnvs"], 2)  # Per process, not divided again.
            saved = deepcopy(self.saved)
            saved["asymmetric_teacher_state"]["experiment_config"] = cfg
            _, restored = self.configure(["--checkpoint", "agent_6.pt"], saved)
            self.assertEqual(restored[0]["asymmetric_teacher"]["world_size"], 2)
            with self.assertRaisesRegex(ValueError, "world_size"):
                self.configure(["--checkpoint", "agent_6.pt"], self.saved)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "world_size"):
                self.configure(["--checkpoint", "agent_6.pt"], saved)
            # A multi-GPU checkpoint can still be evaluated/exported on one GPU.
            _, restored = self.configure(["--checkpoint", "agent_6.pt"], saved, is_eval=True)
            self.assertEqual(restored[0]["asymmetric_teacher"]["world_size"], 2)
            _, migrated = self.configure(["--teacher_init_checkpoint", "agent_6.pt"], saved)
            self.assertEqual(migrated[0]["asymmetric_teacher"]["world_size"], 1)

    def test_visual_checkpoint_uses_global_metadata_not_local_filename_suffix(self):
        original = deepcopy(self.source)
        args, (cfg, origin, completed, global_step) = self.configure(
            ["--teacher_init_checkpoint", "/tmp/visual/checkpoints/agent_48.pt"], self.source)
        self.assertEqual((origin, completed, global_step), (120048, 0, 120048))
        self.assertEqual(cfg["env"]["globalStepCounter"], 120048)
        self.assertEqual(cfg["env"]["maxEpisodeLength"], 137)
        self.assertEqual(cfg["env"]["D1_bench_task_level"], "Level03")
        self.assertEqual(cfg["asymmetric_teacher"]["reward_schedule_steps"], 90000)
        self.assertEqual(cfg["asymmetric_teacher"]["initialization"], "teacher")
        self.assertEqual(cfg["asymmetric_teacher"]["normalization"], "frozen")
        self.assertNotIn("teacher_vision", cfg)
        self.assertEqual(args.timesteps, 16)
        self.assertTrue(args.roboinfo and args.observe_gait_commands and args.rand_control)
        self.assertTrue(cfg["enableCameraSensors"] and cfg["env"]["refreshCameraOnReset"])
        self.assertFalse(cfg["env"]["terminateOnCameraConstraint"])
        self.assertEqual(self.source, original)

    def test_visual_initial_step_override_must_match_metadata(self):
        flags = ["--teacher_init_checkpoint", "agent_48.pt", "--teacher_initial_step"]
        with self.assertRaisesRegex(ValueError, "conflicts with visual-teacher metadata"):
            self.configure(flags + ["48"], self.source)
        _, result = self.configure(flags + ["120048"], self.source)
        self.assertEqual(result[1], 120048)

    def test_explicit_new_config_and_mode_are_supported(self):
        args, (cfg, origin, _, _) = self.configure([
            "--teacher_init_checkpoint", "agent_48.pt", "--asymmetric_config", str(self.path),
            "--asymmetric_mode", "m1", "--asymmetric_eval_interval", "4", "--asymmetric_eval_steps", "3",
            "--vision_learning_epochs", "2", "--seed", "19",
        ], self.source)
        self.assertEqual(origin, 120048)
        self.assertEqual(cfg["env"]["maxEpisodeLength"], 100)
        self.assertEqual(cfg["asymmetric_teacher"]["mode"], "m1")
        self.assertEqual(cfg["asymmetric_teacher"]["seed"], 19)
        self.assertEqual(cfg["asymmetric_teacher"]["learning_epochs"], 2)
        self.assertEqual((cfg["asymmetric_teacher"]["eval_interval"], cfg["asymmetric_teacher"]["eval_steps"]), (4, 3))
        self.assertEqual(args.timesteps, 16)

    def test_resume_restores_exact_config_and_training_origin(self):
        original = deepcopy(self.saved)
        args, (cfg, origin, completed, global_step) = self.configure(
            ["--checkpoint", "agent_6.pt"], self.saved)
        self.assertEqual((origin, completed, global_step), (120048, 6, 120054))
        self.assertEqual(args.seed, 71)
        self.assertEqual(args.timesteps, 16)
        self.assertEqual(cfg["asymmetric_teacher"], self.config["asymmetric_teacher"])
        self.assertEqual(self.saved, original)
        args, result = self.configure(["--checkpoint", "agent_6.pt", "--timesteps", "8"], self.saved)
        self.assertEqual(args.timesteps, 8)
        self.assertEqual(result[2], 6)

    def test_resume_rejects_invalid_budgets_and_training_changes(self):
        invalid = [
            ["--timesteps", "6"], ["--timesteps", "7"], ["--timesteps", "18"], ["--timesteps", "0"],
            ["--asymmetric_config", str(self.path)], ["--asymmetric_mode", "m0"],
            ["--asymmetric_eval_interval", "2"], ["--asymmetric_eval_steps", "4"],
            ["--vision_minibatch_size", "4"], ["--vision_learning_epochs", "2"],
            ["--num_envs", "4"], ["--object_name", "green_bowl"],
            ["--vision_task_level", "Level03"], ["--seed", "99"],
            ["--near_goal_stop"], ["--obj_move_prob", "0.25"], ["--pred_success"],
        ]
        for flags in invalid:
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.configure(["--checkpoint", "agent_6.pt"] + flags, self.saved)

    def test_evaluation_can_change_size_object_and_length_without_changing_training_config(self):
        args, (cfg, _, completed, _) = self.configure([
            "--checkpoint", "agent_6.pt", "--timesteps", "3", "--num_envs", "1",
            "--object_name", "green_bowl", "--vision_task_level", "Level03", "--seed", "19",
        ], self.saved, is_eval=True)
        self.assertEqual(args.timesteps, 3)
        self.assertEqual(cfg["env"]["numEnvs"], 1)
        self.assertEqual(set(cfg["env"]["asset"]["asset_multi"]), {"green_bowl"})
        self.assertEqual(completed, 6)
        self.assertEqual(cfg["asymmetric_teacher"], self.config["asymmetric_teacher"])
        self.assertEqual(args.seed, 19)

    def test_resume_never_treats_visual_teacher_as_asymmetric_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "--teacher_init_checkpoint"):
            self.configure(["--checkpoint", "agent_48.pt"], self.source)

    def test_previous_version_requires_new_experiment_and_inherits_source_global_step(self):
        old = deepcopy(self.saved)
        metadata = old["asymmetric_teacher_state"]
        metadata["version"] = 1
        metadata["experiment_config"]["asymmetric_teacher"].pop("total_steps")
        metadata["experiment_config"]["asymmetric_teacher"].pop("mode")
        metadata["experiment_config"]["asymmetric_teacher"]["schedule"] = {
            "total_steps": 16, "warmup_steps": 4, "anneal_steps": 8, "direct": False}
        with self.assertRaisesRegex(ValueError, "teacher_init_checkpoint"):
            self.configure(["--checkpoint", "agent_6.pt"], old)
        _, (cfg, origin, completed, global_step) = self.configure(
            ["--teacher_init_checkpoint", "agent_6.pt"], old)
        self.assertEqual((origin, completed, global_step), (120054, 0, 120054))
        self.assertEqual(cfg["asymmetric_teacher"]["mode"], "m2")
        self.assertNotIn("schedule", cfg["asymmetric_teacher"])

    def test_version_two_warm_start_starts_new_mode_and_budget(self):
        source = deepcopy(self.saved)
        source["asymmetric_teacher_state"]["experiment_config"]["asymmetric_teacher"]["mode"] = "m0"
        args, (cfg, origin, completed, global_step) = self.configure([
            "--teacher_init_checkpoint", "agent_6.pt", "--asymmetric_mode", "m2",
            "--seed", "19",
        ], source)
        self.assertEqual((origin, completed, global_step), (120054, 0, 120054))
        self.assertEqual(cfg["asymmetric_teacher"]["mode"], "m2")
        self.assertEqual(args.seed, 19)
        self.assertEqual(args.timesteps, 16)

    def test_legacy_provenance_requires_parseable_step_or_explicit_origin(self):
        flags = ["--roboinfo", "--observe_gait_commands", "--asymmetric_allow_legacy"]
        _, result = self.configure(flags + ["--teacher_init_checkpoint", "agent_50000.pt"], {})
        self.assertEqual(result[1:], (50000, 0, 50000))
        with self.assertRaises(ValueError):
            self.configure(flags + ["--teacher_init_checkpoint", "teacher.pt"], {})
        _, result = self.configure(flags + ["--teacher_init_checkpoint", "teacher.pt",
                                           "--teacher_initial_step", "50000"], {})
        self.assertEqual(result[1:], (50000, 0, 50000))

    def test_invalid_layout_camera_and_sample_budget_are_rejected(self):
        cases = [
            ("missing_proprio", lambda cfg: None, {"roboinfo": False}),
            ("camera", lambda cfg: cfg["sensor"]["wrist_camera"].update(resolution=[48, 27]), {}),
            ("resized_camera", lambda cfg: cfg["sensor"].update(resized_resolution=[48, 27]), {}),
            ("floating_base", lambda cfg: cfg["env"].update(floatingBase=True), {}),
            ("minibatch", lambda cfg: cfg["asymmetric_teacher"].update(minibatch_size=3), {}),
        ]
        for name, mutate, options in cases:
            saved = deepcopy(self.saved)
            mutate(saved["asymmetric_teacher_state"]["experiment_config"])
            saved["asymmetric_teacher_state"]["environment_options"].update(options)
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.configure(["--checkpoint", "agent_6.pt"], saved)

    def test_invalid_entry_flags_fail_before_isaacgym_import(self):
        invalid = [
            ["--task", "Other"], ["--use_tanh"], ["--no_feature"],
            ["--pitch_control"], ["--fixed_base"], ["--front_only"],
            ["--wrist_seg"], ["--depth_random"], ["--seperate"], ["--record_video"],
            ["--vision_mode", "zero"], ["--vision_config", "other.yaml"],
            ["--teacher_ckpt_path", "other.pt"],
            ["--teacher_init_checkpoint", "teacher.pt", "--checkpoint", "agent_6.pt"],
        ]
        for flags in invalid:
            with self.subTest(flags=flags), mock.patch.object(sys, "argv", ["m0"] + flags):
                with self.assertRaises(ValueError):
                    get_trainer()


if __name__ == "__main__":
    unittest.main()
