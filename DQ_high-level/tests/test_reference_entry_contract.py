"""Pure BPTT resolution and safe policy reconstruction before simulator startup."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_world_model_entry import arguments, configuration
from utils.wm_entry import add_world_model_arguments, configure_world_model
from utils.world_model.action_checkpoint import TaskContractMetadata, validate_checkpoint_action_contract


class ReferenceEntryTests(unittest.TestCase):
    def resolve(self, **overrides):
        cfg = configuration(mode="hybrid", recipe="reference", **overrides)
        return cfg, configure_world_model(cfg, arguments())

    def test_reference_defaults_have_control_step_units_and_independent_actor(self):
        cfg, settings = self.resolve()
        self.assertEqual(settings["horizon"] * 8, 128)
        self.assertEqual(settings["hidden"], [2048] * 5)
        self.assertEqual(settings["actor_hidden"], [2048, 1024, 512, 256])
        self.assertEqual(settings["world_model_only_iterations"], 1000)
        self.assertEqual(settings["collection_warmup_high_steps"], 64)
        self.assertEqual(settings["collection_updates_per_high_step"], .4)
        self.assertEqual(settings["wm_max_batches"], 0)
        self.assertEqual(settings["wm_epochs"], 4)
        self.assertEqual(settings["value_source"], "imagined")
        self.assertEqual(settings["state_loss_reduction"], "reference")
        self.assertEqual(settings["dynamics_target"], "physical_increment")
        self.assertFalse(settings["fixed_preprocessing"])
        self.assertFalse(cfg["env"]["smallValueSetZero"])

    def test_shipped_task_yaml_does_not_override_recipe_horizon(self):
        import yaml
        path = Path(__file__).resolve().parents[1] / "data/cfg/DQ_teacher.yaml"
        cfg = yaml.safe_load(path.read_text())
        cfg["env"]["tsil"] = {"enabled": False}
        result = configure_world_model(cfg, arguments(world_model="hybrid", wm_recipe="reference"))
        self.assertEqual(result["horizon"], 16)
        self.assertTrue(result["fixed8"])

    def test_saved_reference_evaluation_preserves_architecture_and_needs_no_training_paths(self):
        cfg, settings = self.resolve()
        cfg["env"]["world_model"]["candidate_checkpoint"] = "/missing/training-only.pt"
        evaluated = configure_world_model(cfg, arguments(), is_eval=True)
        self.assertEqual(evaluated["mode"], "off")
        for name in ("actor_hidden", "actor_running_normalization", "policy_amp", "action_noise",
                     "task_contract", "fixed8", "imagined_reward_mode"):
            self.assertEqual(evaluated[name], settings[name])

    def test_reference_actor_cannot_load_a_legacy_gaussian_teacher_or_reverse(self):
        cfg, _ = self.resolve()
        ref = {"policy": {}, "teacher_task_contract": TaskContractMetadata(cfg).state_dict()}
        validate_checkpoint_action_contract(cfg, payload=ref)
        old_cfg = deepcopy(cfg)
        old_cfg["env"]["world_model"]["recipe"] = "manipulation"
        old = {"policy": {}, "teacher_task_contract": TaskContractMetadata(old_cfg).state_dict()}
        for current, checkpoint in ((cfg, old), (old_cfg, ref)):
            with self.assertRaisesRegex(ValueError, "policy_architecture"):
                validate_checkpoint_action_contract(current, payload=checkpoint)

    def test_actor_normalizer_cannot_be_disabled_silently_during_evaluation(self):
        cfg, _ = self.resolve()
        payload = {"policy": {}, "teacher_task_contract": TaskContractMetadata(cfg).state_dict()}
        for key, value in (("actor_running_normalization", False), ("actor_hidden", [128, 64]),
                           ("policy_amp", "none")):
            bad = deepcopy(cfg)
            bad["env"]["world_model"][key] = value
            with self.assertRaisesRegex(ValueError, "policy_network/normalization"):
                validate_checkpoint_action_contract(bad, payload=payload)

    def test_cli_explicit_zero_and_shortened_diagnostic_schedule(self):
        parser = argparse.ArgumentParser()
        add_world_model_arguments(parser)
        parsed = parser.parse_args(["--wm_recipe", "reference", "--wm_world_model_only_iterations", "0",
            "--wm_collection_warmup_high_steps", "4", "--wm_max_batches", "0", "--wm_action_noise", "0"])
        settings = configure_world_model(configuration(mode="hybrid"), arguments(**vars(parsed)))
        self.assertEqual(settings["world_model_only_iterations"], 0)
        self.assertEqual(settings["wm_max_batches"], 0)
        self.assertEqual(settings["collection_warmup_high_steps"], 4)
        self.assertEqual(settings["action_noise"], 0)

    def test_reject_incompatible_reference_components(self):
        for overrides in (dict(state_loss_reduction="sum"), dict(dynamics_target="residual"),
                          dict(value_source="ppo"), dict(running_normalization=False),
                          dict(fixed_preprocessing=True), dict(critic_features_mode="compact"),
                          dict(observation_noise_scale=1), dict(collection_updates_per_high_step=2),
                          dict(world_model_only_iterations=-1), dict(wm_max_batches=-1),
                          dict(actor_hidden=[0]), dict(reference_task_config={"typo": 1})):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.resolve(**overrides)


if __name__ == "__main__":
    unittest.main()
