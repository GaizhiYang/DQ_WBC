"""Resolved task semantics survive recipe selection, model-off and evaluation."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_world_model_entry import arguments, configuration
from utils.wm_entry import add_world_model_arguments, configure_world_model
from utils.world_model.action_checkpoint import TaskContractMetadata, validate_checkpoint_action_contract


class ManipulationEntryTests(unittest.TestCase):
    def resolve(self, mode="hybrid", **settings):
        cfg = configuration(mode=mode, recipe="manipulation", **settings)
        cfg["env"]["smallValueSetZero"] = True
        result = configure_world_model(cfg, arguments())
        return cfg, result

    def test_new_launcher_recipe_has_independent_stages_and_optional_experiments(self):
        cfg, settings = self.resolve()
        self.assertEqual(settings["updates_per_low_step"], 0)
        self.assertEqual(settings["wm_prepare_boundaries"], 4)
        self.assertEqual(settings["value_prepare_boundaries"], 2)
        self.assertEqual(settings["actor_max_updates"], 4)
        self.assertEqual(settings["wm_max_batches"], 512)
        self.assertEqual(settings["holdout_mode"], "environment")
        self.assertTrue(settings["allow_partial_windows"])
        self.assertEqual(settings["task_contract"], "manipulation_v2")
        self.assertFalse(cfg["env"]["smallValueSetZero"])
        self.assertFalse(settings["running_normalization"] or settings["layer_norm"] or settings["critic_memory"])
        self.assertEqual(settings["amp"], "none")

    def test_model_off_retains_exact_real_policy_initialization_preprocessing_and_task(self):
        _, hybrid = self.resolve()
        cfg, off = self.resolve(mode="off")
        for key in ("initial_action_std", "actor_last_gain", "fixed_preprocessing",
                    "observation_noise_scale", "task_contract", "action_coordinates", "action_rate_mode"):
            self.assertEqual(off[key], hybrid[key])
        self.assertFalse(cfg["env"]["smallValueSetZero"])
        evaluated = configure_world_model(deepcopy(cfg), arguments(), is_eval=True)
        self.assertEqual(off, evaluated)

    def test_rejects_unknown_nested_settings_and_incompatible_budget(self):
        for setting in (dict(task_aux_config={"typo": 1}), dict(value_source="ppo"),
                        dict(holdout_mode="episode"), dict(updates_per_low_step=.4),
                        dict(wm_epochs=0), dict(wm_prepare_boundaries=0),
                        dict(actor_stage1_updates=3, actor_stage2_updates=2),
                        dict(running_normalization="true")):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.resolve(**setting)

    def test_cli_explicit_false_zero_and_neural_bf16(self):
        parser = argparse.ArgumentParser()
        add_world_model_arguments(parser)
        parsed = parser.parse_args(["--wm_recipe", "manipulation", "--wm_layer_norm", "true",
                                   "--wm_running_normalization", "false", "--wm_amp", "bf16",
                                   "--wm_task_aux_alpha", "0", "--wm_diagnostics_interval", "0"])
        settings = configure_world_model(configuration(mode="hybrid"), arguments(**vars(parsed)))
        self.assertTrue(settings["layer_norm"])
        self.assertFalse(settings["running_normalization"])
        self.assertEqual(settings["task_aux_alpha"], 0)
        self.assertEqual(settings["amp"], "bf16")

    def test_pure_ppo_and_world_model_checkpoints_reject_semantic_mismatch(self):
        cfg, _ = self.resolve(mode="off")
        metadata = TaskContractMetadata(cfg)
        payload = {"policy": {}, "teacher_task_contract": metadata.state_dict()}
        validate_checkpoint_action_contract(cfg, payload=payload)
        metadata.load_state_dict(payload["teacher_task_contract"])
        for mutation in ("legacy", "deadzone"):
            bad = deepcopy(cfg)
            if mutation == "legacy":
                bad["env"]["world_model"]["task_contract"] = "legacy"
            else:
                bad["env"]["smallValueSetZero"] = True
            with self.assertRaisesRegex(ValueError, "contract"):
                validate_checkpoint_action_contract(bad, payload=payload)
        payload["world_model_training"] = {"metadata": {**metadata.state_dict(), "task_contract": "legacy"}}
        with self.assertRaisesRegex(ValueError, "Contradictory"):
            validate_checkpoint_action_contract(cfg, payload=payload)


if __name__ == "__main__":
    unittest.main()
