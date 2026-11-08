"""Checkpoint action semantics remain explicit even with ordinary PPO eval."""

import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.world_model.action_checkpoint import validate_checkpoint_action_contract
from utils.tsil_checkpoint import read_temporal_checkpoint


def checkpoint(coordinates="normalized_commands", rate="squared_all"):
    return {"policy": {}, "value": {}, "world_model_training": {
        "metadata": {"action_coordinates": coordinates, "action_rate_mode": rate}}}


def configuration(coordinates="normalized_commands", rate="squared_all"):
    return {"env": {"world_model": {
        "mode": "off", "action_coordinates": coordinates, "action_rate_mode": rate}}}


class ActionCheckpointTests(unittest.TestCase):
    def test_matching_contract_reuses_loaded_payload_without_mutating_config_or_weights(self):
        payload, cfg = checkpoint(), configuration()
        original_payload, original_cfg = deepcopy(payload), deepcopy(cfg)
        with patch("torch.load", side_effect=AssertionError("checkpoint loaded again")):
            validate_checkpoint_action_contract(cfg, "already_loaded.pt", payload=payload)
        self.assertEqual(payload, original_payload)
        self.assertEqual(cfg, original_cfg)

    def test_moved_normalized_checkpoint_rejects_fallback_config_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_24.pt"
            torch.save(checkpoint(), path)
            with self.assertRaisesRegex(ValueError, "action_coordinates.*--cfg_file"):
                validate_checkpoint_action_contract({"env": {}}, path)
            validate_checkpoint_action_contract(configuration(), path)

    def test_both_contract_fields_are_checked_independently_and_in_both_directions(self):
        for name, legacy in (("action_coordinates", "physical"), ("action_rate_mode", "base_l2")):
            with self.subTest(field=name):
                mismatched_cfg = configuration()
                mismatched_cfg["env"]["world_model"][name] = legacy
                with self.assertRaisesRegex(ValueError, "action contract.*" + name):
                    validate_checkpoint_action_contract(mismatched_cfg, payload=checkpoint())
                mismatched_payload = checkpoint()
                mismatched_payload["world_model_training"]["metadata"][name] = legacy
                with self.assertRaisesRegex(ValueError, "action contract.*" + name):
                    validate_checkpoint_action_contract(configuration(), payload=mismatched_payload)

    def test_legacy_checkpoints_keep_physical_commands_and_base_rate(self):
        for payload in ({"policy": {}, "value": {}},
                        {"policy": {}, "world_model_training": {"metadata": {"version": 4}}}):
            with self.subTest(payload=payload):
                validate_checkpoint_action_contract({"env": {}}, payload=payload)
                validate_checkpoint_action_contract(configuration("physical", "base_l2"), payload=payload)
                with self.assertRaisesRegex(ValueError, "Legacy checkpoints.*physical/base_l2"):
                    validate_checkpoint_action_contract(configuration(), payload=payload)

    def test_candidate_or_malformed_metadata_does_not_fall_back_to_legacy(self):
        with self.assertRaisesRegex(ValueError, "cannot be used as --checkpoint"):
            validate_checkpoint_action_contract({"env": {}}, payload={
                "schema": "dq-world-model-candidate-v1", "teacher": {"policy": {}}})
        for training in (None, [], {}, {"metadata": None}, {"metadata": "broken"}):
            with self.subTest(training=training), self.assertRaisesRegex(ValueError, "missing.*metadata"):
                validate_checkpoint_action_contract({"env": {}}, payload={"world_model_training": training})
        invalid = checkpoint()
        invalid["world_model_training"]["metadata"]["action_coordinates"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Invalid checkpoint/config action contract"):
            validate_checkpoint_action_contract(configuration(), payload=invalid)

    def test_temporal_metadata_reader_reuses_payload_and_retains_validation(self):
        temporal = {name: {} for name in ("policy", "value", "state_preprocessor",
                    "value_preprocessor", "tsil_state", "reward_normalizer")}
        temporal["tsil_metadata"] = {"version": 1, "config": {"env": {}}}
        with patch("torch.load", side_effect=AssertionError("checkpoint loaded again")):
            self.assertIsNone(read_temporal_checkpoint("already_loaded.pt", payload=checkpoint()))
            self.assertEqual(read_temporal_checkpoint(payload=temporal), temporal["tsil_metadata"])
            validate_checkpoint_action_contract({"env": {}}, payload=temporal)
            incomplete = deepcopy(temporal)
            del incomplete["tsil_state"]
            with self.assertRaisesRegex(ValueError, "Incomplete TSIL checkpoint"):
                read_temporal_checkpoint(payload=incomplete)

    def test_real_entry_checks_contract_after_resolution_and_before_simulator_creation(self):
        tree = ast.parse((ROOT / "train_multistate_DQ_teacher.py").read_text())
        trainer = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "get_trainer")
        calls = {node.func.id: node for node in ast.walk(trainer)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                 and node.func.id in ("configure_teacher_experiment", "create_env",
                                      "validate_checkpoint_action_contract")}
        self.assertLess(calls["configure_teacher_experiment"].lineno,
                        calls["validate_checkpoint_action_contract"].lineno)
        self.assertLess(calls["validate_checkpoint_action_contract"].lineno,
                        calls["create_env"].lineno)
        guards = [node for node in ast.walk(trainer) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Attribute) and node.test.attr == "checkpoint"
                  and calls["validate_checkpoint_action_contract"] in list(ast.walk(node))]
        self.assertEqual(len(guards), 1)  # Fresh training never loads a checkpoint.


if __name__ == "__main__":
    unittest.main()
