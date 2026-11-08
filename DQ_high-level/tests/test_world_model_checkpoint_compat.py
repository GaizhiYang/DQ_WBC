"""Teacher checkpoint, probe configuration and fixed-scale compatibility.

These are tensor-only tests. The real probe session factory is exercised with
a lightweight teacher constructor, without creating an Isaac Gym simulator.
"""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_teacher_world_model as probe
from skrl.agents.torch.ppo import PPO
from skrl.memories.torch import RandomMemory
from test_world_model_aligned import make_aligned
from test_world_model_fixed8_ppo import make_fixed8_agent, collect_fixed8
from test_world_model_ppo import TinyPolicy, TinyValue
from utils.wm_entry import configure_world_model, reject_candidate_checkpoint
from utils.tsil_checkpoint import read_temporal_checkpoint
from utils.world_model.policy_inputs import FixedScale, teacher_observation_vectors
from utils.world_model.recipe import SUPERMIMIC_CONFIG


def make_checkpoint_agent():
    # Production resolves the recipe before constructing the environment and
    # adapter. The lightweight fixture must also establish its action contract
    # before metadata is captured, rather than changing it only during export.
    return make_aligned("cpu", **{
        name: SUPERMIMIC_CONFIG[name]
        for name in ("action_coordinates", "action_rate_mode")})


def add_saved_preprocessing(agent):
    scale, _ = teacher_observation_vectors(device=agent.device)
    fixed = FixedScale(1284, agent.device, scale)
    agent._state_preprocessor = fixed
    agent.checkpoint_modules["state_preprocessor"] = fixed
    agent.wm_imagination.state_preprocessor = fixed
    agent.wm_env.cfg["env"]["world_model"] = deepcopy(agent.wm_config)
    agent.wm_teacher_config = deepcopy(agent.wm_env.cfg)
    agent.wm_teacher_arguments = {"task": "B1Z1PickMulti", "timesteps": 100,
                                  "sgs_mode": "disabled", "rand_control": False,
                                  "wm_recipe": "supermimic"}


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_candidate_schema_is_rejected_before_generic_policy_loading(self):
        candidate = {"schema": "dq-world-model-candidate-v1", "teacher": {"policy": {}}}
        with self.assertRaisesRegex(ValueError, "cannot be used as --checkpoint"):
            reject_candidate_checkpoint(payload=candidate)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            torch.save(candidate, path)
            with self.assertRaisesRegex(ValueError, "full teacher agent_"):
                reject_candidate_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "cannot be used as --checkpoint"):
                read_temporal_checkpoint(path)
            teacher_path = Path(directory) / "agent_1.pt"
            torch.save({"policy": {}, "value": {}}, teacher_path)
            self.assertIsNone(read_temporal_checkpoint(teacher_path))
        # The existing metadata loader already has a CPU payload: avoid a
        # redundant load of the large model/optimizer checkpoint in this path.
        with patch("torch.load", side_effect=AssertionError("checkpoint loaded twice")):
            reject_candidate_checkpoint(payload={"policy": {}, "value": {},
                                                  "world_model_training": {}})
            reject_candidate_checkpoint(payload={"tsil_metadata": {"version": 2}})

    def test_full_teacher_checkpoint_contains_independent_critic_and_rejects_old_dynamics_schema(self):
        agent = make_checkpoint_agent()
        add_saved_preprocessing(agent)
        collect_fixed8(agent, high_steps=1)
        agent._update_world_model()
        with tempfile.TemporaryDirectory() as directory:
            teacher_path = Path(directory) / "agent_1.pt"
            agent.save(str(teacher_path))
            saved = torch.load(teacher_path, map_location="cpu", weights_only=False)
            reject_candidate_checkpoint(payload=saved)
            independent = saved["world_model_training"]["imagination"]["imagined_critic"]
            self.assertIn("critic", independent)
            self.assertTrue(independent["optimizer"]["state"])
            self.assertEqual(independent["updates"], 2)
            self.assertIn("_fixed_scale_version", saved["state_preprocessor"])
            self.assertNotIn("value_preprocessor", saved)

            bundle = agent.candidate_payload()
            self.assertIn("_fixed_scale_version", bundle["teacher"]["state_preprocessor"])
            self.assertIsNone(bundle["teacher"]["value_preprocessor"])
            self.assertTrue(bundle["teacher_config"]["env"]["world_model"]["fixed_preprocessing"])
            # Diagnostic bundles carry the exact independent value weights,
            # but no value optimizer; only full checkpoints back up training.
            self.assertNotIn("optimizer", bundle["imagined_critic"])
            self.assertEqual(bundle["imagined_critic_updates"], 2)
            for name, value in agent.wm_imagination.imagined_critic.state_dict().items():
                torch.testing.assert_close(value, bundle["imagined_critic"][name], rtol=0, atol=0)
            old = deepcopy(bundle)
            old["metadata"]["version"] = 4
            old["metadata"]["schema"] = "dq-sgs-ekf-fixed8-physical64-v4"
            old_path = Path(directory) / "old_candidate.pt"
            torch.save(old, old_path)
            with self.assertRaisesRegex(ValueError, "schema/task contract"):
                agent.load_candidate(old_path)

            restored = make_checkpoint_agent()
            add_saved_preprocessing(restored)
            restored.load(str(teacher_path))
            for name, value in agent.wm_imagination.imagined_critic.state_dict().items():
                torch.testing.assert_close(value, restored.wm_imagination.imagined_critic.state_dict()[name],
                                           rtol=0, atol=0)
            self.assertEqual(restored.wm_imagination.imagined_critic_trainer.updates, 2)
            torch.testing.assert_close(agent._state_preprocessor.scale, restored._state_preprocessor.scale)

            # Evaluation reconstructs ordinary PPO (no model replay or imagined
            # Critic). Its policy/value/fixed-scale modules must still load.
            evaluation = PPO(
                models={"policy": TinyPolicy(1284, "cpu"), "value": TinyValue(1284, "cpu")},
                memory=None, observation_space=1284, action_space=9, device="cpu",
                cfg=dict(state_preprocessor=FixedScale,
                         state_preprocessor_kwargs={"size": 1284, "device": "cpu"},
                         value_preprocessor=None,
                         experiment={"write_interval": 0, "checkpoint_interval": 0}))
            evaluation.load(str(teacher_path))
            for name, value in agent.policy.state_dict().items():
                torch.testing.assert_close(value, evaluation.policy.state_dict()[name], rtol=0, atol=0)
            torch.testing.assert_close(agent._state_preprocessor.scale, evaluation._state_preprocessor.scale)
            torch.testing.assert_close(evaluation._value_preprocessor(torch.tensor([7.])), torch.tensor([7.]))

    def test_probe_factory_preserves_fixed_scale_and_absent_value_scaler_when_mode_is_off(self):
        source = make_checkpoint_agent()
        add_saved_preprocessing(source)
        bundle = source.candidate_payload()
        for name in ("action_coordinates", "action_rate_mode"):
            self.assertEqual(bundle["metadata"][name], SUPERMIMIC_CONFIG[name])
            self.assertEqual(bundle["teacher_config"]["env"]["world_model"][name],
                             bundle["metadata"][name])
        raw_env = make_fixed8_agent("cpu").wm_env
        built = {}
        teacher = SimpleNamespace(get_params=lambda: None)

        def get_trainer(is_eval=False, *, enable_logging=True):
            self.assertFalse(is_eval)
            self.assertFalse(enable_logging)
            args = teacher.get_params()
            config = yaml.safe_load(Path(args.cfg_file).read_text())
            settings = configure_world_model(config, args, is_eval=is_eval)
            built["settings"] = settings
            self.assertEqual(settings["mode"], "off")
            self.assertTrue(settings["fixed8"])
            self.assertTrue(settings["fixed_preprocessing"])
            self.assertEqual(settings["value_source"], "imagined")
            self.assertEqual(settings["observation_noise_scale"], source.wm_config["observation_noise_scale"])
            raw_env.cfg = config
            scale, _ = teacher_observation_vectors(device="cpu")
            agent = PPO(
                models={"policy": TinyPolicy(1284, "cpu"), "value": TinyValue(1284, "cpu")},
                memory=RandomMemory(3, raw_env.num_envs, device="cpu"),
                observation_space=1284, action_space=9, device="cpu",
                cfg=dict(state_preprocessor=FixedScale,
                         state_preprocessor_kwargs={"size": 1284, "device": "cpu", "scale": scale},
                         value_preprocessor=None,
                         experiment={"write_interval": 0, "checkpoint_interval": 0}))
            agent.init()
            return SimpleNamespace(agents=agent,
                                   env=SimpleNamespace(_env=raw_env, reset=lambda: None))

        teacher.get_trainer = get_trainer
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(sys.modules, {"train_multistate_DQ_teacher": teacher}):
            cli = SimpleNamespace(num_envs=12, seed=43, device="cpu", workdir=directory,
                                  worker="plan", family_sampling="balanced")
            session = probe._new_session(cli, bundle, torch)
            # Mode=off must retain the saved action contract, and changing it
            # must still fail the real factory's strict metadata validation.
            for name, incompatible in (("action_coordinates", "physical"),
                                       ("action_rate_mode", "base_l2")):
                with self.subTest(incompatible_field=name):
                    mismatched = deepcopy(bundle)
                    mismatched["teacher_config"]["env"]["world_model"][name] = incompatible
                    with self.assertRaisesRegex(ValueError, "schema does not match candidate metadata"):
                        probe._new_session(cli, mismatched, torch)
        self.assertIsInstance(session.trainer.agents._state_preprocessor, FixedScale)
        self.assertFalse(hasattr(session.trainer.agents._value_preprocessor, "load_state_dict"))
        for name, value in source.policy.state_dict().items():
            torch.testing.assert_close(value, session.trainer.agents.policy.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(source._state_preprocessor.scale,
                                   session.trainer.agents._state_preprocessor.scale, rtol=0, atol=0)
        # This repository's PPO override sets module modes; probe decisions
        # directly call policy.compute and do not consult Agent.training.
        self.assertFalse(session.trainer.agents.policy.training)


if __name__ == "__main__":
    unittest.main()
