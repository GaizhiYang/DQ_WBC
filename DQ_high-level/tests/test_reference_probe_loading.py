"""Isolated diagnostics must restore the exact pure-reference numerical contract."""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_teacher_world_model import (
    _load_probe_model, _verify_probe_metadata, _load_probe_teacher, _probe_mean_action)
from diagnose_manipulation_teacher import _make_imagination
from utils.world_model import HierarchicalDynamics
from utils.world_model.imagination import ImaginationTrainer, ImaginationConfig
from utils.world_model.reference_policy import ReferenceTeacherPolicy
from utils.world_model.reference_recipe import REFERENCE_CONFIG
from utils.wm_reference import reference_metadata
import test_world_model_manipulation_data as data_tests


class ReferenceProbeLoadingTests(unittest.TestCase):
    def fixture(self):
        spec, state, post, action, context, _ = data_tests.PhysicalNormalizationTests.physical_pair(self)
        config = dict(REFERENCE_CONFIG, fixed8=True, hidden=[16, 16], contact_hidden=[8],
                      actor_hidden=[16], critic_hidden=[16, 8], horizon=2,
                      amp="bf16", policy_amp="bf16")
        model = HierarchicalDynamics(spec, hidden=config["hidden"], contact_hidden=config["contact_hidden"],
            layer_norm=True, running_normalization=True, amp=config["amp"], dynamics_target="physical_increment")
        model.observe_transition(state, action, post, context, .02,
                                 train_mask=torch.ones(len(state), dtype=torch.bool))
        observations = spec.observation(state, context)
        policy = ReferenceTeacherPolicy(observations.shape[-1], 9, "cpu", hidden=[16], amp=config["policy_amp"])
        policy.observe_real_next(observations)
        agent = SimpleNamespace(policy=policy, value=None, _state_preprocessor=None,
                                _value_preprocessor=None, wm_config=config)
        agent.set_running_mode = lambda mode: agent.policy.eval() if mode == "eval" else agent.policy.train()
        trainer = ImaginationTrainer(model, spec, policy, None, ImaginationConfig(
            horizon=2, dt=.16, value_source="imagined", reference_mode=True, objective_reduction="sum",
            critic_running_norm=True, critic_activation="elu", critic_amp=config["policy_amp"],
            critic_hidden=tuple(config["critic_hidden"])))
        trainer.imagined_critic.norm.update(spec.critic_features(state, context))
        bundle = dict(config=config, metadata=reference_metadata(spec, config), model=deepcopy(model.state_dict()),
            teacher=dict(policy=deepcopy(policy.state_dict()), value=None,
                         state_preprocessor=None, value_preprocessor=None),
            imagined_critic=deepcopy(trainer.imagined_critic.state_dict()), imagined_gamma=.99,
            imagined_use_tail_value=True, imagined_bootstrap_timeouts=False)
        session = SimpleNamespace(adapter=spec, trainer=SimpleNamespace(agents=agent))
        return session, bundle, model, trainer, state, context

    def test_reference_model_restores_physical_increment_statistics_and_coordinates(self):
        session, bundle, original, _, state, context = self.fixture()
        model = _load_probe_model(session.adapter, bundle, "cpu")
        self.assertEqual(model.dynamics_target, "physical_increment")
        self.assertEqual(model.amp, "bf16")
        self.assertEqual(model.coordinate_metadata(), bundle["metadata"]["model_coordinates"])
        self.assertFalse(model.training)
        action = torch.cat((torch.zeros(len(state), 12), session.adapter.get(state,"held_ee_target"),
                            session.adapter.get(state,"held_gripper_target")), -1)
        torch.testing.assert_close(model.low_step(state, action, context), original.low_step(state, action, context))
        bad = deepcopy(bundle)
        bad["metadata"]["model_coordinates"]["frame"] = "world"
        with self.assertRaisesRegex(ValueError, "dynamics coordinates"):
            _load_probe_model(session.adapter, bad, "cpu")
        del bad["metadata"]["model_coordinates"]
        with self.assertRaisesRegex(ValueError, "missing.*coordinate"):
            _load_probe_model(session.adapter, bad, "cpu")

    def test_training_and_probe_share_reference_metadata_constructor(self):
        session, bundle, *_ = self.fixture()
        _verify_probe_metadata(session.adapter, bundle)
        for field, value in (("amp", "none"), ("dynamics_target", "residual")):
            bad = deepcopy(bundle)
            bad["config"][field] = value
            with self.assertRaisesRegex(ValueError, "schema does not match"):
                _verify_probe_metadata(session.adapter, bad)

    def test_pure_teacher_has_no_value_and_restores_actor_normalizer_without_mutation(self):
        session, bundle, _, _, state, context = self.fixture()
        agent = session.trainer.agents
        obs = session.adapter.observation(state, context)
        expected = _probe_mean_action(agent, obs).detach()
        with torch.no_grad():
            agent.policy.obs_norm._mean.add_(100)
        _load_probe_teacher(agent, bundle)
        buffers = {name: value.clone() for name, value in agent.policy.named_buffers()}
        torch.testing.assert_close(_probe_mean_action(agent, obs), expected)
        torch.testing.assert_close(_probe_mean_action(agent, obs), expected)
        self.assertIsNone(agent.value)
        self.assertFalse(agent.policy.training)
        for name, value in agent.policy.named_buffers():
            torch.testing.assert_close(value, buffers[name], atol=0, rtol=0)
        agent.value = torch.nn.Linear(1, 1)
        with self.assertRaisesRegex(ValueError, "value=None"):
            _load_probe_teacher(agent, bundle)

    def test_full_diagnostic_reconstructs_critic_norm_elu_and_mlp_precision(self):
        session, bundle, _, original, state, context = self.fixture()
        loaded = _make_imagination(session, bundle, "cpu")
        self.assertTrue(loaded.cfg.reference_mode)
        self.assertEqual(loaded.cfg.objective_reduction, "sum")
        self.assertIsNone(loaded.critic)
        self.assertEqual(loaded.imagined_critic.amp, "bf16")
        self.assertEqual(sum(isinstance(m, torch.nn.ELU) for m in loaded.imagined_critic.modules()), 2)
        features = session.adapter.critic_features(state, context)
        torch.testing.assert_close(loaded.imagined_critic(features), original.imagined_critic(features))
        self.assertTrue(all(not p.requires_grad for p in loaded.model.parameters()))
        self.assertTrue(all(not p.requires_grad for p in loaded.imagined_critic.parameters()))
        changed = deepcopy(bundle)
        changed["config"]["critic_running_normalization"] = False
        with self.assertRaises(RuntimeError):
            _make_imagination(session, changed, "cpu")


if __name__ == "__main__":
    unittest.main()
