"""Real skrl PPO integration, report gating and checkpoint ownership on CUDA.

The tensor environment fixture uses the production adapter equations. Probe
reports in these unit tests are synthetic evidence for the verification API;
they are never persisted as deployment approvals or called simulator evidence.
"""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from test_world_model_env import make_env
from utils.wm_ppo import WorldModelPPO
from utils.world_model.probes import (ActionEffectProbeConfig, analyze_action_effects,
                                     model_fingerprint)


class TinyPolicy(torch.nn.Module):
    def __init__(self, observation_dim, device):
        super().__init__()
        self.device, self.num_actions = torch.device(device), 9
        self.linear = torch.nn.Linear(observation_dim, 9)
        torch.nn.init.normal_(self.linear.weight, std=0.001)
        torch.nn.init.zeros_(self.linear.bias)
        self.log_std = torch.nn.Parameter(torch.full((9,), -3.))
        self.to(device)

    def compute(self, inputs, role="policy"):
        return self.linear(inputs["states"]), self.log_std, {}

    def act(self, inputs, role="policy"):
        mean, log_std, _ = self.compute(inputs, role)
        self.distribution = torch.distributions.Normal(mean, log_std.exp())
        action = self.distribution.sample()
        log_probability = self.distribution.log_prob(inputs.get("taken_actions", action)).sum(-1, keepdim=True)
        return action, log_probability, {"mean_actions": mean}

    def get_entropy(self, role="policy"):
        return self.distribution.entropy().sum(-1)

    def set_mode(self, mode):
        self.train(mode == "train")


class TinyValue(torch.nn.Module):
    def __init__(self, observation_dim, device):
        super().__init__()
        self.device = torch.device(device)
        self.linear = torch.nn.Linear(observation_dim, 1)
        torch.nn.init.normal_(self.linear.weight, std=0.001)
        torch.nn.init.zeros_(self.linear.bias)
        self.to(device)

    def compute(self, inputs, role="value"):
        return self.linear(inputs["states"]), {}

    def act(self, inputs, role="value"):
        return self.linear(inputs["states"]), None, {}

    def set_mode(self, mode):
        self.train(mode == "train")


def make_agent(device="cpu", **options):
    environment = make_env(device, n=12)
    config = dict(rollouts=3, learning_epochs=1, mini_batches=1, learning_rate=1e-4,
                  state_preprocessor=RunningStandardScaler,
                  state_preprocessor_kwargs={"size": 1284, "device": device},
                  value_preprocessor=RunningStandardScaler,
                  value_preprocessor_kwargs={"size": 1, "device": device},
                  experiment={"write_interval": 0, "checkpoint_interval": 0})
    model_config = dict(mode="hybrid", replay_envs=8, replay_capacity=12,
                        window=2, horizon=2, warmup_steps=2, hidden=[16, 16],
                        term_hidden=[8], batch_size=8, batches_per_update=1,
                        validation_batch_size=8, imagination_batch=4,
                        holdout_fraction=0.5, max_validation_error=1000., max_start_error=1000.)
    model_config.update(options)
    agent = WorldModelPPO(
        models={"policy": TinyPolicy(1284, device), "value": TinyValue(1284, device)},
        memory=RandomMemory(3, environment.num_envs, device=device),
        observation_space=1284, action_space=9, device=device, cfg=config,
        wm_env=environment, wm_config=model_config)
    agent.init()
    return agent


def collect(agent, count=3, update=True):
    adapter = agent.wm_adapter
    state, context = adapter.capture()
    env_ids = agent.wm_env_ids
    n = state.shape[0]
    zero = torch.zeros(n, device=state.device)
    for step in range(count):
        with torch.no_grad():
            observation = adapter.observation(state, context)
            action, _, _ = agent.act(observation, step, 100)
            duration = torch.full((n, 1), .14, device=state.device)
            future = adapter.integrate(state, torch.zeros(n, adapter.delta_dim, device=state.device),
                                       action, context, duration)
            next_observation = adapter.observation(future, context)
            rewards = adapter.reward(state, future, action, context)
            transition = {"state": state[env_ids], "next_state": future[env_ids],
                          "context": context[env_ids], "actions": action[env_ids],
                          "dt": duration[env_ids], "reward": rewards[env_ids],
                          "terminated": zero[env_ids].bool(), "truncated": zero[env_ids].bool(),
                          "success": zero[env_ids].bool(),
                          "family": torch.arange(len(env_ids), device=state.device) % 3,
                          "episode_id": zero[env_ids].long()}
            agent.record_transition(observation, action, rewards[:, None], next_observation,
                                    zero[:, None].bool(), zero[:, None].bool(),
                                    {"world_model": transition}, step, 100)
        if update:
            agent.post_interaction(step, 100)
        state = future


def synthetic_report(agent):
    n = 64
    plus = torch.linspace(.2, .6, n, device=agent.device)
    zero = torch.zeros_like(plus)
    return analyze_action_effects(
        plus, -plus, zero, zero, plus, -plus,
        zero, zero, zero, torch.ones_like(zero, dtype=torch.bool),
        model_fingerprint(agent.wm_candidate), agent.wm_metadata,
        ActionEffectProbeConfig(),
        metadata={"simulator": "IsaacGym", "protocol": "independent_simulator_prefix_replay",
                  "directions_cover": [0, 1, 2, 3, 4, 5, 7, 8], "gripper_perturbed": False})


class IntegrationChecks:
    device = "cpu"

    def assert_state_equal(self, first, second):
        self.assertEqual(set(first), set(second))
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)

    def test_real_ppo_and_candidate_fit_continue_without_probe(self):
        agent = make_agent(self.device)
        actor_before = deepcopy(agent.policy.state_dict())
        critic_before = deepcopy(agent.value.state_dict())
        collect(agent)
        self.assertEqual(agent.wm_replay.steps, 3)
        self.assertEqual(agent.wm_replay.num_envs, 8)
        self.assertEqual(agent.memory.num_envs, 12)
        self.assertEqual(agent.wm_trainer.updates, 1)
        self.assertEqual(agent.wm_status, "waiting_for_probe")
        self.assertEqual(agent.wm_imagination.accepted_updates, 0)
        self.assertTrue(any(not torch.equal(value, actor_before[name])
                            for name, value in agent.policy.state_dict().items()))
        self.assertTrue(any(not torch.equal(value, critic_before[name])
                            for name, value in agent.value.state_dict().items()))
        self.assertEqual(agent.last_world_model_metrics["on_gpu"], float(self.device.startswith("cuda")))

    def test_probe_binds_snapshot_and_extra_update_preserves_critic_and_scalers(self):
        agent = make_agent(self.device)
        collect(agent)
        payload, report = agent.candidate_payload(), synthetic_report(agent)
        self.assertTrue(report["passed"])
        self.assertTrue(agent.activate_validated(payload, report))
        snapshot = deepcopy(agent.wm_validated.state_dict())
        critic = deepcopy(agent.value.state_dict())
        observation_scaler = deepcopy(agent._state_preprocessor.state_dict())
        value_scaler = deepcopy(agent._value_preprocessor.state_dict())
        metrics = agent._update_world_model()
        self.assertEqual(agent.wm_trainer.updates, 2)
        self.assertEqual(agent.wm_imagination.accepted_updates, 1)
        self.assertEqual(metrics["bptt/accepted"], 1.)
        self.assertGreater(metrics["bptt/actor_grad_norm"], 0.)
        self.assert_state_equal(snapshot, agent.wm_validated.state_dict())
        self.assert_state_equal(critic, agent.value.state_dict())
        self.assert_state_equal(observation_scaler, agent._state_preprocessor.state_dict())
        self.assert_state_equal(value_scaler, agent._value_preprocessor.state_dict())
        # Fitted candidate weights have changed; an old report cannot approve
        # them, and a failed promotion must retain the previous good snapshot.
        self.assertFalse(agent.activate_validated(agent.candidate_payload(), report))
        self.assertIn("model_snapshot_changed", agent.wm_probe_rejection_reasons)
        self.assert_state_equal(snapshot, agent.wm_validated.state_dict())

    def test_checkpoint_restores_optimizers_but_rebuilds_replay_and_warmup(self):
        agent = make_agent(self.device)
        collect(agent)
        self.assertTrue(agent.activate_validated(agent.candidate_payload(), synthetic_report(agent)))
        agent._update_world_model()
        training = deepcopy(agent.checkpoint_modules["world_model_training"].state_dict())
        replacement = make_agent(self.device)
        replacement.checkpoint_modules["world_model_training"].load_state_dict(training)
        self.assertEqual(replacement.wm_trainer.updates, agent.wm_trainer.updates)
        self.assertEqual(replacement.wm_imagination.accepted_updates, 1)
        self.assertEqual(replacement.wm_replay.steps, 0)
        self.assertEqual(replacement.wm_replay.size, 0)
        self.assertTrue(replacement.wm_has_validated_snapshot)
        self.assertEqual(replacement.wm_status, "warming_up_after_resume")
        self.assert_state_equal(agent.wm_validated.state_dict(), replacement.wm_validated.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            payload = agent.save_candidate(path)
            replacement.load_candidate(path)
            self.assert_state_equal(payload["model"], replacement.wm_candidate.state_dict())
            self.assertEqual(set(payload["teacher"]),
                             {"policy", "value", "state_preprocessor", "value_preprocessor"})

    def test_passed_probe_does_not_bypass_current_heldout_error_gate(self):
        agent = make_agent(self.device, max_validation_error=.25)
        collect(agent)
        self.assertTrue(agent.activate_validated(agent.candidate_payload(), synthetic_report(agent)))
        actor = deepcopy(agent.policy.state_dict())
        # Simulate an unseen physical effect in held-out real data only. The
        # previously accepted snapshot must stop supplying policy gradients.
        index = agent.wm_adapter.slices["object"].start
        heldout = agent.wm_replay.valid & agent.wm_replay.holdout
        agent.wm_replay.next_states[..., index][heldout] += 10.
        agent._update_world_model()
        self.assertEqual(agent.wm_status, "heldout_error_rejected")
        self.assertEqual(agent.wm_imagination.accepted_updates, 0)
        self.assert_state_equal(actor, agent.policy.state_dict())

    def test_relative_contact_error_rejects_heldout_and_starts_despite_low_physical_average(self):
        agent = make_agent(self.device, max_validation_error=.25, max_start_error=.25)
        collect(agent)
        self.assertTrue(agent.activate_validated(agent.candidate_payload(), synthetic_report(agent)))
        actor = deepcopy(agent.policy.state_dict())
        physical_targets = agent.wm_replay.next_states.clone()
        obj_index = agent.wm_adapter.slices["object"].start
        ee_index = agent.wm_adapter.slices["ee"].start

        def separate_contact(mask):
            # Two 6 cm world-pose errors are diluted among the physical state
            # groups, but together produce a 12 cm object/EE alignment error.
            agent.wm_replay.next_states[..., obj_index][mask] += .06
            agent.wm_replay.next_states[..., ee_index][mask] -= .06

        separate_contact(agent.wm_replay.valid & agent.wm_replay.holdout)
        metrics = agent._update_world_model()
        self.assertLess(metrics["validated/nse_max"], .25)
        self.assertGreater(metrics["validated/relative_error_max"], .25)
        self.assertGreater(metrics["validated/confidence_error_max"], .25)
        self.assertEqual(agent.wm_status, "heldout_error_rejected")
        self.assert_state_equal(actor, agent.policy.state_dict())

        # With validation restored, the same bad geometry at sampled training
        # starts must also receive zero fixed trust weight.
        agent.wm_replay.next_states.copy_(physical_targets)
        separate_contact(agent.wm_replay.valid & ~agent.wm_replay.holdout)
        metrics = agent._update_world_model()
        self.assertLess(metrics["validated/start_nse_mean"], .25)
        self.assertGreater(metrics["validated/start_relative_error_mean"], .25)
        self.assertEqual(metrics["validated/trusted_start_fraction"], 0.)
        self.assertEqual(agent.wm_status, "start_error_rejected")
        self.assertEqual(agent.wm_imagination.accepted_updates, 0)
        self.assert_state_equal(actor, agent.policy.state_dict())

    def test_missing_collector_and_changed_task_metadata_fail_explicitly(self):
        agent = make_agent(self.device)
        payload = agent.candidate_payload()
        payload["metadata"]["state_dim"] += 1
        with self.assertRaisesRegex(ValueError, "task contract"):
            agent.activate_validated(payload, synthetic_report(agent))
        state, context = agent.wm_adapter.capture()
        observation = agent.wm_adapter.observation(state, context)
        with torch.no_grad():
            action, _, _ = agent.act(observation, 0, 10)
            zero = torch.zeros(len(state), 1, device=state.device)
            with self.assertRaisesRegex(RuntimeError, "pre-reset"):
                agent.record_transition(observation, action, zero, observation,
                                        zero.bool(), zero.bool(), {}, 0, 10)


class CPUWorldModelPPOTests(IntegrationChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAWorldModelPPOTests(IntegrationChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
