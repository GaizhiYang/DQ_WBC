"""Aligned single-model scheduling and independent imagined value ownership.

The real fixed8 adapter and skrl modules are used. Scheduling tests replace
expensive optimizer calls with counters so the 20 ms UTD clock can be checked
without running tens of full world-model rounds per assertion.
"""

from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skrl.memories.torch import RandomMemory
from test_world_model_fixed8_ppo import make_fixed8_agent, collect_fixed8
from test_world_model_ppo import TinyPolicy, TinyValue
from utils.wm_ppo import WorldModelPPO
from utils.world_model.recipe import SUPERMIMIC_CONFIG, cosine_ratio


def make_aligned(device="cpu", **overrides):
    # Reuse the production fixed8 environment/low-controller fixture without
    # altering the legacy fixture's defaults or its own integration tests.
    legacy = make_fixed8_agent(device)
    env = legacy.wm_env
    config = dict(recipe="supermimic", mode="hybrid", fixed8=True,
                  replay_envs=12, replay_capacity=6, low_replay_capacity=32,
                  window=8, horizon=1, hidden=[16, 16], contact_hidden=[8],
                  batch_size=4, batches_per_update=1, validation_batch_size=4,
                  imagination_batch=4, imagination_chunk_size=2,
                  model_microbatch_size=2, critic_hidden=[16, 8],
                  critic_epochs=1, critic_minibatches=2,
                  holdout_fraction=.5, warmup_steps=0,
                  updates_per_low_step=.125, observation_noise_scale=0.)
    config.update(overrides)
    env.cfg["env"]["world_model"] = dict(config)
    agent = WorldModelPPO(
        models={"policy": TinyPolicy(1284, device), "value": TinyValue(1284, device)},
        memory=RandomMemory(3, env.num_envs, device=device),
        observation_space=1284, action_space=9, device=device,
        cfg=dict(rollouts=3, learning_epochs=1, mini_batches=1, learning_rate=1e-4,
                 experiment={"write_interval": 0, "checkpoint_interval": 0}),
        wm_env=env, wm_config=config)
    agent.init()
    return agent


class AlignedChecks:
    device = "cpu"

    def assert_module_equal(self, old, module):
        self.assertEqual(old.keys(), module.state_dict().keys())
        for name, value in module.state_dict().items():
            torch.testing.assert_close(old[name], value, rtol=0, atol=0)

    def fake_updates(self, agent):
        state, context = agent.wm_adapter.capture()
        starts = SimpleNamespace(state=state[:4], context=context[:4])

        def fit(replay):
            self.assertIs(replay, agent.wm_low_replay)
            agent.wm_trainer.updates += agent.wm_config["batches_per_update"]
            return {"state_loss": 1e6, "updates": float(agent.wm_trainer.updates)}

        def imagine(start_state, start_context, **kwargs):
            # Uniform starts: no per-sample error weights, high-level dt fixed.
            self.assertNotIn("weights", kwargs)
            self.assertEqual(kwargs["dt"], .16)
            agent.wm_imagination.accepted_updates += 1
            return {"accepted": 1., "objective": 1.}

        return starts, fit, imagine

    def test_recipe_resolves_single_model_and_independent_value_without_changing_gated_default(self):
        aligned = make_aligned(self.device)
        self.assertTrue(aligned.wm_aligned)
        self.assertIs(aligned.wm_candidate, aligned.wm_validated)
        self.assertIs(aligned.wm_imagination.model, aligned.wm_candidate)
        self.assertTrue(all(parameter.requires_grad for parameter in aligned.wm_candidate.parameters()))
        self.assertFalse(aligned.wm_has_validated_snapshot)
        self.assertIsNotNone(aligned.wm_imagination.imagined_critic)
        self.assertIs(aligned.wm_imagination.critic, aligned.value)
        self.assertEqual(aligned.wm_imagination.cfg.value_source, "imagined")
        self.assertFalse(aligned.wm_imagination.cfg.kl_rollback)
        self.assertEqual(aligned.wm_imagination.optimizer.param_groups[0]["betas"], (.9, .95))
        self.assertLess(aligned.wm_low_replay.states.shape[-1], aligned.wm_adapter.state_dim)
        self.assertEqual(aligned.wm_config["hidden"], [16, 16])  # explicit overrides win
        self.assertEqual(aligned.wm_config["low_replay_capacity"], 32)
        self.assertEqual(SUPERMIMIC_CONFIG["low_replay_capacity"], 1000)
        self.assertEqual(SUPERMIMIC_CONFIG["replay_envs"], 4096)

        legacy = make_fixed8_agent(self.device)
        self.assertFalse(legacy.wm_aligned)
        self.assertIsNot(legacy.wm_candidate, legacy.wm_validated)
        self.assertTrue(all(not parameter.requires_grad for parameter in legacy.wm_validated.parameters()))
        self.assertIsNone(legacy.wm_imagination.imagined_critic)
        self.assertTrue(legacy.wm_imagination.cfg.kl_rollback)

    def test_utd_counts_low_steps_carries_fraction_and_does_not_recount_same_data(self):
        agent = make_aligned(self.device, warmup_steps=5, updates_per_low_step=.4)
        starts, fit, imagine = self.fake_updates(agent)
        with patch.object(agent.wm_low_replay, "can_sample", return_value=True), \
                patch.object(agent.wm_replay, "can_sample", return_value=True), \
                patch.object(agent.wm_replay, "sample_starts", return_value=starts) as sampling, \
                patch.object(agent.wm_trainer, "update", side_effect=fit) as fitting, \
                patch.object(agent.wm_imagination, "update", side_effect=imagine) as bptt:
            for low_steps, expected_rounds, expected_credit in (
                    (40, 0, 0.), (48, 3, .2), (56, 3, .4),
                    (64, 3, .6), (64, 0, .6), (80, 7, 0.)):
                agent.wm_low_replay.steps = low_steps
                # Deliberately unrelated high count: UTD must use 20 ms data.
                agent.wm_replay.steps += 1000
                metrics = agent._update_world_model()
                self.assertEqual(metrics["rounds_this_ppo_update"], expected_rounds)
                self.assertAlmostEqual(agent._wm_update_credit, expected_credit, places=6)
            self.assertEqual(fitting.call_count, 16)
            self.assertEqual(bptt.call_count, 16)
            self.assertEqual(agent.wm_update_rounds, 16)
            self.assertEqual(sampling.call_args.kwargs["balance_families"], False)
            self.assertEqual(sampling.call_args.kwargs["recent_fraction"], 0.)

    def test_missing_windows_do_not_discard_accrued_update_credit(self):
        agent = make_aligned(self.device, updates_per_low_step=.4)
        agent.wm_low_replay.steps = 8
        starts, fit, imagine = self.fake_updates(agent)
        with patch.object(agent.wm_low_replay, "can_sample", return_value=False), \
                patch.object(agent.wm_trainer, "update", side_effect=fit) as fitting:
            metrics = agent._update_world_model()
            self.assertEqual(metrics["rounds_this_ppo_update"], 0.)
            self.assertEqual(fitting.call_count, 0)
            self.assertAlmostEqual(agent._wm_update_credit, 3.2, places=6)
        with patch.object(agent.wm_low_replay, "can_sample", return_value=True), \
                patch.object(agent.wm_replay, "can_sample", return_value=True), \
                patch.object(agent.wm_replay, "sample_starts", return_value=starts), \
                patch.object(agent.wm_trainer, "update", side_effect=fit), \
                patch.object(agent.wm_imagination, "update", side_effect=imagine):
            metrics = agent._update_world_model()
            self.assertEqual(metrics["rounds_this_ppo_update"], 3.)
            self.assertAlmostEqual(agent._wm_update_credit, .2, places=6)

    def test_cosine_uses_low_clock_for_all_three_optimizers_and_clamps_after_decay(self):
        agent = make_aligned(self.device, warmup_steps=100000, lr_decay_low_steps=100,
                             learning_rate=3e-4, actor_learning_rate=2e-4,
                             critic_learning_rate=1e-4)
        for low_steps, expected_ratio in ((0, 1.), (50, .515), (100, .03), (300, .03)):
            with self.subTest(low_steps=low_steps):
                agent.wm_low_replay.steps = low_steps
                agent.wm_replay.steps = 0
                metrics = agent._update_world_model()
                self.assertAlmostEqual(metrics["lr_ratio"], expected_ratio, places=10)
                self.assertAlmostEqual(agent.wm_trainer.optimizer.param_groups[0]["lr"],
                                       3e-4 * expected_ratio)
                self.assertAlmostEqual(agent.wm_imagination.optimizer.param_groups[0]["lr"],
                                       2e-4 * expected_ratio)
                self.assertAlmostEqual(agent.wm_imagination.imagined_critic_trainer.optimizer.param_groups[0]["lr"],
                                       1e-4 * expected_ratio)
        self.assertEqual(cosine_ratio(-1), 1.)
        self.assertAlmostEqual(cosine_ratio(250000), .03)
        self.assertAlmostEqual(cosine_ratio(500000), .03)

    def test_aligned_bptt_needs_no_report_error_gate_or_real_ppo_critic(self):
        agent = make_aligned(self.device, max_validation_error=1e-20, max_start_error=1e-20)
        collect_fixed8(agent, high_steps=1)
        self.assertEqual(agent.wm_low_replay.steps, 8)
        self.assertFalse(agent.wm_has_validated_snapshot)
        before_ppo = deepcopy(agent.value.state_dict())
        before_std = agent.policy.log_std.detach().clone()
        before_imagined = deepcopy(agent.wm_imagination.imagined_critic.state_dict())
        before_low = deepcopy(agent.wm_adapter.low_actor.state_dict())
        with patch.object(agent, "_validated_errors", side_effect=AssertionError("prediction gate called")), \
                patch.object(agent, "_prediction_errors", side_effect=AssertionError("start weights called")), \
                patch.object(agent.value, "compute", side_effect=AssertionError("real PPO Critic queried")), \
                patch.object(agent.value, "act", side_effect=AssertionError("real PPO Critic queried")):
            metrics = agent._update_world_model()
        self.assertEqual(metrics["rounds_this_ppo_update"], 1.)
        self.assertEqual(metrics["bptt/accepted_this_ppo_update"], 1.)
        self.assertEqual(metrics["waiting_for_probe"], 0.)
        self.assertEqual(metrics["bptt/critic_updates"], 2.)
        self.assertEqual(agent.wm_status, "hybrid_updated")
        self.assert_module_equal(before_ppo, agent.value)
        self.assert_module_equal(before_low, agent.wm_adapter.low_actor)
        torch.testing.assert_close(before_std, agent.policy.log_std, rtol=0, atol=0)
        self.assertIsNone(agent.policy.log_std.grad)
        self.assertTrue(any(not torch.equal(before_imagined[name], value) for name, value in
                            agent.wm_imagination.imagined_critic.state_dict().items()))
        self.assertTrue(all(parameter.requires_grad for parameter in agent.wm_candidate.parameters()))

    def test_fit_only_and_zero_auxiliary_strength_are_valid_without_bptt(self):
        for options in ({"mode": "fit"}, {"auxiliary_strength": 0.}):
            with self.subTest(options=options):
                agent = make_aligned(self.device, **options)
                collect_fixed8(agent, high_steps=1)
                before = deepcopy(agent.policy.state_dict())
                metrics = agent._update_world_model()
                self.assertEqual(metrics["candidate/updates"], 1.)
                self.assertEqual(metrics["bptt/accepted_this_ppo_update"], 0.)
                self.assert_module_equal(before, agent.policy)
                if options.get("auxiliary_strength") == 0.:
                    self.assertEqual(agent.wm_imagination.optimizer.param_groups[0]["lr"], 0.)
                    self.assertGreater(agent.wm_imagination.imagined_critic_trainer.optimizer.param_groups[0]["lr"], 0.)

    def test_optional_failed_probe_is_diagnostic_but_legacy_recipe_still_waits_for_report(self):
        aligned = make_aligned(self.device, probe_interval=1)
        aligned.wm_low_replay.steps = 8
        aligned.wm_replay.steps = 1
        starts, fit, imagine = self.fake_updates(aligned)
        callbacks = []

        def failed_probe(agent):
            callbacks.append(agent.wm_imagination.accepted_updates)
            agent.wm_probe_rejection_reasons = ["synthetic failed diagnostic"]

        aligned.maybe_validate = failed_probe
        with patch.object(aligned.wm_low_replay, "can_sample", return_value=True), \
                patch.object(aligned.wm_replay, "can_sample", return_value=True), \
                patch.object(aligned.wm_replay, "sample_starts", return_value=starts), \
                patch.object(aligned.wm_trainer, "update", side_effect=fit), \
                patch.object(aligned.wm_imagination, "update", side_effect=imagine):
            metrics = aligned._update_world_model()
        self.assertEqual(callbacks, [1])  # diagnostics follow an already-accepted update
        self.assertEqual(metrics["bptt/accepted_this_ppo_update"], 1.)
        self.assertEqual(aligned.wm_status, "hybrid_updated")
        self.assertFalse(aligned.wm_has_validated_snapshot)

        legacy = make_fixed8_agent(self.device)
        legacy.wm_config["mode"] = "hybrid"
        collect_fixed8(legacy, high_steps=1)
        with patch.object(legacy.wm_imagination, "update", side_effect=AssertionError("legacy gate bypassed")):
            legacy._update_world_model()
        self.assertEqual(legacy.wm_status, "waiting_for_probe")
        self.assertEqual(legacy.wm_imagination.accepted_updates, 0)


class CPUAlignedTests(AlignedChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAAlignedTests(AlignedChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
