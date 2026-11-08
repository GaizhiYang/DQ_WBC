"""Independent manipulation budgets, value preparation and real-data ownership."""

from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.world_model.schedule import ManipulationSchedule, ScheduleConfig
from utils.world_model.recipe import MANIPULATION_CONFIG, SUPERMIMIC_CONFIG
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
from utils.world_model.probes import configuration_fingerprint
from test_world_model_aligned import make_aligned
from test_world_model_fixed8_ppo import collect_fixed8
from test_world_model_imagined_critic import Actor, PPOCritic, Model, Adapter, tree_equal


def make_manipulation(**overrides):
    settings = dict(recipe="manipulation", updates_per_low_step=0., wm_max_batches=1,
                    wm_prepare_boundaries=1, value_prepare_boundaries=1,
                    value_prepare_batches=1, actor_stage1_boundaries=1,
                    actor_stage2_boundaries=1, diagnostics_interval=1000,
                    diagnostics_batch_size=4)
    settings.update(overrides)
    return make_aligned(**settings)


class ScheduleTests(unittest.TestCase):
    def test_default_stages_have_independent_actor_budget(self):
        schedule = ManipulationSchedule()
        expected = [("W", 0)] * 4 + [("V", 0)] * 2 + [("B1", 1)] * 4 + [("B2", 2)] * 4 + [("B3", 4)]
        for index, (phase, actors) in enumerate(expected, 1):
            plan = schedule.plan(index * 2048)
            self.assertEqual((plan.phase, plan.actor_updates), (phase, actors))
            self.assertEqual(plan.model_batches, 1)
            self.assertEqual(plan.value_batches, 8 if phase == "V" else 0)
            self.assertTrue(schedule.complete(plan, model_updates=1,
                                             value_batches=plan.value_batches, actor_updates=actors))
        self.assertEqual(MANIPULATION_CONFIG["updates_per_low_step"], 0.)
        self.assertEqual(SUPERMIMIC_CONFIG["updates_per_low_step"], .4)

    def test_cap_and_missing_windows_discard_integer_debt_not_fraction(self):
        schedule = ManipulationSchedule(ScheduleConfig(batch_size=10, wm_max_batches=2))
        first = schedule.plan(35)
        self.assertEqual((first.model_batches, first.discarded_batches), (2, 1))
        self.assertAlmostEqual(schedule.credit, .5)
        unavailable = schedule.plan(65, can_sample=False)
        self.assertEqual((unavailable.model_batches, unavailable.discarded_batches), (0, 3))
        fresh = schedule.plan(75)
        self.assertEqual(fresh.model_batches, 1)
        repeated = schedule.plan(75)
        self.assertEqual((repeated.model_batches, repeated.actor_updates), (0, 0))
        self.assertAlmostEqual(schedule.credit, .5)
        with self.assertRaisesRegex(ValueError, "backwards"):
            schedule.plan(0)

    def test_failed_work_does_not_advance_or_accumulate_actor_steps(self):
        schedule = ManipulationSchedule(ScheduleConfig(wm_prepare_boundaries=1, value_prepare_boundaries=1))
        plan = schedule.plan(2048)
        self.assertFalse(schedule.complete(plan, model_updates=0))
        self.assertEqual(schedule.phase, "W")
        plan = schedule.plan(4096)
        schedule.complete(plan, model_updates=1)
        plan = schedule.plan(6144)
        self.assertFalse(schedule.complete(plan, model_updates=1, value_batches=7))
        self.assertEqual(schedule.phase, "V")
        plan = schedule.plan(8192)
        self.assertEqual(plan.value_batches, 8)
        schedule.complete(plan, model_updates=1, value_batches=8)
        plan = schedule.plan(10240)
        self.assertEqual(plan.actor_updates, 1)
        self.assertFalse(schedule.complete(plan, model_updates=1, actor_updates=0))
        plan = schedule.plan(12288)
        self.assertEqual(plan.actor_updates, 1)
        schedule.reset_replay()
        self.assertEqual(schedule.phase, "W")
        self.assertEqual(schedule.state_dict()["seen_transitions"], 0)

    def test_invalid_schedule_is_rejected(self):
        for kwargs in ({"wm_epochs": float("nan")}, {"wm_max_batches": 0},
                       {"actor_max_updates": 5}, {"actor_stage1_updates": 3},
                       {"value_prepare_batches": True}):
            with self.assertRaises(ValueError):
                ScheduleConfig(**kwargs)


class ValuePreparationTests(unittest.TestCase):
    def make(self, adapter=None):
        return ImaginationTrainer(Model(), adapter or Adapter(), Actor(), PPOCritic(),
            ImaginationConfig(value_source="imagined", horizon=2, strength=1.,
                              critic_hidden=[8], critic_epochs=1, critic_minibatches=2,
                              chunk_size=2, kl_rollback=False))

    def test_value_only_keeps_actor_optimizer_std_and_frozen_modules_unchanged(self):
        trainer = self.make()
        state, context = torch.zeros(5, 2), torch.zeros(5, 1)
        trainer.update(state, context)  # Create nonzero Actor Adam moments first.
        actor = deepcopy(trainer.actor.state_dict())
        optimizer = deepcopy(trainer.optimizer.state_dict())
        model = deepcopy(trainer.model.state_dict())
        value = deepcopy(trainer.imagined_critic.state_dict())
        flags = [p.requires_grad for p in trainer.actor.parameters()]
        with patch.object(trainer.optimizer, "step", side_effect=AssertionError("Actor step during V preparation")):
            result = trainer.update_value(state, context)
        self.assertEqual(result["value_prepared"], 1.)
        self.assertEqual(result["actor_updates"], 0.)
        tree_equal(self, actor, trainer.actor.state_dict())
        tree_equal(self, optimizer, trainer.optimizer.state_dict())
        tree_equal(self, model, trainer.model.state_dict())
        self.assertEqual(flags, [p.requires_grad for p in trainer.actor.parameters()])
        self.assertTrue(any(not torch.equal(value[k], v) for k, v in trainer.imagined_critic.state_dict().items()))

    def test_raw_latent_cost_is_shared_by_actor_rollout_and_detached_value_targets(self):
        class CostAdapter(Adapter):
            def raw_action_cost(self, action, latent_mean):
                return latent_mean[:, 6].square()
        trainer = self.make(CostAdapter())
        with torch.no_grad():
            trainer.actor.bias[6] = 10.
        state, context = torch.zeros(3, 2), torch.zeros(3, 1)
        rollout = trainer.imagine(state, context)
        torch.testing.assert_close(rollout.rewards[0], torch.full((3,), -98.))
        self.assertTrue(rollout.action_costs.requires_grad)
        gradient, = torch.autograd.grad(rollout.rewards.sum(), trainer.actor.bias)
        self.assertLess(float(gradient[6]), 0.)
        batch = trainer.imagined_critic_trainer.prepare(rollout, context)
        self.assertFalse(batch.targets.requires_grad)
        self.assertTrue(bool((batch.targets < -90).all()))


class ManipulationIntegrationTests(unittest.TestCase):
    def test_collection_counts_train_rows_and_updates_stats_once(self):
        agent = make_manipulation(running_normalization=True)
        with patch.object(agent.wm_candidate, "observe_transition", wraps=agent.wm_candidate.observe_transition) as observe:
            collect_fixed8(agent, high_steps=2)
            self.assertEqual(observe.call_count, 16)
        train_envs = len(agent.wm_env_ids) - agent.wm_low_replay.holdout_envs
        expected = 16 * train_envs
        self.assertEqual(int(agent.wm_low_replay.valid_train_transitions), expected)
        self.assertEqual(float(agent.wm_candidate.in_norm.count), expected)
        self.assertEqual(float(agent.wm_candidate.out_norm.count), expected)
        self.assertEqual(agent.wm_replay.holdout_mode, "environment")
        self.assertFalse(agent.wm_replay.allow_partial_windows)
        self.assertTrue(agent.wm_low_replay.allow_partial_windows)
        with patch.object(agent.wm_candidate, "observe_transition", side_effect=AssertionError("Repeated observation")):
            agent._update_world_model()
            agent._manipulation_diagnostics()
        self.assertEqual(float(agent.wm_candidate.in_norm.count), expected)

    def test_production_schedule_does_not_invoke_bptt_until_w_and_v_complete(self):
        agent = make_manipulation()
        calls = {"actor": 0, "value": 0, "wm": 0}
        def fit(replay, *, batches, **kwargs):
            calls["wm"] += batches
            agent.wm_trainer.updates += batches
            return {"updates": float(agent.wm_trainer.updates)}
        def value(*args, **kwargs):
            calls["value"] += 1
            return {"value_prepared": 1., "critic_updates": 2.}
        def actor(*args, **kwargs):
            calls["actor"] += 1
            return {"accepted": 1., "critic_updates": 2.}
        with patch.object(agent.wm_trainer, "update", side_effect=fit), \
             patch.object(agent.wm_imagination, "update_value", side_effect=value), \
             patch.object(agent.wm_imagination, "update", side_effect=actor):
            for expected_phase, expected_actor in ((0, 0), (1, 0), (2, 1), (3, 2), (4, 4)):
                collect_fixed8(agent, high_steps=1)
                metrics = agent._update_world_model()
                self.assertEqual(metrics["schedule/phase"], expected_phase)
                self.assertEqual(metrics["schedule/actor_accepted"], expected_actor)
                self.assertEqual(metrics["schedule/wm_updates"], 1.)
            self.assertEqual(calls, {"actor": 7, "value": 1, "wm": 5})
            agent._update_world_model()  # No new data: no Actor debt is payable.
            self.assertEqual(calls, {"actor": 7, "value": 1, "wm": 5})

    def test_legacy_missing_task_metadata_is_not_reinterpreted_as_v2(self):
        agent = make_aligned()
        old = deepcopy(agent.wm_metadata)
        for key in ("task_contract", "task_aux_alpha", "task_objective", "raw_action_bound_weight",
                    "raw_gripper_latent_bound", "reward_contract", "critic_memory", "small_commands_zero"):
            old.pop(key, None)
        agent._check_metadata(old)
        old["task_contract"] = "manipulation_v2"
        with self.assertRaisesRegex(ValueError, "schema/task"):
            agent._check_metadata(old)

    def test_zero_diagnostic_interval_disables_extra_sampling(self):
        agent = make_manipulation(diagnostics_interval=0)
        collect_fixed8(agent, high_steps=1)
        with patch.object(agent, "_manipulation_diagnostics", side_effect=AssertionError("disabled diagnostics")):
            agent._update_world_model()

    def test_checkpoint_restarts_local_preparation_but_keeps_global_lr_clock(self):
        agent = make_manipulation()
        collect_fixed8(agent, high_steps=1)
        checkpoint = agent.checkpoint_modules["world_model_training"]
        payload = deepcopy(checkpoint.state_dict())
        self.assertEqual(payload["total_low_steps"], 8)
        self.assertIn("manipulation_schedule", payload)
        checkpoint.load_state_dict(payload)
        self.assertEqual(agent._wm_total_low_steps, 8)
        self.assertEqual(agent.wm_schedule.phase, "W")
        self.assertEqual(agent.wm_schedule.seen_transitions, 0)
        self.assertEqual(agent.wm_low_replay.steps, 0)
        self.assertEqual(int(agent.wm_low_replay.valid_train_transitions), 0)

    def test_checkpoint_audit_captures_rng_and_sample_counts_without_restoring_rng(self):
        agent = make_manipulation()
        collect_fixed8(agent, high_steps=2)
        python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        checkpoint = agent.checkpoint_modules["world_model_training"]
        payload = checkpoint.state_dict()
        audit = payload["audit"]
        self.assertEqual(audit["reference_source_commit"], "eda7da480c6d92a68cc41c9628b7efdae08cae23")
        self.assertEqual(audit["resolved_config_hash"], configuration_fingerprint(agent.wm_config))
        self.assertEqual(audit["rng_state"]["python"], python_rng)
        np.testing.assert_array_equal(audit["rng_state"]["numpy"][1], numpy_rng[1])
        torch.testing.assert_close(audit["rng_state"]["torch_cpu"], torch_rng)
        self.assertIsNone(audit["rng_state"]["torch_cuda"])
        self.assertFalse(audit["rng_restore_supported"])
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), torch_rng)
        counts = audit["sampling_counts"]
        self.assertEqual(counts["high_vector_steps"], 2)
        self.assertEqual(counts["low_vector_steps"], 16)
        self.assertEqual(counts["selected_low_rows"], 16 * len(agent.wm_env_ids))
        self.assertEqual(counts["low_valid_train_rows_since_replay_reset"],
                         int(agent.wm_low_replay.valid_train_transitions))
        torch.rand(3)
        current_rng = torch.get_rng_state()
        checkpoint.load_state_dict(payload)
        torch.testing.assert_close(torch.get_rng_state(), current_rng)
        self.assertEqual(agent._wm_total_high_steps, 2)

    def test_checkpoint_rejects_same_shape_execution_configuration_mismatch(self):
        agent = make_manipulation()
        checkpoint = agent.checkpoint_modules["world_model_training"]
        original = checkpoint.state_dict()
        for name, changed in (("amp", "bf16"), ("layer_norm", True), ("running_normalization", True)):
            for location in ("config", "metadata"):
                payload = deepcopy(original)
                target = payload[location] if location == "config" else payload[location]["model_execution"]
                target[name] = changed
                with self.assertRaises(ValueError), patch.object(agent.wm_trainer, "load_state_dict",
                                                                 side_effect=AssertionError("loaded before contract check")):
                    checkpoint.load_state_dict(payload)
        bundle = agent.candidate_payload()
        bundle["config"]["amp"] = "bf16"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.pt"
            torch.save(bundle, path)
            with self.assertRaisesRegex(ValueError, "execution configuration"):
                agent.load_candidate(path)


if __name__ == "__main__":
    unittest.main()
