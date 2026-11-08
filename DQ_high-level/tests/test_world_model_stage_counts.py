"""Real high-transition counters are diagnostic, scoped, and RNG independent."""

from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.world_model.stage_counts import RealStageTransitionCounts, captured_stage_flags, STAGES
from test_world_model_manipulation_schedule import make_manipulation
from test_world_model_fixed8_ppo import collect_fixed8


class RealStageCountTests(unittest.TestCase):
    def test_family_entries_exits_and_eligible_denominators_exclude_invalid_rows(self):
        counter = RealStageTransitionCounts(("full_task", "pregrasp", "near_goal"), "cpu")
        before = {name: torch.tensor([False, True, False, False, False]) for name in STAGES}
        after = {name: torch.tensor([True, False, True, True, False]) for name in STAGES}
        counter.update(torch.tensor([0, 1, 2, 2, 8]), torch.tensor([1, 1, 1, 0, 1]),
                       before, after, torch.tensor([False, True, False, True, True]))
        metrics = counter.metrics()
        self.assertEqual(metrics["real_stage/all/valid_transitions"], 4.)
        self.assertEqual(metrics["real_stage/all/lifted_proxy_entries"], 2.)
        self.assertEqual(metrics["real_stage/all/lifted_proxy_eligible"], 3.)
        self.assertAlmostEqual(metrics["real_stage/all/lifted_proxy_entry_rate"], 2/3)
        self.assertEqual(metrics["real_stage/pregrasp/held_proxy_exits"], 1.)
        self.assertEqual(metrics["real_stage/near_goal/held_proxy_entries"], 1.)
        self.assertEqual(metrics["real_stage/all/failure_terminals"], 2.)
        self.assertEqual(metrics["real_stage/unknown_family/failure_terminals"], 1.)
        self.assertIn("not contact truth", counter.state_dict()["definitions"]["lifted_proxy"])

    def test_each_pair_is_independent_of_previous_episode_and_reset_statistics(self):
        counter = RealStageTransitionCounts(("near_goal",), "cpu")
        present = {name: torch.ones(1, dtype=torch.bool) for name in STAGES}
        absent = {name: torch.zeros(1, dtype=torch.bool) for name in STAGES}
        # The reset begins inside a proxy stage: its initial occupancy is not
        # credited as something achieved by the preceding episode's action.
        counter.update(torch.zeros(1), torch.ones(1), present, present, torch.zeros(1))
        self.assertEqual(counter.metrics()["real_stage/all/completed_entries"], 0.)
        counter.update(torch.zeros(1), torch.ones(1), absent, present, torch.zeros(1))
        self.assertEqual(counter.metrics()["real_stage/all/completed_entries"], 1.)
        snapshot = counter.state_dict()
        counter.reset()
        self.assertEqual(counter.metrics()["real_stage/all/valid_transitions"], 0.)
        self.assertEqual(int(snapshot["counts"].sum()), 8)

    def test_captured_flags_use_measured_ee_and_actual_gripper_without_fk_or_rng(self):
        adapter = SimpleNamespace(slices={"object": slice(0, 13), "ee": slice(13, 26),
                                          "table_height": slice(26, 27), "hold": slice(27, 28),
                                          "completed": slice(28, 29)}, lift_threshold=.1)
        state, action, context = torch.zeros(3, 29), torch.zeros(3, 9), torch.zeros(3, 3)
        state[:, 2] = .3
        state[:, 15] = .3
        state[:, 26] = .1
        context[:, 2] = .1
        action[1, 6] = 1.  # Height/proximity alone is not the task's lift proxy.
        state[2, 15] = 1.
        state[0, 27], state[0, 28] = 1., 1.
        python_rng, numpy_rng, rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        flags = captured_stage_flags(adapter, state, action, context)
        self.assertEqual(flags["lifted_proxy"].tolist(), [True, False, False])
        self.assertEqual(flags["held_proxy"].tolist(), [True, False, False])
        self.assertEqual(flags["completed"].tolist(), [True, False, False])
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        torch.testing.assert_close(torch.get_rng_state(), rng)

    def test_real_collector_counts_once_per_high_step_and_exports_audit(self):
        agent = make_manipulation()
        with patch.object(agent.wm_stage_counts, "update", wraps=agent.wm_stage_counts.update) as update:
            collect_fixed8(agent, high_steps=2)
        self.assertEqual(update.call_count, 2)  # Not the sixteen low transitions.
        metrics = agent.wm_stage_counts.metrics()
        self.assertEqual(metrics["real_stage/all/valid_transitions"], 24.)
        # Both fixed train and held-out selected slots contribute to diagnostics.
        self.assertGreater(metrics["real_stage/all/valid_transitions"],
                           int(agent.wm_replay.valid_train_transitions))
        checkpoint = agent.checkpoint_modules["world_model_training"]
        payload = checkpoint.state_dict()
        audit = payload["audit"]["real_stage_transitions"]
        self.assertEqual(int(audit["counts"][:, 0].sum()), 24)
        self.assertIn("train_and_holdout", audit["scope"])
        checkpoint.load_state_dict(payload)
        self.assertEqual(agent.wm_stage_counts.metrics()["real_stage/all/valid_transitions"], 0.)

    def test_actual_success_and_timeout_are_not_counted_as_failure_terminals(self):
        agent = make_manipulation()
        state, context = agent.wm_adapter.capture()
        n = len(state)
        transition = dict(state=state, next_state=state.clone(), context=context,
                          actions=torch.zeros(n, 9), family=torch.zeros(n, dtype=torch.long),
                          terminated=torch.zeros(n, dtype=torch.bool),
                          truncated=torch.zeros(n, dtype=torch.bool), success=torch.zeros(n, dtype=torch.bool))
        transition["terminated"][:3] = True
        transition["success"][0] = True
        transition["truncated"][1] = True
        agent.wm_replay.steps = 1
        agent.wm_replay.valid[0].fill_(True)
        with patch.object(agent.wm_adapter, "fk", side_effect=AssertionError("No extra FK for real counters")):
            agent._record_real_stage_transitions(transition)
        metrics = agent.wm_stage_counts.metrics()
        self.assertEqual(metrics["real_stage/all/failure_terminals"], 1.)
        self.assertEqual(metrics["real_stage/all/completed_entries"], 1.)


if __name__ == "__main__":
    unittest.main()
