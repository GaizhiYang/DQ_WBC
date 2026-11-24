"""Scout & Relay components and the real environment reset path on CPU stand-ins.

The environment methods are extracted from envs/b1z1_pickmulti.py by AST, as in
test_sgs_env.py, so no Isaac Gym import or CUDA context is required.
"""

import ast
import math
import unittest
from unittest.mock import patch

import torch

from utils.pick_success import RollingEpisodeSuccess
from utils.scout_relay import (ISOverlapTracker, RelayBank, SnapshotBuffer, branch_layout,
                               scout_probabilities)
from utils.sgs import SGSSampler, sgs_logits

from test_sgs_env import NAMESPACE, TestEnvironment, make_bank, method_node


def specs():
    return {"robot_root": ((13,), torch.float32), "flag": ((), torch.bool),
            "interval": ((), torch.long)}


def relay_bank(capacity=4, alpha_max=0.3, sampling="sgs", num_configs=6):
    return RelayBank(capacity, specs(), num_configs, [0.25, 0.5], history_size=50,
                     target_success_rate=0.5, concentration=1.0, epsilon=1e-4, temperature=2.0,
                     alpha_max=alpha_max, sampling=sampling, max_chains_per_config=2,
                     master_rate=0.9, master_count=20, useless_count=40, device="cpu")


def entry_states(markers):
    count = len(markers)
    roots = torch.zeros(count, 13)
    roots[:, 0] = torch.tensor(markers, dtype=torch.float32)
    roots[:, 6] = 1
    return {"robot_root": roots, "flag": torch.ones(count, dtype=torch.bool),
            "interval": torch.full((count,), 45, dtype=torch.long)}


def check_chain_bookkeeping(test, bank):
    alive = bank.chain_alive
    test.assertEqual(int(alive.sum()), int(bank.active.sum()))
    for chain in (alive > 0).nonzero().flatten().tolist():
        members = bank.active & (bank.chain_slot == chain)
        test.assertEqual(int(members.sum()), int(alive[chain]))
        test.assertTrue(bool((bank.source_config[members] == bank.chain_source[chain]).all()))
    expected = torch.zeros_like(bank.config_chains)
    for chain in (alive > 0).nonzero().flatten().tolist():
        expected[bank.chain_source[chain]] += 1
    test.assertTrue(torch.equal(expected, bank.config_chains))


class BranchAndCurriculumTests(unittest.TestCase):
    def test_branch_layout(self):
        self.assertEqual(branch_layout(4, 1, 0.5, "cpu").tolist(), [0, 0, 0, 0])
        self.assertEqual(branch_layout(10, 2, 0.5, "cpu").tolist(), [0] * 5 + [1] * 5)
        self.assertEqual(branch_layout(10, 3, 0.5, "cpu").tolist(), [0] * 5 + [1] * 2 + [2] * 3)
        self.assertEqual(branch_layout(8, 3, 0.25, "cpu").bincount().tolist(), [2, 3, 3])
        for arguments in ((4, 2, 0.0), (4, 2, 1.0), (2, 3, 0.5), (4, 0, 0.5)):
            with self.assertRaises(ValueError):
                branch_layout(arguments[0], arguments[1], arguments[2], "cpu")

    def test_kernel_refactor_keeps_the_original_probabilities(self):
        sampler = SGSSampler(5, history_size=4)
        sampler.observe(torch.tensor([0, 0, 1, 2, 2, 2, 3]), torch.tensor([1, 0, 1, 0, 0, 1, 1]))
        rates = sampler.success_rates()
        log_weight = (0.5 * torch.log(rates + 1e-4) + 0.5 * torch.log(1 - rates + 1e-4))
        log_epsilon = torch.full_like(log_weight, math.log(1e-4))
        original = torch.softmax(torch.logaddexp(torch.maximum(log_weight, log_epsilon), log_epsilon) / 2.0, 0)
        torch.testing.assert_close(sampler.probabilities(), original)
        torch.testing.assert_close(sampler.logits(), sgs_logits(rates, 0.5, 1.0, 1e-4))

    def test_reset_entries_forgets_only_the_given_slots(self):
        sampler = SGSSampler(3, history_size=4, p_init=0.5)
        sampler.observe(torch.tensor([0, 1, 1]), torch.tensor([1, 0, 1]), step=7)
        sampler.reset_entries(torch.tensor([1]))
        self.assertEqual(sampler.count.tolist(), [1, 0, 0])
        self.assertEqual(sampler.success_sum.tolist(), [1, 0, 0])
        self.assertEqual(sampler.last_seen.tolist(), [7, -1, -1])
        self.assertEqual(sampler.success_rates().tolist(), [1.0, 0.5, 0.5])
        state = sampler.state_dict()
        SGSSampler(3, history_size=4, p_init=0.5).load_state_dict(state)  # still self-consistent

    def test_scout_curriculum_prefers_configurations_the_leader_fails(self):
        scout = SGSSampler(3)
        leader_rates = torch.tensor([0.0, 1.0, 0.5])
        torch.testing.assert_close(scout_probabilities(scout, leader_rates, 0.0), scout.probabilities())
        probabilities = scout_probabilities(scout, leader_rates, 1.0)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)
        self.assertGreater(probabilities[0].item(), probabilities[2].item())
        self.assertGreater(probabilities[2].item(), probabilities[1].item())
        uniform = SGSSampler(3, mode="uniform")
        torch.testing.assert_close(scout_probabilities(uniform, leader_rates, 1.0), torch.full((3,), 1 / 3))


class SnapshotBufferTests(unittest.TestCase):
    def test_nearest_snapshot_before_the_hold_with_deduplication(self):
        buffer = SnapshotBuffer(torch.tensor([1, 3]), 4, 30, 5, specs(), "cpu")
        self.assertEqual(buffer.num_slots, 5)
        for progress in (5, 10, 15, 20):
            values = entry_states([100 + progress])
            buffer.capture(torch.tensor([1]), torch.tensor([progress]), values)
        # Leader env 0 is not a source; capture ignores it.
        buffer.capture(torch.tensor([0]), torch.tensor([5]), entry_states([1]))
        fractions = torch.tensor([0.2, 0.4, 0.6, 0.8])
        rows, slots, ok = buffer.choose(torch.tensor([1]), torch.tensor([21]), fractions)
        # Targets 4.2/8.4/12.6/16.8 -> steps 5/10/15/15; the duplicate 15 keeps f = 0.6.
        self.assertEqual(ok.tolist(), [[True, True, True, False]])
        self.assertEqual(slots[ok].tolist(), [0, 1, 2])
        markers = buffer.gather(rows[:, None].expand(-1, 4)[ok], slots[ok])["robot_root"][:, 0]
        self.assertEqual(markers.tolist(), [105., 110., 115.])
        # Snapshots at or after the horizon never qualify.
        _, slots, ok = buffer.choose(torch.tensor([1]), torch.tensor([10]), torch.tensor([0.9]))
        self.assertEqual(ok.tolist(), [[True]])
        self.assertEqual(buffer.step[rows[0], slots[0, 0]].item(), 5)
        buffer.clear(torch.tensor([1, 2]))
        _, _, ok = buffer.choose(torch.tensor([1]), torch.tensor([21]), fractions)
        self.assertFalse(bool(ok.any()))


class RelayBankTests(unittest.TestCase):
    def test_insert_evict_and_chain_bookkeeping(self):
        bank = relay_bank(capacity=4)
        slots = bank.insert(entry_states([1, 2, 3, 4]), torch.tensor([0, 0, 5, 5]),
                            torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), step=3)
        self.assertEqual(sorted(slots.tolist()), [0, 1, 2, 3])
        self.assertEqual(bank.config_chains[[0, 5]].tolist(), [1, 1])
        self.assertEqual(bank.chain_allowance(torch.tensor([0, 1])).tolist(), [1, 2])
        check_chain_bookkeeping(self, bank)
        # Make the entry with marker 2 clearly mastered-looking (lowest SGS weight).
        low = slots[1].item()
        bank.stats.observe(torch.full((15,), low), torch.ones(15, dtype=torch.bool))
        new = bank.insert(entry_states([9]), torch.tensor([3]), torch.tensor([0]), torch.tensor([0]), step=4)
        self.assertEqual(new.tolist(), [low])
        self.assertEqual(bank.storage["robot_root"][low, 0].item(), 9.)
        self.assertEqual(int(bank.counters["evicted"]), 1)
        self.assertEqual(bank.stats.count[low].item(), 0)  # fresh statistics for the new entry
        self.assertEqual(bank.config_chains[[0, 3, 5]].tolist(), [1, 1, 1])
        check_chain_bookkeeping(self, bank)
        self.assertEqual(bank.uid.unique().numel(), 4)

    def test_observe_guards_quarantines_and_retires(self):
        bank = relay_bank(capacity=4)
        slots = bank.insert(entry_states([1, 2, 3]), torch.tensor([0, 0, 1]),
                            torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0]), step=0)
        uids = bank.uid[slots].clone()
        # A stale uid (slot reused since the episode started) is ignored.
        bank.observe(slots[:1], uids[:1] + 100, torch.tensor([True]), torch.tensor([False]), 1)
        self.assertEqual(bank.stats.count[slots[0]].item(), 0)
        # A physical failure right after the restore quarantines the entry.
        bank.observe(slots[2:], uids[2:], torch.tensor([False]), torch.tensor([True]), 1)
        self.assertFalse(bank.active[slots[2]].item())
        self.assertEqual(int(bank.counters["quarantined"]), 1)
        self.assertEqual(bank.config_chains[1].item(), 0)
        # Twenty successes master the f = 0.25 entry (front of its chain).
        bank.observe(slots[:1].repeat(20), uids[:1].repeat(20), torch.ones(20, dtype=torch.bool),
                     torch.zeros(20, dtype=torch.bool), 2)
        self.assertFalse(bank.active[slots[0]].item())
        self.assertEqual(int(bank.counters["mastered"]), 1)
        self.assertEqual(int(bank.counters["front_mastered"]), 1)
        # Forty failures retire an entry as useless.
        bank.observe(slots[1:2].repeat(40), uids[1:2].repeat(40), torch.zeros(40, dtype=torch.bool),
                     torch.zeros(40, dtype=torch.bool), 3)
        self.assertFalse(bank.active[slots[1]].item())
        self.assertEqual(int(bank.counters["useless"]), 1)
        self.assertEqual(int(bank.counters["episodes"]), 60)
        self.assertEqual(int(bank.counters["successes"]), 20)
        self.assertEqual(int(bank.active.sum()), 0)
        self.assertEqual(int(bank.config_chains.sum()), 0)
        check_chain_bookkeeping(self, bank)

    def test_leader_distribution_caps_the_relay_mass(self):
        bank_logits = sgs_logits(torch.zeros(6), 0.5, 1.0, 1e-4)  # unvisited bank: low weight
        bank = relay_bank(capacity=4, alpha_max=0.3)
        empty = bank.leader_distribution(bank_logits)
        self.assertEqual(empty[6:].sum().item(), 0.0)
        self.assertAlmostEqual(empty.sum().item(), 1.0, places=5)
        bank.insert(entry_states([1, 2, 3]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]),
                    torch.tensor([0, 1, 0]), step=0)
        probabilities = bank.leader_distribution(bank_logits)
        self.assertAlmostEqual(probabilities.sum().item(), 1.0, places=5)
        self.assertAlmostEqual(probabilities[6:].sum().item(), 0.3, places=5)
        self.assertEqual(probabilities[6:][~bank.active].sum().item(), 0.0)
        uncapped = relay_bank(capacity=4, alpha_max=1.0)
        uncapped.insert(entry_states([1, 2, 3]), torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]),
                        torch.tensor([0, 1, 0]), step=0)
        self.assertGreater(uncapped.leader_distribution(bank_logits)[6:].sum().item(), 0.3)
        uniform = relay_bank(capacity=4, sampling="uniform")
        slots = uniform.insert(entry_states([1, 2]), torch.tensor([0, 1]), torch.tensor([0, 1]),
                               torch.tensor([0, 0]), step=0)
        uniform.stats.observe(slots[:1].repeat(10), torch.ones(10, dtype=torch.bool))
        relay = uniform.leader_distribution(bank_logits)[6:]
        self.assertAlmostEqual(relay[slots[0]].item(), relay[slots[1]].item(), places=6)
        self.assertAlmostEqual(relay.sum().item(), uniform.last_mass.item(), places=6)
        metrics = bank.metrics()
        self.assertTrue(all(isinstance(value, torch.Tensor) for value in metrics.values()))
        self.assertEqual(int(metrics["Relay / active_entries"]), 3)


class OverlapTrackerTests(unittest.TestCase):
    def test_kish_ess_per_configuration(self):
        tracker = ISOverlapTracker(3, device="cpu", min_samples=4)
        configs = torch.tensor([0] * 8 + [1] * 8 + [2] * 2)
        log_mu = torch.cat((torch.zeros(8), torch.tensor([3., -3.] * 4), torch.zeros(2)))
        tracker.update(configs, log_mu)
        ess = tracker.ess()
        self.assertAlmostEqual(ess[0].item(), 1.0, places=5)
        expected = ((math.exp(3) + math.exp(-3)) / 2) ** 2 / ((math.exp(6) + math.exp(-6)) / 2)
        self.assertAlmostEqual(ess[1].item(), expected, places=4)
        self.assertTrue(math.isnan(ess[2].item()))


def relay_environment_class():
    methods = ("_init_scout_relay", "_relay_state_values", "_capture_relay_snapshots",
               "scout_relay_gap", "_reset_scout_relay_envs", "_create_relay_chains",
               "_scout_relay_metrics")
    cls = ast.ClassDef(name="RelayEnvironment", bases=[ast.Name(id="TestEnvironment", ctx=ast.Load())],
                       keywords=[], decorator_list=[],
                       body=[method_node("b1z1_pickmulti.py", "B1Z1PickMulti", name) for name in methods])
    namespace = dict(NAMESPACE, TestEnvironment=TestEnvironment, RollingEpisodeSuccess=RollingEpisodeSuccess)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "b1z1_pickmulti.py", "exec"), namespace)
    return namespace["RelayEnvironment"]


RelayEnvironment = relay_environment_class()


class RelayEnvironmentTests(unittest.TestCase):
    def environment(self, branches=2, relay=True, source="scouts", eval_mode=False):
        env = RelayEnvironment(num_envs=4, bank=make_bank())
        env.eval = eval_mode
        env.hold_steps = 3
        env.cfg = {"env": {"cameraVisibilityReset": False,
                           "sapg": {"branches": branches, "leader_fraction": 0.5, "scout_need_beta": 1.0},
                           "relay": {"enabled": relay, "source": source, "capacity": 16,
                                     "snapshot_interval": 2, "fractions": [0.25, 0.5, 0.75]}}}
        env._init_scout_relay()
        return env

    def finish(self, env, ids, progress, success=False, restore_failure=False):
        ids = torch.as_tensor(ids)
        env.progress_buf[ids] = progress
        env.pick_success_recorded[ids] = success
        if restore_failure:
            env.restore_failure_buf[ids] = True
        env.reset_buf[ids] = 1
        env._sgs_pending_done[ids] = True

    def test_evaluation_and_defaults_disable_the_method(self):
        env = self.environment(eval_mode=True)
        self.assertFalse(env.scout_relay_active)
        self.assertEqual(env.sapg_num_branches, 1)
        self.assertIsNone(env.relay_bank)
        env = self.environment(branches=1, relay=False)
        self.assertFalse(env.scout_relay_active)
        with self.assertRaisesRegex(ValueError, "scouts"):
            self.environment(branches=1, relay=True, source="scouts")
        env = self.environment(branches=1, relay=True, source="leader")
        self.assertEqual(env.relay_snapshots.source_env_ids.tolist(), [0, 1, 2, 3])

    def test_scout_success_becomes_a_leader_relay_start(self):
        env = self.environment()
        self.assertEqual(env.sapg_branch_id.tolist(), [0, 0, 1, 1])
        env.reset()  # first starts; nothing is credited
        self.assertEqual(int(env.sgs_sampler.count.sum()), 0)
        self.assertTrue(bool((env.active_config_id < env.sgs_bank.num_configs).all()))
        scout_config = env.active_config_id[2].item()

        # Scout env 2 runs, snapshots every 2 steps, and succeeds at progress 10.
        for progress in (2, 4, 6, 8):
            env.progress_buf[2] = progress
            env._robot_root_states[2, 0] = 100. + progress
            env._capture_relay_snapshots()
        self.finish(env, [2], progress=10, success=True)
        env.reset()
        # Credited to the scout only; the hold began at 10 - 3 + 1 = 8, so the
        # chain uses steps 2/4/6 (fractions 0.25/0.5/0.75 of 8).
        self.assertEqual(int(env.sgs_sampler.count.sum()), 0)
        self.assertEqual(env.scout_samplers[0].count[scout_config].item(), 1)
        bank = env.relay_bank
        markers = sorted(bank.storage["robot_root"][bank.active, 0].tolist())
        self.assertEqual(markers, [102., 104., 106.])
        self.assertEqual(bank.config_chains[scout_config].item(), 1)
        self.assertEqual(int(bank.counters["created_chains"]), 1)
        check_chain_bookkeeping(self, bank)

        # Force leader env 0 onto the relay entry with marker 104.
        slot = (bank.active & (bank.storage["robot_root"][:, 0] == 104.)).nonzero().item()
        union = env.sgs_bank.num_configs + bank.capacity
        one_hot = torch.zeros(union)
        one_hot[env.sgs_bank.num_configs + slot] = 1
        leader_config = env.active_config_id[0].item()
        self.finish(env, [0], progress=5)
        with patch.object(bank, "leader_distribution", return_value=one_hot):
            env.reset()
        self.assertEqual(env.sgs_sampler.count[leader_config].item(), 1)  # leader's bank failure
        self.assertEqual(env.active_config_id[0].item(), env.sgs_bank.num_configs + slot)
        self.assertEqual(env.active_relay_uid[0].item(), bank.uid[slot].item())
        self.assertEqual(env._robot_root_states[0, 0].item(), 104.)
        self.assertEqual(env.progress_buf[0].item(), 0)

        # The leader's relay episode succeeds: credited to the entry, not the bank.
        bank_counts = env.sgs_sampler.count.clone()
        self.finish(env, [0], progress=12, success=True)
        env.reset()
        self.assertEqual(bank.stats.count[slot].item(), 1)
        self.assertEqual(bank.stats.success_sum[slot].item(), 1)
        self.assertTrue(torch.equal(env.sgs_sampler.count, bank_counts))

        # A physical failure within the quarantine window removes the entry.
        self.finish(env, [0], progress=4)
        with patch.object(bank, "leader_distribution", return_value=one_hot):
            env.reset()
        self.finish(env, [0], progress=2, restore_failure=True)
        env.reset()
        self.assertFalse(bank.active[slot].item())
        self.assertEqual(int(bank.counters["quarantined"]), 1)
        self.assertEqual(bank.config_chains[scout_config].item(), 1)  # two entries remain
        check_chain_bookkeeping(self, bank)

        env._record_sgs_metrics()
        snapshots = env.extras["scalar_snapshots"]
        for key in ("SGS / concurrent_copies", "Relay / active_entries", "Relay / leader_reset_share",
                    "Scout / leader_success_of_sampled", "Gap / valid_configs",
                    "Relay / success_f0.50"):
            self.assertIn(key, snapshots)
            self.assertTrue(math.isfinite(snapshots[key]))
        self.assertEqual(snapshots["Relay / active_entries"], 2)

    def test_scouts_never_start_from_relay_and_need_weighting_is_used(self):
        env = self.environment()
        env.reset()
        env.sgs_sampler.observe(torch.arange(6), torch.tensor([1, 1, 1, 0, 0, 0]))
        with patch("utils.scout_relay.scout_probabilities", wraps=__import__(
                "utils.scout_relay", fromlist=["scout_probabilities"]).scout_probabilities) as spy:
            self.finish(env, [2, 3], progress=5)
            env.reset()
        self.assertTrue(bool((env.active_config_id[2:] < env.sgs_bank.num_configs).all()))
        self.assertGreaterEqual(spy.call_count, 1)
        expected = scout_probabilities(env.scout_samplers[0], env.sgs_sampler.success_rates(), 1.0)
        torch.testing.assert_close(env._scout_last_probabilities[0], expected)


class LeaderExportTests(unittest.TestCase):
    def test_folding_the_leader_latent_is_exact(self):
        from export_sapg_leader import fold_branch_latent

        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.Linear(5 + 3, 7), torch.nn.ELU(), torch.nn.Linear(7, 2))
        latent, log_std = torch.randn(2, 3), torch.randn(2, 2)
        state = {"net." + key: value for key, value in net.state_dict().items()}
        state.update(branch_latent=latent, log_std_parameter=log_std)
        folded = fold_branch_latent(state)
        single = torch.nn.Sequential(torch.nn.Linear(5, 7), torch.nn.ELU(), torch.nn.Linear(7, 2))
        single.load_state_dict({key[4:]: value for key, value in folded.items() if key.startswith("net.")})
        x = torch.randn(11, 5)
        expected = net(torch.cat((x, latent[0].expand(11, -1)), dim=-1))
        torch.testing.assert_close(single(x), expected)
        torch.testing.assert_close(folded["log_std_parameter"], log_std[0])
        self.assertNotIn("branch_latent", folded)
        with self.assertRaises(ValueError):
            fold_branch_latent(folded)


if __name__ == "__main__":
    unittest.main()
