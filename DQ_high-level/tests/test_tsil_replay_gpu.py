"""CPU-oracle checks for the batched TSIL replay backend, including CUDA storage.

These tests compare the scientific replay rules and episode ownership against
the existing implementation. Sampling draws may differ across CPU and CUDA;
membership, probabilities, budgets, rewards and archived data must agree.
"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tsil import TemporalReplay, TemporalState, Trajectory
from utils.tsil_replay_gpu import GPUTemporalReplay


def trajectory(uid, config=0, length=4, success=False, elapsed=4., score=0.):
    observations = torch.arange(length * 7, dtype=torch.float32).reshape(length, 7) / 7
    observations[:, 3] = torch.arange(length)
    observations[:, 4] = torch.arange(length) / 10
    observations[:, 5] = 1
    actions = torch.stack((torch.arange(length).float(), torch.full((length,), float(uid[1]))), -1)
    return Trajectory(uid, config, observations, actions,
                      torch.linspace(-2., .5, length), torch.linspace(.2, 3., length),
                      torch.linspace(-1., 2., length), success, elapsed, score)


class ReplayEquivalenceChecks:
    device = "cpu"

    def pair(self, configurations=5, envs=6, topk=3, maximum=8, prefix=0, gamma=.99):
        kwargs = dict(num_envs=envs, observation_size=7, action_size=2, gamma=gamma,
                      topk=topk, time_indices=(4, 5), max_episode_steps=maximum,
                      static_prefix_size=prefix, success_reward_scale=50.)
        return (TemporalReplay(TemporalState(configurations, 10.), **kwargs),
                GPUTemporalReplay(TemporalState(configurations, 10., self.device), **kwargs))

    def assert_archives_equal(self, reference, candidate):
        for name in ("fastest", "high_return"):
            left, right = getattr(reference, name), getattr(candidate, name)
            self.assertEqual(set(left), set(right), name)
            for config in left:
                self.assertEqual([t.uid for t in left[config]], [t.uid for t in right[config]])
                for expected, actual in zip(left[config], right[config]):
                    self.assertEqual((expected.config_id, expected.success, len(expected)),
                                     (actual.config_id, actual.success, len(actual)))
                    self.assertAlmostEqual(expected.elapsed, actual.elapsed, places=6)
                    self.assertAlmostEqual(expected.return_score, actual.return_score, places=6)
                    torch.testing.assert_close(actual.full_observations().cpu(),
                                               expected.full_observations(), rtol=0, atol=0)
                    for field in ("actions", "reward_fixed", "reward_command_raw", "collected_rewards"):
                        torch.testing.assert_close(getattr(actual, field).cpu(), getattr(expected, field),
                                                   rtol=0, atol=0)
        self.assertEqual([t.uid for t in reference.iter_selected()],
                         [t.uid for t in candidate.iter_selected()])
        self.assertEqual(reference.stats, candidate.stats)
        torch.testing.assert_close(candidate.state.pending_min.cpu(), reference.state.pending_min)

    def record_both(self, pair, configs, episodes, step, done, success=None,
                    raw=None, elapsed=None, natural=None, marker=0., unequal_prefix=False):
        n = len(configs)
        step = torch.as_tensor(step, dtype=torch.float32).expand(n)
        states = torch.zeros(n, 7)
        states[:, :2] = torch.tensor([8., 9.])
        if unequal_prefix:
            states[:, 0] += torch.arange(n)
        states[:, 2] = marker + torch.arange(n) * 10 + step
        states[:, 3] = step
        states[:, 4] = step / 10
        states[:, 5] = 1
        states[:, 6] = marker - step
        actions = torch.stack((states[:, 2], -states[:, 2]), -1)
        info = {"config_id": torch.as_tensor(configs), "episode_id": torch.as_tensor(episodes),
                "elapsed": step + 1 if elapsed is None else torch.as_tensor(elapsed).float(),
                "success": torch.as_tensor(success if success is not None else [False] * n),
                "reward_fixed": -step - 1, "reward_command_raw": step * 2 + .5}
        if natural is not None:
            info["natural_done"] = torch.as_tensor(natural)
        rewards = torch.as_tensor(raw if raw is not None else torch.arange(n) + 1).float()
        done = torch.as_tensor(done)
        for memory in pair:
            device = memory.state.device
            copied_states = states.to(device).clone()
            copied_actions = actions.to(device).clone()
            copied_rewards = rewards.to(device).clone()
            copied_info = {name: value.to(device).clone() for name, value in info.items()}
            memory.record(copied_states, copied_actions, copied_rewards, done.to(device), copied_info)
            # Simulate Isaac Gym immediately reusing all transition buffers.
            copied_states.zero_()
            copied_actions.zero_()
            copied_rewards.zero_()
            for value in copied_info.values():
                value.zero_()
        return states

    def test_union_fastest_first_return_fallback_uid_ties_and_duplicate(self):
        pair = self.pair()
        cases = [(9, True, 8., 100.), (7, False, 4., 100.), (4, True, 3., -20.),
                 (6, True, 3., -30.), (1, False, 6., 100.), (3, True, 2., -10.),
                 (2, True, 3., -40.), (5, False, 4., 200.), (8, False, 8., 99.)]
        for config in range(5):
            for serial, success, elapsed, score in cases:
                sample = trajectory((config, serial), config=config, success=success,
                                    elapsed=elapsed, score=score, length=1 + serial % 5)
                self.assertEqual(pair[0].insert(sample), pair[1].insert(sample))
                self.assert_archives_equal(*pair)
        for memory in pair:
            self.assertFalse(memory.insert(trajectory((0, 3), success=True, elapsed=.1, score=9999)))
            self.assertEqual([t.uid for t in memory.selected(0)], [(0, 3), (0, 2), (0, 4)])
        self.assert_archives_equal(*pair)

    def test_simultaneous_completions_preserve_all_causal_data_and_pending_targets(self):
        pair = self.pair(prefix=2)
        configs = [0, 0, 0, 1, 1, 1]
        for wave in range(9):
            length = 1 + wave % 4
            for step in range(length):
                self.record_both(pair, configs, [wave] * 6, step,
                                 [step == length - 1] * 6,
                                 success=[(wave + env) % 3 == 0 for env in range(6)],
                                 raw=[wave * 3 + env + .25 for env in range(6)], marker=wave * 100.)
            self.assert_archives_equal(*pair)
        for memory in pair:
            self.assertEqual(memory.stats["completed_episodes"], 54)
            self.assertEqual(memory.stats["aborted_episodes"], 0)
            torch.testing.assert_close(memory.state.targets.cpu(), torch.full((5,), 10.))
            self.assertEqual(memory.static_prefix_size, 2)
            self.assertEqual(memory.selected(0)[0].observations.shape[1], 5)
            memory.state.commit_targets()
        torch.testing.assert_close(pair[0].state.targets, pair[1].state.targets.cpu())

    def test_record_threshold_ties_keep_sequential_admission_semantics(self):
        pair = self.pair(configurations=1, envs=4, topk=2)
        # Fill both lists first with higher environment IDs. A later tied score
        # is not admitted merely because its UID would sort before a survivor.
        for wave in range(3):
            self.record_both(pair, [0] * 4, [wave] * 4, 0, [True] * 4,
                             success=[False, False, True, True],
                             raw=[0., 0., 2., 2.], elapsed=[3., 3., 2., 2.])
            self.assert_archives_equal(*pair)
        self.record_both(pair, [0] * 4, [3] * 4, 0, [True] * 4,
                         success=[True, True, False, False],
                         raw=[2., 2., 2., 2.], elapsed=[2., 2., 2., 2.])
        self.assert_archives_equal(*pair)

    def test_asynchronous_episode_lengths_and_ring_reuse_match_reference(self):
        pair = self.pair(envs=4, topk=2, maximum=5, prefix=2)
        serials, steps = torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)
        budgets = torch.tensor([1, 2, 3, 5])
        for collection_step in range(17):
            done = steps + 1 == budgets
            self.record_both(pair, [0, 0, 1, 1], serials, steps, done,
                             success=[True, False, True, False],
                             raw=(serials * 5 + torch.arange(4) + 1).float(),
                             marker=collection_step * 100.)
            self.assert_archives_equal(*pair)
            serials += done.long()
            steps = torch.where(done, torch.zeros_like(steps), steps + 1)
        for memory in pair:
            self.assertEqual(memory.stats["completed_episodes"], 17 + 8 + 5 + 3)
            self.assertEqual(memory.stats["aborted_episodes"], 0)

    def test_randomized_tied_batch_admission_matches_sequential_oracle(self):
        for topk in (1, 2):
            pair = self.pair(configurations=3, envs=6, topk=topk)
            generator = torch.Generator().manual_seed(912 + topk)
            for serial in range(12):
                configs = torch.randint(0, 3, (6,), generator=generator)
                success = torch.randint(0, 2, (6,), generator=generator).bool()
                elapsed = torch.randint(1, 4, (6,), generator=generator).float()
                rewards = torch.randint(-1, 3, (6,), generator=generator).float()
                self.record_both(pair, configs, [serial] * 6, 0, [True] * 6,
                                 success=success, raw=rewards, elapsed=elapsed,
                                 marker=serial * 100.)
                self.assert_archives_equal(*pair)

    def test_same_batch_improvement_prevents_later_uid_only_tie_admission(self):
        pair = self.pair(configurations=1, envs=12, topk=2)
        for memory in pair:
            memory.insert(trajectory((10, 0), score=2.))
            memory.insert(trajectory((11, 0), score=1.))
        self.record_both(pair, [0, 0] + [-1] * 10, [1] * 12, 0, [True] * 12,
                         raw=[2., 2.] + [0.] * 10)
        self.assert_archives_equal(*pair)
        for memory in pair:
            self.assertEqual([t.uid for t in memory.high_return[0]], [(0, 1), (10, 0)])

    def test_configuration_changes_and_episode_overflow_are_rejected(self):
        pair = self.pair(envs=1)
        self.record_both(pair, [0], [0], 0, [False])
        for memory in pair:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                self.record_both((memory,), [1], [0], 1, [False])
        pair = self.pair(envs=1, maximum=1)
        self.record_both(pair, [0], [0], 0, [False])
        for memory in pair:
            with self.assertRaisesRegex(ValueError, "episode exceeded"):
                self.record_both((memory,), [0], [0], 1, [True])

    def test_administrative_resets_mid_episode_attach_and_explicit_abort(self):
        pair = self.pair(envs=3)
        self.record_both(pair, [0, 1, 2], [0, 0, 0], [0, 0, 5], [False] * 3)
        self.record_both(pair, [0, 3, 2], [0, 1, 0], [1, 0, 6], [True] * 3,
                         success=[True] * 3, natural=[True, False, True])
        self.assert_archives_equal(*pair)
        for memory in pair:
            self.assertEqual(memory.stats["completed_episodes"], 1)
            self.assertEqual(memory.stats["aborted_episodes"], 2)
            self.assertEqual([t.uid for t in memory.iter_selected()], [(0, 0)])
        self.record_both(pair, [0, 1, 2], [1, 2, 1], 0, [False] * 3)
        for memory in pair:
            memory.abort(torch.tensor([0, 2], device=memory.state.device))
            memory.abort()
            self.assertEqual(memory.stats["aborted_episodes"], 5)
        self.assert_archives_equal(*pair)

    def test_dynamic_prefix_disables_compression_and_static_prefix_change_rejects(self):
        pair = self.pair(envs=2, prefix=2)
        self.record_both(pair, [0, 1], [0, 0], 0, [True, True], unequal_prefix=True)
        self.assert_archives_equal(*pair)
        for memory in pair:
            self.assertEqual(memory.static_prefix_size, 0)
        pair = self.pair(envs=2, prefix=2)
        self.record_both(pair, [0, 1], [0, 0], 0, [False, False])
        states = torch.zeros(2, 7)
        info = {"config_id": torch.tensor([0, 1]), "episode_id": torch.tensor([0, 0]),
                "elapsed": torch.ones(2), "success": torch.zeros(2, dtype=torch.bool),
                "reward_fixed": torch.zeros(2), "reward_command_raw": torch.zeros(2)}
        for memory in pair:
            device = memory.state.device
            with self.assertRaisesRegex(ValueError, "static feature"):
                memory.record(states.to(device), torch.zeros(2, 2, device=device),
                              torch.zeros(2, device=device), torch.ones(2, dtype=torch.bool, device=device),
                              {name: value.to(device) for name, value in info.items()})

    def test_relabel_and_prepared_returns_match_reference_for_discount_extremes(self):
        for gamma in (0., 1e-8, .001, .99, 1.):
            pair = self.pair(configurations=3, maximum=64, gamma=gamma)
            for config, length, success, elapsed, target in ((0, 1, True, 4., 2.),
                                                           (1, 53, True, 8., 10.),
                                                           (2, 41, False, 6., 3.)):
                sample = trajectory((config, 1), config, length, success, elapsed, 100. - config)
                for memory in pair:
                    memory.insert(sample)
                    memory.state.targets[config] = target
            for command_gate, scale in ((0., 2.75), (1., .3)):
                expected = {}
                for sample in pair[0].iter_selected():
                    obs, returns, efficiency = pair[0].relabel(
                        sample, pair[0].state.targets[sample.config_id], scale, command_gate)
                    expected[sample.uid] = (obs, returns, efficiency, sample.actions)
                selected = pair[1].prepare(lambda obs, ret: torch.ones_like(ret),
                                           reward_scale=scale, command_gate=command_gate,
                                           transitions_per_trajectory=64, screen_batch_size=13)
                self.assertEqual(len(selected), 95)
                batch = {name: value.cpu() for name, value in selected.get_batch(
                    torch.arange(len(selected), device=self.device), device=self.device).items()}
                for row, uid in enumerate(selected.uids.cpu().tolist()):
                    obs, returns, efficiency, actions = expected[tuple(uid[:2])]
                    step = uid[2]
                    torch.testing.assert_close(batch["states"][row], obs[step], rtol=0, atol=0)
                    torch.testing.assert_close(batch["actions"][row], actions[step], rtol=0, atol=0)
                    torch.testing.assert_close(batch["returns"][row, 0], returns[step],
                                               rtol=2e-6, atol=2e-5)
                    self.assertAlmostEqual(float(batch["efficiency"][row, 0]), efficiency, places=6)
                self.assert_archives_equal(*pair)

    def test_sampling_covers_all_configurations_positive_only_without_replacement(self):
        pair = self.pair(configurations=19, maximum=48)
        for config in range(19):
            for serial in (0, 1):
                sample = trajectory((config, serial), config=config, length=1 + (config * 7 + serial) % 48,
                                    success=serial == 0, elapsed=8., score=serial)
                for memory in pair:
                    memory.insert(sample)
        sizes = []
        def gap(obs, returns):
            self.assertEqual(obs.device.type, torch.device(self.device).type)
            self.assertEqual(returns.device, obs.device)
            sizes.append(len(obs))
            step = obs[:, 3]
            return torch.where(step.long() % 4 == 1, -torch.ones_like(step), step + 1)
        selection = pair[1].prepare(gap, transitions_per_trajectory=32, screen_batch_size=13)
        self.assertLessEqual(max(sizes), 13)
        self.assertEqual(sum(sizes), sum(len(t) for t in pair[0].iter_selected()))
        self.assertEqual(set(selection.config_id.cpu().tolist()), set(range(19)))
        self.assertEqual(selection.uids.unique(dim=0).shape[0], len(selection))
        rows = selection.uids.cpu()
        expected_steps = expected_positive = 0
        for sample in pair[0].iter_selected():
            expected = {i for i in range(len(sample)) if i % 4 != 1}
            actual = rows[(rows[:, 0] == sample.uid[0]) & (rows[:, 1] == sample.uid[1]), 2].tolist()
            self.assertEqual(len(actual), min(32, len(expected)))
            self.assertTrue(set(actual).issubset(expected))
            if len(expected) <= 32:
                self.assertEqual(set(actual), expected)
            expected_steps += min(32, len(expected))
            expected_positive += len(expected)
        self.assertEqual(selection.stats["replay_steps"], expected_steps)
        self.assertEqual(selection.stats["replay_positive_available"], expected_positive)
        self.assertEqual(selection.stats["replay_trajectories"], 38)
        for field in ("observations", "actions", "returns", "efficiency", "config_id",
                      "success", "elapsed", "uids"):
            self.assertEqual(getattr(selection, field).device.type, torch.device(self.device).type)
        indices = torch.randperm(len(selection), device=self.device)
        before = selection.get_batch(indices, device=self.device)
        pair[1].state.targets.fill_(.1)
        pair[1].insert(trajectory((0, 10), config=0, success=True, elapsed=.1, score=1000.))
        after = selection.get_batch(indices, device=self.device)
        for field in before:
            self.assertEqual(after[field].device.type, torch.device(self.device).type)
            torch.testing.assert_close(before[field], after[field], rtol=0, atol=0)

    def test_empty_archive_and_no_positive_gap_return_valid_empty_gpu_batches(self):
        _, memory = self.pair()
        for populated in (False, True):
            if populated:
                memory.insert(trajectory((0, 1)))
            selection = memory.prepare(lambda obs, ret: torch.zeros_like(ret))
            self.assertEqual(len(selection), 0)
            self.assertEqual(selection.stats["replay_trajectories"], int(populated))
            self.assertEqual(selection.stats["replay_positive_available"], 0)
            batch = selection.get_batch(torch.empty(0, dtype=torch.long, device=self.device),
                                        device=self.device)
            self.assertEqual(batch["states"].shape, (0, 7))
            self.assertEqual(batch["returns"].shape, (0, 1))
            for value in batch.values():
                self.assertEqual(value.device.type, torch.device(self.device).type)

    def test_empty_unallocated_archive_does_not_assume_static_prefix(self):
        _, memory = self.pair(envs=1, prefix=2)
        selection = memory.prepare(lambda obs, ret: torch.ones_like(ret))
        self.assertEqual(selection.get_batch([], device=self.device)["states"].shape, (0, 7))
        self.assertEqual(memory.static_prefix_size, 2)
        self.record_both((memory,), [0], [0], 0, [True])
        self.assertEqual(memory.static_prefix_size, 2)
        selection = memory.prepare(lambda obs, ret: torch.ones_like(ret))
        torch.testing.assert_close(selection.get_batch([0], device=self.device)["states"][0, :2].cpu(),
                                   torch.tensor([8., 9.]), rtol=0, atol=0)

    def test_screen_rejects_nonfinite_or_wrong_length_gaps(self):
        _, memory = self.pair()
        memory.insert(trajectory((0, 1)))
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(ValueError, "invalid value gaps"):
                memory.prepare(lambda obs, ret: torch.full_like(ret, invalid))
        with self.assertRaisesRegex(ValueError, "invalid value gaps"):
            memory.prepare(lambda obs, ret: ret[:-1])

    def test_sampling_uses_gap_proportions_independent_of_success_efficiency(self):
        # Independent equal-shaped trajectories provide enough draws to detect
        # uniform sampling or mistakenly multiplying probabilities by rewards.
        _, memory = self.pair(configurations=1200, envs=1200, topk=1, maximum=3)
        for step in range(3):
            self.record_both((memory,), list(range(1200)), [0] * 1200, step,
                             [step == 2] * 1200, success=[i % 2 == 0 for i in range(1200)])
        torch.manual_seed(29017)
        selection = memory.prepare(lambda obs, ret: obs.new_tensor([1., 3., 6.])[obs[:, 3].long()],
                                   transitions_per_trajectory=1, screen_batch_size=127)
        self.assertEqual(len(selection), 1200)
        steps, successful = selection.uids[:, 2].cpu(), selection.success.cpu()
        for mask in (successful, ~successful):
            frequencies = torch.bincount(steps[mask], minlength=3).float() / mask.sum()
            torch.testing.assert_close(frequencies, torch.tensor([.1, .3, .6]), rtol=0, atol=.07)


class BatchedReplayCPUReferenceTests(ReplayEquivalenceChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for replay residency tests")
class BatchedReplayCUDAReferenceTests(ReplayEquivalenceChecks, unittest.TestCase):
    device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"

    def test_explicit_replay_device_owns_payload_with_separate_temporal_state_device(self):
        state_devices = ["cpu"] + (["cuda:0"] if torch.cuda.device_count() > 1 else [])
        for state_device in state_devices:
            for input_device in ("cpu", self.device):
                with self.subTest(state_device=state_device, input_device=input_device):
                    state = TemporalState(2, 10., device=state_device)
                    memory = GPUTemporalReplay(
                        state, num_envs=2, observation_size=7, action_size=2,
                        gamma=.5, topk=1, time_indices=(4, 5), max_episode_steps=2,
                        static_prefix_size=2, device=self.device)
                    self.assertIs(memory.state, state)
                    self.assertEqual(memory.device, torch.device(self.device))
                    states = torch.tensor([[8., 9., 100., 0., 0., 1., .1],
                                           [8., 9., 200., 0., 0., 1., .2]], device=input_device)
                    actions = torch.tensor([[1., 2.], [3., 4.]], device=input_device)
                    original_states, original_actions = states.cpu().clone(), actions.cpu().clone()
                    info = {
                        "config_id": torch.tensor([0, 1], device=state_device),
                        "episode_id": torch.tensor([0, 0], device=state_device),
                        "elapsed": torch.tensor([2., 3.], device=state_device),
                        "success": torch.ones(2, dtype=torch.bool, device=state_device),
                        "reward_fixed": torch.tensor([-1., -2.], device=state_device),
                        "reward_command_raw": torch.tensor([.5, 1.], device=state_device),
                    }
                    memory.record(states, actions, torch.tensor([1., 2.], device=input_device),
                                  torch.ones(2, dtype=torch.bool, device=input_device), info)
                    states.zero_()
                    actions.zero_()
                    for value in info.values():
                        value.zero_()
                    self.assertEqual(state.pending_min.device, torch.device(state_device))
                    self.assertEqual(state.targets.device, torch.device(state_device))
                    torch.testing.assert_close(state.pending_min.cpu(), torch.tensor([2., 3.]))
                    torch.testing.assert_close(state.targets.cpu(), torch.tensor([10., 10.]))
                    self.assertEqual(state.commit_targets(), 2)
                    self.assertEqual(memory._ring.device, torch.device(self.device))
                    self.assertEqual(memory._archive.device, torch.device(self.device))
                    def gap(observations, returns):
                        self.assertEqual(observations.device, torch.device(self.device))
                        self.assertEqual(returns.device, torch.device(self.device))
                        return torch.ones_like(returns)
                    selected = memory.prepare(gap, reward_scale=2., command_gate=.5)
                    self.assertEqual(len(selected), 2)
                    indices = torch.argsort(selected.config_id)
                    batch = selected.get_batch(indices)
                    for value in batch.values():
                        self.assertEqual(value.device, torch.device(self.device))
                    original_states[:, 5] = torch.tensor([.2, .3])
                    torch.testing.assert_close(batch["states"].cpu(), original_states, rtol=0, atol=0)
                    torch.testing.assert_close(batch["actions"].cpu(), original_actions, rtol=0, atol=0)
                    torch.testing.assert_close(batch["returns"].cpu(), torch.tensor([[49.625], [49.25]]))
                    torch.testing.assert_close(batch["efficiency"].cpu(), torch.full((2, 1), 2.))


if __name__ == "__main__":
    unittest.main()
