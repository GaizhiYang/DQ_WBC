"""Compare the independent monitoring windows against sequential deque oracles."""

from collections import deque
import random
import unittest

import torch

from utils.episode_statistics import RecentEpisodeStatistics


class RecentEpisodeStatisticsTests(unittest.TestCase):
    def assert_reference(self, statistics, families):
        self.assertEqual(statistics.family_count.tolist(), [len(values) for values in families])
        self.assertEqual(statistics.family_successes.tolist(), [sum(values) for values in families])
        torch.testing.assert_close(
            statistics.family_rates().cpu(),
            torch.tensor([sum(values) / max(len(values), 1) for values in families]))

    def exercise_reference(self, device):
        statistics = RecentEpisodeStatistics(4, device=device)
        families = [deque(maxlen=100) for _ in range(4)]
        rng = random.Random(621)
        # Uneven asynchronous batches, ties, wraparound, and a single batch
        # larger than each window. Family 3 is deliberately undersampled.
        for batch_size in (0, 1, 23, 101, 1001, 19, 2700, 3, 0, 193, 32, 1100):
            ids = rng.choices(range(4), weights=(9, 4, 2, 1), k=batch_size)
            outcomes = [rng.random() < (0.15 + family * 0.2) for family in ids]
            statistics.record(torch.tensor(ids, dtype=torch.long, device=device),
                              torch.tensor(outcomes, dtype=torch.bool, device=device))
            for family, success in zip(ids, outcomes):
                families[family].append(success)
            self.assert_reference(statistics, families)
        for name, value in vars(statistics).items():
            if isinstance(value, torch.Tensor):
                self.assertEqual(value.device, torch.device(device), name)

    def test_matches_deque_after_each_batch_cpu(self):
        self.exercise_reference("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_matches_deque_after_each_batch_cuda(self):
        self.exercise_reference("cuda:0")

    def test_empty_windows_are_not_padded_with_failure(self):
        stats = RecentEpisodeStatistics(4, "cpu")
        self.assertEqual(stats.family_count.tolist(), [0] * 4)
        stats.record(torch.tensor([2]), torch.tensor([True]))
        self.assertEqual(stats.family_count.tolist(), [0, 0, 1, 0])
        self.assertEqual(stats.family_rates().tolist(), [0.0, 0.0, 1.0, 0.0])

    def test_full_batch_overwrite_expires_old_successes(self):
        stats = RecentEpisodeStatistics(4, "cpu")
        stats.record(torch.zeros(1400, dtype=torch.long), torch.ones(1400, dtype=torch.bool))
        self.assertEqual(stats.family_successes.tolist(), [100, 0, 0, 0])
        stats.record(torch.zeros(1273, dtype=torch.long), torch.zeros(1273, dtype=torch.bool))
        self.assertEqual(stats.family_successes.tolist(), [0] * 4)
        self.assertEqual(stats.family_count.tolist(), [100, 0, 0, 0])

    def test_inputs_and_rng_unchanged(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                ids = torch.tensor([3, 0, 2, 0, 1, 3], device=device)
                values = torch.tensor([1, 0, 1, 1, 0, 1], dtype=torch.bool, device=device)
                ids_before, values_before = ids.clone(), values.clone()
                cpu_rng = torch.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state(device).clone() if device.startswith("cuda") else None
                stats = RecentEpisodeStatistics(4, device)
                stats.record(ids, values)
                stats.family_rates()
                self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                if cuda_rng is not None:
                    self.assertTrue(torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))
                self.assertTrue(torch.equal(ids, ids_before))
                self.assertTrue(torch.equal(values, values_before))

    def test_small_windows_and_interleaved_batches(self):
        stats = RecentEpisodeStatistics(4, "cpu", family_window=1)
        stats.record(torch.tensor([1, 0, 1]), torch.tensor([True, True, False]))
        stats.record(torch.tensor([2]), torch.tensor([True]))
        self.assertEqual(stats.family_count.tolist(), [1, 1, 1, 0])
        self.assertEqual(stats.family_successes.tolist(), [1, 0, 1, 0])

    def test_monitoring_does_not_change_sgs_samples_or_history(self):
        from utils.sgs import SGSSampler

        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                baseline = SGSSampler(8, device=device, seed=43)
                monitored = SGSSampler(8, device=device, seed=43)
                stats = RecentEpisodeStatistics(4, device)
                for step in range(5):
                    ids = torch.arange(1237, device=device) % 8
                    outcomes = (torch.arange(1237, device=device) + step) % 3 == 0
                    baseline.observe(ids, outcomes, step)
                    monitored.observe(ids, outcomes, step)
                    stats.record(ids % 4, outcomes)
                    stats.family_rates()
                    self.assertTrue(torch.equal(baseline.sample(97), monitored.sample(97)))
                    self.assertTrue(torch.equal(baseline.probabilities(), monitored.probabilities()))
                    for name in baseline._TENSOR_STATE:
                        self.assertTrue(torch.equal(getattr(baseline, name), getattr(monitored, name)), name)
                    self.assertTrue(torch.equal(baseline.generator.get_state(), monitored.generator.get_state()))

    def test_invalid_shapes_and_values(self):
        for kwargs in ({"num_families": 0}, {"num_families": True},
                       {"family_window": 1.5}, {"family_window": -1}):
            parameters = dict(num_families=4, device="cpu")
            parameters.update(kwargs)
            with self.assertRaises(ValueError):
                RecentEpisodeStatistics(**parameters)
        stats = RecentEpisodeStatistics(4, "cpu")
        for ids, values in ((torch.tensor([[0]]), torch.tensor([True])),
                            (torch.tensor([0]), torch.tensor([True, False])),
                            (torch.tensor([0.0]), torch.tensor([True]))):
            with self.assertRaises(ValueError):
                stats.record(ids, values)
        # Invalid CUDA assertions poison the context, so validate error cases on CPU.
        for ids, values in ((torch.tensor([4]), torch.tensor([True])),
                            (torch.tensor([-1]), torch.tensor([False])),
                            (torch.tensor([0]), torch.tensor([2]))):
            with self.assertRaises(RuntimeError):
                stats.record(ids, values)
        self.assertEqual(stats.family_count.tolist(), [0] * 4)


if __name__ == "__main__":
    unittest.main()
