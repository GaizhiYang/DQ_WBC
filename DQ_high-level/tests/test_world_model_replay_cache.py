"""CPU-only eligibility/distribution checks for reuse between replay appends."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.replay import TransitionReplay


def reference_mask(replay, k, split):
    """Independent chronological oracle, without rolled tensors or cache."""
    mask = torch.zeros_like(replay.valid)
    for row in range(replay.capacity):
        for env in range(replay.num_envs):
            if not bool(replay.valid[row, env]):
                continue
            holdout = bool(replay.holdout[row, env])
            if ((split == "train" and holdout)
                    or (split in ("holdout", "validation") and not holdout)):
                continue
            eligible = True
            for offset in range(1, k):
                future = (row + offset) % replay.capacity
                previous = (row + offset - 1) % replay.capacity
                if (not bool(replay.valid[future, env])
                        or replay.episode_ids[future, env] != replay.episode_ids[row, env]
                        or replay.family[future, env] != replay.family[row, env]
                        or replay.stamp[future, env] != replay.stamp[row, env] + offset
                        or bool(replay.terminated[previous, env] | replay.truncated[previous, env])):
                    eligible = False
                    break
            mask[row, env] = eligible
    return mask


def reference_draw(replay, mask, n, balance, recent_fraction, recent_steps):
    """Pre-cache sampler formula; identical seeds must select identical rows."""
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    families = replay.family.reshape(-1)[indices]
    weights = torch.ones(indices.numel())
    if balance:
        counts = torch.bincount(families, minlength=replay.num_families)
        weights /= counts[families].clamp_min(1)
    weights /= weights.sum()
    if recent_fraction > 0:
        recent = (replay.stamp.reshape(-1)[indices] >=
                  replay.steps - max(1, int(recent_steps))).float()
        if balance:
            counts = torch.zeros(replay.num_families)
            counts.scatter_add_(0, families, recent)
            recent /= counts[families].clamp_min(1)
        recent = torch.where(recent.sum() > 0,
                             recent / recent.sum().clamp_min(1e-12), weights)
        weights = (1 - recent_fraction) * weights + recent_fraction * recent
    selected = indices[torch.multinomial(weights, n, replacement=True)]
    return selected // replay.num_envs, selected % replay.num_envs


class ReplayCacheCPUTests(unittest.TestCase):
    def replay(self, capacity=7, envs=12):
        return TransitionReplay(envs, capacity, 2, 1, 1, "cpu", holdout_fraction=.4)

    def append(self, replay, *, terminal=False, truncated=False, episodes=None, valid=None):
        step = replay.steps
        env = torch.arange(replay.num_envs)
        state = torch.stack((torch.full_like(env, step), env), -1).float()
        future = state + torch.tensor([1., 0.])
        replay.add(state, state[:, :1], future, state[:, 1:], .02, state[:, 0],
                   torch.full((len(env),), terminal, dtype=torch.bool),
                   torch.full((len(env),), truncated, dtype=torch.bool),
                   env % 3, episode_ids=episodes, valid=valid)

    def populated(self):
        replay = self.replay()
        for step in range(15):
            valid = torch.ones(replay.num_envs, dtype=torch.bool)
            if step == 11:
                valid[1] = False
            self.append(replay, terminal=step % 5 == 3, truncated=step % 5 == 4, valid=valid)
        return replay

    def assert_masks(self, replay):
        for k in (1, 2, 3):
            for split in ("train", "validation", "holdout", "all"):
                expected = reference_mask(replay, k, split)
                torch.testing.assert_close(replay.window_mask(k, split), expected)
                self.assertEqual(replay.can_sample(k, split), bool(expected.any()))

    def test_repeated_draws_preserve_family_recent_distribution_and_rng(self):
        replay = self.populated()
        for k in (1, 2, 3):
            for split in ("train", "validation", "all"):
                mask = reference_mask(replay, k, split)
                if not bool(mask.any()):
                    continue
                for balance in (False, True):
                    for fraction, recent in ((0., 3), (.5, 1), (.5, 3), (1., 100)):
                        for seed in (17, 39):
                            torch.manual_seed(seed)
                            row, env = reference_draw(replay, mask, 48, balance, fraction, recent)
                            reference_rng = torch.get_rng_state().clone()
                            torch.manual_seed(seed)
                            batch = replay.sample_windows(48, k, split, balance, fraction, recent)
                            torch.testing.assert_close(batch.env_ids, env, rtol=0, atol=0)
                            torch.testing.assert_close(batch.states[0], replay.states[row, env], rtol=0, atol=0)
                            torch.testing.assert_close(torch.get_rng_state(), reference_rng, rtol=0, atol=0)
                            if k == 1:
                                torch.manual_seed(seed)
                                starts = replay.sample_starts(48, split, balance, fraction, recent)
                                torch.testing.assert_close(starts.state, replay.states[row, env], rtol=0, atol=0)
                                torch.testing.assert_close(starts.env_ids, env, rtol=0, atol=0)

    def test_update_block_reuses_candidate_scan_and_distribution(self):
        replay = self.populated()
        with mock.patch.object(replay, "_split_mask", wraps=replay._split_mask) as build_mask, \
                mock.patch.object(replay, "_sampling_weights", wraps=replay._sampling_weights) as build_weights:
            self.assertTrue(replay.can_sample(2, "all"))
            for _ in range(10):
                replay.sample_windows(16, 2, "all", recent_fraction=.5)
            self.assertEqual(build_mask.call_count, 1)
            self.assertEqual(build_weights.call_count, 1)
            # k=1 eligibility is shared by availability checks and start draws.
            self.assertTrue(replay.can_sample(1, "all"))
            replay.sample_starts(16, "all", recent_fraction=.5)
            replay.sample_windows(16, 1, "all", recent_fraction=.5)
            self.assertEqual(build_mask.call_count, 2)
            self.assertEqual(build_weights.call_count, 2)
            replay.sample_starts(16, "all", recent_fraction=1.)
            self.assertEqual(build_mask.call_count, 2)
            self.assertEqual(build_weights.call_count, 3)

    def test_append_overwrite_terminal_and_external_reset_invalidate_all_splits(self):
        replay = self.replay(capacity=4)
        self.assert_masks(replay)  # Also cache the initially empty pools.
        for step in range(11):
            valid = torch.ones(replay.num_envs, dtype=torch.bool)
            if step == 7:
                valid[0] = False
            # Explicit reset without a preceding terminal is also a boundary.
            episodes = torch.full((replay.num_envs,), step // 3, dtype=torch.long)
            self.append(replay, terminal=step == 5, truncated=step == 8,
                        episodes=episodes, valid=valid)
            self.assertFalse(replay._sampling_cache)
            self.assert_masks(replay)
        sampled = replay.sample_windows(32, 2, "all")
        self.assertTrue(bool((sampled.actions[0, :, 0] >= 7).all()))
        self.assertFalse(bool(sampled.terminated[:-1].any() | sampled.truncated[:-1].any()))

    def test_public_index_tensor_edits_and_replacement_do_not_reuse_stale_pools(self):
        edits = {
            "valid": lambda x: x.zero_(),
            "holdout": lambda x: x.logical_not_(),
            "episode_ids": lambda x: x[3].add_(50),
            "family": lambda x: x[3].fill_(2),
            "stamp": lambda x: x[3].add_(50),
            "terminated": lambda x: x[3].fill_(True),
            "truncated": lambda x: x[3].fill_(True),
        }
        for name, edit in edits.items():
            with self.subTest(field=name):
                replay = self.populated()
                self.assert_masks(replay)
                old_entry = replay._sampling_cache[(2, "all")]
                edit(getattr(replay, name))
                self.assert_masks(replay)
                self.assertIsNot(replay._sampling_cache[(2, "all")], old_entry)
        replay = self.populated()
        replay.sample_starts(4, "all")
        replay.valid = torch.zeros_like(replay.valid)
        self.assertFalse(replay.can_sample(1, "all"))
        with self.assertRaisesRegex(RuntimeError, "no sampleable"):
            replay.sample_starts(4, "all")

    def test_mutating_returned_mask_does_not_modify_cached_eligibility(self):
        replay = self.populated()
        expected = reference_mask(replay, 2, "all")
        mask = replay.window_mask(2, "all")
        mask.zero_()
        torch.testing.assert_close(replay.window_mask(2, "all"), expected)
        self.assertTrue(replay.can_sample(2, "all"))

    def test_no_recent_candidates_preserves_original_fallback(self):
        replay = self.populated()
        replay.valid[replay.stamp >= replay.steps - 2] = False
        mask = reference_mask(replay, 1, "all")
        torch.manual_seed(42)
        row, env = reference_draw(replay, mask, 128, True, 1., 1)
        torch.manual_seed(42)
        starts = replay.sample_starts(128, "all", recent_fraction=1., recent_steps=1)
        torch.testing.assert_close(starts.state, replay.states[row, env], rtol=0, atol=0)

    def test_cached_tensor_payloads_are_included_in_memory_accounting(self):
        replay = self.populated()
        before = replay.memory_bytes
        replay.sample_windows(16, 2, "all", recent_fraction=.5)
        cached = sum(value.numel() * value.element_size()
                     for entry in replay._sampling_cache.values() for value in entry.values()
                     if torch.is_tensor(value))
        self.assertGreater(cached, 0)
        self.assertEqual(replay.memory_bytes, before + cached)
        self.append(replay)
        self.assertEqual(replay.memory_bytes, before)

    def test_inference_mode_without_mutation_versions_rebuilds_conservatively(self):
        with torch.inference_mode():
            replay = self.populated()
            replay.sample_starts(4, "all")
            replay.valid.zero_()
            self.assertFalse(replay.can_sample(1, "all"))


if __name__ == "__main__":
    unittest.main()
