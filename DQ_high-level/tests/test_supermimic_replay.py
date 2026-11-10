"""Replay regression tests that do not import Isaac Gym or start a simulator."""
import importlib.util
from pathlib import Path
import unittest

import torch


_spec = importlib.util.spec_from_file_location(
    "supermimic_replay", Path(__file__).resolve().parents[1] / "supermimic/replay_buffer.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ReplayBuffer = _module.ReplayBuffer
PrioritizedReplayBuffer = _module.PrioritizedReplayBuffer


class ReplayChecks:
    device = "cpu"

    def buffer(self, capacity=100):
        return ReplayBuffer(capacity, state_dim=2, action_dim=1, device=self.device)

    def record(self, buffer, step, num_envs=3, done=None, native_reward=None):
        # State explicitly labels environment identity and chronological time.
        state = torch.stack((torch.arange(num_envs), torch.full((num_envs,), step)), -1).double()
        next_state = state + torch.tensor([0., 1.])
        buffer.add(state, torch.full((num_envs, 1), float(step), dtype=torch.float64),
                   torch.arange(num_envs, dtype=torch.float64).reshape(-1, 1), next_state,
                   torch.zeros(num_envs, 1, dtype=torch.long) if done is None else done,
                   native_reward)

    def assert_consecutive(self, sequence):
        states = sequence['states']
        self.assertTrue(torch.equal(states[:, :, 0], states[:, :1, 0].expand_as(states[:, :, 0])))
        self.assertTrue(torch.all(states[:, 1:, 1] - states[:, :-1, 1] == 1).item())
        self.assertTrue(torch.equal(sequence['next_states'][:, :-1], states[:, 1:]))
        self.assertFalse(sequence['dones'][:, :-1].any().item())

    def test_long_done_and_column_rewards_are_normalized(self):
        buffer = self.buffer()
        self.record(buffer, 0, done=torch.tensor([[0], [1], [2]]))
        self.assertEqual(buffer.dones.dtype, torch.bool)
        self.assertEqual(buffer.dones[:3].tolist(), [False, True, True])
        self.assertEqual(buffer.rewards[:3].tolist(), [0., 1., 2.])
        self.assertEqual(buffer.states.dtype, torch.float32)

    def test_singleton_shapes_and_storage_do_not_keep_autograd_graphs(self):
        buffer = self.buffer()
        state = torch.tensor([[1., 2.]], requires_grad=True)
        buffer.add(state, state[:, :1], state[:, :1], state + 1,
                   torch.tensor([[1]], dtype=torch.long))
        self.assertEqual(buffer.dones[:1].tolist(), [True])
        self.assertEqual(buffer.rewards[:1].tolist(), [1.])
        for name in ('states', 'actions', 'rewards', 'next_states'):
            self.assertFalse(getattr(buffer, name).requires_grad)
        with torch.no_grad():
            state.fill_(99)
        self.assertEqual(buffer.states[0].tolist(), [1., 2.])
        self.assertEqual(buffer.get_stats()['std_reward'], 0.)

    def test_missing_native_reward_overwrites_reused_slot(self):
        buffer = self.buffer(capacity=3)
        self.record(buffer, 0, native_reward=torch.ones(3, 1))
        self.record(buffer, 1)
        self.assertEqual(buffer.native_rewards.tolist(), [0., 0., 0.])

    def test_sequence_requires_eight_timesteps_per_environment(self):
        buffer = self.buffer()
        for step in range(7):
            self.record(buffer, step)
        self.assertFalse(buffer.can_sample_sequences(8))
        with self.assertRaisesRegex(ValueError, "No valid 8-step"):
            buffer.sample_sequences(4, 8)
        self.record(buffer, 7)
        self.assertTrue(buffer.can_sample_sequences(8))
        sequences = buffer.sample_sequences(32, 8)
        self.assert_consecutive(sequences)
        self.assertTrue(torch.all(sequences['states'][:, 0, 1] == 0).item())
        self.assertTrue(torch.all(sequences['states'][:, -1, 1] == 7).item())

    def test_sequence_crosses_physical_ring_end_without_time_discontinuity(self):
        # A capacity not divisible by num_envs leaves a partial oldest timestep.
        # Every retained 3-step sequence crosses the physical end of this ring.
        buffer = self.buffer(capacity=7)
        for step in range(5):
            self.record(buffer, step, num_envs=2)
        sequences = buffer.sample_sequences(64, 3)
        self.assert_consecutive(sequences)
        starts = sequences['states'][:, 0].tolist()
        self.assertTrue(all(tuple(start) in {(1., 1.), (0., 2.), (1., 2.)} for start in starts))
        self.assertEqual(buffer.ptr, 3)
        self.assertEqual(buffer.total_added, 10)

    def test_terminal_is_allowed_only_at_last_transition(self):
        buffer = self.buffer()
        for step in range(5):
            self.record(buffer, step, num_envs=2,
                        done=torch.tensor([step in (0, 3), True], dtype=torch.long))
        # Exactly one candidate survives, regardless of requested sample count.
        sequences = buffer.sample_sequences(128, 3)
        self.assert_consecutive(sequences)
        self.assertTrue(torch.all(sequences['states'][:, :, 0] == 0).item())
        self.assertTrue(torch.all(sequences['states'][:, 0, 1] == 1).item())
        self.assertTrue(sequences['dones'][:, -1].all().item())

    def test_no_valid_episode_sequence_reports_not_ready(self):
        buffer = self.buffer()
        for step in range(10):
            self.record(buffer, step, done=torch.ones(3, dtype=torch.long))
        self.assertFalse(buffer.can_sample_sequences(2))
        self.assertTrue(buffer.can_sample_sequences(1))
        with self.assertRaisesRegex(ValueError, "No valid 2-step"):
            buffer.sample_sequences(1, 2)

    def test_overwrite_invalidates_cached_sequence_starts(self):
        buffer = self.buffer(capacity=6)
        for step in range(3):
            self.record(buffer, step, num_envs=2)
        self.assertTrue(buffer.can_sample_sequences(3))
        for step in range(3, 6):
            self.record(buffer, step, num_envs=2, done=torch.ones(2, dtype=torch.long))
        self.assertFalse(buffer.can_sample_sequences(3))

    def test_clear_resets_environment_layout_and_sequence_cache(self):
        buffer = self.buffer()
        for step in range(3):
            self.record(buffer, step)
        self.assertTrue(buffer.can_sample_sequences(3))
        buffer.clear()
        self.assertFalse(buffer.can_sample_sequences(1))
        self.assertEqual(buffer.total_added, 0)
        for step in range(3):
            self.record(buffer, step, num_envs=1)
        self.assert_consecutive(buffer.sample_sequences(2, 3))

    def test_batch_larger_than_capacity_keeps_newest_entries(self):
        buffer = self.buffer(capacity=4)
        self.record(buffer, 0, num_envs=6)
        self.assertEqual(set(buffer.states[:, 0].tolist()), {2., 3., 4., 5.})
        self.assertEqual(len(buffer), 4)
        self.assertEqual(buffer.ptr, 2)
        self.assertFalse(buffer.can_sample_sequences(2))
        self.assertTrue(buffer.can_sample_sequences(1))

    def test_changing_environment_batch_is_rejected_before_writing(self):
        buffer = self.buffer()
        self.record(buffer, 0)
        with self.assertRaisesRegex(ValueError, "Expected 3 environments"):
            self.record(buffer, 1, num_envs=2)
        self.assertEqual(len(buffer), 3)
        self.assertEqual(buffer.total_added, 3)

    def test_prioritized_buffer_accepts_integer_done(self):
        buffer = PrioritizedReplayBuffer(10, state_dim=2, action_dim=1, device=self.device)
        self.record(buffer, 0, done=torch.tensor([0, 1, 0]))
        batch, indices, weights = buffer.sample(2)
        self.assertEqual(batch['dones'].dtype, torch.bool)
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertEqual(indices.numel(), 2)


class CPUReplayTests(ReplayChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAReplayTests(ReplayChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == '__main__':
    unittest.main()
