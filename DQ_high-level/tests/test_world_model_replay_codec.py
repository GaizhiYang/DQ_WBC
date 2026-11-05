"""Compact low-level storage preserves real physical windows and GPU residency."""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.replay import SliceStateCodec, TransitionReplay


FULL_DIM = 922
SLICES = {"robot": slice(0, 13), "object": slice(13, 26), "table": slice(26, 39),
          "dof_pos": slice(52, 71), "dof_vel": slice(71, 90),
          "foot_contacts": slice(844, 848), "table_height": slice(913, 914)}
NAMES = ("robot", "object", "dof_pos", "dof_vel", "table", "table_height", "foot_contacts")


class CodecChecks:
    device = "cpu"

    def codec(self):
        return SliceStateCodec(FULL_DIM, SLICES, NAMES)

    def replay(self, codec=True, capacity=5, envs=2):
        return TransitionReplay(envs, capacity, FULL_DIM, 20, 26, self.device,
                                holdout_fraction=0, state_codec=self.codec() if codec else None)

    def add(self, replay, value, next_value=None, terminal=False):
        states = torch.full((replay.num_envs, FULL_DIM), float(value), device=self.device)
        future = torch.full_like(states, float(value + 1 if next_value is None else next_value))
        replay.add(states, states[:, :20], future, states[:, :26], .02,
                   torch.ones(replay.num_envs, device=self.device),
                   torch.full((replay.num_envs,), terminal, device=self.device),
                   torch.zeros(replay.num_envs, device=self.device, dtype=torch.bool),
                   torch.zeros(replay.num_envs, device=self.device, dtype=torch.long))

    def test_codec_roundtrip_arbitrary_leading_dimensions_and_gradients(self):
        codec = self.codec()
        self.assertEqual(codec.storage_dim, 82)
        selected = torch.zeros(FULL_DIM, device=self.device, dtype=torch.bool)
        for field in SLICES.values():
            selected[field] = True
        for leading in ((), (3,), (2, 3)):
            full = torch.randn(*leading, FULL_DIM, device=self.device,
                               dtype=torch.float64, requires_grad=True)
            compact = codec.encode(full)
            reconstructed = codec.decode(compact)
            self.assertEqual(compact.shape, (*leading, 82))
            self.assertEqual(reconstructed.device, full.device)
            self.assertEqual(reconstructed.dtype, full.dtype)
            torch.testing.assert_close(reconstructed[..., selected], full[..., selected], rtol=0, atol=0)
            self.assertEqual(float(reconstructed[..., ~selected].abs().sum()), 0.)
            gradient, = torch.autograd.grad(reconstructed.sum(), full)
            torch.testing.assert_close(gradient, selected.to(full.dtype).expand_as(full), rtol=0, atol=0)

    def test_storage_memory_accounting_and_sample_contract(self):
        compact, ordinary = self.replay(), self.replay(codec=False)
        self.assertEqual(compact.state_dim, FULL_DIM)
        self.assertEqual(compact.storage_state_dim, 82)
        self.assertEqual(compact.states.shape[-1], 82)
        self.assertEqual(ordinary.states.shape[-1], FULL_DIM)
        saved = 2 * compact.capacity * compact.num_envs * (FULL_DIM - 82) * 4
        self.assertEqual(ordinary.memory_bytes - compact.memory_bytes, saved)
        # No codec-owned tensors or hidden uncompressed resident copies exist.
        self.assertEqual(compact.memory_bytes,
                         sum(value.numel() * value.element_size()
                             for value in vars(compact).values() if torch.is_tensor(value)))
        self.add(compact, 7)
        starts = compact.sample_starts(8, split="all")
        self.assertEqual(starts.state.shape, (8, FULL_DIM))
        self.assertEqual(starts.next_state.shape, (8, FULL_DIM))
        torch.testing.assert_close(starts.state[:, :13], torch.full_like(starts.state[:, :13], 7.))
        torch.testing.assert_close(starts.next_state[:, :13], torch.full_like(starts.next_state[:, :13], 8.))
        self.assertEqual(float(starts.state[:, 39:52].abs().sum()), 0.)
        for name, value in vars(starts).items():
            self.assertEqual(value.device, compact.device, name)

    def test_terminal_next_state_survives_reset_without_crossing_episode(self):
        replay = self.replay(envs=1)
        self.add(replay, 0)
        self.add(replay, 1)
        self.add(replay, 2, next_value=99, terminal=True)
        self.add(replay, 1000)
        mask = replay.window_mask(3, split="all")
        self.assertEqual(int(mask.sum()), 1)
        batch = replay.sample_windows(16, 3, split="all")
        expected = torch.tensor([0., 1., 2., 99.], device=self.device)
        torch.testing.assert_close(batch.states[:, :, 0], expected[:, None].expand(-1, 16))
        self.assertTrue(bool(batch.terminated[-1].all()))
        self.assertFalse(bool(batch.terminated[:-1].any()))
        self.assertEqual(float(batch.states[:, :, 39:52].abs().sum()), 0.)

    def test_windows_across_ring_boundary_remain_chronological(self):
        replay = self.replay(capacity=4, envs=1)
        for value in range(6):
            self.add(replay, value)
        batch = replay.sample_windows(128, 3, split="all")
        self.assertEqual(batch.states.shape, (4, 128, FULL_DIM))
        self.assertTrue(bool((batch.states[0, :, 0] >= 2).all()))
        torch.testing.assert_close(batch.states[1:, :, 0] - batch.states[:-1, :, 0],
                                   torch.ones(3, 128, device=self.device))
        torch.testing.assert_close(batch.actions[:, :, 0], batch.states[:-1, :, 0])

    def test_no_codec_keeps_all_fields_and_explicit_terminal_payload(self):
        replay = self.replay(codec=False, envs=1)
        self.add(replay, 3, next_value=10, terminal=True)
        batch = replay.sample_windows(4, 1, split="all")
        torch.testing.assert_close(batch.states[0], torch.full_like(batch.states[0], 3.))
        torch.testing.assert_close(batch.states[1], torch.full_like(batch.states[1], 10.))
        self.assertIsNone(replay.state_codec)
        self.assertEqual(replay.storage_state_dim, FULL_DIM)

    def test_invalid_codec_layout_and_replay_width_fail_explicitly(self):
        for names in ((), ("robot", "robot"), ("missing",)):
            with self.assertRaises(ValueError):
                SliceStateCodec(FULL_DIM, SLICES, names)
        with self.assertRaisesRegex(ValueError, "overlap"):
            SliceStateCodec(4, {"a": slice(0, 3), "b": slice(2, 4)}, ("a", "b"))
        with self.assertRaisesRegex(ValueError, "state_dim"):
            TransitionReplay(1, 2, 3, 1, 1, self.device, state_codec=self.codec())
        with self.assertRaisesRegex(ValueError, "encode"):
            self.codec().encode(torch.zeros(3, device=self.device))
        with self.assertRaisesRegex(ValueError, "decode"):
            self.codec().decode(torch.zeros(3, device=self.device))


class CPUCodecTests(CodecChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDACodecTests(CodecChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
