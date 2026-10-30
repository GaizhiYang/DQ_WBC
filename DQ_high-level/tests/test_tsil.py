"""TSIL configuration identity, episode semantics and paper replay invariants."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.tsil import (TemporalReplay, TemporalState, Trajectory, geometry_descriptors,
                        GEOMETRY_DESCRIPTOR_SCHEMA, GEOMETRY_DESCRIPTOR_NAMES,
                        LOOKUP_NEAREST, LOOKUP_NO_SUCCESS, LOOKUP_OUT_OF_SUPPORT,
                        LOOKUP_INVALID)


def quaternion(roll=0., pitch=0., yaw=0.):
    angles = torch.tensor([roll, pitch, yaw]) / 2
    sr, sp, sy = angles.sin()
    cr, cp, cy = angles.cos()
    return torch.stack((sr * cp * cy - cr * sp * sy,
                        cr * sp * cy + sr * cp * sy,
                        cr * cp * sy - sr * sp * cy,
                        cr * cp * cy + sr * sp * sy))


def trajectory(uid, config=0, success=False, elapsed=1.0, score=0.0, length=4):
    obs = torch.arange(length * 7, dtype=torch.float32).reshape(length, 7) / 10
    obs[:, 4] = torch.arange(length) / 10
    obs[:, 5] = 1
    return Trajectory((0, uid), config, obs, torch.full((length, 2), float(uid)),
                      torch.full((length,), -1.0), torch.full((length,), 2.0),
                      torch.zeros(length), success, elapsed, score)


def replay(state=None, envs=1, topk=5, prefix=0, maximum=100):
    return TemporalReplay(state or TemporalState(2, 10), envs, 7, 2,
                          gamma=0.5, topk=topk, time_indices=(4, 5),
                          static_prefix_size=prefix, max_episode_steps=maximum)


def transition(memory, ids, episodes, start, end, done, success=None, raw=None,
               fixed=None, command=None):
    n = len(ids)
    states = torch.zeros(n, 7)
    states[:, :2] = torch.tensor([8., 9.])
    states[:, 2] = torch.arange(n) + 20  # a causal observation channel, e.g. EKF
    states[:, 4] = torch.as_tensor(start) / memory.state.t_ref
    states[:, 5] = 1
    info = {"config_id": torch.tensor(ids), "episode_id": torch.tensor(episodes),
            "elapsed": torch.tensor(end), "success": torch.tensor(success or [False] * n),
            "reward_fixed": torch.tensor(fixed if fixed is not None else [-1.] * n),
            "reward_command_raw": torch.tensor(command if command is not None else [2.] * n)}
    memory.record(states, torch.ones(n, 2), torch.tensor(raw or [1.] * n), torch.tensor(done), info)
    return states, info


class TemporalStateTests(unittest.TestCase):
    def test_repeated_configuration_success_commits_only_after_update(self):
        state = TemporalState(3, 16)
        active_episode_target = state.targets[1].clone()
        state.observe_success([1, 1, 0], [8, 6, 10])
        torch.testing.assert_close(state.targets, torch.full((3,), 16.))
        self.assertFalse(bool(state.seen_success.any()))
        self.assertEqual(state.commit_targets(), 2)
        torch.testing.assert_close(state.targets, torch.tensor([10., 6., 16.]))
        self.assertEqual(active_episode_target.item(), 16)
        state.observe_success([1], [12])
        state.commit_targets()
        self.assertEqual(state.targets[1].item(), 6)

    def test_eval_only_successful_full_task_neighbors_and_ood_fallback(self):
        descriptors = torch.tensor([[0., 5.], [1., 5.], [2., 5.], [1., 5.]])
        state = TemporalState(4, 16, family_ids=[0, 0, 0, 2], descriptors=descriptors)
        state.observe_success([0, 2, 3], [6, 10, 1])
        state.commit_targets()
        result = state.lookup_targets(torch.tensor([[1., 5.], [8., 5.], [1., 5.1]]))
        torch.testing.assert_close(result, torch.tensor([8., 16., 16.]))
        # A checkpoint contains all lookup data; no bank is required at evaluation.
        loaded = TemporalState(0, 16)
        loaded.load_state_dict(state.state_dict())
        torch.testing.assert_close(loaded.lookup_targets(torch.tensor([[1., 5.]])), torch.tensor([8.]))

    def test_five_neighbors_and_checkpoint_validation(self):
        state = TemporalState(7, 16, bank_fingerprint="bank-a",
                              descriptors=torch.arange(7.).reshape(-1, 1))
        state.observe_success(torch.arange(7), torch.arange(7.) + 1)
        state.commit_targets()
        self.assertEqual(state.lookup_targets(torch.tensor([[0.]])).item(), 3.)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            TemporalState(7, 16, bank_fingerprint="bank-b").load_state_dict(state.state_dict())
        with self.assertRaisesRegex(ValueError, "time reference"):
            TemporalState(0, 10).load_state_dict(state.state_dict())
        corrupt = copy.deepcopy(state.state_dict())
        corrupt["descriptor_max"] += 1
        with self.assertRaisesRegex(ValueError, "scale"):
            TemporalState(0, 16).load_state_dict(corrupt)

    def test_geometry_ignores_hidden_velocity_and_world_translation_yaw(self):
        robot = torch.zeros(2, 13)
        obj = torch.zeros(2, 13)
        robot[:, 6] = obj[:, 6] = 1
        obj[:, :3] = torch.tensor([2., 0., 1.])
        robot[:, 2] = .5
        robot[1, :2] = torch.tensor([100., 200.])
        obj[1, :2] = robot[1, :2] + torch.tensor([0., 2.])
        robot[1, 5:7] = obj[1, 5:7] = torch.tensor([2 ** -.5, 2 ** -.5])
        robot[1, 7:] = 100
        obj[1, 7:] = -100
        result = geometry_descriptors(robot, obj, torch.ones(2), torch.zeros(2, 19))
        torch.testing.assert_close(result[0], result[1], atol=1e-6, rtol=1e-6)

    def test_configuration_geometry_ignores_reset_settling_but_keeps_object_yaw(self):
        robot = torch.zeros(3, 13)
        obj = torch.zeros(3, 13)
        robot[:, :3] = torch.tensor([0., 0., .515])
        obj[:, :3] = torch.tensor([2., 0., .61])
        robot[0, 3:7] = quaternion(.0002, -.0001, .1)
        obj[0, 3:7] = quaternion(1.57, .01, -.2)
        # Ordinary reset is airborne with different roll/pitch, object tilt,
        # arm and leg positions. These are the same full-task configuration.
        robot[1:, 2] = .6
        robot[1:, 3:7] = quaternion(-.0003, .0005, .1)
        obj[1:, 2] = .615
        obj[1, 3:7] = quaternion(1.575, -.007, -.2)
        obj[2, 3:7] = quaternion(1.575, -.007, .2)
        joints = torch.randn(3, 19)
        descriptors = geometry_descriptors(robot, obj, torch.full((3,), .55), joints)
        self.assertEqual(descriptors.shape, (3, 5))
        torch.testing.assert_close(descriptors[0], descriptors[1], atol=1e-6, rtol=1e-6)
        self.assertFalse(torch.allclose(descriptors[1, 2:4], descriptors[2, 2:4]))
        state = TemporalState(1, 16, descriptors=descriptors[:1],
                              descriptor_schema=GEOMETRY_DESCRIPTOR_SCHEMA)
        state.observe_success([0], [8.])
        state.commit_targets()
        result, status = state.lookup_targets(descriptors[1:], return_status=True)
        torch.testing.assert_close(result, torch.tensor([8., 16.]))
        self.assertEqual(status.tolist(), [LOOKUP_NEAREST, LOOKUP_OUT_OF_SUPPORT])

    def test_lookup_status_is_per_query_stateless_and_preserves_real_ood(self):
        descriptors = torch.tensor([[1.94, -.5, .90, -.4, .5],
                                    [2.00, .5, 1., 0., .6],
                                    [1.97, 0., .90, .4, .55]])
        descriptors[:, 2:4] /= descriptors[:, 2:4].norm(dim=-1, keepdim=True)
        state = TemporalState(3, 16, descriptors=descriptors,
                              descriptor_schema=GEOMETRY_DESCRIPTOR_SCHEMA)
        queries = torch.tensor([[1.97, 0., 1., 0., .55],
                                [3., 0., 1., 0., .55],  # distance
                                [1.97, 0., 1., 0., .8],  # table height
                                [1.97, 0., 0., 1., .55],  # object orientation
                                [float("nan"), 0., 1., 0., .55]])
        _, before = state.lookup_targets(queries, return_status=True)
        self.assertEqual(before.tolist(), [LOOKUP_NO_SUCCESS] * 4 + [LOOKUP_INVALID])
        state.observe_success([0, 1, 2], [6., 8., 10.])
        state.commit_targets()
        original = copy.deepcopy(state.state_dict())
        targets, statuses = state.lookup_targets(queries, return_status=True)
        torch.testing.assert_close(targets, torch.tensor([8., 16., 16., 16., 16.]))
        self.assertEqual(statuses.tolist(), [LOOKUP_NEAREST] + [LOOKUP_OUT_OF_SUPPORT] * 3
                         + [LOOKUP_INVALID])
        for key, value in original.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, state.state_dict()[key])
            else:
                self.assertEqual(value, state.state_dict()[key])

    def test_legacy_dq_checkpoint_migrates_without_bank_or_lookup_data_loss(self):
        legacy_descriptors = torch.randn(3, 32)
        legacy_descriptors[:, :2] = torch.tensor([[1.95, -.2], [2., .2], [.7, .1]])
        legacy_descriptors[:, 3:5] = torch.tensor([[.98, -.1], [.98, .1], [.9, .2]])
        legacy_descriptors[:, 11] = torch.tensor([.5, .6, .55])
        original = TemporalState(3, 16, family_ids=[0, 0, 2], descriptors=legacy_descriptors)
        original.observe_success([0, 1, 2], [6., 10., 1.])
        original.commit_targets()
        legacy = original.state_dict()
        legacy["version"] = 1
        legacy["descriptor_names"] = []
        del legacy["descriptor_schema"]
        loaded = TemporalState(0, 16)
        loaded.load_state_dict(legacy)
        self.assertEqual(loaded.descriptor_schema, GEOMETRY_DESCRIPTOR_SCHEMA)
        self.assertEqual(loaded.descriptor_names, list(GEOMETRY_DESCRIPTOR_NAMES))
        expected = legacy_descriptors[:, [0, 1, 3, 4, 11]].clone()
        expected[:, 2:4] /= expected[:, 2:4].norm(dim=-1, keepdim=True)
        torch.testing.assert_close(loaded.descriptors, expected)
        torch.testing.assert_close(loaded.targets, original.targets)
        result, status = loaded.lookup_targets(expected[:1], return_status=True)
        self.assertEqual(result.item(), 8.)  # excludes successful near_goal
        self.assertEqual(status.item(), LOOKUP_NEAREST)
        self.assertEqual(loaded.state_dict()["version"], 2)
        reloaded = TemporalState(0, 16)
        reloaded.load_state_dict(loaded.state_dict())
        torch.testing.assert_close(reloaded.descriptors, expected)
        corrupt = copy.deepcopy(legacy)
        corrupt["descriptor_min"][12] += 1  # even discarded old columns are validated
        with self.assertRaisesRegex(ValueError, "scale"):
            TemporalState(0, 16).load_state_dict(corrupt)

    def test_schema_validation_and_legacy_generic_compatibility(self):
        legacy = TemporalState(2, 16, descriptors=torch.tensor([[0.], [1.]])).state_dict()
        legacy["version"] = 1
        legacy["descriptor_names"] = []
        del legacy["descriptor_schema"]
        loaded = TemporalState(0, 16)
        loaded.load_state_dict(legacy)
        self.assertEqual(loaded.descriptor_schema, "generic_v1")
        self.assertEqual(loaded.descriptor_names, ["feature_0"])
        for mutation in ({"descriptor_schema": "unrecognized"},
                         {"descriptor_names": ["x", "y"]}):
            corrupt = dict(loaded.state_dict(), **mutation)
            with self.assertRaisesRegex(ValueError, "schema|names"):
                TemporalState(0, 16).load_state_dict(corrupt)
        with self.assertRaisesRegex(ValueError, "schema"):
            TemporalState(2, 16, descriptors=torch.zeros(2, 3),
                          descriptor_schema=GEOMETRY_DESCRIPTOR_SCHEMA)
        robot = torch.zeros(1, 13)
        obj = torch.zeros(1, 13)
        robot[:, 6] = 1
        obj[:, 3:7] = quaternion(pitch=torch.pi / 2)
        descriptor = geometry_descriptors(robot, obj, torch.ones(1))
        self.assertFalse(torch.isfinite(descriptor).all())


class ArchiveTests(unittest.TestCase):
    def test_fast_success_first_then_same_configuration_return_fallback(self):
        memory = replay(topk=3)
        for uid, success, elapsed, score in [(1, True, 8, -20), (2, False, 10, 100),
                                            (3, True, 3, -30), (4, False, 10, 90),
                                            (5, True, 7, -40), (6, True, 2, -50)]:
            memory.insert(trajectory(uid, success=success, elapsed=elapsed, score=score))
        self.assertEqual([t.uid[1] for t in memory.selected(0)], [6, 3, 5])
        self.assertEqual([t.uid[1] for t in memory.high_return[0]], [2, 4, 1])
        self.assertLessEqual(memory.stats["archive_trajectories"], 6)
        fallback_memory = replay(topk=3)
        fallback_memory.insert(trajectory(1, success=True, score=100))
        fallback_memory.insert(trajectory(2, score=90))
        fallback_memory.insert(trajectory(3, score=80))
        fallback_memory.insert(trajectory(4, config=1, score=1000))
        self.assertEqual([t.uid[1] for t in fallback_memory.selected(0)], [1, 2, 3])
        self.assertFalse(fallback_memory.insert(trajectory(1, success=True, score=500)))

    def test_relabel_preserves_ekf_negative_rewards_and_fixed_ranking(self):
        memory = replay()
        t = trajectory(1, success=True, elapsed=8, score=123, length=3)
        original = t.observations.clone()
        obs, returns, weight = memory.relabel(t, target=6, reward_scale=2, command_gate=0)
        self.assertEqual(weight, 1.75)
        torch.testing.assert_close(obs[:, 5], torch.full((3,), .6))
        torch.testing.assert_close(obs[:, [0, 1, 2, 3, 4, 6]], original[:, [0, 1, 2, 3, 4, 6]])
        # Rewards are [-1, -1, -1+87.5]/2; negative intermediate rewards survive.
        torch.testing.assert_close(returns, torch.tensor([10.0625, 21.125, 43.25]))
        _, current, _ = memory.relabel(t, target=6, reward_scale=2, command_gate=1)
        torch.testing.assert_close(current - returns, torch.tensor([1.75, 1.5, 1.]))
        torch.testing.assert_close(t.observations, original)
        self.assertEqual(t.return_score, 123)

    def test_complete_episode_spans_collection_blocks_and_owns_terminal_data(self):
        memory = replay(envs=2, prefix=2)
        states, info = transition(memory, [0, 0], [1, 1], [0, 0], [1, 1], [False, False], raw=[2, 2])
        self.assertEqual(len(list(memory.iter_selected())), 0)
        states.zero_()
        info["reward_fixed"].zero_()
        transition(memory, [0, 0], [1, 1], [1, 1], [2, 2], [True, True],
                   success=[True, False], raw=[4, 4])
        self.assertEqual(memory.completed_episodes, 2)
        episodes = memory.selected(0)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0].return_score, 4)  # 2 + gamma * 4
        self.assertEqual(len(episodes[0]), 2)
        self.assertEqual(episodes[0].full_observations()[0, 2], 20)
        self.assertEqual(episodes[0].reward_fixed[0], -1)
        self.assertEqual(memory.state.targets[0], 10)
        memory.state.commit_targets()
        self.assertEqual(memory.state.targets[0], 2)

    def test_administrative_reset_drops_partial_episode_and_mid_episode_attach(self):
        memory = replay()
        transition(memory, [0], [1], [0], [1], [False])
        # Reset counter changes, so the old unfinished episode is discarded.
        transition(memory, [1], [2], [0], [1], [True])
        self.assertEqual(memory.aborted_episodes, 1)
        self.assertEqual(memory.selected(0), [])
        self.assertEqual(len(memory.selected(1)), 1)
        attached = replay()
        transition(attached, [0], [7], [5], [6], [True], success=[True])
        self.assertEqual(len(list(attached.iter_selected())), 0)
        self.assertTrue(bool(torch.isinf(attached.state.pending_min).all()))

    def test_sampling_all_configurations_once_no_efficiency_in_probability(self):
        memory = replay(TemporalState(40, 10), topk=2)
        for c in range(40):
            memory.insert(trajectory(c + 1, config=c, success=c % 2 == 0, elapsed=8, score=c, length=4))
        callback_sizes = []
        def gap(states, returns):
            callback_sizes.append(len(states))
            return torch.tensor([1., 2., 0., 3.]).repeat(len(states) // 4).reshape(-1, 1)
        original_multinomial = torch.multinomial
        probabilities = []
        def draw(weights, count, replacement):
            probabilities.append(weights.clone())
            self.assertFalse(replacement)
            return original_multinomial(weights, count, replacement=replacement)
        with patch("torch.multinomial", side_effect=draw):
            selected = memory.prepare(gap, transitions_per_trajectory=2, screen_batch_size=100)
        self.assertEqual(len(selected), 80)
        self.assertEqual(set(selected.config_id.tolist()), set(range(40)))
        self.assertEqual(callback_sizes, [100, 60])
        for weights in probabilities:
            torch.testing.assert_close(weights, torch.tensor([1., 2., 3.]))
        self.assertEqual(selected.uids.unique(dim=0).shape[0], 80)
        snapshot = selected.get_batch(torch.arange(len(selected)))
        memory.state.targets.fill_(1)
        self.assertTrue(bool((selected.get_batch(torch.arange(80))["states"][:, 5] == 1).all()))
        torch.testing.assert_close(snapshot["actions"], selected.get_batch(torch.arange(80))["actions"])

    def test_no_positive_gap_produces_empty_replay(self):
        memory = replay()
        memory.insert(trajectory(1, success=False))
        selection = memory.prepare(lambda obs, ret: torch.zeros_like(ret))
        self.assertEqual(len(selection), 0)
        self.assertEqual(selection.stats["replay_trajectories"], 1)

    def test_trajectory_longer_than_screen_microbatch_is_not_truncated(self):
        memory = replay()
        memory.insert(trajectory(1, length=9))
        sizes = []
        def gap(obs, returns):
            sizes.append(len(obs))
            return torch.ones_like(returns)
        selection = memory.prepare(gap, screen_batch_size=2, transitions_per_trajectory=32)
        self.assertEqual(sizes, [2, 2, 2, 2, 1])
        self.assertEqual(len(selection), 9)
        self.assertEqual(set(selection.uids[:, 2].tolist()), set(range(9)))

    def test_compressed_observations_round_trip_without_loss(self):
        memory = replay(prefix=2)
        states, _ = transition(memory, [0], [1], [0], [1], [True])
        selection = memory.prepare(lambda obs, ret: torch.ones_like(ret))
        self.assertEqual(memory.selected(0)[0].observations.shape, (1, 5))
        torch.testing.assert_close(selection.get_batch([0])["states"], states)


if __name__ == "__main__":
    unittest.main()
