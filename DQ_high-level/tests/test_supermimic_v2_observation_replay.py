"""Full-state observation replay ownership and differentiable imagination."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic_v2 import DQRobotMDP, SuperMimicAgentV2


class ObservationReplayChecks:
    device = 'cpu'

    def make_agent(self, obs_dim=15, capacity=16, window=2):
        env = SimpleNamespace(
            num_envs=2, num_actions=9, num_obs=obs_dim, num_gripper_joints=1,
            num_features=0, max_episode_length=100,
            ee_pos=torch.zeros(2, 3, device=self.device),
            ee_orn=torch.tensor([[0., 0., 0., 1.]], device=self.device).repeat(2, 1),
            dof_limits_lower=torch.tensor([-1.5707963], device=self.device),
            dof_limits_upper=torch.tensor([0.], device=self.device),
        )
        mdp = DQRobotMDP(env, device=self.device)
        agent = SuperMimicAgentV2(
            mdp, device=self.device, wm_hidden=(), policy_hidden=(), critic_hidden=(),
            wm_ensemble_size=1, wm_window=window, horizon=2, wm_batch_size=2,
            policy_batch_size=2, replay_capacity=capacity, min_buffer_size=1,
            warmup_iterations=1, critic_epochs=1, critic_minibatches=1,
        )
        return env, agent

    def snapshot(self, env, mdp, steps, episodes=(0, 0)):
        physical = torch.zeros(2, 34, device=self.device)
        physical[:, 0] = torch.arange(2, device=self.device)
        physical[:, 1] = torch.as_tensor(steps, device=self.device)
        physical[:, 2] = torch.as_tensor(episodes, device=self.device)
        physical[:, 6] = 1
        physical[:, 7:10] = torch.tensor([.3, .2, .6], device=self.device)
        physical[:, 10] = 1
        physical[:, 14] = 1
        physical[:, 22] = -.5
        physical[:, 33] = physical[:, 1] / env.max_episode_length
        marker = physical[:, 0] * 1000 + physical[:, 2] * 100 + physical[:, 1] * 10
        obs = marker[:, None] + torch.arange(mdp.obs_dim, device=self.device)[None, :].float()
        env.ee_pos.copy_(torch.stack((physical[:, 0], physical[:, 1] * .01,
                                     physical[:, 2] + .6), -1))
        state = mdp.capture_state(physical, obs)
        return state, obs

    def record(self, agent, state, obs, next_state, next_obs, done=(0, 0)):
        agent.observe_transition(
            state=state, action=torch.zeros(2, 9, device=self.device),
            reward=torch.ones(2, 1, device=self.device), next_state=next_state,
            done=torch.tensor(done, dtype=torch.long, device=self.device).reshape(2, 1),
            context=torch.zeros(2, 16, device=self.device), obs=obs, next_obs=next_obs,
        )

    def test_replay_observations_share_full_state_storage_without_extra_allocations(self):
        # Match the teacher's large feature/observation footprint, not only a toy
        # vector where accidentally allocating duplicate obs is hard to notice.
        _, agent = self.make_agent(obs_dim=1284, capacity=11)
        replay = agent.replay_buffer
        self.assertEqual(agent.mdp.state_dim, 34 + 7 + agent.mdp.obs_dim)
        for state_key, obs_key in (('state', 'obs'), ('next_state', 'next_obs')):
            state, obs = replay[state_key], replay[obs_key]
            self.assertEqual(state.untyped_storage().data_ptr(), obs.untyped_storage().data_ptr())
            self.assertEqual(obs.storage_offset(), state.storage_offset() + 41)
            self.assertEqual(obs.stride(0), state.stride(0))
            self.assertEqual(tuple(obs.shape), (11, 1284))
        unique = {value.untyped_storage().data_ptr(): value.untyped_storage().nbytes()
                  for value in replay.values()}
        expected = sum(replay[name].numel() * replay[name].element_size()
                       for name in ('state', 'next_state', 'action', 'reward', 'done', 'context'))
        self.assertEqual(len(unique), 6)
        self.assertEqual(sum(unique.values()), expected)

    def test_capture_and_replay_preserve_every_real_observation_column(self):
        env, agent = self.make_agent(obs_dim=1284)
        state, obs = self.snapshot(env, agent.mdp, (0, 0))
        next_state, next_obs = self.snapshot(env, agent.mdp, (1, 1))
        expected_state, expected_next = state.clone(), next_state.clone()
        self.record(agent, state, obs, next_state, next_obs)
        torch.testing.assert_close(agent.mdp.observe(state, torch.zeros(2, 16, device=self.device),
                                                     torch.zeros(2, 9, device=self.device)), obs)
        torch.testing.assert_close(agent.replay_buffer['obs'][:2], obs)
        torch.testing.assert_close(agent.replay_buffer['next_obs'][:2], next_obs)
        # Simulator tensors and observation dictionaries are mutable and reused.
        for value in (env.ee_pos, env.ee_orn, state, next_state, obs, next_obs):
            value.fill_(-999)
        torch.testing.assert_close(agent.replay_buffer['state'][:2], expected_state)
        torch.testing.assert_close(agent.replay_buffer['next_state'][:2], expected_next)
        torch.testing.assert_close(agent.replay_buffer['obs'][:2], expected_state[:, 41:])
        self.assertTrue(all(not value.requires_grad for value in agent.replay_buffer.values()))

    def test_terminal_next_state_and_reset_initial_state_remain_distinct(self):
        env, agent = self.make_agent()
        state, obs = self.snapshot(env, agent.mdp, (0, 0))
        terminal, terminal_obs = self.snapshot(env, agent.mdp, (1, 1))
        self.record(agent, state, obs, terminal, terminal_obs, done=(1, 0))
        reset_state, reset_obs = self.snapshot(env, agent.mdp, (0, 1), episodes=(1, 0))
        next_state, next_obs = self.snapshot(env, agent.mdp, (1, 2), episodes=(1, 0))
        self.record(agent, reset_state, reset_obs, next_state, next_obs)
        replay = agent.replay_buffer
        torch.testing.assert_close(replay['next_state'][0], terminal[0])
        torch.testing.assert_close(replay['state'][2], reset_state[0])
        torch.testing.assert_close(replay['next_obs'][0], terminal_obs[0])
        torch.testing.assert_close(replay['obs'][2], reset_obs[0])
        self.assertFalse(torch.equal(replay['next_obs'][0], replay['obs'][2]))
        self.assertNotEqual(replay['next_state'][0, 36].item(), replay['state'][2, 36].item())

    def test_inconsistent_observation_is_rejected_before_shared_storage_is_written(self):
        env, agent = self.make_agent()
        state, obs = self.snapshot(env, agent.mdp, (0, 0))
        next_state, next_obs = self.snapshot(env, agent.mdp, (1, 1))
        with self.assertRaises(ValueError):
            self.record(agent, state, obs + 1, next_state, next_obs)
        self.assertEqual(agent.buffer_size, 0)
        self.assertEqual(agent.buffer_ptr, 0)
        self.assertEqual(agent.replay_buffer['state'].count_nonzero().item(), 0)
        self.assertEqual(agent.replay_buffer['next_state'].count_nonzero().item(), 0)

    def test_windows_keep_observation_slice_and_terminal_masks_consistent(self):
        env, agent = self.make_agent()
        state, obs = self.snapshot(env, agent.mdp, (0, 0))
        terminal, terminal_obs = self.snapshot(env, agent.mdp, (1, 1))
        self.record(agent, state, obs, terminal, terminal_obs, done=(1, 0))
        reset_state, reset_obs = self.snapshot(env, agent.mdp, (0, 1), episodes=(1, 0))
        next_state, next_obs = self.snapshot(env, agent.mdp, (1, 2), episodes=(1, 0))
        self.record(agent, reset_state, reset_obs, next_state, next_obs)
        # Select one window from each environment to check both sides of reset.
        starts = torch.tensor([0, 1], device=self.device)
        with patch('supermimic_v2.agent_v2.torch.randint', return_value=starts):
            windows = agent.sample_windows(2)
        torch.testing.assert_close(windows['mask'], torch.tensor([[1., 0.], [1., 1.]], device=self.device))
        torch.testing.assert_close(windows['s1'][0, 0, 41:], terminal_obs[0])
        torch.testing.assert_close(windows['s'][0, 1, 41:], reset_obs[0])
        torch.testing.assert_close(windows['s1'][1, 0], windows['s'][1, 1])
        for key in ('s', 's1'):
            full = windows[key].reshape(-1, agent.mdp.state_dim)
            decoded = agent.mdp.observe(full, windows['context'].reshape(-1, 16),
                                         windows['a'].reshape(-1, 9))
            torch.testing.assert_close(decoded, full[:, 41:])

    def test_ring_wrap_preserves_full_state_and_observation_chronology(self):
        env, agent = self.make_agent(capacity=7)
        for step in range(5):
            state, obs = self.snapshot(env, agent.mdp, (step, step))
            next_state, next_obs = self.snapshot(env, agent.mdp, (step + 1, step + 1))
            self.record(agent, state, obs, next_state, next_obs)
        windows = agent.sample_windows(64)
        torch.testing.assert_close(windows['mask'], torch.ones(64, 2, device=self.device))
        torch.testing.assert_close(windows['s1'][:, 0], windows['s'][:, 1])
        torch.testing.assert_close(windows['s'][:, :, 41],
                                   windows['s'][:, :, 0] * 1000 + windows['s'][:, :, 1] * 10)
        batch = agent.sample_batch(7)
        torch.testing.assert_close(batch['obs'], batch['state'][:, 41:])
        torch.testing.assert_close(batch['next_obs'], batch['next_state'][:, 41:])

    def test_imagined_observation_dynamics_backpropagate_into_the_previous_action(self):
        env, agent = self.make_agent()
        state, obs = self.snapshot(env, agent.mdp, (0, 0))
        dynamic_column = agent.mdp.obs_dim - 10  # Not the analytic previous-action tail.
        with torch.no_grad():
            for parameter in agent.wm.parameters():
                parameter.zero_()
            for parameter in agent.policy.parameters():
                parameter.zero_()
            # The only learned action dependency is in the observation section
            # of the state. Physical/EE deltas cannot provide another path.
            agent.wm.models[0].net[0].weight[41 + dynamic_column, agent.mdp.state_dim] = 2.
            agent.policy.net[0].weight[0, dynamic_column] = 3.
        action = torch.full((2, 9), .1, device=self.device, requires_grad=True)
        context = torch.zeros(2, 16, device=self.device)
        context[:, 12] = .1
        next_state, _ = agent.wm.imagine(state, action, context, agent.mdp)
        next_obs = agent.mdp.advance_obs(obs, next_state, action, context)
        torch.testing.assert_close(next_obs, next_state[:, 41:])
        next_action = agent.policy(next_obs)
        gradient = torch.autograd.grad(next_action[:, 0].sum(), action)[0]
        expected = torch.zeros_like(action)
        expected[:, 0] = 6.
        torch.testing.assert_close(gradient, expected, rtol=1e-5, atol=1e-5)


class CPUObservationReplayTests(ObservationReplayChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
class CUDAObservationReplayTests(ObservationReplayChecks, unittest.TestCase):
    device = 'cuda:0'


if __name__ == '__main__':
    unittest.main()
