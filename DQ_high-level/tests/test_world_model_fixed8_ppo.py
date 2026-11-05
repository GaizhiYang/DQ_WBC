"""Production adapter and PPO integration with distinct low/high replay clocks."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from skrl.memories.torch import RandomMemory
from test_world_model_env import make_env
from test_world_model_ppo import TinyPolicy, TinyValue
from utils.wm_ppo import WorldModelPPO
from utils.world_model import HierarchicalDynamics
from utils.world_model.kinematics import DEFAULT_DOF_NAMES, DEFAULT_URDF


class FrozenLow(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.linear = torch.nn.Linear(781, 18).to(device)
        with torch.no_grad():
            self.linear.weight.mul_(.01)
            self.linear.bias.zero_()

    def forward(self, observation, hist_encoding=True):
        return self.linear(observation).tanh()


class LowOwner:
    def __init__(self, device):
        self.actor = FrozenLow(device)

    def act_inference(self, observation, hist_encoding=True):
        return self.actor(observation, hist_encoding=hist_encoding)


def make_fixed8_agent(device):
    env = make_env(device, n=12)
    env.rand_control, env.mask_arm, env.table_dimz = False, False, .2
    env.low_obs_history_buf = torch.zeros(12, 10, 71, device=device)
    env._initial_dof_pos = env._dof_pos.clone()
    env.dof_limits_upper = torch.ones(19, device=device)
    env.dof_limits_lower = -torch.ones(19, device=device)
    env.dof_limits_upper[-1] = 0
    env.dof_names = DEFAULT_DOF_NAMES
    env.low_level_policy = LowOwner(device).act_inference
    env.envs, env.robot_handles = [0], [0]
    env.gym = SimpleNamespace(get_actor_rigid_body_properties=lambda *_: [
        SimpleNamespace(com=SimpleNamespace(x=.3, y=0., z=.1155))])
    env.cfg["env"]["asset"] = {"assetRoot": str(DEFAULT_URDF.parent),
                                "assetFileRobot": DEFAULT_URDF.name}
    config = dict(rollouts=3, learning_epochs=1, mini_batches=1, learning_rate=1e-4,
                  experiment={"write_interval": 0, "checkpoint_interval": 0})
    agent = WorldModelPPO(
        models={"policy": TinyPolicy(1284, device), "value": TinyValue(1284, device)},
        memory=RandomMemory(3, env.num_envs, device=device),
        observation_space=1284, action_space=9, device=device, cfg=config,
        wm_env=env, wm_config=dict(mode="fit", fixed8=True, replay_envs=12,
            replay_capacity=6, low_replay_capacity=24, window=8, horizon=2,
            hidden=[16, 16], contact_hidden=[8], batch_size=4, batches_per_update=1,
            validation_batch_size=4, holdout_fraction=.5, warmup_steps=0))
    agent.init()
    return agent


def collect_fixed8(agent, high_steps=2):
    adapter = agent.wm_adapter
    state, context = adapter.capture()
    zeros = torch.zeros(len(state), device=state.device)

    def transition(before, after, actions, duration):
        return dict(state=before.detach().clone(), next_state=after.detach().clone(),
                    actions=actions.detach().clone(), context=context.clone(),
                    dt=torch.full((len(state), 1), duration, device=state.device),
                    reward=zeros, terminated=zeros.bool(), truncated=zeros.bool(),
                    success=zeros.bool(), family=zeros.long(), episode_id=zeros.long())

    for index in range(high_steps):
        with torch.no_grad():
            observation = adapter.observation(state, context)
            action, _, _ = agent.act(observation, index, 100)
            current = adapter.prepare_high(state, action)
            low_steps = []
            for _ in range(8):
                prepared, low_action = adapter.low_control(current, context)
                future = agent.wm_candidate.low_step(prepared, low_action, context)
                low_steps.append(transition(current, future, low_action, .02))
                current = future
            future = adapter.finish_high(state, current, action, context)
            high = transition(state, future, action, .16)
            high["low_steps"] = low_steps
            next_observation = adapter.observation(future, context)
            agent.record_transition(observation, action, zeros[:, None], next_observation,
                                    zeros[:, None].bool(), zeros[:, None].bool(),
                                    {"world_model": high}, index, 100)
            state = future


class Fixed8PPOChecks:
    device = "cpu"

    def test_collector_routes_20ms_targets_and_160ms_starts_to_separate_replays(self):
        agent = make_fixed8_agent(self.device)
        collect_fixed8(agent)
        self.assertIsInstance(agent.wm_candidate, HierarchicalDynamics)
        self.assertEqual(agent.wm_replay.steps, 2)
        self.assertEqual(agent.wm_low_replay.steps, 16)
        self.assertEqual(agent.wm_replay.action_dim, 9)
        self.assertEqual(agent.wm_low_replay.action_dim, 20)
        self.assertEqual(agent.wm_imagination.cfg.dt, .16)
        low = agent.wm_low_replay.sample_windows(4, 8, split="all")
        torch.testing.assert_close(low.dt, torch.full_like(low.dt, .02))
        high = agent.wm_replay.sample_starts(4, split="all")
        torch.testing.assert_close(high.dt, torch.full_like(high.dt, .16))
        # Candidate fitting must receive 20-D recorded actuators. Accidentally
        # handing it high replay triggers the real low_step input check.
        metrics = agent._update_world_model()
        self.assertEqual(agent.wm_status, "fit_only")
        self.assertEqual(metrics["candidate/updates"], 1)
        self.assertIn("candidate/contact_bce", metrics)
        self.assertEqual(agent.wm_low_replay.device.type, torch.device(self.device).type)

    def test_high_validation_horizon_is_not_interpreted_as_microstep_window(self):
        agent = make_fixed8_agent(self.device)
        collect_fixed8(agent)
        # There are two high records (horizon=2), fewer than window=8 microsteps.
        errors = agent._validated_errors()
        self.assertIsNotNone(errors)
        self.assertEqual(errors["nse"].shape, (2,))
        metadata = agent.wm_metadata
        self.assertEqual(metadata["low_steps"], 8)
        self.assertEqual(metadata["low_action_dim"], 20)
        self.assertNotEqual(metadata["schema"], "dq-teacher-state-v1")
        mismatched = dict(metadata, low_steps=7)
        with self.assertRaisesRegex(ValueError, "schema/task"):
            agent._check_metadata(mismatched)


class CPUFixed8PPOTests(Fixed8PPOChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "CUDA device unavailable")
class CUDAFixed8PPOTests(Fixed8PPOChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
