"""CPU-only checks for bounded evaluation without any PPO training callbacks."""

from pathlib import Path
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DQ_high-level"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "skrl"))

from skrl.trainers.torch import SequentialTrainer
from learning.teacher_vision_trainer import TeacherVisionTrainer


class FakeEvaluationEnv:
    num_agents = 1
    num_envs = 2

    def __init__(self, terminated_steps=(), truncated_steps=()):
        self.terminated_steps = set(terminated_steps)
        self.truncated_steps = set(truncated_steps)
        self.steps = 0
        self.resets = 0
        self.renders = 0
        self.grad_enabled = []
        self.action_requires_grad = []

    def reset(self):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.resets += 1
        return torch.full((2, 3), -float(self.resets), requires_grad=True), {}

    def step(self, actions):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.action_requires_grad.append(actions.requires_grad)
        self.steps += 1
        terminated = torch.tensor([[self.steps in self.terminated_steps], [False]])
        truncated = torch.tensor([[False], [self.steps in self.truncated_steps]])
        return torch.full((2, 3), float(self.steps), requires_grad=True), torch.zeros(2, 1), terminated, truncated, {}

    def render(self):
        self.grad_enabled.append(torch.is_grad_enabled())
        self.renders += 1


class FakeEvaluationAgent:
    """Expose accidental rollout recording or optimizer callbacks as failures."""

    def __init__(self):
        self.model = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1.)
        self.calls = []
        self.init_calls = 0
        self.running_mode = None

    def init(self, trainer_cfg):
        self.init_calls += 1

    def set_running_mode(self, mode):
        self.running_mode = mode
        self.model.eval()

    def act(self, states, timestep, timesteps):
        self.calls.append((timestep, timesteps, states.detach().clone(), torch.is_grad_enabled()))
        return self.model(states), None, {}

    def record_transition(self, **kwargs):
        raise AssertionError("Evaluation must not populate training rollout memory")

    def pre_interaction(self, **kwargs):
        raise AssertionError("Evaluation must not invoke training callbacks")

    def post_interaction(self, **kwargs):
        raise AssertionError("Evaluation must not invoke PPO updates or checkpoint callbacks")


class TeacherVisionTrainerTests(unittest.TestCase):
    def make_trainer(self, env, agent, timesteps, headless=True, initial_timestep=0):
        return TeacherVisionTrainer(env=env, agents=agent, cfg={
            "timesteps": timesteps,
            "initial_timestep": initial_timestep,
            "headless": headless,
            "disable_progressbar": True,
            "close_environment_at_exit": False,
        })

    def test_exact_budget_termination_reset_and_no_training(self):
        env = FakeEvaluationEnv(terminated_steps=(2,), truncated_steps=(4,))
        agent = FakeEvaluationAgent()
        weights = {name: value.detach().clone() for name, value in agent.model.state_dict().items()}
        trainer = self.make_trainer(env, agent, timesteps=5, initial_timestep=2400)
        trainer.eval()
        self.assertEqual(agent.init_calls, 1)
        self.assertEqual(agent.running_mode, "eval")
        self.assertEqual(env.steps, 5)
        self.assertEqual(env.resets, 3)
        self.assertEqual(env.renders, 0)
        self.assertEqual([call[0] for call in agent.calls], list(range(5)))
        self.assertTrue(all(call[1] == 5 for call in agent.calls))
        self.assertEqual([call[2][0, 0].item() for call in agent.calls], [-1., 1., -2., 3., -3.])
        self.assertFalse(any(call[3] for call in agent.calls))
        self.assertFalse(any(env.grad_enabled))
        self.assertFalse(any(env.action_requires_grad))
        for name, value in agent.model.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in agent.model.parameters()))
        self.assertEqual(agent.optimizer.state_dict()["state"], {})

    def test_non_headless_render_count_matches_budget(self):
        env, agent = FakeEvaluationEnv(), FakeEvaluationAgent()
        self.make_trainer(env, agent, timesteps=3, headless=False).eval()
        self.assertEqual(env.steps, 3)
        self.assertEqual(env.renders, 3)
        self.assertEqual(env.resets, 1)

    def test_zero_budget_does_not_step_and_training_remains_inherited(self):
        env, agent = FakeEvaluationEnv(), FakeEvaluationAgent()
        self.make_trainer(env, agent, timesteps=0).eval()
        self.assertEqual(env.steps, 0)
        self.assertEqual(agent.calls, [])
        self.assertIs(TeacherVisionTrainer.single_agent_train, SequentialTrainer.single_agent_train)
        self.assertIs(TeacherVisionTrainer.train, SequentialTrainer.train)


if __name__ == "__main__":
    unittest.main()
