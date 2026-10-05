"""CPU checks of pure-Actor PPO budgets and M2 evaluation isolation."""

from pathlib import Path
import json
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "DQ_high-level"))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "skrl"))

from learning.asymmetric_teacher_trainer import AsymmetricTeacherTrainer


class FakeState:
    def __init__(self, env, completed=0):
        self.env = env
        self.completed_steps = completed
        self.best_eval_score = -float("inf")
        self.last_eval_step = -1
        self.marked = []

    def mark_completed(self, step):
        self.completed_steps = step
        getattr(self.env, "_env", self.env).global_step_counter = 100 + step
        self.marked.append(step)

    def state_dict(self):
        return {"version": 2, "completed_steps": self.completed_steps,
                "best_eval_score": self.best_eval_score, "last_eval_step": self.last_eval_step}


class FakeScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("count", torch.tensor(12.))

    def forward(self, states, train=False):
        if train:
            self.count += 1
        return states


class FakePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([0.2, 0.4]))
        self.calls = []

    def compute(self, inputs, role):
        # Deliberately exposes only the Gaussian actor interface and sensor
        # input. The trainer cannot request privileged targets or transition
        # branches from this policy.
        assert set(inputs) == {"states"}
        assert inputs["states"].shape == (2, 3)
        assert role == "policy"
        self.calls.append((inputs["states"].clone(), torch.is_grad_enabled()))
        return self.bias.expand(inputs["states"].shape[0], -1), torch.zeros(2), {}


class FakeEnv:
    num_envs = 2
    num_agents = 1
    device = torch.device("cpu")

    def __init__(self, fail_eval=False, episode_length=2):
        self.global_step_counter = 100
        self.local_step_counter = 10
        self.eval = False
        self.fail_eval = fail_eval
        self.episode_length = episode_length
        self.progress = torch.zeros(2, dtype=torch.long)
        self.episode_counter = torch.tensor([8, 9])
        self.success_counter = torch.tensor([2, 3])
        self.success_onetime_counter = torch.tensor([1, 2])
        self.episode_step_sum = torch.tensor([16., 18.])
        self.one_epoch_step = torch.tensor([1., 1.])
        self.episode_sums = {"reward": torch.tensor([3., 4.])}
        self.episode_metric_sums = {"distance": torch.tensor([5., 6.])}
        self.extras = {"saved": "training"}
        self.resets = []
        self.steps = []
        self.renders = 0
        self.reset_serial = 0

    def reset_idx(self):
        self.resets.append(("all", self.eval))
        self.episode_counter += 1
        self.episode_step_sum += self.one_epoch_step
        self.one_epoch_step.zero_()
        self.progress.zero_()
        self.reset_serial += 1

    def reset(self):
        self.resets.append(("partial", self.eval))
        done = self.progress >= self.episode_length
        self.episode_counter[done] += 1
        self.progress[done] = 0
        self.one_epoch_step[done] = 0
        return torch.full((2, 3), float(self.reset_serial)), {}

    def step(self, actions):
        self.steps.append((self.eval, actions.clone(), self.global_step_counter))
        if self.eval and self.fail_eval:
            raise RuntimeError("simulated evaluation failure")
        self.global_step_counter += 1
        self.local_step_counter += 1
        self.progress += 1
        self.one_epoch_step += 1
        self.episode_sums["reward"] += 1
        self.extras = {"saved": "eval" if self.eval else "training"}
        done = self.progress >= self.episode_length
        if self.eval:
            # One of two robots succeeds; both terminate on the same step.
            self.success_counter[0] += int(done[0])
            self.success_onetime_counter[0] += int(done[0])
            random.random()
            np.random.random()
            torch.rand(1)
        return (torch.full((2, 3), float(self.reset_serial)), torch.ones(2, 1),
                done[:, None], torch.zeros(2, 1, dtype=torch.bool), self.extras)

    def render(self):
        self.renders += 1


class FakePerceptionWrapper:
    """Stateful M2 boundary, with persistent measurements/filter/history."""

    def __init__(self, env):
        self._env = env
        self.measurements = torch.full((2, 3), 99.)
        self.filter_state = torch.full((2, 3), 99.)
        self.history = torch.full((2, 3), 99.)
        self.runtime_resets = []
        self.generator = torch.Generator().manual_seed(751)
        self.runtime_seeds = []
        self.perception_steps = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset_runtime(self):
        self._env.resets.append(("runtime", self._env.eval))
        self.runtime_resets.append((self._env.reset_serial, self._env.eval))
        self.perception_steps = 0
        for buffer in (self.measurements, self.filter_state, self.history):
            buffer.zero_()

    def get_runtime_rng_state(self):
        return self.generator.get_state()

    def set_runtime_rng_state(self, state):
        self.generator.set_state(state)

    def seed_runtime(self, seed):
        self.runtime_seeds.append(seed)
        self.generator.manual_seed(seed)

    def reset(self):
        # Ordinary termination resets only affected episode histories; a full
        # simulator reset has already cleared progress, so it requires the
        # trainer's explicit reset_runtime call before this method.
        done = self._env.progress >= self._env.episode_length
        for buffer in (self.measurements, self.filter_state, self.history):
            buffer[done] = 0
        return self._env.reset()

    def step(self, actions):
        torch.rand(1, generator=self.generator)
        for buffer in (self.measurements, self.filter_state, self.history):
            buffer.add_(1)
        states, reward, terminated, truncated, info = self._env.step(actions)
        # The perception bonus is for training only; evaluation keeps the
        # original task reward and never contributes a PPO transition.
        if not self._env.eval:
            reward = reward + 0.25
        self.perception_steps += 1
        info = dict(info, perception={
            "prediction_error_m": None if self.perception_steps == 1 else self.perception_steps / 10,
            "initialized_fraction": float(self.perception_steps > 1),
            "visible_fraction": .5,
            "reward_mean": .25,
        })
        return states, reward, terminated, truncated, info


class FakeAgent:
    def __init__(self, env, directory, completed=0):
        self.policy = FakePolicy()
        self.models = {"policy": self.policy, "value": torch.nn.Linear(3, 1)}
        self._state_preprocessor = FakeScaler()
        self.state = FakeState(env, completed)
        self.checkpoint_modules = {"asymmetric_teacher_state": self.state,
                                   "state_preprocessor": self._state_preprocessor}
        self.experiment_dir = directory
        self._rollouts = 2
        self._rollout = 0
        self._learning_starts = 0
        self.checkpoint_interval = 1
        self.training = True
        self.records = []
        self.actions = []
        self.updates = []
        self.saved = []
        self.metrics = []
        self._current_log_prob = None
        self._cumulative_rewards = torch.ones(2, 1) * 7
        self._cumulative_timesteps = torch.ones(2, 1) * 5

    def init(self, trainer_cfg):
        self.set_mode("eval")

    def set_mode(self, mode):
        for model in self.models.values():
            model.train(mode == "train")

    def set_running_mode(self, mode):
        self.training = mode == "train"
        self.set_mode(mode)

    def pre_interaction(self, **kwargs):
        pass

    def act(self, states, timestep, timesteps):
        self.actions.append((timestep, states.clone()))
        self._current_log_prob = torch.tensor(float(timestep))
        return self.policy.compute({"states": states}, "policy")

    def record_transition(self, **kwargs):
        self.records.append((kwargs["timestep"], self._current_log_prob.clone(), kwargs["rewards"].clone()))

    def post_interaction(self, timestep, timesteps):
        self._rollout += 1
        if self._rollout % self._rollouts == 0:
            self.set_mode("train")
            self.updates.append(timestep + 1)
            self.set_mode("eval")

    def track_data(self, name, value):
        self.metrics.append((name, value))

    def save(self, path):
        payload = self.state.state_dict()
        self.saved.append((Path(path).name, dict(payload)))
        torch.save(payload, path)


class AsymmetricTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def make(self, *, total=12, completed=0, interval=4,
             eval_steps=4, fail_eval=False, episode_length=2, callback=None,
             perception=True):
        env = FakeEnv(fail_eval, episode_length)
        env.global_step_counter = 100 + completed
        if perception:
            env = FakePerceptionWrapper(env)
        agent = FakeAgent(env, self.temp.name, completed)
        trainer = AsymmetricTeacherTrainer(env, agent, cfg={
            "timesteps": total, "headless": True, "disable_progressbar": True,
            "close_environment_at_exit": False, "eval_interval": interval,
            "eval_steps": eval_steps, "checkpoint_interval": 4,
        }, evaluation_callback=callback)
        return trainer, env, agent

    def test_pure_actor_trains_from_first_rollout_and_saves_completed_updates(self):
        trainer, env, agent = self.make()
        trainer.train()
        self.assertEqual([item[0] for item in agent.records], list(range(12)))
        self.assertEqual(agent.updates, [2, 4, 6, 8, 10, 12])
        self.assertEqual(agent.state.marked, [2, 4, 6, 8, 10, 12])
        self.assertEqual(env.global_step_counter, 112)
        self.assertEqual(agent.state.completed_steps, 12)
        self.assertEqual(agent.checkpoint_interval, 0)
        self.assertEqual(sum(name == "best_agent.pt" for name, _ in agent.saved), 1)
        self.assertEqual([name for name, _ in agent.saved if name.startswith("agent_")],
                         ["agent_4.pt", "agent_8.pt", "agent_12.pt"])
        self.assertEqual([payload["completed_steps"] for name, payload in agent.saved
                          if name.startswith("agent_")], [4, 8, 12])
        self.assertFalse(hasattr(agent.policy, "transition"))
        for _, actions, _ in env.steps:
            torch.testing.assert_close(actions, agent.policy.bias.detach().expand(2, -1))

    def test_evaluation_never_records_or_updates_and_resets_cached_states(self):
        trainer, env, agent = self.make()
        trainer.train()
        self.assertEqual(len(agent.records), 12)
        self.assertEqual(len(agent.actions), 12)
        self.assertEqual(len(agent.updates), 6)
        self.assertEqual([float(item[1]) for item in agent.records], list(range(12)))
        self.assertEqual(sum(is_eval for is_eval, _, _ in env.steps), 12)
        self.assertEqual(sum(not is_eval for is_eval, _, _ in env.steps), 12)
        # Each evaluation is bracketed by complete resets. The next training
        # action receives that new reset observation, never a stale eval state.
        self.assertEqual([float(agent.actions[i][1][0, 0]) for i in (0, 4, 8)], [1., 3., 5.])
        self.assertEqual(len(trainer.evaluation_history), 3)
        for metrics in trainer.evaluation_history:
            self.assertEqual(metrics["gsr"], .5)
            self.assertEqual(metrics["ossr"], .5)
            self.assertEqual(metrics["tsc"], 2.)
            self.assertEqual(metrics["completed_episodes"], 4)
            self.assertEqual(metrics["mean_reward"], 1.)
        self.assertTrue(all(torch.equal(row[2], torch.full((2, 1), 1.25)) for row in agent.records))
        self.assertFalse(env.eval)
        self.assertTrue(agent.training)
        self.assertEqual(float(agent._state_preprocessor.count), 12.)
        self.assertEqual(agent._cumulative_rewards.sum(), 0)
        self.assertEqual(agent._cumulative_timesteps.sum(), 0)
        self.assertEqual(env.episode_sums["reward"].sum(), 0)
        self.assertEqual(env.episode_metric_sums["distance"].sum(), 0)
        for is_eval, actions, _ in env.steps:
            if is_eval:
                torch.testing.assert_close(actions, agent.policy.bias.detach().expand(2, -1))

    def test_resume_uses_completed_steps_and_remaining_budget(self):
        trainer, env, agent = self.make(completed=8)
        trainer.train()
        self.assertEqual([item[0] for item in agent.records], [8, 9, 10, 11])
        self.assertEqual(agent.state.marked, [10, 12])
        self.assertEqual(env.global_step_counter, 112)

    def test_training_without_perception_runtime_is_supported(self):
        trainer, env, agent = self.make(perception=False, interval=0)
        trainer.train()
        self.assertEqual(len(agent.records), 12)
        self.assertEqual(agent.updates, [2, 4, 6, 8, 10, 12])
        self.assertEqual(trainer.evaluation_history, [])
        self.assertEqual(env.global_step_counter, 112)

    def test_full_reset_clears_perception_after_physics_before_observations(self):
        trainer, env, _ = self.make(total=4, interval=4, episode_length=10)
        trainer.train()
        self.assertEqual(env.runtime_resets, [(1, False), (2, True), (3, False)])
        for index, (kind, mode) in enumerate(env.resets):
            if kind == "all":
                self.assertEqual(env.resets[index:index + 3],
                                 [("all", mode), ("runtime", mode), ("partial", mode)])
        for buffer in (env.measurements, env.filter_state, env.history):
            torch.testing.assert_close(buffer, torch.zeros_like(buffer))
        self.assertEqual(float(trainer._resume_states[0, 0]), 3.)

    def test_partial_episode_reset_leaves_other_robot_perception_intact(self):
        trainer, env, _ = self.make()
        trainer._full_reset()
        env._env.progress[:] = torch.tensor([2, 1])
        for buffer in (env.measurements, env.filter_state, env.history):
            buffer.fill_(7)
        env.reset()
        for buffer in (env.measurements, env.filter_state, env.history):
            torch.testing.assert_close(buffer[0], torch.zeros(3))
            torch.testing.assert_close(buffer[1], torch.full((3,), 7.))
        self.assertEqual(len(env.runtime_resets), 1)

    def test_evaluation_preserves_independent_perception_random_stream(self):
        trainer, env, agent = self.make()
        expected_generator = torch.Generator()
        expected_generator.set_state(env.get_runtime_rng_state())
        expected = torch.rand(5, generator=expected_generator)
        trainer.evaluate_policy(4)
        actual = torch.rand(5, generator=env.generator)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(env.runtime_seeds, [trainer.eval_seed])
        self.assertEqual(agent.records, [])

    def test_rejects_partial_training_budget_and_intervals(self):
        trainer, _, _ = self.make(total=11)
        with self.assertRaisesRegex(ValueError, "rollout boundary"):
            trainer.train()
        trainer, _, _ = self.make(completed=1)
        with self.assertRaisesRegex(ValueError, "rollout boundary"):
            trainer.train()
        with self.assertRaisesRegex(ValueError, "eval_interval"):
            self.make(interval=3)

    def test_empty_completed_episodes_do_not_select_best(self):
        trainer, _, agent = self.make(total=4, eval_steps=1, episode_length=10)
        trainer.train()
        self.assertIsNone(trainer.last_evaluation["gsr"])
        self.assertEqual(trainer.last_evaluation["completed_episodes"], 0)
        self.assertFalse(any(name == "best_agent.pt" for name, _ in agent.saved))
        self.assertEqual(agent.state.best_eval_score, -float("inf"))

    def test_eval_restores_statistics_rng_modes_and_scaler_after_failure(self):
        trainer, env, agent = self.make(fail_eval=True)
        agent.policy.train(True)
        agent.models["value"].train(False)
        before = {name: getattr(env, name).clone() for name in (
            "episode_counter", "success_counter", "success_onetime_counter", "episode_step_sum")}
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        with self.assertRaisesRegex(RuntimeError, "evaluation failure"):
            trainer.evaluate_policy(4)
        actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(expected[0:2], actual[0:2])
        torch.testing.assert_close(expected[2], actual[2])
        self.assertEqual(env.global_step_counter, 100)
        self.assertEqual(env.local_step_counter, 10)
        self.assertEqual(env.extras, {"saved": "training"})
        self.assertFalse(env.eval)
        self.assertTrue(agent.policy.training)
        self.assertFalse(agent.models["value"].training)
        self.assertFalse(trainer._is_evaluating)
        self.assertIsNotNone(trainer._resume_states)
        for name, value in before.items():
            torch.testing.assert_close(getattr(env, name), value)
        self.assertEqual(agent.records, [])
        self.assertEqual(env.runtime_resets, [(1, True), (2, False)])
        for buffer in (env.measurements, env.filter_state, env.history):
            torch.testing.assert_close(buffer, torch.zeros_like(buffer))

    def test_periodic_evaluation_callback_and_json_record_success_metrics(self):
        calls = []
        trainer, _, _ = self.make(callback=lambda trainer, metrics, step: calls.append((metrics, step)))
        trainer.train()
        self.assertEqual([step for _, step in calls], [4, 8, 12])
        path = Path(self.temp.name) / "evaluations.jsonl"
        metrics = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual([row["completed_steps"] for row in metrics], [4, 8, 12])
        self.assertTrue(all(row["gsr"] == .5 for row in metrics))
        self.assertEqual([row["best"] for row in metrics], [True, False, False])

    def test_perception_logs_training_separately_and_averages_available_eval_values(self):
        trainer, _, agent = self.make(total=4)
        trainer.train()
        logged_error = [value for name, value in agent.metrics if name == "Perception / prediction_error_m"]
        self.assertEqual(logged_error, [.2, .3, .4])
        initialized = [value for name, value in agent.metrics if name == "Perception / initialized_fraction"]
        self.assertEqual(initialized, [0., 1., 1., 1.])
        row = json.loads((Path(self.temp.name) / "evaluations.jsonl").read_text())
        self.assertAlmostEqual(row["perception"]["prediction_error_m"], .3)
        self.assertEqual(row["perception"]["initialized_fraction"], .75)
        self.assertEqual(row["perception"]["visible_fraction"], .5)
        self.assertEqual(row["perception"]["reward_mean"], .25)
        self.assertEqual(row["mean_reward"], 1.)
        self.assertEqual((row["gsr"], row["ossr"], row["best"]), (.5, .5, True))
        self.assertEqual(len(agent.records), 4)
        evaluation_errors = [value for name, value in agent.metrics
                             if name == "Evaluation perception / prediction_error_m"]
        self.assertEqual(len(evaluation_errors), 1)
        self.assertAlmostEqual(evaluation_errors[0], .3)

    def test_missing_perception_estimates_are_not_logged_as_zero(self):
        trainer, _, agent = self.make(total=2, interval=2, eval_steps=1)
        trainer.train()
        row = json.loads((Path(self.temp.name) / "evaluations.jsonl").read_text())
        self.assertNotIn("prediction_error_m", row["perception"])
        self.assertEqual(row["perception"]["initialized_fraction"], 0.)
        self.assertFalse(any(name == "Evaluation perception / prediction_error_m" for name, _ in agent.metrics))
        self.assertIsNone(row["gsr"])
        self.assertFalse(row["best"])
        plain, _, _ = self.make(perception=False)
        self.assertEqual(plain.evaluate_policy(2)["perception"], {})

    def test_no_evaluation_inside_unfinished_rollout(self):
        trainer, _, agent = self.make()
        agent._rollout = 1
        with self.assertRaisesRegex(ValueError, "complete PPO rollout"):
            trainer.evaluate_policy(4)

    def test_standalone_evaluation_ignores_resume_training_count(self):
        trainer, env, agent = self.make(total=3, completed=8)
        trainer.eval()
        self.assertEqual(len(env.steps), 3)
        self.assertEqual(agent.state.completed_steps, 8)
        self.assertEqual(agent.records, [])
        self.assertEqual(agent.saved, [])
        self.assertEqual(trainer.last_evaluation["vector_steps"], 3)
        self.assertIn("current_initial_pose", trainer.last_evaluation["protocol"])


if __name__ == "__main__":
    unittest.main()
