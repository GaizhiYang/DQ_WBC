"""CPU regression tests for the real SKRL logging/trainer path; no Isaac Gym."""

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "skrl"))
from skrl.agents.torch import Agent
from skrl.trainers.torch import SequentialTrainer


RATE = "SuccessRate / sugar_box"
COUNT = "SuccessWindowEpisodes / sugar_box"


class MemoryWriter:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))

    def flush(self):
        self.flush_count += 1

    def values(self, tag):
        return [(value, step) for key, value, step in self.scalars if key == tag]


class LoggingAgent(Agent):
    """Keep real recording/post-interaction logic; avoid models and disk I/O."""

    def init(self, trainer_cfg=None):
        self.writer = MemoryWriter()

    def act(self, states, timestep, timesteps):
        return torch.zeros_like(states), None, None


class CompletedEpisodeEnv:
    """Supply scalar snapshots on the step that completes each scheduled episode."""

    num_envs = 1
    num_agents = 1
    max_episode_length = 26

    def __init__(self, events, initial_step=0):
        self.events = events
        self.step_number = initial_step
        self.outcomes = []

    def reset(self):
        return torch.zeros(1, 1), {}

    def step(self, actions):
        self.step_number += 1
        done = self.step_number in self.events
        if done:
            self.outcomes.append(self.events[self.step_number])
        count = len(self.outcomes)
        infos = {"success_rate": {
            RATE: sum(self.outcomes) / max(count, 1), COUNT: count,
        }}
        return (torch.zeros(1, 1), torch.tensor([[float(self.step_number)]]),
                torch.tensor([[done]]), torch.tensor([[False]]), infos)

    def render_record(self, mode):
        return None


def make_trainer(events, timesteps=24, write_interval=24, initial_step=0, **cfg):
    env = CompletedEpisodeEnv(events, initial_step=initial_step)
    agent = LoggingAgent({}, device="cpu", cfg={"experiment": {
        "write_interval": write_interval, "checkpoint_interval": 0,
    }})
    trainer_cfg = {
        "timesteps": timesteps, "headless": True, "disable_progressbar": True,
        "close_environment_at_exit": False, "initial_timestep": initial_step,
    }
    trainer_cfg.update(cfg)
    return SequentialTrainer(env=env, agents=agent, cfg=trainer_cfg), agent


class SuccessLoggingTests(unittest.TestCase):
    def test_latest_rate_and_count_preserve_reward_aggregation(self):
        trainer, agent = make_trainer({10: False, 20: True})
        trainer.train()
        self.assertEqual(agent.writer.values(RATE), [(0.5, 24)])
        self.assertEqual(agent.writer.values(COUNT), [(2., 24)])
        self.assertEqual(agent.writer.values("Reward / Instantaneous reward (mean)"), [(12.5, 24)])
        self.assertEqual(agent.writer.values("Reward / Instantaneous reward (min)"), [(1., 24)])
        self.assertEqual(agent.writer.values("Reward / Instantaneous reward (max)"), [(24., 24)])

    def test_final_step_success_on_logging_boundary_is_written_once(self):
        trainer, agent = make_trainer({24: True})
        trainer.train()
        self.assertEqual(agent.writer.values(RATE), [(1., 24)])
        self.assertEqual(agent.writer.values(COUNT), [(1., 24)])
        self.assertEqual(agent.writer.flush_count, 1)

    def test_partial_final_interval_includes_final_step_success(self):
        trainer, agent = make_trainer({10: False, 25: True}, timesteps=25)
        trainer.train()
        self.assertEqual(agent.writer.values(RATE), [(0., 24), (0.5, 25)])
        self.assertEqual(agent.writer.values(COUNT), [(1., 24), (2., 25)])
        self.assertEqual(agent.writer.flush_count, 1)

    def test_one_step_run_still_writes_final_snapshot(self):
        trainer, agent = make_trainer({1: True}, timesteps=1)
        trainer.train()
        self.assertEqual(agent.writer.values(RATE), [(1., 1)])

    def test_disabled_logging_remains_disabled(self):
        trainer, agent = make_trainer({1: True}, timesteps=1, write_interval=0)
        trainer.train()
        self.assertEqual(agent.writer.scalars, [])
        self.assertEqual(agent.writer.flush_count, 0)

    def test_resume_uses_absolute_step_for_final_write(self):
        trainer, agent = make_trainer({25: True}, timesteps=25, initial_step=23)
        trainer.train()
        self.assertEqual(agent.writer.values(RATE), [(0., 24), (1., 25)])

    def test_expired_category_is_removed_without_removing_reward_data(self):
        _, agent = make_trainer({})
        zeros = torch.zeros(1, 1)
        flags = torch.zeros(1, 1, dtype=torch.bool)
        for step, snapshot in enumerate(({RATE: 1., COUNT: 1.}, {"SuccessRate / other": 0.})):
            agent.record_transition(zeros, zeros, torch.ones(1, 1), zeros,
                                    flags, flags, {"success_rate": snapshot}, step, 2)
            agent.post_interaction(step, 2)
        self.assertEqual(agent.writer.values(RATE), [])
        self.assertEqual(agent.writer.values(COUNT), [])
        self.assertEqual(agent.writer.values("SuccessRate / other"), [(0., 2)])
        self.assertEqual(agent.writer.values("Reward / Instantaneous reward (mean)"), [(1., 2)])

    def test_eval_flushes_at_actual_video_length(self):
        trainer, agent = make_trainer({26: True}, timesteps=120000,
                                      record_video=True, video_name="test", log_dir="unused")
        # Exercise the real short evaluation loop without creating media/files.
        with mock.patch("imageio.get_writer") as get_writer, mock.patch("os.path.exists", return_value=True):
            trainer.eval()
        self.assertEqual(agent.writer.values(RATE), [(0., 24), (1., 26)])
        self.assertEqual(agent.writer.flush_count, 1)
        get_writer.return_value.close.assert_called_once_with()

    def test_eval_flushes_at_actual_nonvideo_length(self):
        trainer, agent = make_trainer({50000: True}, timesteps=120000, initial_step=49999)
        # Skip the prefix of the fixed 50000-step evaluation; execute its real
        # final iteration with a final-step success and the actual endpoint.
        with mock.patch("skrl.trainers.torch.base.range", create=True,
                        side_effect=lambda start, stop: range(stop - 1, stop)) as loop_range:
            trainer.eval()
        loop_range.assert_called_once_with(0, 50000)
        self.assertEqual(agent.writer.values(RATE), [(1., 50000)])
        self.assertEqual(agent.writer.flush_count, 1)


if __name__ == "__main__":
    unittest.main()
