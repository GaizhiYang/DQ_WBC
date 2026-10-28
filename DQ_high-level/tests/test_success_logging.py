"""CPU regression tests for the TensorBoard metrics synchronized to W&B."""

import sys
import tempfile
import unittest
from pathlib import Path

# Exercise this repository's patched SKRL even if another copy is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "skrl"))

try:
    import torch
    from skrl.agents.torch.base import Agent
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:
    raise unittest.SkipTest("Run these tests in the dqwbc environment: {}".format(exc))


class SuccessLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.agent = Agent(
            models={},
            device="cpu",
            cfg={"experiment": {
                "write_interval": 24,
                "checkpoint_interval": 0,
                "directory": self.directory.name,
                "experiment_name": "success_logging",
                "wandb": False,
            }},
        )
        self.agent.init()
        self.addCleanup(self.agent.writer.close)

    def record(self, timestep, success_rate, rewards):
        values = torch.zeros((2, 1), dtype=torch.float32)
        self.agent.record_transition(
            states=values,
            actions=values,
            rewards=torch.tensor(rewards, dtype=torch.float32).reshape(2, 1),
            next_states=values,
            terminated=torch.zeros((2, 1), dtype=torch.bool),
            truncated=torch.zeros((2, 1), dtype=torch.bool),
            infos={"success_rate": success_rate},
            timestep=timestep,
            timesteps=48,
        )
        self.agent.post_interaction(timestep=timestep, timesteps=48)

    def read_scalars(self):
        self.agent.writer.flush()
        events = EventAccumulator(self.agent.experiment_dir)
        events.Reload()
        return {
            tag: [(event.step, event.value) for event in events.Scalars(tag)]
            for tag in events.Tags()["scalars"]
        }

    def test_write_interval_records_latest_window_and_keeps_reward_aggregation(self):
        for timestep in range(24):
            self.agent.track_data("Loss / policy loss", float(timestep))
            self.record(
                timestep,
                {"SuccessRate / Total": timestep / 1000,
                 "SuccessCount / Total": timestep,
                 "SuccessWindow / Episodes": 1000},
                [timestep + 1, timestep + 3],
            )

        scalars = self.read_scalars()
        self.assertEqual(scalars["SuccessRate / Total"][0][0], 24)
        self.assertAlmostEqual(scalars["SuccessRate / Total"][0][1], .023)
        self.assertEqual(scalars["SuccessCount / Total"], [(24, 23)])
        self.assertEqual(scalars["SuccessWindow / Episodes"], [(24, 1000)])
        self.assertEqual(scalars["Reward / Instantaneous reward (mean)"], [(24, 13.5)])
        self.assertEqual(scalars["Reward / Instantaneous reward (min)"], [(24, 1)])
        self.assertEqual(scalars["Reward / Instantaneous reward (max)"], [(24, 26)])
        self.assertEqual(scalars["Loss / policy loss"], [(24, 11.5)])

    def test_unchanged_window_remains_available_across_write_intervals(self):
        # An interval with no new completed episodes still publishes the
        # current window, including after the logger clears its own buffers.
        for timestep in range(48):
            snapshot = {"SuccessRate / Total": .75, "SuccessCount / Total": 750}
            self.record(timestep, snapshot, [0, 0])

        scalars = self.read_scalars()
        self.assertEqual(scalars["SuccessRate / Total"], [(24, .75), (48, .75)])
        self.assertEqual(scalars["SuccessCount / Total"], [(24, 750), (48, 750)])


if __name__ == "__main__":
    unittest.main()
