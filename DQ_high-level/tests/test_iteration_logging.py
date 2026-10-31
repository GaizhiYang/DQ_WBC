"""Iteration-axis regressions through the real SKRL scalar writer and trainer."""

import unittest
from unittest.mock import Mock, call, patch

from test_success_logging import Agent, MemoryWriter, make_trainer


class IterationLoggingTests(unittest.TestCase):
    def test_iteration_is_written_first_without_changing_rewards_or_environment_steps(self):
        for timesteps, expected_axis, expected_rewards in (
                (480, [(10., 240), (20., 480)], [(120.5, 240), (360.5, 480)]),
                (241, [(10., 240), (241. / 24., 241)], [(120.5, 240), (241., 241)])):
            with self.subTest(timesteps=timesteps):
                trainer, agent = make_trainer({}, timesteps=timesteps, write_interval=240)
                agent.cfg["experiment"]["steps_per_iteration"] = 24
                trainer.train()
                self.assertEqual(agent.writer.values("iteration"), expected_axis)
                self.assertEqual(agent.writer.values("Reward / Instantaneous reward (mean)"),
                                 expected_rewards)
                for iteration, step in expected_axis:
                    records = [record for record in agent.writer.scalars if record[2] == step]
                    # TensorBoard synchronization sends scalar events separately;
                    # step_sync must see this window's axis before its metrics.
                    self.assertEqual(records[0], ("iteration", iteration, step))
                    self.assertEqual(sum(tag == "iteration" for tag, _, _ in records), 1)
                self.assertEqual(agent.write_interval, 240)
                self.assertEqual(agent.writer.flush_count, 1)

    def test_iteration_axis_is_opt_in(self):
        for steps_per_iteration in (None, 0):
            with self.subTest(steps_per_iteration=steps_per_iteration):
                trainer, agent = make_trainer({}, timesteps=241, write_interval=240)
                if steps_per_iteration is not None:
                    agent.cfg["experiment"]["steps_per_iteration"] = steps_per_iteration
                trainer.train()
                self.assertEqual(agent.writer.values("iteration"), [])
                self.assertEqual(agent.writer.values("Reward / Instantaneous reward (mean)"),
                                 [(120.5, 240), (241., 241)])

    def test_wandb_initialization_defines_iteration_axis_only_when_configured(self):
        for steps_per_iteration in (None, 0, 24):
            with self.subTest(steps_per_iteration=steps_per_iteration):
                config = {"write_interval": 240, "checkpoint_interval": 0, "wandb": True,
                          "wandb_kwargs": {"project": "test-only"}}
                if steps_per_iteration is not None:
                    config["steps_per_iteration"] = steps_per_iteration
                agent = Agent({}, device="cpu", cfg={"experiment": config})
                fake_wandb = Mock()
                with patch.dict("sys.modules", {"wandb": fake_wandb}), \
                        patch("skrl.agents.torch.base.SummaryWriter", return_value=MemoryWriter()):
                    agent.init()
                fake_wandb.init.assert_called_once()
                self.assertTrue(fake_wandb.init.call_args.kwargs["sync_tensorboard"])
                expected = ([call("iteration"), call("*", step_metric="iteration", step_sync=True)]
                            if steps_per_iteration else [])
                self.assertEqual(fake_wandb.define_metric.call_args_list, expected)


if __name__ == "__main__":
    unittest.main()
