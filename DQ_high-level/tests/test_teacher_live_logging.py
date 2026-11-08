"""CPU coverage of committed teacher scalars using real local TensorBoard files.

W&B is replaced before Agent.init, so these tests never authenticate or upload.
"""

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, call, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "skrl"))
from skrl.agents.torch import Agent


@contextmanager
def logged_agent(**overrides):
    with tempfile.TemporaryDirectory() as directory:
        experiment = {
            "directory": directory, "experiment_name": "teacher",
            "write_interval": 24, "checkpoint_interval": 0,
            "steps_per_iteration": 24, "wandb": True,
            "wandb_direct_logging": True,
            # Direct mode must override even an explicit sync request.
            "wandb_kwargs": {"project": "test-only", "sync_tensorboard": True},
        }
        experiment.update(overrides)
        fake_wandb = Mock()
        agent = Agent({}, device="cpu", cfg={"experiment": experiment})
        with patch.dict("sys.modules", {"wandb": fake_wandb}):
            try:
                agent.init()
                yield agent, fake_wandb
            finally:
                if hasattr(agent, "writer"):
                    agent.writer.close()


class TeacherLiveLoggingTests(unittest.TestCase):
    def test_first_window_commits_one_complete_row_and_flushes_tensorboard(self):
        with logged_agent() as (agent, wandb):
            for value in (1., 5.):
                agent.track_data("Reward / reward (mean)", value)
                agent.track_data("Reward / reward (min)", value)
                agent.track_data("Reward / reward (max)", value)
            agent.write_tracking_data(24, 10000)
            expected = {
                "iteration": 1., "global_step": 24,
                "Reward / reward (mean)": 3.,
                "Reward / reward (min)": 1.,
                "Reward / reward (max)": 5.,
            }
            # A single window is committed immediately, with no explicit W&B
            # step and no second TensorBoard step needed to publish it.
            wandb.init.return_value.log.assert_called_once_with(expected, commit=True)
            self.assertFalse(wandb.init.call_args.kwargs["sync_tensorboard"])
            self.assertFalse(agent.tracking_data)
            events = EventAccumulator(agent.experiment_dir).Reload()
            for tag, value in expected.items():
                self.assertEqual([(row.step, row.value) for row in events.Scalars(tag)], [(24, value)])

    def test_direct_axes_separate_model_updates_from_environment_progress(self):
        with logged_agent() as (_, wandb):
            self.assertEqual(wandb.define_metric.call_args_list, [
                call("global_step"),
                call("world_model_update"),
                call("*", step_metric="global_step", step_sync=True),
                call("World model live / *", step_metric="world_model_update", step_sync=True),
                call("iteration"),
            ])

    def test_snapshots_advance_during_fixed_environment_step_without_consuming_window(self):
        with logged_agent() as (agent, wandb):
            agent.track_data("Reward / reward (mean)", 2.)
            agent.track_data("Reward / reward (mean)", 6.)
            agent._track_rewards.append(4.)
            agent._track_timesteps.append(8)
            before = deepcopy(agent.tracking_data)
            metrics = {"world_model_update": 1, "World model live / loss": 7.}
            agent.write_scalar_snapshot(metrics, 1, global_step=24)
            self.assertEqual(metrics, {"world_model_update": 1, "World model live / loss": 7.})
            self.assertEqual(agent.tracking_data, before)
            self.assertEqual(list(agent._track_rewards), [4.])
            self.assertEqual(list(agent._track_timesteps), [8])
            first_events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual([(row.step, row.value) for row in first_events.Scalars("global_step")], [(1, 24.)])
            wandb.init.return_value.log.assert_called_once_with({**metrics, "global_step": 24}, commit=True)

            agent.write_scalar_snapshot({"world_model_update": 2, "World model live / loss": 5.},
                                        2, global_step=24)
            agent.write_tracking_data(24, 10000)
            calls = wandb.init.return_value.log.call_args_list
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(item.kwargs == {"commit": True} for item in calls))
            self.assertEqual([item.args[0]["global_step"] for item in calls], [24, 24, 24])
            self.assertEqual(calls[-1].args[0]["Reward / reward (mean)"], 4.)
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual([(row.step, row.value) for row in events.Scalars("World model live / loss")],
                             [(1, 7.), (2, 5.)])

    def test_no_wandb_keeps_immediate_local_scalars(self):
        with logged_agent(wandb=False) as (agent, wandb):
            agent.write_scalar_snapshot({"world_model_update": 1, "World model live / loss": 3.},
                                        1, global_step=24)
            wandb.init.assert_not_called()
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual(events.Scalars("World model live / loss")[0].value, 3.)

    def test_disabled_logging_creates_no_writer_or_wandb_run(self):
        with patch("skrl.agents.torch.base.SummaryWriter") as writer:
            with logged_agent(write_interval=0) as (agent, wandb):
                agent.track_data("Reward / reward (mean)", 2.)
                agent.write_scalar_snapshot({"World model live / loss": 3.}, 1)
                agent.write_tracking_data(24, 10000)
                self.assertFalse(hasattr(agent, "writer"))
                self.assertFalse(Path(agent.experiment_dir).exists())
                writer.assert_not_called()
                wandb.init.assert_not_called()

    def test_legacy_tensorboard_sync_and_iteration_axis_remain_unchanged(self):
        with logged_agent(wandb_direct_logging=False) as (agent, wandb):
            self.assertTrue(wandb.init.call_args.kwargs["sync_tensorboard"])
            self.assertEqual(wandb.define_metric.call_args_list,
                             [call("iteration"), call("*", step_metric="iteration", step_sync=True)])
            agent.track_data("Reward / reward (mean)", 2.)
            with patch.object(agent.writer, "flush", wraps=agent.writer.flush) as flush:
                agent.write_tracking_data(24, 10000)
                flush.assert_not_called()
            wandb.init.return_value.log.assert_not_called()
            agent.writer.flush()
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual(set(events.Tags()["scalars"]), {"iteration", "Reward / reward (mean)"})
            self.assertEqual(events.Scalars("iteration")[0].value, 1.)


if __name__ == "__main__":
    unittest.main()
