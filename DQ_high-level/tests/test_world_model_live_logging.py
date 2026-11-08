"""CPU regressions for live metrics during paused environment collection."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "skrl"))

from test_teacher_live_logging import logged_agent
from test_success_logging import MemoryWriter
from test_world_model_aligned import make_aligned
from test_world_model_fixed8_ppo import make_fixed8_agent
from utils.world_model.live_logging import (
    close_teacher_logging, flush_before_model_update, log_model_snapshot,
)
from utils.world_model.reference_policy import ReferenceTeacherPolicy
from utils.world_model.reference_recipe import REFERENCE_CONFIG
from utils.wm_ppo import PPO
from utils.wm_reference import ReferenceTeacherAgent


def enable_live_logging(agent, events):
    """Keep production Agent writes, replacing only filesystem/network sinks."""
    class OrderedWriter(MemoryWriter):
        def flush(self):
            events.append("flush")
            super().flush()

    agent.cfg["experiment"].update(
        write_interval=24, steps_per_iteration=3, wandb_direct_logging=True,
        world_model_live_logging=True, world_model_log_seconds=10.)
    agent.write_interval = 24
    agent.writer = OrderedWriter()
    agent._wandb_run = Mock()


class LiveLoggingHelperTests(unittest.TestCase):
    def test_first_snapshot_is_immediate_then_throttled_with_force_and_real_axes(self):
        with logged_agent(world_model_live_logging=True) as (agent, wandb):
            agent.wm_replay = SimpleNamespace(steps=24)
            agent.wm_trainer = SimpleNamespace(updates=10)
            agent.track_data("Reward / mean", 2.)
            with patch("utils.world_model.live_logging.time.monotonic", return_value=100.) as clock:
                self.assertTrue(log_model_snapshot(agent, {"loss": 4.}))
                agent.wm_trainer.updates = 11
                clock.return_value = 109.999
                self.assertFalse(log_model_snapshot(agent, {"loss": 3.}))
                clock.return_value = 110.
                self.assertTrue(log_model_snapshot(agent, {"loss": 2.}))
                agent.wm_trainer.updates = 12
                clock.return_value = 110.1
                self.assertTrue(log_model_snapshot(agent, {"loss": 1.}, force=True))
                clock.return_value = 120.099
                self.assertFalse(log_model_snapshot(agent, {"loss": 0.}))
            rows = [item.args[0] for item in wandb.init.return_value.log.call_args_list]
            self.assertEqual([row["world_model_update"] for row in rows], [10, 11, 12])
            self.assertEqual([row["global_step"] for row in rows], [24, 24, 24])
            self.assertEqual([row["World model live / collected_high_steps"] for row in rows], [24, 24, 24])
            self.assertEqual(agent.wm_replay.steps, 24)
            self.assertEqual(agent.wm_trainer.updates, 12)
            self.assertEqual(agent.tracking_data["Reward / mean"], [2.])
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual([(row.step, row.value) for row in events.Scalars("World model live / loss")],
                             [(10, 4.), (11, 2.), (12, 1.)])

    def test_disabled_live_logging_and_probe_are_noops(self):
        for interval, enabled in ((24, False), (0, True)):
            agent = SimpleNamespace(write_interval=interval,
                cfg={"experiment": {"world_model_live_logging": enabled}},
                tracking_data={"reward": [1.]}, write_tracking_data=Mock(),
                write_scalar_snapshot=Mock())
            with self.subTest(interval=interval, enabled=enabled):
                self.assertFalse(log_model_snapshot(agent, {"loss": 1.}, force=True))
                flush_before_model_update(agent, 24)
                agent.write_tracking_data.assert_not_called()
                agent.write_scalar_snapshot.assert_not_called()
                self.assertFalse(hasattr(agent, "_wm_last_metric_write"))

    def test_flush_publishes_pending_window_once_before_any_model_work(self):
        with logged_agent(world_model_live_logging=True) as (agent, wandb):
            agent.track_data("Reward / mean", 2.)
            agent.track_data("Reward / mean", 6.)
            flush_before_model_update(agent, 24)
            wandb.init.return_value.log.assert_called_once_with(
                {"iteration": 1., "Reward / mean": 4., "global_step": 24}, commit=True)
            self.assertFalse(agent.tracking_data)
            flush_before_model_update(agent, 24)
            self.assertEqual(wandb.init.return_value.log.call_count, 1)
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual(events.Scalars("Reward / mean")[0].value, 4.)

    def test_close_finishes_with_exit_code_after_writer(self):
        events = []
        agent = SimpleNamespace(writer=Mock(), _wandb_run=Mock())
        agent.writer.close.side_effect = lambda: events.append("close")
        agent._wandb_run.finish.side_effect = lambda **kwargs: events.append(("finish", kwargs))
        close_teacher_logging(agent, exit_code=130)
        self.assertEqual(events, ["close", ("finish", {"exit_code": 130})])
        close_teacher_logging(SimpleNamespace())

    def test_failed_writer_close_warns_and_still_finishes(self):
        agent = SimpleNamespace(writer=Mock(), _wandb_run=Mock())
        agent.writer.close.side_effect = RuntimeError("disk unavailable")
        with self.assertWarnsRegex(UserWarning, "Could not close teacher TensorBoard writer"):
            close_teacher_logging(agent, exit_code=1)
        agent._wandb_run.finish.assert_called_once_with(exit_code=1)
        agent._wandb_run.finish.side_effect = RuntimeError("upload unavailable")
        agent.writer.close.side_effect = None
        with self.assertWarnsRegex(UserWarning, "Could not finish teacher W&B upload"):
            close_teacher_logging(agent, exit_code=1)


class ModelBlockLoggingTests(unittest.TestCase):
    def test_supermimic_flushes_before_fit_and_forces_first_last_round(self):
        agent = make_aligned("cpu", mode="fit")
        events = []
        enable_live_logging(agent, events)
        agent.wm_replay.steps = 3
        agent.wm_low_replay.steps = 24  # .125 updates per low step => three rounds.
        agent.track_data("Reward / mean", 2.)
        # Even a recently published snapshot must not hide either endpoint.
        agent._wm_last_metric_write = 100.

        def ppo_update(current, timestep, timesteps):
            events.append("ppo")
            current.track_data("Loss / Policy loss", 5.)

        def fit(replay):
            self.assertIs(replay, agent.wm_low_replay)
            events.append("model")
            agent.wm_trainer.updates += 1
            return {"state_loss": float(4-agent.wm_trainer.updates)}

        with patch.object(PPO, "_update", new=ppo_update), \
                patch.object(agent.wm_low_replay, "can_sample", return_value=True), \
                patch.object(agent.wm_trainer, "update", side_effect=fit) as fitting, \
                patch("utils.world_model.live_logging.time.monotonic", return_value=100.):
            agent._update(2, 100)

        self.assertEqual(events[:3], ["ppo", "flush", "model"])
        self.assertEqual(fitting.call_count, 3)
        rows = [item.args[0] for item in agent._wandb_run.log.call_args_list]
        self.assertEqual(len(rows), 3)  # reward/PPO, first round, final round.
        self.assertEqual(rows[0]["Reward / mean"], 2.)
        self.assertEqual(rows[0]["Loss / Policy loss"], 5.)
        self.assertEqual([row["World model live / round_in_block"] for row in rows[1:]], [1., 3.])
        self.assertEqual([row["world_model_update"] for row in rows[1:]], [1, 3])
        self.assertEqual([row["global_step"] for row in rows], [3, 3, 3])
        self.assertEqual(agent.wm_replay.steps, 3)
        self.assertIn("World model / candidate/state_loss", agent.tracking_data)

    def test_reference_flushes_rewards_and_publishes_model_progress_during_fit(self):
        env = make_fixed8_agent("cpu").wm_env
        config = {**REFERENCE_CONFIG, "mode": "fit", "hidden": [8], "contact_hidden": [8],
            "amp": "none", "policy_amp": "none", "horizon": 2, "window": 2,
            "batch_size": 4, "model_microbatch_size": 4, "imagination_batch": 4,
            "imagination_chunk_size": 2, "critic_hidden": [8], "critic_epochs": 1,
            "critic_minibatches": 1, "replay_envs": 12, "replay_capacity": 8,
            "low_replay_capacity": 32, "validation_batch_size": 4,
            "collection_warmup_high_steps": 1, "world_model_only_iterations": 0,
            "wm_max_batches": 3, "diagnostics_interval": 0}
        policy = ReferenceTeacherPolicy(1284, 9, "cpu", hidden=(8,), amp="none")
        agent = ReferenceTeacherAgent(models={"policy": policy}, memory=None,
            observation_space=1284, action_space=9, device="cpu", wm_env=env, wm_config=config,
            cfg={"experiment": {"write_interval": 0, "checkpoint_interval": 0}})
        events = []
        enable_live_logging(agent, events)
        agent.wm_replay.steps = agent._wm_total_high_steps = 3
        agent._wm_total_low_steps = 24
        agent.track_data("Reward / mean", 2.)
        plan = {"iteration": 0, "model_batches": 3, "actor": False,
                "new_valid_low": 12, "discarded_batches": 0}

        def fit(replay, *, batches, progress_callback):
            events.append("model")
            self.assertEqual(agent._wandb_run.log.call_count, 1)
            for done in range(1, batches + 1):
                agent.wm_trainer.updates += 1
                progress_callback(done, batches)
                if done == 1:
                    # The callback publishes while fit is still running.
                    self.assertEqual(agent._wandb_run.log.call_count, 2)
            return {"state_loss": 1., "updates": float(agent.wm_trainer.updates)}

        with patch.object(agent.reference_schedule, "plan", return_value=plan), \
                patch.object(agent.wm_trainer, "update", side_effect=fit), \
                patch("utils.world_model.live_logging.time.monotonic", return_value=100.):
            agent._update_reference()

        self.assertEqual(events[:2], ["flush", "model"])
        rows = [item.args[0] for item in agent._wandb_run.log.call_args_list]
        progress = rows[1]
        self.assertEqual(progress["World model live / fit/batches_completed"], 1.)
        self.assertEqual(progress["world_model_update"], 1)
        self.assertEqual(len(rows), 3)  # final aggregate bypasses the ten-second throttle.
        self.assertEqual(rows[-1]["World model live / candidate/state_loss"], 1.)
        self.assertEqual(rows[-1]["world_model_update"], 3)
        self.assertTrue(all(row["global_step"] == 3 for row in rows))
        self.assertEqual(agent.wm_replay.steps, 3)
        self.assertEqual(agent.wm_trainer.updates, 3)
        self.assertIn("World model / candidate/state_loss", agent.tracking_data)


if __name__ == "__main__":
    unittest.main()
