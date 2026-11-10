"""SuperMimic logging windows without Isaac Gym, W&B, or a simulator."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch


_spec = importlib.util.spec_from_file_location(
    "supermimic_logging", Path(__file__).resolve().parents[1] / "supermimic/logging.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
SuperMimicMetrics = _module.SuperMimicMetrics

NATIVE = "Reward / Instantaneous reward (mean)"
PROXY = "Reward / Geometric proxy reward (mean)"
RATE = "SuccessRate / full_task"


def component(name):
    return f"Reward components / {name} (mean)"


class MetricsChecks:
    device = "cpu"

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    def test_window_averages_native_proxy_and_weighted_components(self):
        metrics = SuperMimicMetrics(self.device)
        for step in range(1, 4):
            metrics.record_step(self.tensor([step, 2 * step]), self.tensor([10 * step, 20 * step]), {
                "reward_components": {"approaching": self.tensor(2 * step),
                                      "lifting": self.tensor(.5 * step),
                                      "termination": self.tensor(-step)},
            })
        self.assertEqual(metrics.pending_steps, 3)
        logged = metrics.flush()
        self.assertEqual(logged[NATIVE], 3.)
        self.assertEqual(logged[PROXY], 30.)
        self.assertEqual(logged[component("approaching")], 4.)
        self.assertEqual(logged[component("lifting")], 1.)
        self.assertEqual(logged[component("termination")], -2.)
        self.assertEqual(sum(logged[component(name)] for name in ("approaching", "lifting", "termination")),
                         logged[NATIVE])
        self.assertEqual(metrics.pending_steps, 0)
        self.assertTrue(all(isinstance(value, (int, float)) for value in logged.values()))

    def test_vector_step_means_have_equal_weight(self):
        metrics = SuperMimicMetrics(self.device)
        metrics.record_step(self.tensor([0., 2.]), self.tensor([2., 4.]), {})
        metrics.record_step(self.tensor([9.]), self.tensor([5.]), {})
        logged = metrics.flush()
        self.assertEqual(logged[NATIVE], 5.)
        self.assertEqual(logged[PROXY], 4.)

    def test_episodes_from_all_steps_have_equal_weight_and_population_std(self):
        metrics = SuperMimicMetrics(self.device)
        for step in range(4):
            metrics.record_step(self.tensor([1., 2.]), self.tensor([0., 0.]), {})
            if step == 0:
                metrics.record_episodes(self.tensor([[1.], [3.]]), self.tensor([[2], [4]]))
            if step == 2:
                metrics.record_episodes(self.tensor([-2.]), self.tensor([9]))
        logged = metrics.flush()
        expected = self.tensor([1., 3., -2.])
        self.assertEqual(logged["episode_count"], 3)
        self.assertAlmostEqual(logged["episode_reward_mean"], expected.mean().item())
        self.assertAlmostEqual(logged["episode_reward_std"], expected.std(unbiased=False).item())
        self.assertEqual(logged["episode_length_mean"], 5.)
        self.assertEqual(logged["Reward / Total reward (mean)"], logged["episode_reward_mean"])
        self.assertEqual(logged["Episode / Total timesteps (mean)"], logged["episode_length_mean"])

    def test_no_episodes_emits_count_without_fabricating_episode_rewards(self):
        metrics = SuperMimicMetrics(self.device)
        metrics.record_step(self.tensor([1.]), self.tensor([2.]), {})
        metrics.record_episodes(self.tensor([]), self.tensor([]))
        logged = metrics.flush()
        self.assertEqual(logged["episode_count"], 0)
        for name in ("episode_reward_mean", "episode_reward_std", "episode_length_mean",
                     "Reward / Total reward (mean)", "Episode / Total timesteps (mean)"):
            self.assertNotIn(name, logged)

    def test_single_completed_episode_without_step_can_be_flushed(self):
        metrics = SuperMimicMetrics(self.device)
        metrics.record_episodes(self.tensor([5.]), self.tensor([10]))
        logged = metrics.flush()
        self.assertEqual(logged["episode_count"], 1)
        self.assertEqual(logged["episode_reward_std"], 0.)
        self.assertNotIn(NATIVE, logged)
        self.assertEqual(metrics.flush(), {})

    def test_snapshots_are_latest_and_scalar_snapshots_take_precedence(self):
        metrics = SuperMimicMetrics(self.device)
        for rate, count in ((1., 1), (.25, 4)):
            metrics.record_step(self.tensor([2.]), self.tensor([3.]), {
                "success_rate": {RATE: .8, "SuccessWindow / completed": count},
                "scalar_snapshots": {RATE: self.tensor(rate), "SGS / episodes_full_task": count},
            })
        logged = metrics.flush()
        self.assertEqual(logged[RATE], .25)
        self.assertEqual(logged["SuccessWindow / completed"], 4)
        self.assertEqual(logged["SGS / episodes_full_task"], 4)

    def test_snapshot_groups_replace_removed_keys_without_inventing_unknown_rates(self):
        metrics = SuperMimicMetrics(self.device)
        metrics.record_step(self.tensor([1.]), self.tensor([1.]), {
            "success_rate": {RATE: .5, "SuccessRate / near_goal": 1.},
            "scalar_snapshots": {"SGS / episodes_near_goal": 3},
        })
        metrics.update_snapshots({
            "success_rate": {RATE: .75},
            "scalar_snapshots": {"SGS / episodes_near_goal": 0},
        })
        logged = metrics.flush()
        self.assertEqual(logged[RATE], .75)
        self.assertNotIn("SuccessRate / near_goal", logged)
        self.assertNotIn("SuccessRate / pregrasp", logged)
        self.assertEqual(logged["SGS / episodes_near_goal"], 0)
        metrics.record_step(self.tensor([2.]), self.tensor([2.]), {
            "success_rate": {}, "scalar_snapshots": {},
        })
        logged = metrics.flush()
        self.assertNotIn(RATE, logged)
        self.assertNotIn("SGS / episodes_near_goal", logged)

    def test_retained_snapshots_do_not_create_empty_window_records(self):
        metrics = SuperMimicMetrics(self.device)
        self.assertEqual(metrics.flush(), {})
        metrics.update_snapshots({"scalar_snapshots": {RATE: .5}})
        self.assertEqual(metrics.flush(), {})
        metrics.record_step(self.tensor([1.]), self.tensor([2.]), {})
        self.assertEqual(metrics.flush()[RATE], .5)
        self.assertEqual(metrics.flush(), {})
        metrics.update_snapshots({"scalar_snapshots": {RATE: .5}})
        self.assertEqual(metrics.flush(), {})
        metrics.record_step(self.tensor([3.]), self.tensor([4.]), {})
        self.assertEqual(metrics.flush()[RATE], .5)

    def test_partial_final_window_is_flushed_once_and_does_not_reuse_old_rewards(self):
        metrics = SuperMimicMetrics(self.device)
        for value in (1., 2., 3.):
            metrics.record_step(self.tensor([value]), self.tensor([2 * value]), {})
        self.assertEqual(metrics.flush()[NATIVE], 2.)
        metrics.record_step(self.tensor([10.]), self.tensor([20.]), {})
        metrics.update_snapshots({"scalar_snapshots": {RATE: 1.}})
        final = metrics.flush()
        self.assertEqual(final[NATIVE], 10.)
        self.assertEqual(final[PROXY], 20.)
        self.assertEqual(final[RATE], 1.)
        self.assertEqual(metrics.pending_steps, 0)
        self.assertEqual(metrics.flush(), {})

    def test_mutating_environment_buffers_after_recording_does_not_change_logs(self):
        metrics = SuperMimicMetrics(self.device)
        reward = self.tensor([1., 3.]).requires_grad_()
        proxy = self.tensor([4., 6.]).requires_grad_()
        term = self.tensor(2.).requires_grad_()
        rate = self.tensor(.5).requires_grad_()
        returns = self.tensor([7., 9.]).requires_grad_()
        lengths = self.tensor([2., 4.])
        infos = {"reward_components": {"approaching": term}, "scalar_snapshots": {RATE: rate}}
        metrics.record_step(reward, proxy, infos)
        metrics.record_episodes(returns, lengths)
        with torch.no_grad():
            for value in (reward, proxy, term, rate, returns, lengths):
                value.fill_(99.)
        infos["scalar_snapshots"].clear()
        infos["reward_components"].clear()
        logged = metrics.flush()
        self.assertEqual(logged[NATIVE], 2.)
        self.assertEqual(logged[PROXY], 5.)
        self.assertEqual(logged[component("approaching")], 2.)
        self.assertEqual(logged[RATE], .5)
        self.assertEqual(logged["episode_reward_mean"], 8.)
        self.assertEqual(logged["episode_length_mean"], 3.)

    def test_recording_never_reads_tensor_scalars_back_to_cpu(self):
        metrics = SuperMimicMetrics(self.device)
        reward, proxy, term, rate = (self.tensor(value) for value in ([1., 3.], [2., 4.], 2., .5))
        returns, lengths = self.tensor([7.]), self.tensor([3.])
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("per-step item")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("per-step cpu")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("per-step tolist")):
            metrics.record_step(reward, proxy, {
                "reward_components": {"approaching": term}, "scalar_snapshots": {RATE: rate},
            })
            metrics.record_episodes(returns, lengths)
        self.assertEqual(metrics.flush()[NATIVE], 2.)

    def test_large_snapshot_counts_and_component_windows_preserve_values(self):
        metrics = SuperMimicMetrics(self.device)
        count = 2 ** 40 + 1
        metrics.record_step(self.tensor([1.]), self.tensor([2.]), {
            "reward_components": {"approaching": self.tensor(3.)},
            "scalar_snapshots": {"SuccessWindow / completed": count},
        })
        self.assertEqual(metrics.flush()["SuccessWindow / completed"], count)
        metrics.record_step(self.tensor([4.]), self.tensor([5.]), {
            "reward_components": {"termination": self.tensor(-1.)},
        })
        logged = metrics.flush()
        self.assertNotIn(component("approaching"), logged)
        self.assertEqual(logged[component("termination")], -1.)


class CPUMetricsTests(MetricsChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAMetricsTests(MetricsChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
