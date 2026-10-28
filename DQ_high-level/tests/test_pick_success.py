"""CPU regression checks for lift eligibility and consecutive grasp completion."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.pick_success import advance_pick_hold, lifted_grasp_mask, RollingEpisodeSuccess


class RollingSuccessTests(unittest.TestCase):
    def test_empty_window_does_not_report_an_undefined_rate(self):
        metrics = RollingEpisodeSuccess().metrics(["sugar_box"])
        self.assertEqual(metrics, {
            "SuccessWindow / completed": 0,
            "SuccessWindow / successes": 0,
        })

    def test_warmup_uses_completed_episodes_and_snapshots_are_independent(self):
        window = RollingEpisodeSuccess()
        window.extend([(0, False), (0, True)])
        before = window.metrics(["sugar_box"])
        self.assertEqual(before["SuccessRate / sugar_box"], .5)
        self.assertEqual(before["SuccessWindow / completed"], 2)
        window.extend([(0, True)])
        self.assertEqual(window.metrics(["sugar_box"])["SuccessRate / sugar_box"], 2 / 3)
        self.assertEqual(before["SuccessRate / sugar_box"], .5)

    def test_old_failures_leave_the_window_as_new_successes_arrive(self):
        window = RollingEpisodeSuccess()
        window.extend([(0, False)] * 1000)
        window.extend([(0, True)] * 250)
        metrics = window.metrics(["sugar_box"])
        self.assertEqual(metrics["SuccessWindow / completed"], 1000)
        self.assertEqual(metrics["SuccessWindow / successes"], 250)
        self.assertEqual(metrics["SuccessRate / sugar_box"], .25)
        window.extend([(0, True)] * 750)
        self.assertEqual(window.metrics(["sugar_box"])["SuccessRate / sugar_box"], 1.)

    def test_batch_larger_than_window_keeps_only_its_last_thousand(self):
        window = RollingEpisodeSuccess()
        window.extend([(0, True)] * 1000)
        window.extend([(0, True)] * 5 + [(0, False)] * 1000)
        metrics = window.metrics(["sugar_box"])
        self.assertEqual(metrics["SuccessWindow / completed"], 1000)
        self.assertEqual(metrics["SuccessWindow / successes"], 0)
        self.assertEqual(metrics["SuccessRate / sugar_box"], 0.)

    def test_object_breakdowns_share_global_window_and_expire(self):
        window = RollingEpisodeSuccess()
        window.extend([(0, True)] * 500 + [(1, False)] * 500)
        metrics = window.metrics(["sugar_box", "mug"])
        self.assertEqual(metrics["SuccessRate / total"], .5)
        self.assertEqual(metrics["SuccessRate / sugar_box"], 1.)
        self.assertEqual(metrics["SuccessRate / mug"], 0.)
        window.extend([(1, True)] * 500)
        metrics = window.metrics(["sugar_box", "mug"])
        self.assertNotIn("SuccessRate / sugar_box", metrics)
        self.assertEqual(metrics["SuccessWindow / sugar_box / completed"], 0)
        self.assertEqual(metrics["SuccessRate / mug"], .5)
        self.assertEqual(metrics["SuccessRate / total"], .5)


class LiftedGraspTests(unittest.TestCase):
    def test_sugar_box_old_reward_window_is_below_success_height(self):
        # A sugar box at rest has its center 4.5 cm above the tabletop.
        # These positions represent net lifts of 17, 18, 19.9, and 20.1 cm.
        mask = lifted_grasp_mask(
            torch.tensor([.715, .725, .744, .746]),
            torch.full((4,), .5),
            torch.full((4,), .045),
            torch.full((4,), .05),
            torch.full((4,), -1.),
            .20,
        )
        self.assertEqual(mask.tolist(), [False, False, False, True])
        self.assertEqual(mask.dtype, torch.bool)

    def test_exactly_twenty_centimeters_qualifies(self):
        table_height = torch.tensor([.5, .5])
        rest_height = torch.tensor([.045, .045])
        target = table_height + rest_height + .20
        object_height = target.clone()
        object_height[0] = torch.nextafter(target[0], torch.tensor(-torch.inf))
        mask = lifted_grasp_mask(
            object_height, table_height, rest_height,
            torch.tensor([.05, .05]), torch.tensor([-1., -1.]), .20,
        )
        self.assertEqual(mask.tolist(), [False, True])

    def test_table_and_object_rest_heights_are_per_environment(self):
        # First two centers have risen 21 cm despite different table/object sizes;
        # the third large object has risen only 10 cm.
        mask = lifted_grasp_mask(
            torch.tensor([.755, 1.11, 1.]),
            torch.tensor([.5, .8, .8]),
            torch.tensor([.045, .1, .1]),
            torch.full((3,), .05), torch.full((3,), -1.), .20,
        )
        self.assertEqual(mask.tolist(), [True, True, False])

    def test_open_gripper_and_distant_end_effector_do_not_qualify(self):
        mask = lifted_grasp_mask(
            torch.full((5,), .8), torch.full((5,), .5),
            torch.full((5,), .045),
            torch.tensor([.05, .05, .05, .2, .21]),
            torch.tensor([-1., 0., 1., -1., -1.]), .20,
        )
        self.assertEqual(mask.tolist(), [True, False, False, False, False])


class PickHoldTests(unittest.TestCase):
    def setUp(self):
        self.count = torch.zeros(1, dtype=torch.long)
        self.best = torch.zeros(1, dtype=torch.long)
        self.completed = torch.zeros(1, dtype=torch.bool)

    def step(self, lifted=True, interrupted=False, hold_steps=10):
        self.count, self.best, progress, success = advance_pick_hold(
            torch.tensor([lifted]), torch.tensor([interrupted]),
            self.count, self.best, self.completed, hold_steps,
        )
        # The environment retains completion until its next episode reset.
        self.completed |= success
        return bool(progress.item()), bool(success.item())

    def test_success_on_tenth_consecutive_step(self):
        for expected_count in range(1, 10):
            self.assertEqual(self.step(), (True, False))
            self.assertEqual(self.count.item(), expected_count)
        self.assertEqual(self.step(), (True, True))
        self.assertEqual(self.count.item(), 10)
        self.assertEqual(self.best.item(), 10)

    def test_configurable_one_or_fifteen_step_target(self):
        for hold_steps in (1, 15):
            with self.subTest(hold_steps=hold_steps):
                self.setUp()
                outcomes = [self.step(hold_steps=hold_steps)[1]
                            for _ in range(hold_steps)]
                self.assertEqual(outcomes, [False] * (hold_steps - 1) + [True])

    def test_lost_grasp_restarts_count_without_repaying_prior_progress(self):
        for _ in range(4):
            self.assertEqual(self.step(), (True, False))
        self.assertEqual(self.step(lifted=False), (False, False))
        self.assertEqual(self.count.item(), 0)
        self.assertEqual(self.best.item(), 4)
        for expected_count in range(1, 5):
            self.assertEqual(self.step(), (False, False))
            self.assertEqual(self.count.item(), expected_count)
        self.assertEqual(self.step(), (True, False))
        self.assertEqual(self.best.item(), 5)
        for _ in range(4):
            self.assertFalse(self.step()[1])
        self.assertEqual(self.step(), (True, True))

    def test_failure_takes_precedence_on_would_be_success_step(self):
        for _ in range(9):
            self.step()
        self.assertEqual(self.step(interrupted=True), (False, False))
        self.assertEqual(self.count.item(), 0)
        self.assertEqual(self.best.item(), 9)
        self.assertFalse(self.completed.item())
        for _ in range(9):
            self.assertEqual(self.step(), (False, False))
        self.assertEqual(self.step(), (True, True))

    def test_completed_episode_never_emits_another_success_or_progress(self):
        for _ in range(10):
            self.step()
        for lifted in (True, False, True, True):
            self.assertEqual(self.step(lifted=lifted), (False, False))
            self.assertEqual(self.count.item(), 0)
            self.assertEqual(self.best.item(), 10)

    def test_reset_allows_a_new_episode_to_earn_progress_and_success(self):
        for _ in range(10):
            self.step()
        self.assertTrue(self.completed.item())
        self.count.zero_()
        self.best.zero_()
        self.completed.zero_()
        for _ in range(9):
            self.assertEqual(self.step(), (True, False))
        self.assertEqual(self.step(), (True, True))

    def test_batched_environments_do_not_share_hold_or_failure_state(self):
        count, best, progress, success = advance_pick_hold(
            torch.tensor([True, True, False, True, True, True]),
            torch.tensor([False, False, False, True, False, False]),
            torch.tensor([9, 2, 8, 9, 10, 0]),
            torch.tensor([9, 4, 8, 9, 10, 0]),
            torch.tensor([False, False, False, False, True, False]),
            10,
        )
        self.assertEqual(count.tolist(), [10, 3, 0, 0, 0, 1])
        self.assertEqual(best.tolist(), [10, 4, 8, 9, 10, 1])
        self.assertEqual(progress.tolist(), [True, False, False, False, False, True])
        self.assertEqual(success.tolist(), [True, False, False, False, False, False])
        self.assertEqual(progress.dtype, torch.bool)
        self.assertEqual(success.dtype, torch.bool)


if __name__ == "__main__":
    unittest.main()
