"""Grasp-and-lift criteria and episode statistics shared by train and eval."""

from collections import deque

import torch


class RollingEpisodeSuccess:
    """Keep the most recent completed episodes across all parallel environments.

    Outcomes are (object_index, success) pairs in completion order. The caller
    breaks simultaneous-completion ties by ascending environment index. Object
    breakdowns use this SAME global window, not separate per-object histories.
    """

    def __init__(self, window_size=1000):
        if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        self.window_size = window_size
        self._outcomes = deque(maxlen=window_size)

    def extend(self, outcomes):
        self._outcomes.extend((int(obj), bool(success)) for obj, success in outcomes)

    def metrics(self, object_names):
        """Return a snapshot; omit undefined rates when there are no samples."""
        completed = len(self._outcomes)
        successes = sum(success for _, success in self._outcomes)
        metrics = {
            "SuccessWindow / completed": completed,
            "SuccessWindow / successes": successes,
        }
        if not completed:
            return metrics
        if len(object_names) == 1:
            metrics[f"SuccessRate / {object_names[0]}"] = successes / completed
            return metrics

        metrics["SuccessRate / total"] = successes / completed
        counts = [0] * len(object_names)
        wins = [0] * len(object_names)
        for obj, success in self._outcomes:
            counts[obj] += 1
            wins[obj] += success
        for obj, name in enumerate(object_names):
            metrics[f"SuccessWindow / {name} / completed"] = counts[obj]
            metrics[f"SuccessWindow / {name} / successes"] = wins[obj]
            if counts[obj]:
                metrics[f"SuccessRate / {name}"] = wins[obj] / counts[obj]
        return metrics


def lifted_grasp_mask(object_height, table_height, rest_height, ee_distance,
                      gripper_command, lift_threshold):
    """Measure lift above the object's resting height on the current table.

    A close command plus proximity is a grasp proxy, not a contact-force test.
    Compare absolute heights to avoid subtractive rounding at the threshold.
    """
    target_height = table_height + rest_height + lift_threshold
    return ((object_height >= target_height)
            & (ee_distance < 0.2)
            & (gripper_command < 0))


def advance_pick_hold(lifted, interrupted, count, best_count, completed, hold_steps):
    """Advance consecutive holding; reward each holding milestone only once.

    The caller latches ``success`` into ``completed`` until the episode resets.
    Interrupted or incomplete grasps restart the consecutive count at zero.
    """
    eligible = lifted & ~interrupted & ~completed
    next_count = torch.where(eligible, count + 1, torch.zeros_like(count))
    progress = eligible & (next_count > best_count)
    success = eligible & (next_count >= hold_steps)
    return next_count, torch.maximum(best_count, next_count), progress, success
