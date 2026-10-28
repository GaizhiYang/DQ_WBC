"""Success statistics over one shared window of completed episodes."""

from collections import deque


class EpisodeSuccessWindow:
    """Keep the last 1000 episodes across all parallel environments.

    Object rates use only that object's episodes within the same global window.
    Before the window fills, the denominator is the number actually completed.
    Simultaneous completions are appended in ascending environment ID order.
    """

    def __init__(self, object_names):
        self.object_names = tuple(object_names)
        self.episodes = deque(maxlen=1000)

    def append(self, object_name, success, one_time_success=False):
        self.episodes.append((object_name, bool(success), bool(one_time_success)))

    def metrics(self, include_one_time=False):
        counts = {name: 0 for name in self.object_names}
        successes = dict.fromkeys(self.object_names, 0)
        one_time_successes = dict.fromkeys(self.object_names, 0)
        for name, success, one_time_success in self.episodes:
            counts[name] += 1
            successes[name] += success
            one_time_successes[name] += one_time_success

        completed = len(self.episodes)
        successful = sum(successes.values())
        metrics = {
            "SuccessRate / Total": successful / max(completed, 1),
            "SuccessCount / Total": successful,
            "SuccessWindow / Episodes": completed,
        }
        for name in self.object_names:
            metrics[f"SuccessRate / {name}"] = successes[name] / max(counts[name], 1)
            metrics[f"SuccessWindow / {name}"] = counts[name]
        if include_one_time:
            metrics["Onetime_SuccessRate / Total"] = sum(one_time_successes.values()) / max(completed, 1)
            for name in self.object_names:
                metrics[f"Onetime_SuccessRate / {name}"] = one_time_successes[name] / max(counts[name], 1)
        return metrics
