"""Windowed SuperMimic metrics shared by regular and final W&B writes."""

from numbers import Real

import torch


class SuperMimicMetrics:
    """Average reward samples, collect completed episodes, and retain snapshots.

    Call ``record_step`` for every vector step and ``record_episodes`` for every
    completion batch, independently of the logging cadence. Success rates and
    counters are supplied by the environment; they are never averaged in time.
    Tensor inputs are copied without transferring GPU scalars to the host until
    ``flush``. An empty window produces no log record.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self._pending_steps = 0
        self._sums = {}
        self._counts = {}
        self._episode_returns = []
        self._episode_lengths = []
        self._episode_count = 0
        self._snapshots = {"success_rate": {}, "scalar_snapshots": {}}

    @property
    def pending_steps(self):
        """Number of vector steps recorded since the last nonempty flush."""
        return self._pending_steps

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float64).detach()

    def _accumulate(self, key, value):
        if key in self._sums:
            self._sums[key].add_(value)
        else:
            self._sums[key] = value.clone()
            self._counts[key] = 0
        self._counts[key] += 1

    @torch.no_grad()
    def record_step(self, rewards, geometric_rewards, infos):
        """Record per-environment reward means for one vector timestep."""
        native = self._tensor(rewards)
        proxy = self._tensor(geometric_rewards)
        if native.numel() == 0 or proxy.numel() == 0:
            raise ValueError("A recorded vector step must contain rewards")
        self._accumulate("Reward / Instantaneous reward (mean)", native.mean())
        self._accumulate("Reward / Geometric proxy reward (mean)", proxy.mean())
        for name, value in infos.get("reward_components", {}).items():
            component = self._tensor(value)
            if component.numel() != 1:
                raise ValueError(f"Reward component {name!r} must be a scalar mean")
            self._accumulate(f"Reward components / {name} (mean)", component.reshape(()))
        self._pending_steps += 1
        self.update_snapshots(infos)

    @torch.no_grad()
    def record_episodes(self, returns, lengths):
        """Record every finished episode, including completions between writes."""
        returns = self._tensor(returns).reshape(-1)
        lengths = self._tensor(lengths).reshape(-1)
        if returns.numel() != lengths.numel():
            raise ValueError("Episode returns and lengths must have equal counts")
        if returns.numel() == 0:
            return
        self._episode_returns.append(returns.clone())
        self._episode_lengths.append(lengths.clone())
        self._episode_count += returns.numel()

    def update_snapshots(self, infos):
        """Replace supplied snapshot groups; omitted groups keep their latest values.

        An explicit empty group removes its old keys. ``scalar_snapshots`` takes
        precedence over ``success_rate`` for shared tags, matching the PPO writer.
        Updating snapshots alone does not create a new logging window.
        """
        for group in self._snapshots:
            if group not in infos:
                continue
            copied = {}
            for key, value in infos[group].items():
                if isinstance(value, Real):
                    # Environment snapshots commonly contain Python numbers
                    # already transferred as a group by the environment.
                    copied[key] = value
                else:
                    value = torch.as_tensor(value)
                    if value.numel() != 1:
                        raise ValueError(f"Snapshot {key!r} must be scalar")
                    copied[key] = value.detach().reshape(()).clone()
            self._snapshots[group] = copied

    @torch.no_grad()
    def flush(self):
        """Return Python scalars and clear this window, retaining latest snapshots."""
        if not self._pending_steps and not self._episode_count:
            return {}

        values = {key: total / self._counts[key] for key, total in self._sums.items()}
        if self._episode_count:
            returns = torch.cat(self._episode_returns)
            lengths = torch.cat(self._episode_lengths)
            values["episode_reward_mean"] = returns.mean()
            values["episode_reward_std"] = returns.std(unbiased=False)
            values["episode_length_mean"] = lengths.mean()
            values["Reward / Total reward (mean)"] = values["episode_reward_mean"]
            values["Episode / Total timesteps (mean)"] = values["episode_length_mean"]

        for group in ("success_rate", "scalar_snapshots"):
            values.update(self._snapshots[group])
        # Transfer scalar diagnostics together, instead of synchronizing CUDA
        # separately for every reward term, episode, and success-rate snapshot.
        scalars = torch.stack([self._tensor(value).reshape(()) for value in values.values()])
        result = dict(zip(values, scalars.cpu().tolist()))
        result["episode_count"] = self._episode_count

        self._sums.clear()
        self._counts.clear()
        self._episode_returns.clear()
        self._episode_lengths.clear()
        self._episode_count = 0
        self._pending_steps = 0
        return result
