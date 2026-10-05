"""Keep legacy PPO training, but honor the requested visual-teacher eval budget."""

import torch
import tqdm

from skrl.trainers.torch import SequentialTrainer


class TeacherVisionTrainer(SequentialTrainer):
    """Use exactly ``timesteps`` environment steps for single-agent evaluation.

    Training is inherited unchanged. Evaluation starts fresh episodes, uses the
    already configured deterministic policy, and neither records rollout memory
    nor calls training/update/checkpoint callbacks. The environment still owns
    its task-specific success counters and episode-reset behavior.
    """

    @torch.no_grad()
    def single_agent_eval(self):
        if self.num_simultaneous_agents != 1 or self.env.num_agents != 1:
            raise ValueError("Visual-teacher evaluation requires one agent")
        if self.timesteps < 0:
            raise ValueError("Evaluation timesteps must be non-negative")
        states, _ = self.env.reset()
        # Evaluation has its own step budget, independent of the training
        # checkpoint's initial_timestep or the legacy fixed 50000-step loop.
        for timestep in tqdm.tqdm(range(self.timesteps), disable=self.disable_progressbar):
            actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
            next_states, _, terminated, truncated, _ = self.env.step(actions)
            if not self.headless:
                self.env.render()
            if terminated.any() or truncated.any():
                states, _ = self.env.reset()
            else:
                states = next_states
