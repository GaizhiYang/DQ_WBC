#!/usr/bin/env python3
"""CPU shape/count check for the proposed MLPs; NOT training implementation."""
import json
from pathlib import Path
import torch
from torch import nn


def mlp(dims, last_activation=True):
    layers = []
    for i, (left, right) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(left, right))
        if i < len(dims) - 2 or last_activation:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ProposedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.robot = mlp([76, 128, 64])
        self.target = mlp([192, 128, 64])
        self.control = mlp([128, 256, 128, 9], last_activation=False)
        self.log_std = nn.Parameter(torch.zeros(9))

    def forward(self, actor_obs):
        robot, target = actor_obs.split([76, 192], dim=-1)
        mean = self.control(torch.cat([self.robot(robot), self.target(target)], dim=-1))
        return mean, self.log_std.expand_as(mean)


class ProposedCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_observation = mlp([268, 128, 128])
        self.privileged = mlp([80, 256, 128])
        self.value = mlp([256, 256, 128, 1], last_activation=False)

    def forward(self, actor_obs, privileged):
        return self.value(torch.cat([
            self.actor_observation(actor_obs), self.privileged(privileged)], dim=-1))


def count(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def main():
    robot_fields = [19, 18, 9, 9, 3, 9, 3, 3, 3]
    target_fields = [9, 12, 3, 3, 3, 3, 2, 2, 1, 1, 1, 1, 1, 2, 3, 1]
    privileged_fields = [15, 21, 6, 6, 4, 5, 15, 8]
    assert sum(robot_fields) == 76
    assert sum(target_fields) == 48
    assert sum(privileged_fields) == 80
    torch.manual_seed(0)
    actor, critic = ProposedActor(), ProposedCritic()
    actor_obs, privileged = torch.randn(3, 268), torch.randn(3, 80)
    mean, log_std = actor(actor_obs)
    value = critic(actor_obs, privileged)
    assert mean.shape == log_std.shape == (3, 9)
    assert value.shape == (3, 1)
    assert torch.isfinite(mean).all() and torch.isfinite(value).all()
    assert not {id(p) for p in actor.parameters()} & {id(p) for p in critic.parameters()}
    result = {
        'status': 'PROPOSAL ONLY: CPU shape / parameter-count verification; not trained',
        'robot_dim': 76, 'target_frame_dim': 48, 'history_frames': 4,
        'actor_input_dim': 268, 'privileged_dim': 80, 'critic_total_input_dim': 348,
        'actor_mean_shape': list(mean.shape), 'actor_log_std_shape': list(log_std.shape),
        'critic_value_shape': list(value.shape),
        'actor_parameters': count(actor),
        'actor_submodules': {name: count(module) for name, module in actor.named_children()},
        'actor_log_std_parameters': actor.log_std.numel(),
        'critic_parameters': count(critic),
        'critic_submodules': {name: count(module) for name, module in critic.named_children()},
        'total_trainable_parameters': count(actor) + count(critic),
        'parameter_sets_disjoint': True,
        'normalization_statistics_counted_as_parameters': False,
        'torch_version': torch.__version__,
    }
    target = Path(__file__).resolve().parent / 'dimension_verification.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
