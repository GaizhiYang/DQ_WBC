"""Shared command coordinates and action regularization for real/imagined DQ.

SuperMimic uses normalized actions before its physical target scale, a small
orthogonal Actor output layer, and an all-coordinate squared action-rate cost.
The legacy DQ coordinate/reward contracts remain explicit selectable defaults.
"""

import math

import torch
from torch import nn


def validate_task_contract(task_contract="legacy"):
    if task_contract not in ("legacy", "manipulation_v2"):
        raise ValueError("task_contract must be legacy or manipulation_v2")
    return task_contract


def outer_command_deadzone(task_contract, configured):
    """The v2 execution contract removes only the outer high-level gate.

    The frozen low-level walking/gait classifier is deliberately unchanged.
    Config resolution also persists False for v2; this function makes direct
    environment/adapter construction use the same effective execution rule.
    """
    validate_task_contract(task_contract)
    return bool(configured) if task_contract == "legacy" else False


def horizontal_base_alignment(base_position, object_position, heading_direction,
                              minimum_distance=.01):
    """Horizontal heading cosine, shared by real and imagined task rewards."""
    offset = object_position[..., :2] - base_position[..., :2]
    distance = torch.linalg.vector_norm(offset, dim=-1)
    direction = heading_direction[..., :2]
    heading_norm = torch.linalg.vector_norm(direction, dim=-1).clamp_min(1e-8)
    cosine = (offset * direction).sum(-1) / (distance.clamp_min(1e-8) * heading_norm)
    return torch.where(distance >= minimum_distance, cosine, torch.zeros_like(cosine))


def arm_acceleration_penalty(velocity, low_entry_velocity, dt):
    """Existing penalty shape with explicitly matched low-cycle endpoints."""
    acceleration = torch.linalg.vector_norm(velocity - low_entry_velocity, dim=-1) / dt
    return -torch.expm1(-acceleration)


def validate_action_contract(action_coordinates="physical", action_rate_mode="base_l2",
                             use_tanh=False):
    if action_coordinates not in ("physical", "normalized_commands"):
        raise ValueError("action_coordinates must be physical or normalized_commands")
    if action_rate_mode not in ("base_l2", "squared_all"):
        raise ValueError("action_rate_mode must be base_l2 or squared_all")
    if action_coordinates == "normalized_commands" and use_tanh:
        raise ValueError("normalized_commands uses Gaussian coordinates, not a global tanh transform")


def high_level_command_components(action, forward_limit, *, action_coordinates="physical",
                                  use_tanh=False):
    """Return physical EE increments and base commands without changing action.

    The Gaussian sample is retained in policy observations/replay. Coordinate 6
    is already a sigmoid opening and is handled separately by the gripper. The
    normalized path scales before the existing physical actuator limits, so
    interior command derivatives have the proper metre/radian/velocity units.
    """
    validate_action_contract(action_coordinates, use_tanh=use_tanh)
    if action.shape[-1] != 9:
        raise ValueError("The shared teacher command mapping requires nine coordinates")
    if not math.isfinite(float(forward_limit)) or forward_limit <= 0:
        raise ValueError("forward_limit must be positive and finite")
    if action_coordinates == "normalized_commands":
        return ((action[..., :3] * .02).clamp(-.02, .02),
                (action[..., 3:6] * .06).clamp(-.06, .06),
                (action[..., 7] * forward_limit).clamp(-forward_limit, forward_limit),
                (action[..., 8] * .6).clamp(-.6, .6))
    if use_tanh:
        return (action[..., :3] * .02, action[..., 3:6] * .06,
                action[..., 7] * forward_limit, action[..., 8] * .6)
    return (action[..., :3].clamp(-.02, .02), action[..., 3:6].clamp(-.06, .06),
            action[..., 7].clamp(-forward_limit, forward_limit), action[..., 8].clamp(-.6, .6))


def action_rate_cost(action, previous_action, mode="base_l2"):
    """Unweighted rate penalty, in the same stored coordinates as the policy."""
    if mode == "squared_all":
        if action.shape[-1] != 9 or previous_action.shape != action.shape:
            raise ValueError("squared_all action rate requires matching nine-coordinate commands")
        return (action - previous_action).square().sum(-1)
    if mode == "base_l2":
        return torch.linalg.vector_norm(action[..., 7:9] - previous_action[..., 7:9], dim=-1)
    raise ValueError("action_rate_mode must be base_l2 or squared_all")


def initialize_actor_output(output_layer, log_std, *, initial_action_std=None,
                            actor_last_gain=None):
    """Optional source initialization; call in construction, never after load.

    ``None`` leaves legacy initialization untouched. The exploration standard
    deviation belongs to real Gaussian PPO only; mean BPTT never reads it.
    """
    if initial_action_std is not None:
        if not math.isfinite(float(initial_action_std)) or initial_action_std <= 0:
            raise ValueError("initial_action_std must be positive and finite")
        nn.init.constant_(log_std, math.log(initial_action_std))
    if actor_last_gain is not None:
        if not math.isfinite(float(actor_last_gain)) or actor_last_gain <= 0:
            raise ValueError("actor_last_gain must be positive and finite")
        nn.init.orthogonal_(output_layer.weight, gain=actor_last_gain)
        nn.init.zeros_(output_layer.bias)
