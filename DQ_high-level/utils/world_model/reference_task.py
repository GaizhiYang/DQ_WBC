"""SuperMimic manipulation's explicit POST MDP, adapted to mobile picking.

The straight-through gate, counted success, its 0.02 gradient scale, and the
endpoint reaching / velocity / action costs follow clab.tasks.supermimic and
clab.tasks.lemon.supermimic (see THIRD_PARTY_LICENSES.md). This is a task
adaptation, not a contact model: hard forward predicates are DQ's unchanged
height / tool-distance / continuous-opening criteria. Real success labels are
never replaced with these surrogate derivatives.
"""

import math

import torch

from utils.gripper_control import CLOSING_OPENING_THRESHOLD


DEFAULT_REFERENCE_TASK_CONFIG = {
    "reaching_weight": .2,
    "lifting_weight": .2,
    "holding_weight": .2,
    "action_magnitude_weight": 1e-4,
    "action_rate_weight": 1e-3,
    "joint_velocity_weight": 1e-2,
    "reaching_scale_m": 1.,
    "height_temperature_m": .01,
    "distance_temperature_m": .01,
    "opening_temperature": .1,
    "success_count_temperature": .5,
    "success_gradient_scale": .02,
    "relative_linear_scale_m_s": .1,
    "relative_angular_scale_rad_s": .5,
}


def resolve_reference_task_config(config=None):
    result = dict(DEFAULT_REFERENCE_TASK_CONFIG)
    supplied = {} if config is None else dict(config)
    unknown = set(supplied) - set(result)
    if unknown:
        raise ValueError("Unknown reference_task_config fields: " + ", ".join(sorted(unknown)))
    result.update(supplied)
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(key + " must be finite numeric")
        if value < 0 or (value == 0 and not key.endswith("_weight")):
            raise ValueError(key + " must be positive (weights may be zero)")
        result[key] = float(value)
    return result


def straight_through(hard, soft):
    """Exact hard forward, explicitly approximate soft backward."""
    return hard.to(soft.dtype) + (soft - soft.detach())


class ReferencePickTask:
    def __init__(self, *, hold_steps, lift_threshold, success_reward, high_dt, config=None):
        self.config = resolve_reference_task_config(config)
        self.hold_steps = int(hold_steps)
        self.lift_threshold = float(lift_threshold)
        self.success_reward = float(success_reward)
        self.high_dt = float(high_dt)
        if self.hold_steps < 1 or self.lift_threshold <= 0 or self.high_dt <= 0:
            raise ValueError("Reference pick duration and lift threshold must be positive")
        self.metadata = {
            "version": "dq_reference_endpoint_pick_v1",
            "source": "supermimic_manipulation/clab/tasks/supermimic.py + lemon/supermimic.py",
            "config": self.config,
            "success_reward": self.success_reward,
            "success_reward_scaling": "native DQ per-success reward, not multiplied by dt",
            "dense_reward_dt": self.high_dt,
            "candidate_forward": "native height >= threshold AND EE distance < .2 AND opening < .5",
            "candidate_backward": "sigmoid margins; explicit surrogate, not physical contact derivative",
            "success_backward": "counted ST gate with gradient-only scale",
            "imagined_terminal_continuation": "source candidate*(count+1) without native completed latch; hypothetical tail rewards retained for ST derivatives, excluded by survival forward",
            "holding": "relative object/tool linear and angular slip; no world-velocity penalty",
            "cost_actions": "issued continuous high-level actions; gripper excluded from magnitude only",
            "native_task": "real rewards, reset statistics and success labels unchanged",
        }

    def outcomes(self, *, outcome, previous_count, previous_best, previous_completed,
                 object_height, target_height, distance, opening, time_limit=None):
        """Native forward on reachable rows; source continuation after termination.

        The source keeps evaluating hypothetical states after a hard success:
        its counter keeps rising while the candidate holds, or resets when it
        fails. Survival masks their forward reward / Critic samples, but their
        raw rewards are required by the straight-through survival derivative.
        DQ's real completed latch still owns simulation labels and reset stats;
        applying that latch here would silently change this surrogate estimator.
        """
        del previous_completed
        cfg = self.config
        soft = (torch.sigmoid((object_height - target_height) / cfg["height_temperature_m"])
                * torch.sigmoid((.2 - distance) / cfg["distance_temperature_m"])
                * torch.sigmoid((CLOSING_OPENING_THRESHOLD - opening) / cfg["opening_temperature"]))
        candidate = straight_through(outcome["lifted"], soft)
        eligible = (~outcome["failed"]).to(soft.dtype)
        count = candidate * (previous_count + 1) * eligible
        success_soft = torch.sigmoid((count - self.hold_steps) / cfg["success_count_temperature"]) * eligible
        success = straight_through(count >= self.hold_steps, success_soft)
        success = success.detach() + cfg["success_gradient_scale"] * (success - success.detach())
        failed = outcome["failed"].to(soft.dtype)
        timeout = (outcome["timeout"] if time_limit is None else time_limit).to(soft.dtype)
        done = 1 - (1 - failed) * (1 - success) * (1 - timeout)
        return {"hold": count, "best_hold": torch.maximum(previous_best, count),
                "candidate_surrogate": candidate, "success_surrogate": success,
                "imagined_done": done}

    def reward_components(self, *, obj, ee, height, distance, action, previous_action,
                          arm_velocity, outcome):
        cfg = self.config
        reaching = 1 - torch.tanh(distance / cfg["reaching_scale_m"])
        # No new geometry/force proxy: the grasp part of native DQ's candidate.
        soft_held = (torch.sigmoid((.2-distance) / cfg["distance_temperature_m"])
                     * torch.sigmoid((CLOSING_OPENING_THRESHOLD-action[:, 6]) / cfg["opening_temperature"]))
        held = straight_through((distance < .2) & (action[:, 6] < CLOSING_OPENING_THRESHOLD), soft_held)
        lift = (height / self.lift_threshold).clamp(0, 1) * held
        relative = obj[:, :3] - ee[:, :3]
        slip = obj[:, 7:10] - ee[:, 7:10] - torch.cross(ee[:, 10:13], relative, dim=-1)
        angular_slip = obj[:, 10:13] - ee[:, 10:13]
        slip_sq = ((slip / cfg["relative_linear_scale_m_s"]).square().sum(-1)
                   + (angular_slip / cfg["relative_angular_scale_rad_s"]).square().sum(-1))
        holding = outcome["candidate_surrogate"] / (1 + slip_sq)
        pose_base_action = torch.cat((action[:, :6], action[:, 7:]), -1)
        magnitude = pose_base_action.square().sum(-1).clamp_max(1e4)
        rate = (action - previous_action).square().sum(-1).clamp_max(1e4)
        velocity = arm_velocity.square().sum(-1).clamp_max(1e4)
        dense = (cfg["reaching_weight"]*reaching + cfg["lifting_weight"]*lift
                 + cfg["holding_weight"]*holding - cfg["action_magnitude_weight"]*magnitude
                 - cfg["action_rate_weight"]*rate - cfg["joint_velocity_weight"]*velocity)
        success = self.success_reward * outcome["success_surrogate"]
        return {"reference_reaching": reaching, "reference_lifting": lift,
                "reference_holding": holding, "reference_slip_speed": slip.norm(dim=-1),
                "reference_action_magnitude": magnitude, "reference_action_rate": rate,
                "reference_joint_velocity": velocity, "reference_dense": self.high_dt*dense,
                "reference_success": success, "total": self.high_dt*dense+success}
