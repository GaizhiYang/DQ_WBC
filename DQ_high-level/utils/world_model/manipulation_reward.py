"""Differentiable reward surrogates for manipulation imagination.

Following supermimic_manipulation principles (MIT license, see THIRD_PARTY_LICENSES.md):
- Explicit functions of state, never learned heads
- Straight-through gradients for predicates
- Continuous shaping terms dominate
- Hard gates use .detach() to control gradient flow
"""

import math
import torch
from utils.gripper_control import CLOSING_OPENING_THRESHOLD


def straight_through(hard, soft):
    """Exact hard forward, explicitly approximate soft backward."""
    return hard.to(soft.dtype) + (soft - soft.detach())


class ManipulationRewardMDP:
    """Differentiable reward and termination for mobile manipulation BPTT.

    Adapts SuperMimic's RigidObjectMDP pattern to the quadruped picking task.
    Key differences from reference:
    - Mobile base adds locomotion costs
    - Hierarchical control: high-level decisions + frozen low-level
    - Task-specific geometry: height threshold, EE distance, gripper opening

    All continuous reward terms provide gradients. Success predicates use
    straight-through estimators to provide smooth backward gradients despite
    hard forward decisions.
    """

    def __init__(self, adapter, config=None):
        """Initialize with task adapter and configuration.

        Args:
            adapter: Fixed8TeacherWorldModelAdapter instance
            config: Optional dict with reward weights and temperatures
        """
        self.adapter = adapter
        cfg = config or {}

        # Reward weights (match native task proportions)
        self.reaching_weight = float(cfg.get("reaching_weight", 1.0))
        self.lifting_weight = float(cfg.get("lifting_weight", 2.0))
        self.placement_weight = float(cfg.get("placement_weight", 1.0))
        self.stability_weight = float(cfg.get("stability_weight", 0.5))
        self.gripper_weight = float(cfg.get("gripper_weight", 0.5))
        self.success_weight = float(cfg.get("success_weight", 10.0))

        # Action penalties
        self.action_magnitude_weight = float(cfg.get("action_magnitude_weight", 0.0001))
        self.action_rate_weight = float(cfg.get("action_rate_weight", 0.002))
        self.joint_velocity_weight = float(cfg.get("joint_velocity_weight", 0.01))

        # Smooth gate temperature (smaller = sharper sigmoid)
        self.smooth_temperature = float(cfg.get("smooth_temperature", 0.01))

        # Success counting (from SuperMimic)
        self.required_steps = int(cfg.get("required_steps", adapter.hold_steps))
        self.success_gradient_scale = float(cfg.get("success_gradient_scale", 0.02))
        self.success_count_temperature = float(cfg.get("success_count_temperature", 0.5))

        # Geometric thresholds
        self.reach_threshold = float(cfg.get("reach_threshold", 0.05))
        self.distance_temperature = float(cfg.get("distance_temperature", 0.01))
        self.height_temperature = float(cfg.get("height_temperature", 0.01))
        self.velocity_threshold = float(cfg.get("velocity_threshold", 0.1))
        self.opening_temperature = float(cfg.get("opening_temperature", 0.1))

        # Task parameters from adapter
        self.lift_threshold = adapter.lift_threshold
        self.high_dt = adapter.high_dt

    def _smooth_gate(self, margin, temperature=None):
        """Straight-through: hard forward, sigmoid backward.

        Args:
            margin: Tensor where margin >= 0 means condition is satisfied
            temperature: Optional override for sigmoid smoothness

        Returns:
            1.0 where condition met (forward), smooth gradient (backward)
        """
        temp = self.smooth_temperature if temperature is None else temperature
        hard = (margin >= 0).float()
        soft = torch.sigmoid(margin / temp)
        return straight_through(hard, soft)

    def _candidate(self, state, context, outcome):
        """Success candidate predicate with differentiable surrogate gradient.

        Native forward: lifted AND close AND gripper closed
        Differentiable backward: smooth sigmoid approximation

        Args:
            state: Predicted state tensor
            context: Environment context
            outcome: Dict with 'lifted', 'distance', 'opening' from geometry

        Returns:
            Binary success candidate (forward), smooth gradient (backward)
        """
        # Geometric conditions with smooth margins
        height_margin = outcome["object_height"] - self.lift_threshold
        distance_margin = self.reach_threshold - outcome["distance"]
        opening_margin = CLOSING_OPENING_THRESHOLD - outcome["opening"]

        # Smooth sigmoid for each condition
        lifted_soft = torch.sigmoid(height_margin / self.height_temperature)
        close_soft = torch.sigmoid(distance_margin / self.distance_temperature)
        closed_soft = torch.sigmoid(opening_margin / self.opening_temperature)

        # Hard forward: all conditions must be met
        lifted_hard = outcome["lifted"].float()
        close_hard = (outcome["distance"] < self.reach_threshold).float()
        closed_hard = (outcome["opening"] < CLOSING_OPENING_THRESHOLD).float()

        # Combine with straight-through
        candidate_soft = lifted_soft * close_soft * closed_soft
        candidate_hard = lifted_hard * close_hard * closed_hard

        return straight_through(candidate_hard, candidate_soft)

    def _success_term(self, previous_count, candidate, failed):
        """Counted success with scaled straight-through derivative.

        From SuperMimic: count consecutive candidate steps, success when
        count >= required_steps. The gradient scale allows BPTT to affect
        success despite the hard forward decision.

        Args:
            previous_count: Success counter from previous state
            candidate: Current candidate predicate (with ST gradient)
            failed: Failure predicate (stops counting)

        Returns:
            Success indicator with scaled gradient
        """
        # Count increments only when candidate holds and not failed
        eligible = (~failed).float()
        count = candidate * (previous_count + 1) * eligible

        # Success when count reaches threshold
        count_margin = count - self.required_steps
        success_soft = torch.sigmoid(count_margin / self.success_count_temperature) * eligible
        success_hard = (count >= self.required_steps).float()
        success = straight_through(success_hard, success_soft)

        # Scale gradient for BPTT (forward value unchanged)
        success = success.detach() + self.success_gradient_scale * (success - success.detach())

        return success, count

    def reward(self, previous_state, state, action, context, outcome):
        """Compute differentiable reward from imagined transition.

        Args:
            previous_state: State before action
            state: Predicted next state
            action: Executed action (9-dim continuous)
            context: Environment context
            outcome: Dict with geometry and task outcomes

        Returns:
            (reward, done): Both with appropriate gradients
        """
        # Extract geometry
        obj_pos = outcome["object_position"]
        obj_vel = outcome["object_velocity"]
        ee_pos = outcome["ee_position"]
        distance = outcome["distance"]
        height = outcome["object_height"]
        opening = outcome["opening"]

        # Get target from context
        target_height = self.adapter.get(previous_state, "closest")
        if target_height.ndim > 1:
            target_height = target_height.squeeze(-1)
        target_height = torch.where(target_height < 0,
                                    torch.full_like(target_height, self.lift_threshold),
                                    target_height)

        # 1. Reaching reward: EE to object (continuous)
        reaching = 1.0 - torch.tanh(distance / 0.1)

        # 2. Lifting reward: Object height progress (continuous)
        height_progress = (height / self.lift_threshold).clamp(0, 1)
        # Only reward lifting when close to object (detached gate)
        near_object = (distance < 0.1).float().detach()
        lifting = height_progress * near_object

        # 3. Placement reward: Height stability at target (continuous)
        height_error = (height - target_height).abs()
        placement = torch.exp(-height_error / 0.05)

        # 4. Stability reward: Low velocity (continuous with smooth gate)
        velocity_norm = obj_vel.norm(dim=-1)
        stability = self._smooth_gate(self.velocity_threshold - velocity_norm)

        # 5. Gripper coordination: Close when near object (gradient through opening)
        should_close = (distance < self.reach_threshold).float().detach()  # detached gate
        gripper_reward = should_close * (1.0 - opening)  # gradient through opening

        # 6. Success bonus with straight-through gradient
        candidate = self._candidate(state, context, outcome)
        previous_count = self.adapter.get(previous_state, "hold").squeeze(-1)
        failed = outcome.get("failed", torch.zeros_like(candidate).bool())
        success, new_count = self._success_term(previous_count, candidate, failed)

        # 7. Action costs (all differentiable)
        # Exclude gripper from magnitude cost (only pose + base)
        pose_base_action = torch.cat([action[:, :6], action[:, 7:]], dim=-1)
        action_magnitude = pose_base_action.square().sum(-1).clamp_max(1e4)

        # All 9 dimensions in action rate
        previous_action = self.adapter.get(previous_state, "prev_action")
        action_rate = (action - previous_action).square().sum(-1).clamp_max(1e4)

        # Joint velocity from state
        arm_velocity = self.adapter.get(state, "dof_vel")[:, 12:18]  # Arm joints only
        joint_velocity = arm_velocity.square().sum(-1).clamp_max(1e4)

        # Combine all terms (dense rewards scaled by dt, success is not)
        dense_reward = (
            self.reaching_weight * reaching +
            self.lifting_weight * lifting +
            self.placement_weight * placement +
            self.stability_weight * stability +
            self.gripper_weight * gripper_reward -
            self.action_magnitude_weight * action_magnitude -
            self.action_rate_weight * action_rate -
            self.joint_velocity_weight * joint_velocity
        )

        total_reward = self.high_dt * dense_reward + self.success_weight * success

        # Termination: success, failure, or timeout
        # All use straight-through for gradients
        failed_indicator = failed.float()
        timeout = outcome.get("timeout", torch.zeros_like(success).bool()).float()
        done = 1 - (1 - failed_indicator) * (1 - success) * (1 - timeout)

        return total_reward, done

    def metadata(self):
        """Return configuration for logging/checkpointing."""
        return {
            "version": "dq_manipulation_mdp_v1",
            "source": "supermimic_manipulation adaptation",
            "weights": {
                "reaching": self.reaching_weight,
                "lifting": self.lifting_weight,
                "placement": self.placement_weight,
                "stability": self.stability_weight,
                "gripper": self.gripper_weight,
                "success": self.success_weight,
                "action_magnitude": self.action_magnitude_weight,
                "action_rate": self.action_rate_weight,
                "joint_velocity": self.joint_velocity_weight,
            },
            "smooth_temperature": self.smooth_temperature,
            "required_steps": self.required_steps,
            "success_gradient_scale": self.success_gradient_scale,
            "reward_dt": self.high_dt,
            "lift_threshold": self.lift_threshold,
            "predicates": "straight_through_estimators",
            "gradient_flow": "all_continuous_terms_plus_ST_predicates",
        }
