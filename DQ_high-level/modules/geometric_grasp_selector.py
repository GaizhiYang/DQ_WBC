"""Cheap, batched geometric selection of existing DQ grasp candidates.

All scoring happens on physical observations, before normalization or PPO
storage. The table test uses a conservative gripper envelope, not full robot
collision checking or IK. Candidate translations and rotations are never edited.
"""
import math

import torch

from .karl_grasp_selector import rpy_to_quaternion


GEOMETRIC_DEFAULTS = {
    "switch_margin": 0.10,
    "center_weight": 3.0,
    "topdown_weight": 1.0,
    "height_weight": 0.5,
    "distance_weight": 0.1,
    "rotation_weight": 0.1,
    "height_fraction": 0.75,
    "distance_scale": 1.0,       # metres; only a soft movement cost
    "object_padding": 0.025,     # metres outside the object collision bounds
    "table_clearance": 0.002,    # metres around the table proxy
    "lock_distance": 0.08,      # metres to current target when commanded closed
}


def validate_geometric_settings(settings=None):
    settings = {} if settings is None else settings
    unknown = set(settings) - set(GEOMETRIC_DEFAULTS)
    if unknown:
        raise ValueError("Unknown geometric grasp settings: %s" % sorted(unknown))
    result = dict(GEOMETRIC_DEFAULTS, **settings)
    for name, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("geometric %s must be finite and non-negative" % name)
    if result["distance_scale"] <= 0 or not 0 <= result["height_fraction"] <= 1:
        raise ValueError("geometric distance_scale must be positive and height_fraction in [0, 1]")
    if sum(result[name] for name in result if name.endswith("_weight")) <= 0:
        raise ValueError("At least one geometric scoring weight must be positive")
    return result


def quaternion_matrix(quaternion):
    """xyzw -> rotation matrix; supports arbitrary leading dimensions."""
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1] + (3, 3))


def geometric_grasp_costs(grasps, ee_pose, context, settings=None):
    """Return [N,K] costs (invalid=inf) and component tensors.

    Grasps and EE poses are in the same unnormalized robot-base frame. Object
    and table poses use DQ's per-environment world frame. Object bounds are
    local collision-mesh bounds; no uncalibrated CGN contact offset is applied.
    """
    s = validate_geometric_settings(settings)
    finite = torch.isfinite(grasps).all(-1)
    safe = torch.where(finite[..., None], grasps, torch.zeros_like(grasps))
    base_r = quaternion_matrix(context["base_quaternion"])
    object_r = quaternion_matrix(context["object_pose"][:, 3:7])
    table_r = quaternion_matrix(context["table_pose"][:, 3:7])
    position = torch.einsum("nij,nkj->nki", base_r, safe[..., :3]) + context["arm_base"][:, None]
    local_q = rpy_to_quaternion(safe[..., 3:])
    rotation = base_r[:, None] @ quaternion_matrix(local_q)

    local_center = context["object_center_local"]
    half = context["object_half_extents"]
    object_center = context["object_pose"][:, :3] + torch.einsum("nij,nj->ni", object_r, local_center)
    world_half = torch.einsum("nij,nj->ni", object_r.abs(), half)
    horizontal = (position[..., :2] - object_center[:, None, :2]).norm(dim=-1)
    center_cost = horizontal / world_half[:, :2].norm(dim=-1).clamp_min(0.01)[:, None]
    # DQ tool +X is approach. Down is WORLD -Z, independent of base/object roll.
    down_angle = torch.acos((-rotation[..., 2, 0]).clamp(-1.0, 1.0))
    topdown_cost = down_angle / math.pi
    height = (position[..., 2] - (object_center[:, 2] - world_half[:, 2])[:, None]) / (2 * world_half[:, 2]).clamp_min(0.01)[:, None]
    height_cost = (s["height_fraction"] - height).clamp_min(0)
    distance = (safe[..., :3] - ee_pose[:, None, :3]).norm(dim=-1)
    distance_cost = distance / s["distance_scale"]
    ee_q = rpy_to_quaternion(ee_pose[..., 3:])
    rotation_cost = 2 * torch.acos((local_q * ee_q[:, None]).sum(-1).abs().clamp(0.0, 1.0)) / math.pi

    # Proximity to the object is only a broad geometric sanity check. An OBB
    # cannot certify surface contacts, antipodality, aperture or hollow shapes.
    object_local = torch.einsum("nji,nkj->nki", object_r, position - context["object_pose"][:, None, :3])
    object_ok = ((object_local - local_center[:, None]).abs() <= half[:, None] + s["object_padding"]).all(-1)

    # Enclose the transformed gripper in the table frame. This conservative
    # box test includes fingers/palm; a clear TCP alone is not sufficient.
    table_position = torch.einsum("nji,nkj->nki", table_r, position - context["table_pose"][:, None, :3])
    table_rotation = table_r.transpose(-1, -2)[:, None] @ rotation
    corners = torch.einsum("nkij,pj->nkpi", table_rotation, context["gripper_corners"])
    corners = corners + table_position[:, :, None]
    lower, upper = corners.amin(-2), corners.amax(-2)
    table_half = context["table_half_extents"][:, None]
    overlaps = ((lower <= table_half + s["table_clearance"]) &
                (upper >= -table_half - s["table_clearance"])).all(-1)
    table_context_ok = (torch.isfinite(context["table_pose"]).all(-1) &
                        (context["table_pose"][:, 3:7].norm(dim=-1) > 1e-6))
    table_ok = ~overlaps & table_context_ok[:, None]
    clearance = lower[..., 2] - table_half[..., 2]

    costs = (s["center_weight"] * center_cost + s["topdown_weight"] * topdown_cost +
             s["height_weight"] * height_cost + s["distance_weight"] * distance_cost +
             s["rotation_weight"] * rotation_cost)
    context_ok = torch.isfinite(context["arm_base"]).all(-1)
    for q in (context["base_quaternion"], context["object_pose"][:, 3:7]):
        context_ok &= torch.isfinite(q).all(-1) & (q.norm(dim=-1) > 1e-6)
    valid = finite & object_ok & table_ok & torch.isfinite(costs) & context_ok[:, None]
    costs = costs.masked_fill(~valid, float("inf"))
    return costs, {"center_cost": center_cost, "topdown_cost": topdown_cost,
                   "height_cost": height_cost, "distance_cost": distance_cost,
                   "rotation_cost": rotation_cost, "horizontal_distance": horizontal,
                   "topdown_angle_deg": torch.rad2deg(down_angle), "height_fraction": height,
                   "ee_distance": distance, "table_clearance": clearance,
                   "finite": finite, "object_ok": object_ok, "table_ok": table_ok,
                   "context_ok": context_ok & table_context_ok, "valid": valid}


class GeometricGraspSelector:
    """Best initial target, score hysteresis, and a close-command target lock."""
    def __init__(self, num_envs, device, settings=None):
        self.settings = validate_geometric_settings(settings)
        self.indices = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.locked = torch.zeros_like(self.initialized)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.indices.zero_()
            self.initialized.zero_()
            self.locked.zero_()
        else:
            self.indices[env_ids] = 0
            self.initialized[env_ids] = False
            self.locked[env_ids] = False

    @torch.no_grad()
    def select(self, grasps, ee_pose, context):
        if grasps.ndim != 3 or grasps.shape[0] != self.indices.numel() or grasps.shape[-1] != 6 or grasps.shape[1] == 0:
            raise ValueError("Expected nonempty [num_envs, candidates, 6] grasps")
        if ee_pose.shape != (grasps.shape[0], 6):
            raise ValueError("Expected [num_envs, 6] end-effector poses")
        costs, parts = geometric_grasp_costs(grasps, ee_pose, context, self.settings)
        best_cost, best_index = costs.min(-1)
        current_cost = costs.gather(1, self.indices[:, None]).squeeze(1)
        current_distance = parts["ee_distance"].gather(1, self.indices[:, None]).squeeze(1)
        current_valid = torch.isfinite(current_cost)
        has_valid = torch.isfinite(best_cost)
        # Latch until open/reset/invalid; this avoids target changes during lift.
        self.locked &= context["closing"] & current_valid
        self.locked |= (self.initialized & current_valid & context["closing"] &
                        (current_distance <= self.settings["lock_distance"]))
        choose = has_valid & (~self.initialized | ~current_valid |
                             (~self.locked & (current_cost > best_cost + self.settings["switch_margin"])))
        switched = choose & self.initialized & (best_index != self.indices)
        self.indices.copy_(torch.where(choose, best_index, self.indices))
        self.initialized.copy_(has_valid)
        self.locked &= has_valid
        gather = self.indices[:, None]
        selected = grasps.gather(1, gather[..., None].expand(-1, 1, 6)).squeeze(1)
        # This preserves finite PPO inputs, not a certified collision-free grasp.
        fallback = torch.where(torch.isfinite(ee_pose), ee_pose, torch.zeros_like(ee_pose))
        selected = torch.where(has_valid[:, None], selected, fallback)
        metrics = {"geometric_grasp_index": self.indices.masked_fill(~has_valid, -1),
                   "geometric_grasp_switched": switched,
                   "geometric_no_valid_grasp": ~has_valid,
                   "geometric_invalid_context": ~parts["context_ok"],
                   "geometric_locked": self.locked.clone(),
                   "geometric_valid_count": parts["valid"].sum(-1),
                   "geometric_nonfinite_count": (~parts["finite"]).sum(-1),
                   "geometric_object_rejected_count": (parts["finite"] & ~parts["object_ok"]).sum(-1),
                   "geometric_table_rejected_count": (parts["finite"] & ~parts["table_ok"]).sum(-1)}
        # Zero invalid measurements for aggregate logging; always inspect the
        # explicit no-valid rate alongside these conditional measurements.
        for name in ("center_cost", "topdown_cost", "height_cost", "distance_cost",
                     "rotation_cost", "horizontal_distance", "topdown_angle_deg",
                     "height_fraction", "table_clearance"):
            value = parts[name].gather(1, gather).squeeze(1)
            metrics["geometric_" + name] = torch.where(has_valid, value, torch.zeros_like(value))
        value = costs.gather(1, gather).squeeze(1)
        metrics["geometric_score"] = torch.where(has_valid, value, torch.zeros_like(value))
        return selected, metrics
