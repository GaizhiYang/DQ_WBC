"""Batched, non-learned KARL-style grasp selection in raw robot-base coordinates.

The UR5-specific orientation preference is optional: its workspace convention
is not calibrated for the DQ B1+Z1. No UR5 tool offsets or axis flips are used.
"""
import math

import torch


def rpy_to_quaternion(rpy):
    """XYZ roll/pitch/yaw, R = Rz Ry Rx; return Isaac Gym's xyzw quaternion."""
    r, p, y = (rpy * 0.5).unbind(-1)
    cr, cp, cy = r.cos(), p.cos(), y.cos()
    sr, sp, sy = r.sin(), p.sin(), y.sin()
    return torch.stack((sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
                        cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy), -1)


def quaternion_to_rpy(quaternion):
    """Canonical XYZ Euler angles, including a consistent yaw=0 gimbal lock.

    DQ's legacy quat_to_euler_zyx returns YPR and loses the rotation at gimbal
    lock. Used by geometric single-target branches; the GFM baseline is unchanged.
    """
    q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    sin_pitch = (2 * (w*y-z*x)).clamp(-1.0, 1.0)
    r11, r21 = 1-2*(y*y+z*z), 2*(x*y+w*z)
    r32, r33 = 2*(y*z+w*x), 1-2*(x*x+y*y)
    cos_pitch = torch.sqrt(r11.square() + r21.square())
    pitch = torch.atan2(sin_pitch, cos_pitch)
    roll, yaw = torch.atan2(r32, r33), torch.atan2(r21, r11)
    singular = cos_pitch < 1e-6
    roll = torch.where(singular, torch.atan2(2*(w*x-y*z), 1-2*(x*x+z*z)), roll)
    yaw = torch.where(singular, torch.zeros_like(yaw), yaw)
    return torch.stack((roll, pitch, yaw), -1)


def grasp_costs(grasps, ee_pose, orientation_preference="none"):
    """SO(3) angle in radians plus optional original KARL 0/1 heuristic.

    Inputs are [N,K,6] and [N,6], in the SAME unnormalized base frame.
    Position/confidence/IK/critic value do not enter the KARL cost.
    Invalid candidates have infinite cost.
    """
    valid = torch.isfinite(grasps).all(-1)
    safe = torch.where(valid[..., None], grasps, torch.zeros_like(grasps))
    q = rpy_to_quaternion(safe[..., 3:])
    ee_q = rpy_to_quaternion(ee_pose[..., 3:])
    dot = (q * ee_q[:, None]).sum(-1).abs().clamp(0.0, 1.0)
    costs = 2.0 * torch.acos(dot)
    if orientation_preference == "karl":
        roll, pitch, yaw = ((safe[..., 3:] + math.pi) % (2*math.pi) - math.pi).unbind(-1)
        preferred = ((roll >= -math.pi/4) & (roll <= math.pi/4)
                     & (pitch >= -math.pi/8) & (pitch <= math.pi/2)
                     & (yaw.abs() > math.pi/2))
        costs = costs + (~preferred).to(costs.dtype)
    elif orientation_preference != "none":
        raise ValueError("orientation_preference must be 'none' or 'karl'")
    return costs.masked_fill(~valid, float("inf"))


class KarlGraspSelector:
    """One persistent candidate index per environment, outside the PPO model.

    Start with candidate 0, including on episode reset. Switch only when the
    best candidate improves TOTAL cost by more than the margin (KARL: 30 deg).
    Stable candidate ordering is required throughout an episode.
    """
    def __init__(self, num_envs, device, switch_margin_deg=30.0,
                 orientation_preference="none"):
        if not math.isfinite(switch_margin_deg) or switch_margin_deg < 0:
            raise ValueError("switch_margin_deg must be finite and non-negative")
        if orientation_preference not in ("none", "karl"):
            raise ValueError("orientation_preference must be 'none' or 'karl'")
        self.margin = math.radians(switch_margin_deg)
        self.orientation_preference = orientation_preference
        self.indices = torch.zeros(num_envs, dtype=torch.long, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.indices.zero_()
        else:
            self.indices[env_ids] = 0

    @torch.no_grad()
    def select(self, grasps, ee_pose):
        if grasps.ndim != 3 or grasps.shape[0] != self.indices.numel() or grasps.shape[-1] != 6 or grasps.shape[1] == 0:
            raise ValueError("Expected nonempty [num_envs, candidates, 6] grasps")
        if ee_pose.shape != (grasps.shape[0], 6):
            raise ValueError("Expected [num_envs, 6] current end-effector poses")
        costs = grasp_costs(grasps, ee_pose, self.orientation_preference)
        best_cost, best_index = costs.min(-1)
        current_cost = costs.gather(1, self.indices[:, None]).squeeze(1)
        has_valid = torch.isfinite(best_cost)
        switched = has_valid & (current_cost > best_cost + self.margin)
        self.indices.copy_(torch.where(switched, best_index, self.indices))
        selected = grasps.gather(1, self.indices[:, None, None].expand(-1, 1, 6)).squeeze(1)
        # Defensive fallback for a corrupt candidate set. DQ's offline loader
        # already rejects NaN/Inf; expose this event instead of feeding NaNs.
        selected = torch.where(has_valid[:, None], selected, ee_pose)
        return selected, {"karl_grasp_index": self.indices.clone(),
                          "karl_grasp_switched": switched,
                          "karl_no_valid_grasp": ~has_valid}
