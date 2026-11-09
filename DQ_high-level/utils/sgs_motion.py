"""Level02 motion and horizontal heading geometry, without simulator imports."""

import torch


def level02_object_motion_xy(state):
    """World XY motion induced by Level02 at the object's support location.

    Construction holds the table still and clears object velocity. Use saved
    motion commands, including the table heading and enabled yaw rotation,
    to determine the reference direction that resumes after reset.
    """
    table = state["table_root"]
    quaternion = table[:, 3:7]
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quaternion.unbind(-1)
    velocity = torch.stack((2 * (x * y - w * z),
                            1 - 2 * (x.square() + z.square())), dim=1)
    velocity = velocity * state["table_vy"][:, None]
    offset = state["object_root"][:, :2] - table[:, :2]
    yaw_rate = torch.where(state["rotation_enabled"].bool(),
                           state["table_yaw_rate"], 0.)
    tangent = torch.stack((-offset[:, 1], offset[:, 0]), dim=1)
    return velocity + yaw_rate[:, None] * tangent


def base_motion_angle_deg(state):
    """Unsigned angle in [0, 180]; NaN when a horizontal direction is absent."""
    quaternion = state["robot_root"][:, 3:7]
    quaternion = quaternion / quaternion.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = quaternion.unbind(-1)
    forward = torch.stack((1 - 2 * (y.square() + z.square()),
                           2 * (x * y + w * z)), dim=1)
    motion = level02_object_motion_xy(state)
    dot = (forward * motion).sum(dim=1)
    cross = forward[:, 0] * motion[:, 1] - forward[:, 1] * motion[:, 0]
    angle = torch.rad2deg(torch.atan2(cross.abs(), dot))
    scale = forward.norm(dim=1) * motion.norm(dim=1)
    return torch.where(scale > 1e-12, angle, torch.full_like(angle, float("nan")))
