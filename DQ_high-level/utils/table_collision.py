"""GPU-compatible collision checks for the actual table and chassis boxes.

Isaac Gym's net-contact tensor has no other-body IDs. Use the URDF collision
box (not its visual mesh or base-object distance) to identify this body pair.
All fifteen separating axes are tested, including arbitrary roll/pitch/yaw.
"""

import math
import xml.etree.ElementTree as ET

import torch


def read_chassis_box(urdf_path, link_name="trunk"):
    """Read the single box collider and its link-local pose from the URDF."""
    link = ET.parse(urdf_path).getroot().find("link[@name='%s']" % link_name)
    collisions = [] if link is None else link.findall("collision")
    if len(collisions) != 1 or collisions[0].find("geometry/box") is None:
        raise ValueError("table/chassis reset requires one box collision on URDF link "
                         + link_name)
    collision = collisions[0]
    size = [float(v) for v in collision.find("geometry/box").get("size").split()]
    origin = collision.find("origin")
    xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None
                              else "0 0 0").split()]
    rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None
                              else "0 0 0").split()]
    if len(size) != 3 or len(xyz) != 3 or len(rpy) != 3 or min(size) <= 0:
        raise ValueError("invalid chassis box geometry")
    sr, sp, sy = (math.sin(a / 2) for a in rpy)
    cr, cp, cy = (math.cos(a / 2) for a in rpy)
    xyzw = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy]
    return [v / 2 for v in size], xyz, xyzw


@torch.jit.script
def quaternion_box_axes(q: torch.Tensor) -> torch.Tensor:
    """Return world-space box axes as matrix rows; quaternions use xyzw."""
    q = q / q.norm(p=2, dim=-1, keepdim=True).clamp_min(1.e-12)
    x, y, z, w = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(-1, 3, 3)


@torch.jit.script
def box_separation(
    center_a: torch.Tensor, quat_a: torch.Tensor, half_a: torch.Tensor,
    center_b: torch.Tensor, quat_b: torch.Tensor, half_b: torch.Tensor,
) -> torch.Tensor:
    """Maximum signed separating-axis gap in metres (<= 0 means overlap).

    Positive values are an SAT gap, not a Euclidean closest-point distance.
    Parallel cross-product axes are discarded to avoid degenerate constraints.
    """
    axes_a, axes_b = quaternion_box_axes(quat_a), quaternion_box_axes(quat_b)
    cross_axes = torch.cross(axes_a[:, :, None, :].expand(-1, 3, 3, -1),
                             axes_b[:, None, :, :].expand(-1, 3, 3, -1),
                             dim=-1).reshape(-1, 9, 3)
    cross_norm = cross_axes.norm(p=2, dim=-1, keepdim=True)
    cross_axes = cross_axes / cross_norm.clamp_min(1.e-6)
    axes = torch.cat((axes_a, axes_b, cross_axes), dim=1)
    distance = torch.bmm(axes, (center_b - center_a).unsqueeze(-1)).squeeze(-1).abs()
    radius_a = (torch.bmm(axes, axes_a.transpose(1, 2)).abs() * half_a).sum(-1)
    radius_b = (torch.bmm(axes, axes_b.transpose(1, 2)).abs() * half_b).sum(-1)
    gaps = distance - radius_a - radius_b
    gaps[:, 6:] = torch.where(cross_norm.squeeze(-1) > 1.e-6, gaps[:, 6:],
                             torch.full_like(gaps[:, 6:], -float("inf")))
    return gaps.max(dim=-1).values


@torch.jit.script
def table_chassis_contact(
    chassis_center: torch.Tensor, chassis_quat: torch.Tensor, chassis_half: torch.Tensor,
    table_center: torch.Tensor, table_quat: torch.Tensor, table_half: torch.Tensor,
    chassis_contact_force: torch.Tensor, pair_rest_offset: float,
) -> torch.Tensor:
    """Overlap is sufficient, even with cancelling forces; near contact needs force.

    PhysX can hold surfaces apart by their rest offsets. A 2 mm solver tolerance
    covers this case only when the chassis itself has contact force (> 0.01 N).
    Object, feet, arm and gripper contact forces never enter the predicate.
    """
    gap = box_separation(chassis_center, chassis_quat, chassis_half,
                         table_center, table_quat, table_half)
    touching = gap <= 1.e-5  # floating-point tolerance, 0.01 mm
    solver_contact = ((gap <= max(0., pair_rest_offset) + .002)
                      & (chassis_contact_force.norm(p=2, dim=-1) > .01))
    return touching | solver_contact
