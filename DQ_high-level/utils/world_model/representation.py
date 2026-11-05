"""GPU rotation representations, using Isaac's **xyzw** convention.

Adapted from SuperMimic-Reproduced's ``kinematics/rotations.py`` (MIT,
Copyright (c) 2026 Yuxi Liu). The full license is retained in the adjacent
``THIRD_PARTY_LICENSES.md``. These helpers avoid a MuJoCo/mjlab dependency.
"""

import torch


def rotation_6d(quaternion):
    """First two rotation-matrix columns, row-major, invariant to q versus -q."""
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1 - 2 * (y*y + z*z), 2 * (x*y - w*z),
                        2 * (x*y + w*z), 1 - 2 * (x*x + z*z),
                        2 * (x*z - w*y), 2 * (y*z + w*x)), -1)


def rotation_vector(quaternion):
    """Shortest quaternion logarithm, with finite first derivatives at identity.

    The sign is canonicalized to w >= 0. At exactly pi the logarithm has the
    unavoidable axis-sign ambiguity; no artificial smoothness is implied there.
    """
    q = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[..., 3:] < 0, -q, q)
    vector, scalar = q[..., :3], q[..., 3:]
    squared = vector.square().sum(-1, keepdim=True)
    # Clamping only the inactive exact formula avoids sqrt(0)'s infinite
    # derivative in torch.where's backward pass.
    norm = squared.clamp_min(1e-16).sqrt()
    angle = 2 * torch.atan2(norm, scalar)
    factor = torch.where(squared < 1e-8,
                         2 + squared / 3 + 3 * squared.square() / 20,
                         angle / norm)
    return vector * factor
