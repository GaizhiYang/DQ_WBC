"""Invert the fixed8 62-output semi-implicit integrator, without simulator I/O."""

import torch
from .representation import rotation_vector


def residual_targets(spec, pre, post, dt):
    """Return dimensionless residuals in PRE heading coordinates.

    Layout: root[dv,dw,dp,dr], object[dv,dw,dp,dr], q19, qd19.
    These are NOT endpoint error-vector coordinates or plain state increments.
    Both inputs must be real, same-episode PRE and pre-reset POST states.
    """
    from utils.wm_env import quat_conjugate, quat_mul, quat_normalize, quat_inverse_apply
    if spec.delta_dim != 62:
        raise ValueError("Fixed8 residual inversion requires the 62-output physical contract")
    duration = torch.as_tensor(dt, dtype=pre.dtype, device=pre.device)
    if duration.ndim == 0:
        duration = duration.expand(pre.shape[0], 1)
    elif duration.ndim == 1:
        duration = duration[:, None]
    duration = duration.expand(pre.shape[0], 1)
    _, heading = spec._heading(pre)
    parts = []
    for name in ("robot", "object"):
        before, after = spec.get(pre, name), spec.get(post, name)
        velocity, angular = after[:, 7:10], after[:, 10:13]
        origin_velocity = (spec.kinematics.root_origin_velocity(after)
                           if name == "robot" else velocity)
        rotation = quat_mul(quat_normalize(after[:, 3:7]),
                            quat_conjugate(quat_normalize(before[:, 3:7])))
        parts.extend((quat_inverse_apply(heading, velocity - before[:, 7:10]) / .3,
                      quat_inverse_apply(heading, angular - before[:, 10:13]) / .5,
                      quat_inverse_apply(heading, after[:, :3] - before[:, :3]
                                         - origin_velocity * duration) / .01,
                      quat_inverse_apply(heading, rotation_vector(rotation)
                                         - angular * duration) / .02))
    qd = spec.get(post, "dof_vel")
    parts.extend(((spec.get(post, "dof_pos") - spec.get(pre, "dof_pos")
                   - qd * duration) / .02,
                  (qd - spec.get(pre, "dof_vel")) / 2.))
    return torch.cat(parts, -1)
