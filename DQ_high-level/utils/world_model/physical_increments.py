"""Physical increment coordinates shared by WM targets, integration and errors.

The reference manipulation model predicts raw increments, not a correction to
velocity integration.  DQ keeps its PRE base-heading frame, xyzw quaternions,
19 joints and frozen eight-step controller.  Layout is robot/object each
``[dv(3), dw(3), dp(3), drot(3)]``, then ``dq(19), dqd(19)``.
"""

import torch

from .representation import rotation_vector


def physical_increment_targets(spec, pre, post):
    """Raw endpoint displacement in the PRE heading frame, in physical units."""
    return physical_endpoint_error(spec, post, pre, pre)


def physical_endpoint_error(spec, predicted, target, pre):
    """Endpoint error in precisely the output increment's order/frame/units.

    The PRE heading is shared by predicted and target. It is not reconstructed
    from the target, which would give output statistics and loss different axes.
    Angular error is Log(Rpred Rtarget^-1), invariant to quaternion sign.
    """
    from utils.wm_env import quat_conjugate, quat_mul, quat_normalize, quat_inverse_apply
    if spec.delta_dim != 62:
        raise ValueError("Physical increments require the fixed8 62-output layout")
    _, heading = spec._heading(pre)
    pieces = []
    for name in ("robot", "object"):
        predicted_body, target_body = spec.get(predicted, name), spec.get(target, name)
        rotation = quat_mul(quat_normalize(predicted_body[:, 3:7]),
                            quat_conjugate(quat_normalize(target_body[:, 3:7])))
        pieces.extend((quat_inverse_apply(heading, predicted_body[:, 7:10] - target_body[:, 7:10]),
                       quat_inverse_apply(heading, predicted_body[:, 10:13] - target_body[:, 10:13]),
                       quat_inverse_apply(heading, predicted_body[:, :3] - target_body[:, :3]),
                       quat_inverse_apply(heading, rotation_vector(rotation))))
    pieces.extend((spec.get(predicted, "dof_pos") - spec.get(target, "dof_pos"),
                   spec.get(predicted, "dof_vel") - spec.get(target, "dof_vel")))
    return torch.cat(pieces, -1)


def integrate_physical_increments(spec, state, delta, action, context, dt):
    """Direct increment integration; task adapter still advances analytic fields."""
    from utils.wm_env import quat_apply, quat_mul, quat_normalize, from_rotvec
    _, heading = spec._heading(state)
    changes = {}
    for i, name in enumerate(("robot", "object")):
        old, increment = spec.get(state, name), delta[:, i * 12:(i + 1) * 12]
        velocity = old[:, 7:10] + quat_apply(heading, increment[:, :3])
        angular = old[:, 10:13] + quat_apply(heading, increment[:, 3:6])
        position = old[:, :3] + quat_apply(heading, increment[:, 6:9])
        rotation = quat_normalize(quat_mul(from_rotvec(quat_apply(heading, increment[:, 9:12])),
                                           quat_normalize(old[:, 3:7])))
        changes[name] = torch.cat((position, rotation, velocity, angular), -1)
    changes.update(dof_pos=spec.get(state, "dof_pos") + delta[:, 24:43],
                   dof_vel=spec.get(state, "dof_vel") + delta[:, 43:62])
    finish = getattr(spec, "finish_low_physics", None)
    if finish is not None:
        return finish(state, changes, context)
    # Compatibility for adapters predating the shared analytic-field hook.
    # Invert the old integrator rather than silently changing memory semantics.
    from .residual_targets import residual_targets
    desired = spec.replace(state, **changes)
    residual = residual_targets(spec, state, desired, dt)
    return spec.integrate_low(state, residual, action, context, dt)


def physical_increment_metadata():
    """Explicit dimensional floors; these are numerical scales, not accuracy goals."""
    blocks = []
    for body in ("robot", "object"):
        for field, unit, floor in (("dv", "m/s", .01), ("dw", "rad/s", .01),
                                   ("dp", "m", .001), ("drot", "rad", .001)):
            blocks.append({"name": body + "." + field, "width": 3, "unit": unit, "error_floor": floor})
    # All 19 coordinates in b1z1-col/urdf/b1z1.urdf, including z1_jointGripper,
    # are revolute joints. This is joint angle, not Cartesian finger opening.
    blocks.extend(({"name": "dq", "width": 19, "unit": "rad", "error_floor": .01},
                   {"name": "dqd", "width": 19, "unit": "rad/s", "error_floor": .01}))
    return {"dynamics_target": "physical_increment", "version": 1,
            "frame": "PRE robot heading", "rotation": "left-multiplied shortest rotation vector",
            "blocks": blocks, "error_floor_is_accuracy_threshold": False}


def physical_increment_error_floor():
    return [block["error_floor"] for block in physical_increment_metadata()["blocks"]
            for _ in range(block["width"])]
