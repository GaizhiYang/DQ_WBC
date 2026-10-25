"""Ground support checks for B1Z1 SGS states on the z=0 plane."""
import torch

FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
# Collision spheres in b1z1.urdf; the smaller visual spheres are not used here.
FOOT_RADIUS = .04
# The geometry-only builder does not require contact force or a stationary base.
APPROX_GROUND_LIMITS = {
    "min_foot_bottom_m": -.005,
    "max_foot_bottom_m": .02,
}
GROUND_LIMITS = {
    "min_foot_bottom_m": -.003,
    "max_foot_bottom_m": .01,
    "min_contact_force_z_n": 2.,
    "min_support_feet": 3,
    "max_base_linear_speed_m_s": .15,
    "max_base_angular_speed_rad_s": .5,
}
RESET_MIN_SUPPORT_FEET = 2


def foot_indices(env):
    return [env.body_names_to_idx[name] for name in FOOT_NAMES]


def grasp_contact_body_indices(env):
    """The fixed finger is merged into link06 when fixed joints are collapsed."""
    names = env.body_names_to_idx
    fixed = "gripperStator" if "gripperStator" in names else "link06"
    return [names[fixed], names["gripperMover"]]


def grounding_state(env):
    indices = foot_indices(env)
    return (env._rigid_body_pos[:, indices, 2] - FOOT_RADIUS,
            env._contact_forces[:, indices, :])


def approx_grounded_mask(bottom_z):
    """All foot collision bottoms are finite and approximately at ground level."""
    return (torch.isfinite(bottom_z).all(dim=-1)
            & (bottom_z >= APPROX_GROUND_LIMITS["min_foot_bottom_m"]).all(dim=-1)
            & (bottom_z <= APPROX_GROUND_LIMITS["max_foot_bottom_m"]).all(dim=-1))


def grounded_mask(robot_root, bottom_z, forces):
    """All feet near the plane, at least three bearing load, low base velocity."""
    limits = GROUND_LIMITS
    near = ((bottom_z >= limits["min_foot_bottom_m"])
            & (bottom_z <= limits["max_foot_bottom_m"]))
    supported = near & (forces[..., 2] > limits["min_contact_force_z_n"])
    return (torch.isfinite(robot_root).all(dim=-1)
            & torch.isfinite(forces).all(dim=(-1, -2))
            & near.all(dim=-1)
            & (supported.sum(dim=-1) >= limits["min_support_feet"])
            & (robot_root[:, 7:10].norm(dim=-1) <= limits["max_base_linear_speed_m_s"])
            & (robot_root[:, 10:13].norm(dim=-1) <= limits["max_base_angular_speed_rad_s"]))


def reset_supported_mask(bottom_z, forces):
    """After history reset, permit load redistribution while feet stay grounded."""
    near = ((bottom_z >= GROUND_LIMITS["min_foot_bottom_m"])
            & (bottom_z <= GROUND_LIMITS["max_foot_bottom_m"]))
    supported = near & (forces[..., 2] > GROUND_LIMITS["min_contact_force_z_n"])
    return (near.all(dim=-1) & torch.isfinite(forces).all(dim=(-1, -2))
            & (supported.sum(dim=-1) >= RESET_MIN_SUPPORT_FEET))
