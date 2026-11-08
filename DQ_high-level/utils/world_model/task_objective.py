"""Auditable POST-state auxiliary objective for mobile dynamic grasping.

This is a DQ task adaptation, not SuperMimic's native reward or a contact
detector. All continuous arithmetic stays on the input device. Geometry is
loaded once from the same collision assets used by Isaac Gym; no mesh parsing
or host synchronization occurs during an imagined rollout.
"""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


DEFAULT_TASK_AUX_CONFIG = {
    "quality_weights": [.20, .10, .20, .20, .30],
    "object_half_extents_m": {},
    "object_centers_m": {},
    # None means a per-object scale resolved from its real collision bounds.
    "approach_near_scale_m": None,
    "approach_far_scale_m": None,
    "grasp_position_scale_m": None,
    "grasp_width_scale_m": None,
    "orientation_scale_rad": .5,
    "slip_linear_scale_m_s": .10,
    "slip_angular_scale_rad_s": 1.,
    "hold_linear_scale_m_s": .15,
    "hold_angular_scale_rad_s": 1.,
    "base_tilt_scale_rad": .35,
    "slip_cost_weight": .10,
    "unsafe_cost_weight": .10,
    "excess_height_margin_m": .15,
    # B1Z1 URDF collision-pad inner faces, in ee_gripper_link / gripperMover.
    "fixed_pad_ee_m": [0., 0., -.01],
    "moving_pivot_ee_m": [-.086, 0., 0.],
    "moving_pad_m": [.085, 0., -.009],
}


def resolve_task_aux_config(config=None):
    result = dict(DEFAULT_TASK_AUX_CONFIG)
    config = {} if config is None else dict(config)
    unknown = set(config) - set(result)
    if unknown:
        raise ValueError("Unknown task_aux_config fields: " + ", ".join(sorted(unknown)))
    result.update(config)
    weights = result["quality_weights"]
    if len(weights) != 5 or any(not math.isfinite(float(v)) or v < 0 for v in weights) or not math.isclose(sum(weights), 1., abs_tol=1e-6):
        raise ValueError("quality_weights must contain five finite nonnegative weights summing to one")
    for name, value in result.items():
        if name in ("quality_weights", "object_half_extents_m", "object_centers_m"):
            continue
        if name.endswith("_m") and isinstance(value, (tuple, list)):
            if len(value) != 3 or not all(math.isfinite(float(v)) for v in value):
                raise ValueError(name + " must contain three finite coordinates")
        elif value is None:
            if name not in ("approach_near_scale_m", "approach_far_scale_m", "grasp_position_scale_m", "grasp_width_scale_m"):
                raise ValueError(name + " must have an explicit finite scale")
        else:
            if not math.isfinite(float(value)) or value < 0 or (value == 0 and not name.endswith("_weight")):
                raise ValueError(name + " must be finite and positive (cost weights may be zero)")
    for name in ("object_half_extents_m", "object_centers_m"):
        if not isinstance(result[name], dict):
            raise ValueError(name + " must map object names to three coordinates")
        for obj, value in result[name].items():
            if len(value) != 3 or not all(math.isfinite(float(v)) for v in value) or (name == "object_half_extents_m" and min(value) <= 0):
                raise ValueError(name + " contains invalid geometry for " + str(obj))
    return result


def _rpy_matrix(rpy):
    r, p, y = map(float, rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                         [-sp, cp*sr, cp*cr]], dtype=torch.float64)


def collision_object_bounds(urdf_path):
    """Object-frame AABB of URDF collision geometry (not visual geometry).

    Current DQ objects are single-link OBJ collisions. Unsupported assets fail
    explicitly and can supply measured bounds through task_aux_config. DQ's
    asset_multi.scale is not applied by _create_extra, so it is not multiplied
    here; mesh scale and collision origin in the actual URDF are applied.
    """
    path = Path(urdf_path)
    tree = ET.parse(path).getroot()
    if len(tree.findall("link")) != 1:
        raise ValueError("Task objective requires explicit bounds for multi-link object " + str(path))
    vertices = []
    for collision in tree.findall("link/collision"):
        geometry = collision.find("geometry")
        box, mesh = geometry.find("box"), geometry.find("mesh")
        if box is not None:
            half = torch.tensor([float(v)/2 for v in box.get("size").split()], dtype=torch.float64)
            points = torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)], dtype=torch.float64) * half
        elif mesh is not None:
            mesh_path = path.parent / mesh.get("filename")
            if mesh_path.suffix.lower() != ".obj":
                raise ValueError("Task objective needs explicit bounds for non-OBJ collision " + str(mesh_path))
            with mesh_path.open() as source:
                points = torch.tensor([[float(v) for v in line.split()[1:4]]
                                       for line in source if line.startswith("v ")], dtype=torch.float64)
            points *= torch.tensor([float(v) for v in mesh.get("scale", "1 1 1").split()], dtype=torch.float64)
        else:
            raise ValueError("Unsupported object collision geometry: " + str(path))
        origin = collision.find("origin")
        if origin is not None:
            points = points @ _rpy_matrix(origin.get("rpy", "0 0 0").split()).T
            points += torch.tensor([float(v) for v in origin.get("xyz", "0 0 0").split()], dtype=torch.float64)
        vertices.append(points)
    if not vertices:
        raise ValueError("No collision geometry in " + str(path))
    points = torch.cat(vertices)
    minimum, maximum = points.amin(0), points.amax(0)
    return ((minimum + maximum)/2).tolist(), ((maximum - minimum)/2).tolist()


class ManipulationTaskObjective:
    """Bounded D/O/G/GL/GLS quality plus explicit costs, evaluated at POST."""

    def __init__(self, env, config=None):
        self.config = resolve_task_aux_config(config)
        cfg = self.config
        root = Path(__file__).resolve().parents[2]
        assets = env.cfg["env"].get("asset", {})
        asset_root = Path(assets.get("assetRoot", "data/asset"))
        asset_root = asset_root if asset_root.is_absolute() else root / asset_root
        centers, extents, source = [], [], {}
        for name in env.obj_list:
            if name in cfg["object_half_extents_m"]:
                half = cfg["object_half_extents_m"][name]
                center = cfg["object_centers_m"].get(name, [0., 0., 0.])
                source[name] = "explicit_object_frame_collision_bounds"
            else:
                path = asset_root / assets.get("assetFileObj", "obj_set") / name / "model.urdf"
                center, half = collision_object_bounds(path)
                source[name] = str(path)
                center = cfg["object_centers_m"].get(name, center)
            if min(half) <= 0:
                raise ValueError("Degenerate task collision bounds for " + name)
            centers.append(center)
            extents.append(half)
        dtype, device = env._robot_root_states.dtype, env.device
        self.centers = torch.tensor(centers, dtype=dtype, device=device)
        self.extents = torch.tensor(extents, dtype=dtype, device=device)
        self.weights = torch.tensor(cfg["quality_weights"], dtype=dtype, device=device)
        self.fixed_pad = torch.tensor(cfg["fixed_pad_ee_m"], dtype=dtype, device=device)
        self.pivot = torch.tensor(cfg["moving_pivot_ee_m"], dtype=dtype, device=device)
        self.moving_pad = torch.tensor(cfg["moving_pad_m"], dtype=dtype, device=device)
        diameter = 2 * self.extents.amax(-1)
        width = 2 * self.extents.amin(-1)
        self.scales = {name: torch.full_like(width, cfg[name]) if cfg[name] is not None else value
                       for name, value in {
                           "approach_near_scale_m": width.clamp_min(.02),
                           "approach_far_scale_m": (2*diameter).clamp_min(.15),
                           "grasp_position_scale_m": (.2*width).clamp_min(.005),
                           "grasp_width_scale_m": (.2*width).clamp_min(.005)}.items()}
        self.metadata = {"version": "post_bounded_quality_geometry_slip_v1", "config": cfg,
                         "geometry_sources": source,
                         "object_centers_m": dict(zip(env.obj_list, centers)),
                         "object_half_extents_m": dict(zip(env.obj_list, extents)),
                         "resolved_scales": {key: dict(zip(env.obj_list, value.detach().cpu().tolist()))
                                             for key, value in self.scales.items()},
                         "contact_claim": "geometric_and_motion_proxy_not_contact_or_load_truth"}

    def __call__(self, *, robot, obj, ee, object_rotation, ee_rotation,
                 grasp_positions, grasp_rotations, gripper_angle, object_ids,
                 table_height, rest_height, lift_threshold, high_dt):
        cfg, ids = self.config, object_ids
        half, center = self.extents[ids], self.centers[ids]
        delta = grasp_positions - ee[:, None, :3]
        distance2 = delta.square().sum(-1)
        near = self.scales["approach_near_scale_m"][ids, None]
        far = self.scales["approach_far_scale_m"][ids, None]
        selection = torch.softmax(-distance2 / near.square(), -1)
        # A smooth nearest-candidate mixture gives consistent pose association.
        d = (selection * (.5/(1 + distance2/near.square()) + .5/(1 + distance2/far.square()))).sum(-1)
        chord2 = (grasp_rotations - ee_rotation[:, None]).square().sum((-2, -1)) / 2
        orientation = 1/(1 + chord2/cfg["orientation_scale_rad"]**2)
        o = (selection * orientation).sum(-1)
        candidate = (selection * orientation/(1 + distance2/near.square())).sum(-1)

        # Actual moving-jaw angle, not commanded opening. The two points are
        # inner faces of the shipped URDF distal collision pads.
        angle = gripper_angle
        p = self.moving_pad.expand(len(obj), -1)
        moving = torch.stack((angle.cos()*p[:, 0]+angle.sin()*p[:, 2], p[:, 1],
                              -angle.sin()*p[:, 0]+angle.cos()*p[:, 2]), -1) + self.pivot
        tips_local = torch.stack((self.fixed_pad.expand_as(moving), moving), 1)
        tips_world = ee[:, None, :3] + (ee_rotation[:, None] @ tips_local[..., None]).squeeze(-1)
        tips_object = (object_rotation.transpose(-1, -2)[:, None] @
                       (tips_world - obj[:, None, :3])[..., None]).squeeze(-1) - center[:, None]
        box_error = tips_object.abs() - half[:, None]
        surface_distance = torch.linalg.vector_norm(box_error.clamp_min(0), dim=-1) + box_error.amax(-1).clamp_max(0).abs()
        surface = (1/(1+(surface_distance/self.scales["grasp_position_scale_m"][ids,None]).square())).prod(-1)
        jaw_delta = tips_object[:, 1] - tips_object[:, 0]
        gap = torch.linalg.vector_norm(jaw_delta, dim=-1)
        axis = jaw_delta / gap[:, None].clamp_min(1e-6)
        projected_width = 2*(axis.abs()*half).sum(-1)
        width_match = 1/(1+((gap-projected_width)/self.scales["grasp_width_scale_m"][ids]).square())
        midpoint = .5*(tips_object[:, 1]+tips_object[:, 0])
        centered = 1/(1+((midpoint*axis).sum(-1)/(half.amin(-1)+1e-6)).square())
        grasp_geometry = candidate * surface * width_match * centered
        relative = obj[:, :3]-ee[:, :3]
        slip_velocity = obj[:, 7:10]-ee[:, 7:10]-torch.cross(ee[:, 10:13], relative, dim=-1)
        slip_omega = obj[:, 10:13]-ee[:, 10:13]
        slip2 = (slip_velocity/cfg["slip_linear_scale_m_s"]).square().sum(-1) + (slip_omega/cfg["slip_angular_scale_rad_s"]).square().sum(-1)
        slip_quality = 1/(1+slip2)
        g = grasp_geometry * slip_quality
        height = obj[:, 2]-table_height-rest_height
        lift = (height/lift_threshold).clamp(0,1)
        # Body-up tilt is sign-invariant and avoids acos's endpoint derivative.
        q = robot[:, 3:7]
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
        tilt2 = 4*q[:, :2].square().sum(-1)
        steady2 = (obj[:, 7:10]/cfg["hold_linear_scale_m_s"]).square().sum(-1) + (obj[:, 10:13]/cfg["hold_angular_scale_rad_s"]).square().sum(-1)
        stable = 1/(1+steady2+slip2+tilt2/cfg["base_tilt_scale_rad"]**2)
        components = torch.stack((d,o,g,g*lift,g*lift*stable), -1)
        quality = (components*self.weights).sum(-1)
        slip_cost = cfg["slip_cost_weight"]*grasp_geometry*(1-slip_quality)
        excess = ((height-lift_threshold-cfg["excess_height_margin_m"])/lift_threshold).clamp_min(0)
        unsafe_cost = cfg["unsafe_cost_weight"]*(tilt2/cfg["base_tilt_scale_rad"]**2+excess.square())
        aux = high_dt*(quality-slip_cost-unsafe_cost)
        return {"aux": aux, "quality": quality, "approach": d, "orientation": o,
                "grasp": g, "grasp_geometry": grasp_geometry, "lift": lift,
                "hold": stable, "grasp_lift": g*lift, "grasp_lift_hold": g*lift*stable,
                "slip_cost": slip_cost, "unsafe_cost": unsafe_cost,
                "slip_speed": torch.linalg.vector_norm(slip_velocity, dim=-1)}


def raw_action_bound_cost(action, latent_action, forward_limit, *, coordinates, latent_bound=3.):
    """Optional cost with the actual pre-sigmoid Gaussian mean, never logit(u)."""
    if latent_action is None or latent_action.shape != action.shape:
        raise ValueError("raw action bound cost requires the original latent Gaussian mean")
    axes = torch.cat((latent_action[..., :6], latent_action[..., 7:]), -1)
    limits = torch.ones_like(axes) if coordinates == "normalized_commands" else axes.new_tensor([.02]*3+[.06]*3+[forward_limit,.6])
    command_cost = (axes.abs()-limits).clamp_min(0).square().sum(-1)
    latent = latent_action[..., 6]
    return command_cost + (latent.abs()-latent_bound).clamp_min(0).square()
