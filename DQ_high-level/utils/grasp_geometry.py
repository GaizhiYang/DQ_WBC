"""Static asset geometry and live poses for the geometric teacher selector.

Asset loading runs once on the CPU.  Per-step context contains only small
device tensors and does not require Isaac Gym imports or mesh queries.
"""
from itertools import product
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import torch


_SIGNS = np.asarray(list(product((-1.0, 1.0), repeat=3)), dtype=np.float64)


def _numbers(text, count, label):
    values = np.asarray([float(value) for value in text.split()], dtype=np.float64)
    if values.shape != (count,) or not np.isfinite(values).all():
        raise ValueError("{} must contain {} finite values".format(label, count))
    return values


def _origin(element):
    origin = element.find("origin")
    if origin is None:
        return np.eye(3), np.zeros(3)
    xyz = _numbers(origin.get("xyz", "0 0 0"), 3, "URDF origin xyz")
    rpy = _numbers(origin.get("rpy", "0 0 0"), 3, "URDF origin rpy")
    return Rotation.from_euler("xyz", rpy).as_matrix(), xyz


def _read_urdf(urdf_path):
    path = Path(urdf_path).expanduser().resolve(strict=True)
    return path, ET.parse(str(path)).getroot()


def _collision_vertices(link, urdf_path):
    """Return collision vertices in the link frame for supported DQ assets.

    Boxes are exact; mesh support deliberately covers OBJ, the format used by
    the teacher's object and gripper collision assets. Unsupported geometry
    raises instead of silently inventing dimensions.
    """
    parts = []
    for collision in link.findall("collision"):
        geometry = collision.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError("Expected one collision geometry in {}".format(urdf_path))
        shape = geometry[0]
        if shape.tag == "box":
            size = _numbers(shape.get("size", ""), 3, "URDF box size")
            if (size <= 0).any():
                raise ValueError("Collision box dimensions must be positive")
            points = _SIGNS * (size / 2.0)
        elif shape.tag == "mesh":
            filename = shape.get("filename", "")
            if not filename or "://" in filename:
                raise ValueError("Expected a filesystem collision mesh path in {}".format(urdf_path))
            mesh_path = (urdf_path.parent / filename).resolve(strict=True)
            if mesh_path.suffix.lower() != ".obj":
                raise ValueError("Unsupported collision mesh format: {}".format(mesh_path))
            vertices = []
            with mesh_path.open("r") as stream:
                for line in stream:
                    fields = line.split()
                    if fields and fields[0] == "v":
                        if len(fields) < 4:
                            raise ValueError("Invalid OBJ vertex in {}".format(mesh_path))
                        vertices.append([float(value) for value in fields[1:4]])
            points = np.asarray(vertices, dtype=np.float64)
            if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
                raise ValueError("Collision mesh has no valid finite vertices: {}".format(mesh_path))
            scale = _numbers(shape.get("scale", "1 1 1"), 3, "URDF mesh scale")
            if (scale == 0).any():
                raise ValueError("Collision mesh scale must be nonzero")
            points = points * scale
        else:
            raise ValueError("Unsupported collision geometry '{}' in {}".format(shape.tag, urdf_path))
        rotation, translation = _origin(collision)
        parts.append(points @ rotation.T + translation)
    if not parts:
        raise ValueError("Link '{}' has no collision geometry in {}".format(link.get("name"), urdf_path))
    return np.concatenate(parts, axis=0)


def load_object_collision_bounds(urdf_path):
    """Return (lower, upper) local AABB arrays from a single-link object URDF.

    The URDF collision origin and mesh scale are honored. The unrelated
    ``asset_multi.scale`` setting is not applied: the teacher environment
    reads that setting but does not use it to scale the simulated actors.
    """
    path, root = _read_urdf(urdf_path)
    links = root.findall("link")
    if len(links) != 1 or root.findall("joint"):
        raise ValueError("Object geometry requires a single-link URDF: {}".format(path))
    vertices = _collision_vertices(links[0], path)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    if (upper <= lower).any():
        raise ValueError("Object collision bounds must have positive extent: {}".format(path))
    return lower, upper


def load_gripper_envelope_corners(urdf_path, max_joint_step_deg=2.0):
    """Return eight TCP-frame corners conservatively enclosing the Z1 gripper.

    The TCP fixed joint, stator collisions, mover collisions, joint origin,
    axis, and limits all come from the URDF. The mover's full joint range is
    sampled; an analytical displacement bound pads the envelope between
    samples, so it also encloses the continuous motion. This is a bounding
    envelope for table-clearance checks, not an exact collision model.
    """
    if not math.isfinite(max_joint_step_deg) or max_joint_step_deg <= 0:
        raise ValueError("max_joint_step_deg must be positive and finite")
    path, root = _read_urdf(urdf_path)
    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    tcp_joints = [joint for joint in joints
                  if joint.find("child") is not None
                  and joint.find("child").get("link") == "ee_gripper_link"]
    if len(tcp_joints) != 1 or tcp_joints[0].get("type") != "fixed":
        raise ValueError("Expected one fixed ee_gripper_link TCP joint in {}".format(path))
    tcp_joint = tcp_joints[0]
    stator_name = tcp_joint.find("parent").get("link")
    if stator_name not in links:
        raise ValueError("Gripper TCP parent link is missing in {}".format(path))
    tcp_rotation, tcp_translation = _origin(tcp_joint)
    stator = _collision_vertices(links[stator_name], path)

    mover_joints = [joint for joint in joints if joint is not tcp_joint
                    and joint.find("parent") is not None
                    and joint.find("parent").get("link") == stator_name]
    if len(mover_joints) != 1 or mover_joints[0].get("type") != "revolute":
        raise ValueError("Expected one revolute gripper mover under '{}'".format(stator_name))
    mover_joint = mover_joints[0]
    mover_name = mover_joint.find("child").get("link")
    if mover_name not in links or any(joint.find("parent").get("link") == mover_name for joint in joints):
        raise ValueError("Expected one leaf gripper mover link in {}".format(path))
    mover = _collision_vertices(links[mover_name], path)
    joint_rotation, joint_translation = _origin(mover_joint)
    axis_element = mover_joint.find("axis")
    axis = _numbers(axis_element.get("xyz", "1 0 0") if axis_element is not None else "1 0 0",
                    3, "Gripper joint axis")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("Gripper joint axis cannot be zero")
    axis /= norm
    limit = mover_joint.find("limit")
    if limit is None:
        raise ValueError("Gripper revolute joint limits are missing")
    lower, upper = float(limit.get("lower", "nan")), float(limit.get("upper", "nan"))
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Gripper joint limits must be finite and increasing")
    intervals = max(1, int(math.ceil((upper - lower) / math.radians(max_joint_step_deg))))
    angles = np.linspace(lower, upper, intervals + 1)
    rotations = Rotation.from_rotvec(angles[:, None] * axis[None, :]).as_matrix()
    swept = np.einsum("aij,vj->avi", rotations, mover)
    swept = swept @ joint_rotation.T + joint_translation
    points = np.concatenate((stator, swept.reshape(-1, 3)), axis=0)
    # p_stator = R_tcp * p_tcp + t_tcp, therefore row vectors use R_tcp.
    points = (points - tcp_translation) @ tcp_rotation
    radius = np.linalg.norm(mover - np.outer(mover @ axis, axis), axis=1).max()
    sample_spacing = (upper - lower) / intervals
    guard = 2.0 * radius * math.sin(sample_spacing / 4.0)
    bound_min, bound_max = points.min(axis=0) - guard, points.max(axis=0) + guard
    return (bound_min + bound_max) / 2.0 + _SIGNS * (bound_max - bound_min) / 2.0


class TeacherGraspGeometry:
    """Cache asset dimensions and expose live geometric teacher context."""

    def __init__(self, env):
        self.env = env
        self.device = torch.device(env.rl_device)
        asset = env.cfg["env"]["asset"]
        asset_root = Path(asset["assetRoot"]).expanduser().resolve(strict=True)
        object_root = asset_root / asset["assetFileObj"]
        object_names = list(env.obj_list)
        if not object_names:
            raise ValueError("Geometric grasp selection needs at least one object asset")
        bounds = [load_object_collision_bounds(object_root / name / "model.urdf")
                  for name in object_names]
        lower = torch.as_tensor(np.stack([bound[0] for bound in bounds]),
                                dtype=torch.float32, device=self.device)
        upper = torch.as_tensor(np.stack([bound[1] for bound in bounds]),
                                dtype=torch.float32, device=self.device)
        object_indices = torch.arange(env.num_envs, device=self.device) % len(object_names)
        self.object_center_local = ((lower + upper) / 2.0)[object_indices]
        self.object_half_extents = ((upper - lower) / 2.0)[object_indices]
        dimensions = [env.table_dims.x, env.table_dims.y, env.table_dims.z]
        if not all(math.isfinite(value) and value > 0 for value in dimensions):
            raise ValueError("Table dimensions must be positive and finite")
        self.table_half_extents = (torch.tensor(dimensions, dtype=torch.float32, device=self.device)
                                   .unsqueeze(0).expand(env.num_envs, -1) / 2.0)
        self.gripper_corners = torch.as_tensor(
            load_gripper_envelope_corners(asset_root / asset["assetFileRobot"]),
            dtype=torch.float32, device=self.device)

    def context(self):
        """Get current poses without re-reading assets or caching stale state."""
        env = self.env
        return {
            "base_quaternion": env._robot_root_states[:, 3:7].to(self.device),
            "arm_base": env.arm_base.to(self.device),
            "object_pose": env._cube_root_states[:, :7].to(self.device),
            "object_center_local": self.object_center_local,
            "object_half_extents": self.object_half_extents,
            "table_pose": env._table_root_states[:, :7].to(self.device),
            "table_half_extents": self.table_half_extents,
            "gripper_corners": self.gripper_corners,
            "closing": (env.actions[:, 6] < 0).to(self.device),
        }
