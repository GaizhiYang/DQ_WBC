"""Differentiable B1+Z1 link kinematics, without an Isaac Gym dependency.

The URDF is parsed once at construction; every forward operation is a batched
torch operation on the module's device. Quaternions are xyzw. Input ``base13``
is ``[world link position, quaternion, world COM linear velocity, world angular
velocity]`` using the actual Isaac Gym root-state tensor convention. The root
COM velocity is transported to the root link origin before the FK recursion.
Outputs are at link origins. The shipped asset's root ``base`` and ``trunk``
origins coincide, and ``ee_gripper_link`` has a zero inertial offset. Its root
body includes the collapsed arm mount and has a nonzero COM; the randomized
trunk COM (body 1) is a separate rigid body and must not be substituted here.
Pass the live actor's ``rigid_body_properties[0].com`` as ``base_com``: PhysX
may recompute that COM from collision geometry when collapsing fixed joints.
The default is the nominal URDF inertial COM, useful without a simulator but
not guaranteed to reproduce a compiled asset's mass properties.

The forward velocity recursion follows the approach in the local SuperMimic
reproduction's ``src/supermimic/kinematics/g1_fk.py`` (MIT, Copyright (c) 2026
Yuxi Liu; license text in THIRD_PARTY_LICENSES.md). This implementation uses
URDF transforms and Isaac's xyzw/world-angular-velocity conventions instead
of MuJoCo's wxyz/body-angular-velocity free joint conventions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Sequence
import xml.etree.ElementTree as ET

import torch
from torch import Tensor, nn


DEFAULT_URDF = (Path(__file__).resolve().parents[2] / "data" / "asset"
                / "b1z1-col" / "urdf" / "b1z1.urdf")
DEFAULT_DOF_NAMES = tuple(
    name + "_joint"
    for leg in ("FL", "FR", "RL", "RR")
    for name in (leg + "_hip", leg + "_thigh", leg + "_calf")
) + ("z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle",
     "z1_forearm_roll", "z1_wrist_rotate", "z1_jointGripper")
FOOT_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")


def _quat_mul(a: Tensor, b: Tensor) -> Tensor:
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def _quat_apply(q: Tensor, v: Tensor) -> Tensor:
    cross = 2.0 * torch.cross(q[..., :3], v, dim=-1)
    return v + q[..., 3:] * cross + torch.cross(q[..., :3], cross, dim=-1)


def _rpy_quat(rpy):
    r, p, y = (angle * 0.5 for angle in rpy)
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                             math.sin(p), math.cos(y), math.sin(y))
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


def _vector(element, key, default):
    value = element.get(key, default) if element is not None else default
    # The existing B1Z1 asset spells the arm mount height "0.09>>". Isaac's
    # importer accepts its numeric prefix; preserve that transform without
    # silently accepting arbitrary malformed URDF numbers.
    value = value.replace("0.09>>", "0.09")
    result = tuple(float(number) for number in value.split())
    if len(result) != 3 or not all(math.isfinite(number) for number in result):
        raise ValueError("Expected a finite three-component URDF vector: " + value)
    return result


def _collapsed_root_com(root, root_name):
    """URDF root rigid-body COM with Isaac's fixed-joint collapse enabled."""
    links = {link.get("name"): link for link in root.findall("link")}
    children = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed" and joint.get("dont_collapse", "false").lower() != "true":
            children.setdefault(joint.find("parent").get("link"), []).append(joint)
    weighted = torch.zeros(3, dtype=torch.float64)
    total_mass = 0.0

    def visit(name, position, quaternion):
        nonlocal weighted, total_mass
        inertial = links[name].find("inertial")
        if inertial is not None and inertial.find("mass") is not None:
            mass = float(inertial.find("mass").get("value"))
            local_com = torch.tensor(_vector(inertial.find("origin"), "xyz", "0 0 0"),
                                     dtype=torch.float64)
            weighted = weighted + mass * (position + _quat_apply(quaternion, local_com))
            total_mass += mass
        for joint in children.get(name, ()):
            offset = torch.tensor(_vector(joint.find("origin"), "xyz", "0 0 0"), dtype=torch.float64)
            rotation = torch.tensor(_rpy_quat(_vector(joint.find("origin"), "rpy", "0 0 0")),
                                    dtype=torch.float64)
            visit(joint.find("child").get("link"), position + _quat_apply(quaternion, offset),
                  _quat_mul(quaternion, rotation))

    visit(root_name, torch.zeros(3, dtype=torch.float64),
          torch.tensor([0., 0., 0., 1.], dtype=torch.float64))
    if total_mass <= 0:
        raise ValueError("Root COM cannot be inferred from massless fixed links; pass base_com explicitly")
    return weighted / total_mass


@dataclass(frozen=True)
class _Joint:
    parent: str
    child: str
    dof_index: int


class B1Z1Kinematics(nn.Module):
    """Batched FK for the 19-DOF B1Z1 robot used by the teacher.

    ``joint_pos`` / ``joint_vel`` use the raw simulator order, **not** the
    reindexed low-policy order. Pass ``env.dof_names`` as ``dof_names`` when
    constructing against a live simulator to make this mapping explicit.
    Feet are returned in FR, FL, RR, RL order. The ee frame is on the fixed
    gripper stator, so the moving jaw DOF does not change its pose.

    Example::

        props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.robot_handles[0])
        com = props[0].com
        fk = B1Z1Kinematics(device="cuda:0", dof_names=env.dof_names,
                           base_com=(com.x, com.y, com.z))
        result = fk(base13, joint_pos19, joint_vel19)
        ee = fk.end_effector(base13, joint_pos19, joint_vel19)

    Module buffers can be moved with ``.to(device=..., dtype=...)``; no CPU
    transfers, simulator calls, or autograd detaches occur in either path.
    ``end_effector`` skips all leg branches when only EE geometry is needed.
    """

    def __init__(self, urdf_path=None, device=None, *,
                 dof_names: Optional[Sequence[str]] = None,
                 base_com=None,
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self.urdf_path = str(Path(urdf_path or DEFAULT_URDF).resolve())
        root = ET.parse(self.urdf_path).getroot()
        self.dof_names = tuple(dof_names or DEFAULT_DOF_NAMES)
        if len(self.dof_names) != 19 or len(set(self.dof_names)) != 19:
            raise ValueError("B1Z1 FK requires 19 distinct simulator DOF names")
        indices = {name: index for index, name in enumerate(self.dof_names)}
        all_joints = root.findall("joint")
        moving = {joint.get("name") for joint in all_joints
                  if joint.get("type") != "fixed"}
        if moving != set(indices):
            raise ValueError("Simulator DOF names do not match the B1Z1 URDF: "
                             + str(sorted(moving.symmetric_difference(indices))))
        children = {joint.find("child").get("link"): joint for joint in all_joints}
        roots = {link.get("name") for link in root.findall("link")} - set(children)
        if len(roots) != 1:
            raise ValueError("Expected one URDF root link")
        self.root_name = next(iter(roots))
        root_com = _collapsed_root_com(root, self.root_name) if base_com is None else base_com
        self.register_buffer("base_com", torch.as_tensor(root_com, dtype=dtype, device=device))
        self.ee_name = "ee_gripper_link"
        self.foot_names = FOOT_NAMES

        # Visit just the EE/feet ancestors. Fixed decorative links and the
        # moving jaw have no effect on the requested kinematics.
        ordered, visited, visiting = [], {self.root_name}, set()

        def visit(link):
            if link in visited:
                return
            if link in visiting or link not in children:
                raise ValueError("Disconnected or cyclic URDF link: " + link)
            visiting.add(link)
            joint = children[link]
            visit(joint.find("parent").get("link"))
            ordered.append(joint)
            visiting.remove(link)
            visited.add(link)

        for link in (self.ee_name, *self.foot_names):
            visit(link)
        offsets, rotations, axes, joints = [], [], [], []
        for joint in ordered:
            kind = joint.get("type")
            if kind not in ("fixed", "revolute", "continuous"):
                raise ValueError("Unsupported B1Z1 joint type: " + str(kind))
            offsets.append(_vector(joint.find("origin"), "xyz", "0 0 0"))
            rotations.append(_rpy_quat(_vector(joint.find("origin"), "rpy", "0 0 0")))
            axis = _vector(joint.find("axis"), "xyz", "1 0 0")
            norm = math.sqrt(sum(component * component for component in axis))
            if norm == 0:
                raise ValueError("Zero joint axis: " + joint.get("name"))
            axes.append(tuple(component / norm for component in axis))
            joints.append(_Joint(joint.find("parent").get("link"),
                                 joint.find("child").get("link"),
                                 indices[joint.get("name")] if kind != "fixed" else -1))
        self.joints = tuple(joints)
        self._all_joints = tuple(enumerate(self.joints))
        ee_ancestors = set()
        link = self.ee_name
        while link != self.root_name:
            ee_ancestors.add(link)
            link = children[link].find("parent").get("link")
        # Pure Python tree metadata: no extra state_dict entries or device
        # tensors, and no ancestor discovery inside a training forward pass.
        self._ee_joints = tuple((index, joint) for index, joint in self._all_joints
                                if joint.child in ee_ancestors)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=dtype, device=device))
        self.register_buffer("rotations", torch.tensor(rotations, dtype=dtype, device=device))
        self.register_buffer("axes", torch.tensor(axes, dtype=dtype, device=device))

    def root_origin_velocity(self, base13: Tensor, base_com=None) -> Tensor:
        """Transport Isaac COM velocity to the root link position's derivative.

        Optional ``base_com`` is a (...,3) tensor in root-link coordinates,
        useful if the root body COM is randomized independently per environment.
        The shipped task randomizes the separate trunk body, not this offset.
        """
        quaternion = base13[..., 3:7]
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True).clamp_min(1e-12)
        com = self.base_com if base_com is None else base_com
        com_world = _quat_apply(quaternion, com.expand_as(base13[..., :3]))
        return base13[..., 7:10] - torch.cross(base13[..., 10:13], com_world, dim=-1)

    def forward(self, base13: Tensor, joint_pos: Tensor,
                joint_vel: Tensor, base_com=None) -> Dict[str, Tensor]:
        """Return EE pose/twist and all four foot positions/velocities."""
        return self._forward(base13, joint_pos, joint_vel, base_com, include_feet=True)

    def end_effector(self, base13: Tensor, joint_pos: Tensor,
                     joint_vel: Tensor, base_com=None) -> Dict[str, Tensor]:
        """Return only ``ee_pos``, ``ee_quat``, ``ee_vel``, ``ee_omega``.

        Inputs, COM correction and gradients match ``forward``. Only joints
        on the root-to-EE path are evaluated; feet and leg intermediates are
        never constructed. Both paths use the same registered URDF buffers.
        """
        return self._forward(base13, joint_pos, joint_vel, base_com, include_feet=False)

    def _forward(self, base13: Tensor, joint_pos: Tensor,
                 joint_vel: Tensor, base_com, *, include_feet: bool) -> Dict[str, Tensor]:
        if base13.shape[-1] != 13 or joint_pos.shape[-1] != 19 or joint_vel.shape[-1] != 19:
            raise ValueError("Expected base13[...,13], joint_pos[...,19], joint_vel[...,19]")
        if base13.shape[:-1] != joint_pos.shape[:-1] or joint_pos.shape != joint_vel.shape:
            raise ValueError("Base and joint tensors must have matching batch dimensions")
        if (base13.device != self.offsets.device or joint_pos.device != base13.device
                or joint_vel.device != base13.device):
            raise ValueError("Move FK buffers and all inputs onto the same device")
        quat = base13[..., 3:7]
        quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(1e-12)
        positions = {self.root_name: base13[..., :3]}
        quaternions = {self.root_name: quat}
        velocities = {self.root_name: self.root_origin_velocity(base13, base_com)}
        angular = {self.root_name: base13[..., 10:13]}
        joints = self._all_joints if include_feet else self._ee_joints
        for index, joint in joints:
            parent, child = joint.parent, joint.child
            parent_pos, parent_quat = positions[parent], quaternions[parent]
            displacement = _quat_apply(parent_quat, self.offsets[index].expand_as(parent_pos))
            position = parent_pos + displacement
            orientation = _quat_mul(parent_quat, self.rotations[index].expand_as(parent_quat))
            velocity = velocities[parent] + torch.cross(angular[parent], displacement, dim=-1)
            omega = angular[parent]
            if joint.dof_index >= 0:
                axis = self.axes[index].expand_as(parent_pos)
                axis_world = _quat_apply(orientation, axis)
                angle = joint_pos[..., joint.dof_index:joint.dof_index + 1] * 0.5
                rotation = torch.cat((axis * angle.sin(), angle.cos()), dim=-1)
                orientation = _quat_mul(orientation, rotation)
                omega = omega + axis_world * joint_vel[..., joint.dof_index:joint.dof_index + 1]
            positions[child], quaternions[child] = position, orientation
            velocities[child], angular[child] = velocity, omega

        result = {
            "ee_pos": positions[self.ee_name],
            "ee_quat": quaternions[self.ee_name],
            "ee_vel": velocities[self.ee_name],
            "ee_omega": angular[self.ee_name],
        }
        if include_feet:
            result["feet_pos"] = torch.stack([positions[name] for name in self.foot_names], dim=-2)
            result["feet_vel"] = torch.stack([velocities[name] for name in self.foot_names], dim=-2)
        return result
