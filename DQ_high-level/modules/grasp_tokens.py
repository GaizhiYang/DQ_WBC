"""Spatio-temporal grasp tokens for the teacher (direction 8, G1/G2).

Each token is one candidate grasp at one time offset ``h``: the candidate's
fixed object-frame transform applied to the object pose at ``t + h``. All
tokens are expressed in arm frame A at time ``t`` (origin at the arm base,
axes following the full base orientation), the same frame as the existing
candidate block and the EKF forecasts.

Rotations go through matrices, never through concatenated Euler angles: the
object pose uses roll/pitch/yaw while the old candidate block stores
[yaw, pitch, roll]. Pure PyTorch, so the module is testable without Isaac Gym.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


ROTATIONS = ("rot6d", "euler")
POSITION_DIM = 3
REACH_DIM = 2            # horizontal distance to the arm base, bearing angle
Z_CLIP = 0.6             # same clip as the old candidate positions


@dataclass(frozen=True)
class GraspTokenLayout:
    """Observation layout of the token block that replaces the 180-dim grasp block.

    Block = tokens flattened in horizon-major order [h, candidate, feature],
    followed by one ``future_active`` flag (1 when the EKF forecast is valid
    and the object is not lifted; otherwise the model masks tokens with h > 0).
    """

    num_candidates: int = 30
    horizons: Tuple[float, ...] = (0.0,)
    rotation: str = "rot6d"

    def __post_init__(self):
        if self.rotation not in ROTATIONS:
            raise ValueError(f"grasp token rotation must be one of {ROTATIONS}")
        if self.num_candidates <= 0:
            raise ValueError("grasp tokens need at least one candidate")
        if not self.horizons or self.horizons[0] != 0.0:
            raise ValueError("grasp token horizons must start with the current time 0.0")
        if any(b <= a for a, b in zip(self.horizons, self.horizons[1:])):
            raise ValueError("grasp token horizons must be strictly increasing")

    @property
    def rotation_dim(self) -> int:
        return 6 if self.rotation == "rot6d" else 3

    @property
    def feature_dim(self) -> int:
        return POSITION_DIM + self.rotation_dim + REACH_DIM

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)

    @property
    def num_tokens(self) -> int:
        return self.num_candidates * self.num_horizons

    @property
    def width(self) -> int:
        return self.num_tokens * self.feature_dim + 1

    def to_dict(self):
        return {"num_candidates": self.num_candidates, "horizons": list(self.horizons),
                "rotation": self.rotation}


def layout_from_config(cfg, num_candidates=30):
    """Token layout from ``cfg["env"]``, or None when tokens are disabled.

    ``graspTokens.future`` adds the EKF horizons (posePrediction.horizons) after h = 0.
    """
    env_cfg = cfg["env"]
    settings = env_cfg.get("graspTokens", {}) or {}
    enabled = settings.get("enabled", False)
    future = settings.get("future", False)
    rotation = settings.get("rotation", "rot6d")
    for name, value in (("enabled", enabled), ("future", future)):
        if not isinstance(value, bool):
            raise ValueError(f"env.graspTokens.{name} must be a boolean")
    if not enabled:
        if future:
            raise ValueError("env.graspTokens.future requires env.graspTokens.enabled")
        return None
    horizons = (0.0,)
    if future:
        prediction = env_cfg.get("posePrediction", {})
        if not prediction.get("enabled", False):
            raise ValueError("env.graspTokens.future requires EKF future poses (posePrediction.enabled)")
        horizons += tuple(float(h) for h in prediction.get("horizons", [0.30, 0.60]))
    return GraspTokenLayout(num_candidates=num_candidates, horizons=horizons, rotation=rotation)


def quat_xyzw_to_matrix(quat: Tensor) -> Tensor:
    """[..., 4] xyzw quaternions (normalized here) to [..., 3, 3] rotation matrices."""
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
    x, y, z, w = quat.unbind(-1)
    row0 = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), dim=-1)
    row1 = torch.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), dim=-1)
    row2 = torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def rotation_features(rotation: Tensor, mode: str) -> Tensor:
    """[..., 3, 3] matrices to rot6d (first two columns, Zhou et al. 2019) or zyx Euler.

    The Euler option returns [yaw, pitch, roll], the order of the old candidate block.
    """
    if mode == "rot6d":
        return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)
    if mode == "euler":
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        pitch = torch.asin(torch.clamp(-rotation[..., 2, 0], -1.0, 1.0))
        roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
        return torch.stack((yaw, pitch, roll), dim=-1)
    raise ValueError(f"unknown rotation mode {mode!r}")


def build_grasp_tokens(object_position: Tensor, object_quaternion: Tensor,
                       grasp_in_object: Tensor, grasp_correction: Tensor,
                       future_active: Tensor, layout: GraspTokenLayout) -> Tensor:
    """Token block [B, layout.width] in arm frame A at time t.

    object_position:   [B, H, 3] object position in A at t + h (h = layout.horizons)
    object_quaternion: [B, H, 4] xyzw object orientation in A at t + h
    grasp_in_object:   [B, N, 4, 4] fixed candidate transforms in the object frame
    grasp_correction:  [3, 3] gripper-axis correction applied to candidate rotations
                       (PredictPoint: Rx_rot @ Rz_rot @ Rx_rot)
    future_active:     [B] bool, whether tokens with h > 0 may be attended
    """
    batch, horizons = object_position.shape[:2]
    if horizons != layout.num_horizons or grasp_in_object.shape[1] != layout.num_candidates:
        raise ValueError("object poses / candidates do not match the grasp token layout")
    object_rotation = quat_xyzw_to_matrix(object_quaternion)                     # [B, H, 3, 3]
    grasp_rotation = grasp_in_object[..., :3, :3] @ grasp_correction             # [B, N, 3, 3]
    rotation = object_rotation[:, :, None] @ grasp_rotation[:, None]             # [B, H, N, 3, 3]
    offset = grasp_in_object[:, None, :, :3, 3, None]                            # [B, 1, N, 3, 1]
    position = (object_rotation[:, :, None] @ offset).squeeze(-1) + object_position[:, :, None]
    position = torch.cat((position[..., :2], position[..., 2:].clamp(-Z_CLIP, Z_CLIP)), dim=-1)
    distance = torch.linalg.norm(position[..., :2], dim=-1, keepdim=True)
    bearing = torch.atan2(position[..., 1:2], position[..., 0:1])
    features = torch.cat((position, rotation_features(rotation, layout.rotation), distance, bearing), dim=-1)
    flag = future_active.to(features.dtype).reshape(batch, 1)
    return torch.cat((features.reshape(batch, -1), flag), dim=-1)


def split_grasp_tokens(block: Tensor, layout: GraspTokenLayout):
    """Inverse of the flattening: ([..., H*N, F] tokens, [...] future_active flag)."""
    tokens = block[..., :-1].reshape(*block.shape[:-1], layout.num_tokens, layout.feature_dim)
    return tokens, block[..., -1]


def token_horizons(layout: GraspTokenLayout) -> Tensor:
    """[H*N] time offset of each token, in the block's horizon-major order."""
    return torch.tensor(layout.horizons, dtype=torch.float32).repeat_interleave(layout.num_candidates)
