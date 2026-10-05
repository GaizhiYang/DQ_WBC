"""Deployment normalization and exports containing no privileged actor branch."""

import copy
import math
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.nn as nn

from modules.asymmetric_teacher import (
    AsymmetricTeacherPolicy, DeploymentActor, DEPLOY_OBS_DIM, IMAGE_OBS_DIM,
    PROPRIO_DIM, PROPRIO_INDICES, BELIEF_DIM,
)
from .teacher_vision_preprocessor import PrefixRunningStandardScaler


# Only the privileged prefix is standardized; images/task/belief keep their units.
AsymmetricTeacherPreprocessor = PrefixRunningStandardScaler
DEPLOYMENT_FORMAT_VERSION = 2


class DeployProprioNormalizer(nn.Module):
    """Frozen 61-value subset with exactly the legacy scaler's arithmetic."""

    def __init__(self, mean, variance, *, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float64).detach().clone()
        variance = torch.as_tensor(variance, dtype=torch.float64).detach().clone()
        if tuple(mean.shape) != (PROPRIO_DIM,) or tuple(variance.shape) != (PROPRIO_DIM,):
            raise ValueError("Deployment normalization requires exactly 61 means and variances")
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all() or (variance < 0).any():
            raise ValueError("Deployment normalization statistics must be finite with nonnegative variance")
        self.epsilon, self.clip_threshold = float(epsilon), float(clip_threshold)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be finite and positive")
        if not math.isfinite(self.clip_threshold) or self.clip_threshold <= 0:
            raise ValueError("clip_threshold must be finite and positive")
        self.register_buffer("running_mean", mean)
        self.register_buffer("running_variance", variance)

    @classmethod
    def from_teacher_scaler(cls, scaler):
        if tuple(scaler.running_mean.shape) != (1276,) or tuple(scaler.running_variance.shape) != (1276,):
            raise ValueError("Expected the original 1276-value teacher normalization statistics")
        index = list(PROPRIO_INDICES)
        return cls(scaler.running_mean[index], scaler.running_variance[index],
                   epsilon=scaler.epsilon, clip_threshold=scaler.clip_threshold)

    def forward(self, raw_proprio):
        if raw_proprio.ndim != 2 or raw_proprio.shape[-1] != PROPRIO_DIM:
            raise ValueError("Expected raw proprioception shaped [batch, 61]")
        # The existing SKRL scaler uses sqrt(var) + epsilon, not sqrt(var + epsilon).
        standardized = (raw_proprio - self.running_mean.float()) / (
            torch.sqrt(self.running_variance.float()) + self.epsilon)
        return torch.clamp(standardized, -self.clip_threshold, self.clip_threshold)

    def export_state(self):
        return {
            "mean": self.running_mean.detach().cpu().clone(),
            "variance": self.running_variance.detach().cpu().clone(),
            "epsilon": self.epsilon,
            "clip_threshold": self.clip_threshold,
        }


class DeploymentPolicy(nn.Module):
    """Deployment action means from images and *raw* 61-value robot state.

    The saved log standard deviation is retained as distribution metadata. This
    forward path always returns the deterministic mean used for deployment.
    """

    def __init__(self, actor, normalizer, log_std, *, m1=False):
        super().__init__()
        if not isinstance(actor, DeploymentActor) or not isinstance(normalizer, DeployProprioNormalizer):
            raise TypeError("DeploymentPolicy requires the pure actor and its 61-value normalizer")
        log_std = torch.as_tensor(log_std).detach().clone()
        if tuple(log_std.shape) != (9,) or not torch.isfinite(log_std).all():
            raise ValueError("Expected nine finite log standard deviations")
        self.actor = actor
        self.normalizer = normalizer
        if type(m1) is not bool:
            raise ValueError("m1 must be a boolean feature flag")
        self.m1, self.m2 = m1, actor.m2
        self.register_buffer("log_std", log_std)

    def forward(self, images, raw_proprio=None, belief=None):
        if raw_proprio is None:
            expected = DEPLOY_OBS_DIM + BELIEF_DIM * self.m2
            if images.ndim != 2 or images.shape[-1] != expected:
                raise ValueError("Expected flattened deployment observation [batch, {}]".format(expected))
            if belief is not None:
                raise ValueError("Flattened input already contains the complete deployment observation")
            if self.m2:
                belief = images[:, DEPLOY_OBS_DIM:]
            images, raw_proprio = images[:, :IMAGE_OBS_DIM], images[:, IMAGE_OBS_DIM:DEPLOY_OBS_DIM]
        return self.actor(images, self.normalizer(raw_proprio), belief)


def export_deployment(policy, scaler, path=None):
    """Build/save the version 2 deployable actor, normalization, and input contract."""
    if not isinstance(policy, AsymmetricTeacherPolicy):
        raise TypeError("Expected AsymmetricTeacherPolicy")
    from .asymmetric_perception import BELIEF_CONTRACT
    normalizer = DeployProprioNormalizer.from_teacher_scaler(scaler)
    actor = policy.to_deploy_actor()
    payload = {
        "format_version": DEPLOYMENT_FORMAT_VERSION,
        "feature_flags": {"m1": policy.m1, "m2": policy.m2},
        "actor_state_dict": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
        "proprio_normalizer": normalizer.export_state(),
        "log_std": policy.log_std_parameter.detach().cpu().clone(),
        "metadata": {
            "image_shape": [12, 54, 96],
            "image_channels_per_time": ["base_mask", "wrist_mask", "base_target_depth", "wrist_target_depth"],
            "history_order": "oldest_to_newest",
            "proprio_dim": PROPRIO_DIM,
            "belief_contract": copy.deepcopy(BELIEF_CONTRACT) if policy.m2 else None,
            "deployment_flat_order": ["images62208", "raw_proprio61"] + (["belief16"] if policy.m2 else []),
            "joint_velocity_prescale": 0.05,
            "actions": ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw", "gripper", "base_vx", "base_yaw_rate"],
            "action_execution": "Original DQ non-floating 9-action environment clipping and scaling; gripper nonnegative opens",
            "actor_output": "unsquashed Gaussian mean; deterministic deployment",
        },
    }
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
    return payload


def load_deployment(source, device="cpu"):
    """Load the restricted deployment format from a path or in-memory payload."""
    payload = source if isinstance(source, Mapping) else torch.load(source, map_location="cpu", weights_only=True)
    expected = {"format_version", "feature_flags", "actor_state_dict", "proprio_normalizer", "log_std", "metadata"}
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Expected a deployment-only payload, not a training checkpoint")
    if payload["format_version"] != DEPLOYMENT_FORMAT_VERSION:
        raise ValueError("Unsupported deployment format version")
    features = payload["feature_flags"]
    if (not isinstance(features, Mapping) or set(features) != {"m1", "m2"}
            or any(type(flag) is not bool for flag in features.values())):
        raise ValueError("Invalid deployment feature flags")
    from .asymmetric_perception import BELIEF_CONTRACT
    expected_contract = BELIEF_CONTRACT if features["m2"] else None
    if not isinstance(payload["metadata"], Mapping) or payload["metadata"].get("belief_contract") != expected_contract:
        raise ValueError("Incompatible deployment belief contract")
    normalization = payload["proprio_normalizer"]
    if not isinstance(normalization, Mapping) or set(normalization) != {"mean", "variance", "epsilon", "clip_threshold"}:
        raise ValueError("Invalid deployment normalization payload")
    normalizer = DeployProprioNormalizer(
        normalization["mean"], normalization["variance"],
        epsilon=normalization["epsilon"], clip_threshold=normalization["clip_threshold"],
    )
    actor = DeploymentActor(m2=features["m2"])
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return DeploymentPolicy(actor, normalizer, payload["log_std"], m1=features["m1"]).to(device).eval()
