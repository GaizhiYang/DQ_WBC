"""Validate and export the minimal actor as a standalone normalized TorchScript."""
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path

import torch
from torch import nn
from skrl.resources.preprocessors.torch import RunningStandardScaler

from modules.minimal_asymmetric_teacher import (
    ACTOR_INDICES, ACTOR_OBS_DIM, PACKED_OBS_DIM, MinimalAsymmetricTeacherPolicy,
)


class DeploymentPolicy(nn.Module):
    """Raw67 -> frozen SKRL normalization -> mean9, without actuator transforms."""
    def __init__(self, actor, mean, variance, count, epsilon=1e-8, clip_threshold=5.0):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float64).detach().cpu().clone()
        variance = torch.as_tensor(variance, dtype=torch.float64).detach().cpu().clone()
        count = torch.as_tensor(count, dtype=torch.float64).detach().cpu().clone()
        if mean.shape != (ACTOR_OBS_DIM,) or variance.shape != (ACTOR_OBS_DIM,) or count.shape != ():
            raise ValueError("Expected 67-value mean/variance and scalar observation count")
        if (not torch.isfinite(mean).all() or not torch.isfinite(variance).all()
                or (variance < 0).any() or not torch.isfinite(count) or count <= 0):
            raise ValueError("Invalid deployment normalization statistics")
        if not math.isfinite(epsilon) or epsilon <= 0 or not math.isfinite(clip_threshold) or clip_threshold <= 0:
            raise ValueError("Invalid deployment normalization constants")
        self.actor = deepcopy(actor).cpu().eval()
        self.register_buffer("running_mean", mean)
        self.register_buffer("running_variance", variance)
        self.register_buffer("current_count", count)
        self.epsilon = float(epsilon)
        self.clip_threshold = float(clip_threshold)

    def forward(self, raw_observation: torch.Tensor) -> torch.Tensor:
        if raw_observation.dim() != 2 or raw_observation.size(-1) != 67:
            raise ValueError("Minimal deployment policy requires raw [batch, 67] observations")
        normalized = (raw_observation - self.running_mean.float()) / (
            torch.sqrt(self.running_variance.float()) + self.epsilon)
        normalized = torch.clamp(normalized, -self.clip_threshold, self.clip_threshold)
        return self.actor(normalized)


def build_deployment(checkpoint):
    """Accept only a full, identified minimal experiment checkpoint.

    The teacher training entry point uses the unmodified RunningStandardScaler
    defaults. Its state_dict contains mean/variance/count, but not the epsilon
    or clipping constant. Instantiate that same class to recover the constants.
    """
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a minimal asymmetric training checkpoint")
    if "asymmetric_teacher_state" in checkpoint or "teacher_vision_state" in checkpoint:
        raise ValueError("Expected a minimal geometric checkpoint, not a visual/asymmetric M0/M1/M2 checkpoint")
    state = checkpoint.get("grasp_selection_state")
    if not isinstance(state, Mapping) or state.get("version") != 1:
        raise ValueError("Expected version-1 grasp-selection checkpoint metadata")
    settings = state.get("settings")
    if not isinstance(settings, Mapping) or settings.get("teacher_actor") != "minimal" or settings.get("mode") != "geometric":
        raise ValueError("Export requires teacher_actor=minimal and grasp_selector=geometric")
    options = state.get("environment_options")
    if not isinstance(options, Mapping) or type(options.get("use_tanh")) is not bool:
        raise ValueError("Checkpoint must record its use_tanh action convention")
    if options["use_tanh"]:
        raise ValueError("Minimal asymmetric export supports the default unsquashed Gaussian only")
    for key in ("policy", "state_preprocessor"):
        if not isinstance(checkpoint.get(key), Mapping):
            raise ValueError("Checkpoint is missing " + key)
    policy = MinimalAsymmetricTeacherPolicy((PACKED_OBS_DIM,), (9,), "cpu", use_tanh=options["use_tanh"])
    policy.load_state_dict(checkpoint["policy"], strict=True)
    if any(not torch.isfinite(parameter).all() for parameter in policy.parameters()):
        raise ValueError("Actor checkpoint contains non-finite parameters")
    scaler_state = checkpoint["state_preprocessor"]
    expected_shapes = {"running_mean": (1102,), "running_variance": (1102,), "current_count": ()}
    if set(scaler_state) != set(expected_shapes) or any(
            not isinstance(scaler_state[key], torch.Tensor) or tuple(scaler_state[key].shape) != shape
            for key, shape in expected_shapes.items()):
        raise ValueError("Expected the full 1102-value RunningStandardScaler state")
    scaler = RunningStandardScaler(size=PACKED_OBS_DIM, device="cpu")
    scaler.load_state_dict(scaler_state, strict=True)
    indices = list(ACTOR_INDICES)
    deployment = DeploymentPolicy(
        policy.actor, scaler.running_mean[indices], scaler.running_variance[indices],
        scaler.current_count, epsilon=scaler.epsilon, clip_threshold=scaler.clip_threshold,
    ).eval()
    metadata = {
        "format": "dq_minimal_asymmetric_actor_torchscript",
        "format_version": 1,
        "teacher_actor": "minimal",
        "grasp_selector": "geometric",
        "input_shape": ["batch", 67],
        "input_dtype": "float32",
        "input_normalization": "raw values; normalization is included in the exported model",
        "packed_training_indices": indices,
        "fields": [
            {"name": "ee_position_rpy", "slice": [0, 6]},
            {"name": "joint_positions_including_gripper", "slice": [6, 25]},
            {"name": "joint_velocities_excluding_gripper_times_0.05", "slice": [25, 43]},
            {"name": "base_commands", "slice": [43, 46]},
            {"name": "accumulated_ee_goal_position_rpy", "slice": [46, 52]},
            {"name": "previous_high_level_action", "slice": [52, 61],
             "source": "action_history_buf: after global clipActions, before per-channel physical clipping and target integration"},
            {"name": "selected_grasp_position_rpy", "slice": [61, 67]},
        ],
        "pose_frame": "mechanical-arm-base origin, robot-body axes; rotations are XYZ roll/pitch/yaw",
        "joint_order": {
            "position_indices": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18],
            "velocity_indices": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 12, 13, 14, 15, 16, 17],
            "source": "DQ reindex_all applied to the simulation/URDF joint order; match hardware joint names before packing",
            "velocity_scale": 0.05,
        },
        "normalization": {"epsilon": scaler.epsilon, "clip_threshold": scaler.clip_threshold,
                          "formula": "clamp((x-mean.float())/(sqrt(variance.float())+epsilon), -clip, clip)"},
        "actions": ["delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw",
                    "gripper", "base_vx", "base_yaw_rate"],
        "output": "raw Gaussian mean; no tanh, Gaussian sampling, action clipping or actuator scaling",
        "training_use_tanh": options["use_tanh"],
        "environment_options": deepcopy(dict(options)),
        "action_execution": "Apply the saved DQ use_tanh convention, clipping, accumulated-target integration, IK and low-level control externally; gripper >=0 opens",
        "selection_settings": deepcopy(dict(settings)),
        "scope": "Ideal-geometry control experiment. Real target perception and actuator integration are external.",
    }
    return deployment, metadata


def export_deployment(checkpoint, output):
    """Save one file loadable by torch.jit.load with only PyTorch installed."""
    deployment, metadata = build_deployment(checkpoint)
    scripted = torch.jit.script(deployment)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(destination), _extra_files={"metadata.json": json.dumps(metadata, ensure_ascii=False, indent=2)})
    return metadata
