"""Camera-augmented teacher models with compatible privileged-teacher weights.

Inputs are packed as ``[privileged observation (1276), image history (62208)]``.
The caller standardizes only the privileged prefix. The camera branch receives
the environment's mask/depth values and derives its 61-dimensional guidance
from that already-standardized prefix. These models do not import Isaac Gym.
"""

from typing import Mapping

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .feature_extractor import GuidedTransformerBlock, SharedCNNBackbone


PRIVILEGED_OBS_DIM = 1276
IMAGE_OBS_DIM = 2 * 3 * 2 * 54 * 96
PACKED_OBS_DIM = PRIVILEGED_OBS_DIM + IMAGE_OBS_DIM
VISION_MODES = ("images", "zero", "none")
_VISUAL_PREFIXES = (
    "shared_cnn.",
    "transformer_arm.",
    "transformer_base.",
    "arm_proj.",
    "base_proj.",
    "visual_adapter.",
)


class _TeacherBackbone:
    """Preserve every original teacher parameter name, shape and input slice."""

    def _validate_configuration(self, no_feature, pitch_control, floating_base):
        if no_feature or pitch_control or floating_base:
            raise ValueError(
                "Teacher vision requires 1024 object features and the standard "
                "9-action non-floating robot; no_feature, pitch_control and "
                "floating_base are unsupported"
            )
        if self.num_observations != PACKED_OBS_DIM:
            raise ValueError(
                "Expected packed observation size {}, got {}".format(
                    PACKED_OBS_DIM, self.num_observations
                )
            )
        if self.num_actions != 9:
            raise ValueError("Teacher vision requires exactly 9 actions")

    def _build_teacher_backbone(self, output_dim):
        self.num_features = 1024
        self.encode_dim = 128
        self.cube_object_num = 30
        self.notrain_obs_num = 180
        self.key_proj = nn.Linear(6, 64)
        self.value_proj = nn.Linear(6, 64)
        self.query_proj = nn.Linear(134, 64)
        self.output_proj = nn.Linear(64, 6)
        self.feature_encoder = nn.Sequential(
            nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, 128)
        )
        self.net = nn.Sequential(
            nn.Linear(206, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, output_dim),
        )
        # The original Value also registers this unused parameter. Retain it so
        # its state_dict is exactly compatible with the teacher checkpoint.
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def _split_states(self, states):
        if states.ndim != 2 or states.shape[-1] != PACKED_OBS_DIM:
            raise ValueError(
                "Expected states shaped [batch, {}], got {}".format(
                    PACKED_OBS_DIM, tuple(states.shape)
                )
            )
        return states[:, :PRIVILEGED_OBS_DIM], states[:, PRIVILEGED_OBS_DIM:]

    def _teacher_fusion(self, privileged):
        # Negative indices apply to the 1276-dimensional prefix, never to the
        # packed input whose tail now contains images.
        positions = privileged[:, -189:-99].reshape(-1, 30, 3)
        orientations = privileged[:, -99:-9].reshape(-1, 30, 3)
        grasps = torch.cat([positions, orientations], dim=-1)
        pose = privileged[:, 1024:1030]
        encoded = self.feature_encoder(privileged[:, :1024])
        query = self.query_proj(torch.cat([encoded, pose], dim=-1)).unsqueeze(1)
        keys = self.key_proj(grasps)
        values = self.value_proj(grasps)
        attention = torch.softmax(query @ keys.transpose(-2, -1) / 8.0, dim=-1)
        grasp_feature = self.output_proj((attention @ values).squeeze(1))
        return torch.cat(
            [privileged[:, 1024:1087], privileged[:, -9:], encoded, grasp_feature],
            dim=-1,
        )


class TeacherVisionPolicy(_TeacherBackbone, GaussianMixin, Model):
    """Original Gaussian teacher plus a zero-initialized visual contribution.

    ``images`` uses both cameras; ``zero`` preserves the same visual network and
    state guidance while replacing every image with zero; ``none`` constructs
    only the original teacher network. All modes accept the same packed input.
    """

    def __init__(
        self, observation_space, action_space, device="cpu", *,
        vision_mode="images", deterministic=False, use_tanh=False,
        clip_actions=False, clip_log_std=True, min_log_std=-20,
        max_log_std=2, no_feature=False, pitch_control=False, floating_base=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        self._validate_configuration(no_feature, pitch_control, floating_base)
        if vision_mode not in VISION_MODES:
            raise ValueError("vision_mode must be one of {}".format(VISION_MODES))
        if use_tanh:
            raise ValueError("Teacher vision preserves the unsquashed Gaussian teacher; use_tanh is unsupported")
        if clip_actions:
            raise ValueError("Teacher vision keeps action clipping in the environment; clip_actions must be False")
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=clip_log_std,
            min_log_std=min_log_std, max_log_std=max_log_std,
            reduction="sum", transform_func=None, deterministic=deterministic,
        )
        self.vision_mode = vision_mode
        self._build_teacher_backbone(output_dim=9)
        if vision_mode != "none":
            self.shared_cnn = SharedCNNBackbone(feature_dim=64)
            # PPO must evaluate the same policy during rollout and optimization.
            # train/eval switches must not introduce random attention dropout.
            self.transformer_arm = GuidedTransformerBlock(feature_dim=64, dropout=0.0)
            self.transformer_base = GuidedTransformerBlock(feature_dim=64, dropout=0.0)
            self.arm_proj = nn.Linear(64, 64)
            self.base_proj = nn.Linear(64, 64)
            self.visual_adapter = nn.Linear(128, 512, bias=False)
            nn.init.zeros_(self.visual_adapter.weight)
        self.to(self.device)

    def _visual_features(self, privileged, image_flat):
        batch = image_flat.shape[0]
        images = image_flat.reshape(batch, 12, 54, 96)
        if self.vision_mode == "zero":
            images = torch.zeros_like(images)
        guidance = torch.cat([privileged[:, 1030:1082], privileged[:, -9:]], dim=-1)

        def encode_frames(channels):
            frames = images[:, channels].reshape(batch * 3, 2, 54, 96)
            return self.shared_cnn(frames).reshape(batch, 3, 64)

        arm_sequence = encode_frames([1, 3, 5, 7, 9, 11])
        base_sequence = encode_frames([0, 2, 4, 6, 8, 10])
        arm = self.arm_proj(self.transformer_arm(arm_sequence, guidance))
        base = self.base_proj(self.transformer_base(base_sequence, guidance))
        return torch.cat([arm, base], dim=-1)

    def compute(self, inputs, role):
        privileged, images = self._split_states(inputs["states"])
        fusion = self._teacher_fusion(privileged)
        hidden = self.net[0](fusion)
        if self.vision_mode != "none":
            hidden = hidden + self.visual_adapter(self._visual_features(privileged, images))
        for layer in list(self.net.children())[1:]:
            hidden = layer(hidden)
        return hidden, self.log_std_parameter, {}


class TeacherVisionValue(_TeacherBackbone, DeterministicMixin, Model):
    """Unchanged privileged teacher Critic; the packed image suffix is ignored."""

    def __init__(
        self, observation_space, action_space, device="cpu", *,
        no_feature=False, pitch_control=False, floating_base=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        self._validate_configuration(no_feature, pitch_control, floating_base)
        DeterministicMixin.__init__(self)
        self._build_teacher_backbone(output_dim=1)
        self.to(self.device)

    def compute(self, inputs, role):
        privileged, _ = self._split_states(inputs["states"])
        return self.net(self._teacher_fusion(privileged)), {}


def load_legacy_teacher_weights(model, state_dict: Mapping[str, torch.Tensor]):
    """Load an original teacher Actor/Value, allowing only new visual keys.

    This is a model-only warm start. Optimizer/scaler loading belongs to the
    training entry point. Normal augmented-checkpoint resume should instead use
    the model's standard strict ``load_state_dict``.
    """
    if not isinstance(model, (TeacherVisionPolicy, TeacherVisionValue)):
        raise TypeError("Expected TeacherVisionPolicy or TeacherVisionValue")
    expected = model.state_dict()
    legacy_keys = {key for key in expected if not key.startswith(_VISUAL_PREFIXES)}
    supplied_keys = set(state_dict)
    missing = sorted(legacy_keys - supplied_keys)
    unexpected = sorted(supplied_keys - legacy_keys)
    mismatched = [
        key for key in sorted(legacy_keys & supplied_keys)
        if tuple(expected[key].shape) != tuple(state_dict[key].shape)
    ]
    if missing or unexpected or mismatched:
        raise ValueError(
            "Incompatible legacy teacher weights: missing={}, unexpected={}, "
            "shape_mismatch={}".format(missing, unexpected, mismatched)
        )
    # Validate every key and shape before changing the model. strict=False alone
    # would otherwise silently accept missing original teacher parameters.
    return model.load_state_dict(state_dict, strict=False)
