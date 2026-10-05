"""Visual Gaussian actor and privileged Critic for asymmetric M0/M1/M2 PPO.

Training observations are [privileged1276, images62208, task5?, belief16?].
The actor can access only the 61 robot values, images, and optional belief.
"""

import copy
from typing import Mapping

import torch
import torch.nn as nn
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .feature_extractor import GuidedTransformerBlock, SharedCNNBackbone
from .teacher_vision import IMAGE_OBS_DIM, PACKED_OBS_DIM, PRIVILEGED_OBS_DIM, _TeacherBackbone


PROPRIO_DIM = 61
TASK_DIM = 5
BELIEF_DIM = 16
DEPLOY_OBS_DIM = IMAGE_OBS_DIM + PROPRIO_DIM
PROPRIO_INDICES = tuple(range(1030, 1082)) + tuple(range(1267, 1276))
KEEP_FUSION_INDICES = tuple(range(6, 58)) + tuple(range(63, 72))


def _validate_features(m1, m2):
    if type(m1) is not bool or type(m2) is not bool:
        raise ValueError("m1 and m2 must be boolean feature flags")


def packed_obs_dim(m1=False, m2=False):
    _validate_features(m1, m2)
    return PACKED_OBS_DIM + TASK_DIM * m1 + BELIEF_DIM * m2


def _split_states(states, m1, m2):
    expected = packed_obs_dim(m1, m2)
    if states.ndim != 2 or states.shape[-1] != expected:
        raise ValueError("Expected packed states shaped [batch, {}]".format(expected))
    privileged = states[:, :PRIVILEGED_OBS_DIM]
    images = states[:, PRIVILEGED_OBS_DIM:PACKED_OBS_DIM]
    task = states[:, PACKED_OBS_DIM:PACKED_OBS_DIM + TASK_DIM] if m1 else None
    belief_start = PACKED_OBS_DIM + TASK_DIM * m1
    belief = states[:, belief_start:belief_start + BELIEF_DIM] if m2 else None
    return privileged, images, task, belief


class DeploymentActor(nn.Module):
    """Action mean from deployable inputs; no shape encoder, GFM, or GT branch.

    Images have shape [B,12,54,96] or [B,62208]. Proprioception has shape
    [B,61] and is already normalized. M2 requires unnormalized belief [B,16].
    """

    def __init__(self, *, m2=False):
        super().__init__()
        _validate_features(False, m2)
        self.m2 = m2
        self.shared_cnn = SharedCNNBackbone(feature_dim=64)
        self.transformer_arm = GuidedTransformerBlock(feature_dim=64, dropout=0.0)
        self.transformer_base = GuidedTransformerBlock(feature_dim=64, dropout=0.0)
        self.arm_proj = nn.Linear(64, 64)
        self.base_proj = nn.Linear(64, 64)
        self.visual_adapter = nn.Linear(128, 512, bias=False)
        nn.init.zeros_(self.visual_adapter.weight)
        self.proprio_adapter = nn.Linear(PROPRIO_DIM, 512)
        if m2:
            self.belief_adapter = nn.Linear(BELIEF_DIM, 512, bias=False)
            nn.init.zeros_(self.belief_adapter.weight)
        self.action_head = nn.Sequential(
            nn.ELU(), nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 9),
        )

    def visual_features(self, images, normalized_proprio):
        if normalized_proprio.ndim != 2 or normalized_proprio.shape[-1] != PROPRIO_DIM:
            raise ValueError("Expected normalized proprioception shaped [batch, 61]")
        batch = normalized_proprio.shape[0]
        if tuple(images.shape) not in ((batch, IMAGE_OBS_DIM), (batch, 12, 54, 96)):
            raise ValueError("Expected images shaped [batch, 12, 54, 96] or [batch, 62208]")
        images = images.reshape(batch, 12, 54, 96)

        def encode(channels):
            frames = images[:, channels].reshape(batch * 3, 2, 54, 96)
            return self.shared_cnn(frames).reshape(batch, 3, 64)

        arm = self.transformer_arm(encode([1, 3, 5, 7, 9, 11]), normalized_proprio)
        base = self.transformer_base(encode([0, 2, 4, 6, 8, 10]), normalized_proprio)
        return torch.cat([self.arm_proj(arm), self.base_proj(base)], dim=-1)

    def preactivation(self, images, normalized_proprio, belief=None):
        if self.m2:
            if belief is None or tuple(belief.shape) != (normalized_proprio.shape[0], BELIEF_DIM):
                raise ValueError("M2 requires belief shaped [batch, 16]")
        elif belief is not None:
            raise ValueError("Belief is only accepted when m2=True")
        visual = self.visual_features(images, normalized_proprio)
        hidden = self.proprio_adapter(normalized_proprio) + self.visual_adapter(visual)
        if self.m2:
            hidden = hidden + self.belief_adapter(belief)
        return hidden

    def forward(self, images, normalized_proprio, belief=None):
        return self.action_head(self.preactivation(images, normalized_proprio, belief))


def _validate_model_configuration(model, m1, m2, no_feature, pitch_control, floating_base):
    if model.num_observations != packed_obs_dim(m1, m2) or model.num_actions != 9:
        raise ValueError("Asymmetric teacher requires packed observation {} and 9 actions".format(packed_obs_dim(m1, m2)))
    if no_feature or pitch_control or floating_base:
        raise ValueError("Asymmetric teacher requires the standard non-floating 9-action teacher task")


class AsymmetricTeacherPolicy(GaussianMixin, Model):
    """SKRL Gaussian policy whose entire actor uses deployable information."""

    def __init__(
        self, observation_space, action_space, device="cpu", *, m1=False, m2=False,
        deterministic=False, use_tanh=False, clip_actions=False,
        clip_log_std=True, min_log_std=-20, max_log_std=2,
        no_feature=False, pitch_control=False, floating_base=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        _validate_model_configuration(self, m1, m2, no_feature, pitch_control, floating_base)
        if use_tanh or clip_actions:
            raise ValueError("Asymmetric teacher keeps the unsquashed Gaussian and environment action clipping")
        GaussianMixin.__init__(
            self, clip_actions=False, clip_log_std=clip_log_std,
            min_log_std=min_log_std, max_log_std=max_log_std,
            reduction="sum", transform_func=None, deterministic=deterministic,
        )
        self.m1, self.m2 = m1, m2
        self.actor = DeploymentActor(m2=m2)
        self.log_std_parameter = nn.Parameter(torch.zeros(9))
        self.to(self.device)

    def compute(self, inputs, role):
        privileged, images, _, belief = _split_states(inputs["states"], self.m1, self.m2)
        proprio = torch.cat([privileged[:, 1030:1082], privileged[:, -9:]], dim=-1)
        return self.actor(images, proprio, belief), self.log_std_parameter, {}

    def to_deploy_actor(self):
        return copy.deepcopy(self.actor).cpu().eval()


class AsymmetricTeacherValue(_TeacherBackbone, DeterministicMixin, Model):
    """Privileged Critic with optional task-memory and perception-belief inputs."""

    def __init__(
        self, observation_space, action_space, device="cpu", *, m1=False, m2=False,
        no_feature=False, pitch_control=False, floating_base=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        _validate_model_configuration(self, m1, m2, no_feature, pitch_control, floating_base)
        DeterministicMixin.__init__(self)
        self.m1, self.m2 = m1, m2
        self._build_teacher_backbone(output_dim=1)
        if m1:
            self.task_adapter = nn.Linear(TASK_DIM, 512, bias=False)
            nn.init.zeros_(self.task_adapter.weight)
        if m2:
            self.belief_adapter = nn.Linear(BELIEF_DIM, 512, bias=False)
            nn.init.zeros_(self.belief_adapter.weight)
        self.to(self.device)

    def compute(self, inputs, role):
        privileged, _, task, belief = _split_states(inputs["states"], self.m1, self.m2)
        hidden = self.net[0](self._teacher_fusion(privileged))
        if self.m1:
            hidden = hidden + self.task_adapter(task)
        if self.m2:
            hidden = hidden + self.belief_adapter(belief)
        for layer in list(self.net.children())[1:]:
            hidden = layer(hidden)
        return hidden, {}


_PRIVILEGED_SHAPES = {
    "feature_encoder.0.weight": (512, 1024), "feature_encoder.0.bias": (512,),
    "feature_encoder.2.weight": (128, 512), "feature_encoder.2.bias": (128,),
    "key_proj.weight": (64, 6), "key_proj.bias": (64,),
    "value_proj.weight": (64, 6), "value_proj.bias": (64,),
    "query_proj.weight": (64, 134), "query_proj.bias": (64,),
    "output_proj.weight": (6, 64), "output_proj.bias": (6,),
}


def _validate_weights(state_dict, expected_shapes, label, permitted_missing=()):
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a model state_dict mapping")
    supplied, expected = set(state_dict), set(expected_shapes)
    missing = sorted(expected - supplied - set(permitted_missing))
    unexpected = sorted(supplied - expected)
    mismatched = sorted(
        key for key in supplied & expected
        if not isinstance(state_dict[key], torch.Tensor)
        or tuple(state_dict[key].shape) != expected_shapes[key]
    )
    if missing or unexpected or mismatched:
        raise ValueError("Incompatible {} weights: missing={}, unexpected={}, shape_mismatch={}".format(
            label, missing, unexpected, mismatched))


def initialize_policy_from_teacher(policy, state_dict: Mapping[str, torch.Tensor], *, allow_legacy=False):
    """Warm-start the pure actor; strictly validate and discard teacher GT keys.

    The trained visual adapter is copied unchanged. A teacher without any
    visual keys needs explicit ``allow_legacy=True``. New belief weights are
    zero. This operation restores neither optimizer nor scaler state.
    """
    if not isinstance(policy, AsymmetricTeacherPolicy):
        raise TypeError("Expected AsymmetricTeacherPolicy")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a teacher model state_dict mapping")
    destination = policy.state_dict()
    mapping, visual_keys = {"log_std_parameter": "log_std_parameter"}, set()
    for key in destination:
        if not key.startswith("actor."):
            continue
        local = key[len("actor."):]
        if local.startswith(("proprio_adapter.", "belief_adapter.")):
            continue
        if local.startswith("action_head."):
            parts = local.split(".")
            old = "net.{}.{}".format(int(parts[1]) + 1, parts[2])
        else:
            old = local
            visual_keys.add(old)
        mapping[old] = key
    expected = {old: tuple(destination[new].shape) for old, new in mapping.items()}
    expected.update(_PRIVILEGED_SHAPES)
    expected.update({"net.0.weight": (512, 206), "net.0.bias": (512,)})
    legacy = not (set(state_dict) & visual_keys)
    _validate_weights(state_dict, expected, "teacher", visual_keys if legacy and allow_legacy else ())
    migrated = dict(destination)
    for old, new in mapping.items():
        if old in state_dict:
            migrated[new] = state_dict[old]
    migrated["actor.proprio_adapter.weight"] = state_dict["net.0.weight"][:, KEEP_FUSION_INDICES]
    migrated["actor.proprio_adapter.bias"] = state_dict["net.0.bias"]
    if legacy:
        migrated["actor.visual_adapter.weight"] = torch.zeros_like(policy.actor.visual_adapter.weight)
    if policy.m2:
        migrated["actor.belief_adapter.weight"] = torch.zeros_like(policy.actor.belief_adapter.weight)
    return policy.load_state_dict(migrated, strict=True)


def initialize_value_from_teacher(value, state_dict: Mapping[str, torch.Tensor]):
    """Warm-start a teacher/v2 Critic, preserving existing enabled adapters.

    Newly introduced adapters start at zero. Removing an existing adapter is
    rejected rather than silently changing the source Critic's computation.
    """
    if not isinstance(value, AsymmetricTeacherValue):
        raise TypeError("Expected AsymmetricTeacherValue")
    destination = value.state_dict()
    new_keys = {key for key in destination if key.startswith(("task_adapter.", "belief_adapter."))}
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a model state_dict mapping")
    expected = {key: tuple(tensor.shape) for key, tensor in destination.items()
                if key not in new_keys or key in state_dict}
    _validate_weights(state_dict, expected, "teacher Critic")
    migrated = dict(state_dict)
    migrated.update({key: torch.zeros_like(destination[key]) for key in new_keys if key not in state_dict})
    return value.load_state_dict(migrated, strict=True)


def initialize_policy_from_asymmetric(policy, state_dict: Mapping[str, torch.Tensor]):
    """Warm-start a v1/v2 asymmetric actor, preserving existing belief weights.

    Historical auxiliary parameters are checked for completeness and shape,
    then discarded. They are never constructed as runnable modules. Resuming
    an old optimizer or old training metadata is deliberately unsupported.
    """
    if not isinstance(policy, AsymmetricTeacherPolicy):
        raise TypeError("Expected AsymmetricTeacherPolicy")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Expected a model state_dict mapping")
    destination = policy.state_dict()
    historical = "_alpha" in state_dict or any(key.startswith("transition.") for key in state_dict)
    copied = {key: tensor for key, tensor in destination.items()
              if key != "actor.belief_adapter.weight" or (key in state_dict and not historical)}
    expected = {key: tuple(tensor.shape) for key, tensor in copied.items()}
    if historical:
        expected.update({"transition." + key: shape for key, shape in _PRIVILEGED_SHAPES.items()})
        expected.update({"transition.projection.weight": (512, 145), "_alpha": ()})
    _validate_weights(state_dict, expected, "asymmetric teacher")
    migrated = {key: state_dict[key] for key in copied}
    if policy.m2 and "actor.belief_adapter.weight" not in migrated:
        migrated["actor.belief_adapter.weight"] = torch.zeros_like(policy.actor.belief_adapter.weight)
    return policy.load_state_dict(migrated, strict=True)


initialize_policy_from_v1 = initialize_policy_from_asymmetric
