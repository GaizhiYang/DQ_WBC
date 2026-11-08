"""Reject checkpoint/config action contracts before constructing the simulator."""

from collections.abc import Mapping


def task_contract_metadata(config):
    env = config.get("env", {})
    settings = env.get("world_model", {})
    metadata = {"action_coordinates": settings.get("action_coordinates", "physical"),
            "action_rate_mode": settings.get("action_rate_mode", "base_l2"),
            "task_contract": settings.get("task_contract", "legacy"),
            "small_commands_zero": bool(env.get("smallValueSetZero", False))}
    if settings.get("recipe") == "reference":
        metadata["policy_architecture"] = "reference_flat_deterministic_v1"
        metadata["policy_network"] = {
            "hidden": list(settings.get("actor_hidden", [2048, 1024, 512, 256])),
            "amp": settings.get("policy_amp", "bf16"),
            "normalization": ("empirical_real_next" if settings.get("actor_running_normalization", True) else "none"),
            "action_noise": settings.get("action_noise", .1),
            "last_gain": settings.get("actor_last_gain", .01)}
    return metadata


class TaskContractMetadata:
    """Persist real-task semantics even in a pure PPO mode=off checkpoint."""

    def __init__(self, config):
        self.metadata = task_contract_metadata(config)

    def state_dict(self):
        return dict(self.metadata)

    def load_state_dict(self, state):
        if dict(state) != self.metadata:
            raise ValueError("Teacher task contract metadata differs from the resolved configuration")


def validate_checkpoint_action_contract(config, path=None, *, payload=None):
    """Check a full teacher checkpoint without converting learned coordinates.

    Ordinary evaluation constructs PPO, whose generic checkpoint loader skips
    ``world_model_training``. Its metadata must therefore be checked separately.
    Missing action fields in legacy checkpoints mean the original physical
    commands and base-only rate cost. A caller that already read the checkpoint
    can supply its CPU payload to avoid reading the file again.
    """
    if payload is None:
        if path is None:
            raise ValueError("Provide a teacher checkpoint path or its loaded payload")
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("Teacher checkpoint must contain named model states")
    from utils.wm_entry import reject_candidate_checkpoint
    reject_candidate_checkpoint(payload=payload)

    metadata = payload.get("teacher_task_contract", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Teacher task contract metadata must be a mapping")
    if "world_model_training" in payload:
        training = payload["world_model_training"]
        if not isinstance(training, Mapping) or not isinstance(training.get("metadata"), Mapping):
            raise ValueError("World-model teacher checkpoint is missing its action/task metadata")
        wm_metadata = training["metadata"]
        for key, value in metadata.items():
            if key in wm_metadata and wm_metadata[key] != value:
                raise ValueError("Contradictory teacher/world-model task metadata: " + key)
        metadata = {**wm_metadata, **metadata}
    settings = config.get("env", {}).get("world_model", {})
    if not isinstance(settings, Mapping):
        raise ValueError("env.world_model must be a mapping")
    defaults = {"action_coordinates": "physical", "action_rate_mode": "base_l2", "task_contract": "legacy"}
    choices = {"action_coordinates": ("physical", "normalized_commands"),
               "action_rate_mode": ("base_l2", "squared_all"),
               "task_contract": ("legacy", "manipulation_v2")}
    mismatches = []
    saved_architecture = metadata.get("policy_architecture", "dq_attention_gaussian_v1")
    current_architecture = ("reference_flat_deterministic_v1" if settings.get("recipe") == "reference"
                            else "dq_attention_gaussian_v1")
    if saved_architecture != current_architecture:
        mismatches.append("policy_architecture: checkpoint={!r}, config={!r}".format(
            saved_architecture, current_architecture))
    if saved_architecture == current_architecture == "reference_flat_deterministic_v1":
        expected = task_contract_metadata(config)["policy_network"]
        if metadata.get("policy_network") != expected:
            mismatches.append("policy_network/normalization differs from saved reference teacher")
    for name, default in defaults.items():
        saved, current = metadata.get(name, default), settings.get(name, default)
        if saved not in choices[name] or current not in choices[name]:
            raise ValueError("Invalid checkpoint/config action contract {}: saved={!r}, configured={!r}"
                             .format(name, saved, current))
        if saved != current:
            mismatches.append("{}: checkpoint={!r}, config={!r}".format(name, saved, current))
    if "small_commands_zero" in metadata:
        current = bool(config.get("env", {}).get("smallValueSetZero", False))
        if type(metadata["small_commands_zero"]) is not bool or metadata["small_commands_zero"] != current:
            mismatches.append("small_commands_zero: checkpoint={!r}, config={!r}".format(
                metadata["small_commands_zero"], current))
    if settings.get("task_contract") == "manipulation_v2" and config.get("env", {}).get("smallValueSetZero", False):
        mismatches.append("manipulation_v2 requires small_commands_zero=False")
    if mismatches:
        raise ValueError(
            "Teacher checkpoint action contract does not match the selected configuration ({}). "
            "Use --cfg_file with this checkpoint's saved teacher YAML, or keep that YAML "
            "beside its original run/checkpoints directory. A moved checkpoint cannot silently "
            "use fallback action coordinates; learned weights are not converted. Legacy "
            "checkpoints without action metadata require physical/base_l2."
            .format("; ".join(mismatches)))
