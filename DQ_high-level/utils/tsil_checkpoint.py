"""Self-contained temporal-teacher inference metadata (not training resume)."""

from copy import deepcopy
import math

VALUE_NORMALIZATION_SCHEMA = "affine_output_v2"
CONFIGURATION_DESCRIPTOR_SCHEMA = "dq_full_task_planar_v2"
VALUE_ADAPTER_BUFFERS = {"prediction_scale", "prediction_bias", "standard_deviation_floor"}


OBSERVATION_ARGUMENTS = (
    "roboinfo", "observe_gait_commands", "small_value_set_zero", "no_feature",
    "mask_arm", "pitch_control", "rand_control", "arm_delay", "rand_cmd_scale",
    "rand_depth_clip", "stop_pick", "use_tanh", "last_commands", "no_velocity_obs",
    "future_pose_obs", "future_pose_horizons", "ekf_history", "ekf_q_scale", "ekf_r_scale",
    "control_freq",
)


class TemporalCheckpointMetadata:
    def __init__(self, config, arguments, observation_dim, t_ref):
        self.data = {
            "version": 2, "checkpoint_kind": "temporal_teacher_evaluation",
            "value_normalization": VALUE_NORMALIZATION_SCHEMA,
            "configuration_descriptor": CONFIGURATION_DESCRIPTOR_SCHEMA,
            "config": deepcopy(config),
            "arguments": {name: deepcopy(getattr(arguments, name, None))
                          for name in OBSERVATION_ARGUMENTS},
            "observation_dim": int(observation_dim), "time_indices": [-191, -190],
            "time_reference": float(t_ref),
        }

    def state_dict(self):
        return deepcopy(self.data)

    def load_state_dict(self, state):
        if state.get("version") not in (1, 2):
            raise ValueError("Unsupported TSIL checkpoint metadata version")
        if state["version"] == 2:
            _validate_current_metadata(state)
        for key in ("observation_dim", "time_indices"):
            if state[key] != self.data[key]:
                raise ValueError(f"TSIL checkpoint {key} does not match the environment")
        if not math.isclose(state["time_reference"], self.data["time_reference"], abs_tol=1e-6):
            raise ValueError("TSIL checkpoint time reference does not match the environment")
        saved_env, current_env = state["config"]["env"], self.data["config"]["env"]
        for key in ("posePrediction", "noVelocityObs", "holdSteps", "controlFrequencyInv",
                    "controlFrequencyLow", "asset", "liftedSuccessThreshold"):
            if saved_env.get(key) != current_env.get(key):
                raise ValueError(f"TSIL checkpoint env.{key} does not match the environment")
        if state["config"].get("sim", {}).get("dt") != self.data["config"].get("sim", {}).get("dt"):
            raise ValueError("TSIL checkpoint simulator timestep does not match the environment")
        self.data = deepcopy(state)
        if state["version"] == 1:
            # Legacy loading is inference-only; TSILPPO performs the explicit
            # value adapter initialization and TemporalState migrates geometry.
            self.data.update(version=2, migrated_from_version=1, critic_available=False,
                             value_normalization=VALUE_NORMALIZATION_SCHEMA,
                             configuration_descriptor=CONFIGURATION_DESCRIPTOR_SCHEMA)


def _validate_current_metadata(metadata):
    if metadata.get("value_normalization") != VALUE_NORMALIZATION_SCHEMA:
        raise ValueError("TSIL checkpoint value normalization schema is missing or incompatible")
    if metadata.get("configuration_descriptor") != CONFIGURATION_DESCRIPTOR_SCHEMA:
        raise ValueError("TSIL checkpoint configuration descriptor schema is missing or incompatible")


def read_temporal_checkpoint(path):
    """Read a local trusted checkpoint before constructing observation spaces."""
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Teacher checkpoint must contain named model states")
    temporal = "tsil_state" in payload or "tsil_metadata" in payload
    if not temporal:
        # A damaged temporal checkpoint must not silently become a baseline.
        scaler = payload.get("state_preprocessor", {})
        if any("pending" in key or "time_indices" in key for key in scaler):
            raise ValueError("Temporal checkpoint is missing TSIL state/metadata")
        return None
    required = {"policy", "value", "state_preprocessor", "value_preprocessor",
                "tsil_state", "tsil_metadata", "reward_normalizer"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Incomplete TSIL checkpoint: missing {sorted(missing)}")
    metadata = payload["tsil_metadata"]
    if metadata.get("version") not in (1, 2):
        raise ValueError("Unsupported TSIL checkpoint metadata version")
    if metadata["version"] == 2:
        _validate_current_metadata(metadata)
        temporal_state = payload["tsil_state"]
        if (temporal_state.get("version") != 2 or
                temporal_state.get("descriptor_schema") != CONFIGURATION_DESCRIPTOR_SCHEMA):
            raise ValueError("TSIL checkpoint descriptor state does not match metadata schema/version")
        if not VALUE_ADAPTER_BUFFERS.issubset(payload["value_preprocessor"]):
            raise ValueError("Incomplete TSIL checkpoint: missing value prediction adapter buffers")
    else:
        adapter_keys = VALUE_ADAPTER_BUFFERS.intersection(payload["value_preprocessor"])
        if adapter_keys and adapter_keys != VALUE_ADAPTER_BUFFERS:
            raise ValueError("Incomplete TSIL checkpoint: partial value prediction adapter")
    return metadata


def restore_temporal_arguments(args, metadata):
    if getattr(args, "tsil", None) is False:
        raise ValueError("A temporal teacher checkpoint requires TSIL time observations")
    for name in OBSERVATION_ARGUMENTS:
        if name in metadata["arguments"]:
            setattr(args, name, deepcopy(metadata["arguments"][name]))
    args.tsil = True


def reject_temporal_student_teacher(path):
    if path and read_temporal_checkpoint(path) is not None:
        raise ValueError(
            "This student entrypoint does not implement TSIL teacher time conditioning "
            "and frozen preprocessing. Use eval_sgs_teacher.py to evaluate this teacher; "
            "temporal teacher distillation requires adapting the student entrypoint first.")
