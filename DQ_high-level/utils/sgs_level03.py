"""Tensor state for reproducible Level03 SGS starts, without simulator imports.

A bank stores a sampled driver's intention together with both smoothing stages.
Only future random corrections are resampled; restoring a configuration must not
resample its initial direction or forget the filter state. The current builder
uses ``prepare_rest_motion`` so geometry and grasp configurations start at rest.
"""

import math

import torch

from utils.table_motion import RemoteCarDriver


REST_PLANAR_TOLERANCE = 1e-5


MOTION_FLOAT_NAMES = ("speed_stage", "speed", "curvature_stage", "curvature",
                      "target_speed", "target_curvature")
MOTION_BOOL_NAMES = ("command_changed", "manual_pending")
MOTION_STATE_DTYPES = {
    **{"level03_motion_" + name: torch.float32 for name in MOTION_FLOAT_NAMES},
    **{"level03_motion_" + name: torch.bool for name in MOTION_BOOL_NAMES},
    **{"level03_driver_" + name: torch.float32
       for name in RemoteCarDriver.float_state_names},
    **{"level03_driver_" + name: torch.bool
       for name in ("initialized",) + RemoteCarDriver.event_names},
    "level03_driver_phase": torch.long,
}
MOTION_STATE_PREFIXES = ("level03_motion_", "level03_driver_",
                         "level03_x_motion_", "level03_x_driver_")


def _axis_motions(motion):
    yield "level03_", motion
    if motion.x_motion is not None:
        yield "level03_x_", motion.x_motion


def validated_bank_motion_config(settings):
    """Normalize an immutable bank's complete motion config, including old banks.

    Banks written before the driving-speed floor or X motion existed omit
    minSpeed or translationAxes; they retain their zero-floor/local-Y semantics. Other
    missing keys still indicate incomplete provenance and must not be filled
    silently from today's defaults.
    """
    from utils.config import level03_motion_config

    if not isinstance(settings, dict):
        raise ValueError("Level03 bank requires level03_motion_config")
    normalized = dict(settings)
    if "minSpeed" not in normalized and "maxSpeed" in normalized:
        normalized["minSpeed"] = 0.0
    if "translationAxes" not in normalized and "maxSpeed" in normalized:
        normalized["translationAxes"] = "y"
    validated = level03_motion_config({"level03Motion": normalized})
    if normalized != validated:
        raise ValueError("Level03 bank requires complete level03_motion_config")
    return validated


def _state_tensors(motion):
    result = {}
    for prefix, axis in _axis_motions(motion):
        result.update({prefix + "motion_" + name: getattr(axis, name)
                       for name in MOTION_FLOAT_NAMES})
        result[prefix + "motion_command_changed"] = axis.command_changed
        result[prefix + "motion_manual_pending"] = axis._manual_pending
        result.update({prefix + "driver_" + name: getattr(axis.driver, name)
                       for name in RemoteCarDriver.state_names})
    return result


def prepare_rest_motion(motion, dt):
    """Sample initial intentions, then clear all physical/input filter histories.

This leaves a known forward/reverse intention for heading-aware geometry
sampling. No simulator time passes and the actual table starts with zero speed,
curvature and acceleration; the ordinary controller supplies the smooth start.
"""
    dt = float(dt)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Level03 SGS initialization dt must be finite and positive")
    for _, axis in _axis_motions(motion):
        axis.driver_enabled = True
    motion.reset()
    motion.step(dt)
    for _, axis in _axis_motions(motion):
        for name in MOTION_FLOAT_NAMES:
            getattr(axis, name).zero_()
        axis.driver.input_speed.zero_()
        axis.driver.input_curvature.zero_()
        axis.command_changed.zero_()
        axis._manual_pending.zero_()
        for name in axis.driver.event_names:
            getattr(axis.driver, name).zero_()


def snapshot_motion(motion):
    """Copy every per-environment automatic-driver state into flat bank fields."""
    if not all(axis.driver_enabled for _, axis in _axis_motions(motion)):
        raise ValueError("Level03 SGS snapshots require the automatic driver")
    return {name: value.detach().clone() for name, value in _state_tensors(motion).items()}


def restore_motion(motion, state, env_ids=None):
    """Restore bank rows to all slots or to the specified slots, in their order.

``state`` contains already-selected rows (e.g. ``bank.batch(config_ids)``), not
the entire bank. Values are validated once when loading the bank; the hot reset
path checks only structure before any writes and transfers devices if needed.
"""
    destinations = _state_tensors(motion)
    supplied = {name for name in state if name.startswith(MOTION_STATE_PREFIXES)}
    if supplied != set(destinations):
        raise ValueError("Level03 SGS motion fields do not match translation axes; missing={}, unknown={}".format(
            sorted(set(destinations) - supplied), sorted(supplied - set(destinations))))
    if env_ids is None:
        indices, count = slice(None), motion.num_envs
    else:
        indices = torch.as_tensor(env_ids, device=motion.device)
        if indices.dtype != torch.long or indices.ndim != 1:
            raise ValueError("Level03 SGS env_ids must be a one-dimensional int64 tensor")
        count = indices.numel()
    for name, destination in destinations.items():
        value = state.get(name)
        if not isinstance(value, torch.Tensor):
            raise ValueError("missing Level03 SGS tensor: " + name)
        if tuple(value.shape) != (count,):
            raise ValueError("invalid Level03 SGS tensor shape: " + name)
        if value.dtype != destination.dtype:
            raise ValueError("invalid Level03 SGS tensor dtype: " + name)
    # Clone before writes so a source view into the same motion cannot be
    # corrupted by a partial permutation or by an overlapping tensor alias.
    selected = {name: state[name].to(device=value.device).clone()
                for name, value in destinations.items()}
    for name, destination in destinations.items():
        destination[indices] = selected[name]
    for _, axis in _axis_motions(motion):
        axis.driver_enabled = True


def validate_motion_state(tensors, settings):
    """Validate every configured axis and reject state from another motion mode."""
    axes = settings.get("translationAxes", "y")
    if axes not in ("y", "xy"):
        raise ValueError("invalid Level03 SGS translationAxes")
    prefixes = ("level03_", "level03_x_") if axes == "xy" else ("level03_",)
    expected = {prefix + key[len("level03_"):]
                for prefix in prefixes for key in MOTION_STATE_DTYPES}
    fields = {key for key in tensors if key.startswith(MOTION_STATE_PREFIXES)}
    if fields != expected:
        raise ValueError("invalid Level03 SGS fields; missing={}, unknown={}".format(
            sorted(expected - fields), sorted(fields - expected)))
    first = tensors["level03_motion_speed"]
    for prefix in prefixes:
        axis_state = {key: tensors[prefix + key[len("level03_"):]]
                      for key in MOTION_STATE_DTYPES}
        _validate_axis_motion_state(axis_state, settings)
        if axis_state["level03_motion_speed"].shape != first.shape:
            raise ValueError("Level03 SGS axes must share tensor shapes")
        if axis_state["level03_motion_speed"].device != first.device:
            raise ValueError("Level03 SGS tensors must share a device")


def _validate_axis_motion_state(tensors, settings):
    """Reject incomplete, malformed or out-of-range Level03 motion state.

This accepts authentic running states as well as sampled stationary starts.
The bank's declared reset mode additionally controls whether nonzero motion is
permitted. Expired clocks may be negative (the driver handles them next tick),
so timers are checked for finiteness and their sampled upper limits.
"""
    fields = {key for key in tensors if key.startswith(("level03_motion_", "level03_driver_"))}
    expected = set(MOTION_STATE_DTYPES)
    if fields != expected:
        missing, unknown = sorted(expected - fields), sorted(fields - expected)
        raise ValueError("invalid Level03 SGS fields; missing={}, unknown={}".format(missing, unknown))
    first = tensors["level03_motion_speed"]
    if not isinstance(first, torch.Tensor) or first.ndim != 1 or first.numel() == 0:
        raise ValueError("invalid Level03 SGS tensor shape: level03_motion_speed")
    for name, dtype in MOTION_STATE_DTYPES.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.shape != first.shape:
            raise ValueError("invalid Level03 SGS tensor shape: " + name)
        if value.dtype != dtype:
            raise ValueError("invalid Level03 SGS tensor dtype: " + name)
        if value.device != first.device:
            raise ValueError("Level03 SGS tensors must share a device")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError("nonfinite Level03 SGS tensor: " + name)

    speed_limit = float(settings["maxSpeed"])
    yaw_limit = float(settings["maxYawRate"])
    if not math.isfinite(speed_limit) or not math.isfinite(yaw_limit) or min(speed_limit, yaw_limit) <= 0:
        raise ValueError("Level03 SGS motion limits must be finite and positive")
    minimum_speed = float(settings.get("minSpeed", 0.0))
    if not math.isfinite(minimum_speed) or not 0 <= minimum_speed <= speed_limit:
        raise ValueError("Level03 SGS minSpeed must be finite and within [0, maxSpeed]")
    curvature_limit = yaw_limit / speed_limit

    def bounded(name, lower, upper):
        value = tensors[name]
        if not bool(((value >= lower - 1e-6) & (value <= upper + 1e-6)).all()):
            raise ValueError("out-of-range Level03 SGS tensor: " + name)

    for name in MOTION_FLOAT_NAMES:
        limit = curvature_limit if "curvature" in name else speed_limit
        bounded("level03_motion_" + name, -limit, limit)
    bounded("level03_driver_input_speed", -speed_limit, speed_limit)
    bounded("level03_driver_input_curvature", -curvature_limit, curvature_limit)
    for name in ("cruise_throttle", "throttle_goal"):
        bounded("level03_driver_" + name, 0., 1.)
    for name in ("steering_reference", "steering_goal"):
        bounded("level03_driver_" + name, -1., 1.)
    for name, key in (("throttle_time_left", "throttleUpdateTime"),
                      ("steering_time_left", "steeringUpdateTime"),
                      ("turn_time_left", "turnHoldTime")):
        bounded("level03_driver_" + name, -float("inf"), float(settings[key][1]))
    bounded("level03_driver_phase_time_left", -float("inf"),
            max(float(settings["driveDuration"][1]), float(settings["stopHoldTime"][1])))

    phase = tensors["level03_driver_phase"]
    if not bool(((phase >= RemoteCarDriver.DRIVE) & (phase <= RemoteCarDriver.PAUSE)).all()):
        raise ValueError("invalid Level03 SGS driver phase")
    initialized = tensors["level03_driver_initialized"]
    driving = initialized & (phase == RemoteCarDriver.DRIVE)
    for name in ("cruise_throttle", "throttle_goal"):
        speed_goal = tensors["level03_driver_" + name][driving] * speed_limit
        if not bool((speed_goal >= minimum_speed - 1e-6).all()):
            raise ValueError("Level03 SGS driving target below minSpeed: " + name)
    for name in ("direction", "pending_direction"):
        value = tensors["level03_driver_" + name]
        expected_direction = torch.where(initialized, torch.ones_like(value), torch.zeros_like(value))
        if not bool((value.abs() == expected_direction).all()):
            raise ValueError("invalid Level03 SGS driver " + name)
    if bool(tensors["level03_motion_manual_pending"].any()):
        raise ValueError("Level03 SGS state must use the automatic driver")
    # Before initialization all driver tensors must be zero; otherwise a reset
    # could accidentally carry previous-episode input or a pending event.
    for name in RemoteCarDriver.state_names:
        if bool(tensors["level03_driver_" + name][~initialized].any()):
            raise ValueError("uninitialized Level03 SGS driver has nonzero state: " + name)

