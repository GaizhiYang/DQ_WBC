"""Read and validate SGS bank-construction parameters without a simulator.

The builder samples targets from these ranges and checks geometry, collision and
approximate ground contact. Schema 5 keeps full_task, pregrasp and physically
verified near_goal. Legacy schema 4 also includes at_grasp; schema 3 keeps the
original three geometry-only families.
Neither profile requires stable standing. Accepted states need not be uniform.
"""

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path

import yaml


DEFAULT_BUILD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "cfg" / "sgs_bank.yaml"
SINGLE_VALUE_POSITION_TOLERANCE_M = 0.005
SINGLE_VALUE_HEADING_TOLERANCE_DEG = 1.0
BUILD_FAMILIES = {
    3: ("full_task", "pregrasp", "at_grasp"),
    4: ("full_task", "pregrasp", "at_grasp", "near_goal"),
    5: ("full_task", "pregrasp", "near_goal"),
}

_BASE_KEYS = {"x_range_m", "y_range_m", "heading_offset_range_deg"}
_FAMILY_EXTRA = {
    "full_task": "arm_pose",
    "pregrasp": "ee_approach_distance_range_m",
    "at_grasp": "gripper_opening_range",
}
_PREGRASP_OFFSET_KEYS = {
    "ee_lateral_offset_range_m", "ee_vertical_offset_range_m",
    "ee_rotation_offset_range_deg",
}
_ROTATION_AXES = ("roll", "pitch", "yaw")
_VALIDATION_KEYS = {
    "physics_steps", "max_object_displacement_m", "max_contact_force_n",
}
_NEAR_GOAL_POSITIVE_KEYS = {
    "close_time_s", "lift_time_s", "hold_time_s", "max_relative_slip_m",
    "max_relative_rotation_deg", "min_finger_contact_force_n",
    "min_contact_fraction", "max_object_speed_m_s",
}


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Reject misspelled/repeated settings instead of silently overriding them."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("SGS YAML mapping keys must be hashable scalars") from error
        if duplicate:
            raise ValueError(
                f"Duplicate SGS YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _keys(value, expected, path):
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    missing = expected - value.keys()
    unknown = value.keys() - expected
    if missing or unknown:
        details = []
        if missing:
            details.append("missing keys " + ", ".join(sorted(missing)))
        if unknown:
            details.append("unknown keys " + ", ".join(sorted(map(repr, unknown))))
        raise ValueError(f"{path}: {'; '.join(details)}")


def _number(value, path):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite number (not a boolean)")
    try:
        normalized = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{path} must be a finite number") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{path} must be a finite number")
    return normalized


def _range(value, path, *, positive=False, angle=False):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must be [minimum, maximum]")
    result = [_number(item, f"{path}[{index}]") for index, item in enumerate(value)]
    minimum, maximum = result
    if minimum > maximum:
        raise ValueError(f"{path}: minimum must not exceed maximum")
    if positive and minimum <= 0:
        raise ValueError(f"{path} must be strictly positive")
    if angle and (minimum < -180 or maximum > 180):
        raise ValueError(f"{path} must stay within [-180, 180] degrees")
    return result


def _base(value, path):
    _keys(value, _BASE_KEYS, path)
    result = {
        key: _range(value[key], f"{path}.{key}",
                    angle=key == "heading_offset_range_deg")
        for key in ("x_range_m", "y_range_m", "heading_offset_range_deg")
    }
    if result["x_range_m"] == [0., 0.] and result["y_range_m"] == [0., 0.]:
        raise ValueError(f"{path}: x and y cannot both be fixed at zero")
    return result


def validate_build_config(mapping, lifted_threshold=None):
    """Return a detached, normalized config, rejecting ambiguous settings.

    ``lifted_threshold`` is used for schemas 4 and 5: a near-goal state must be
    strictly below the task's success height (default 0.2 m). Schema 3 does not
    pre-lift objects and keeps its earlier threshold-independent behavior.
    """
    if isinstance(mapping, Mapping) and (
        (type(mapping.get("schema_version")) is int and mapping["schema_version"] in (1, 2))
        or "grasped_before_lift" in mapping
    ):
        raise ValueError(
            "Legacy SGS build config: migrate to schema_version: 3; add object, validation, "
            "and pregrasp.ee_lateral_offset_range_m / ee_vertical_offset_range_m / "
            "ee_rotation_offset_range_deg. Use zero yaw/pose offsets and physics_steps: 1 "
            "if preserving the earlier target distribution. Replace any "
            "grasped_before_lift/object_lift_height_range_m with "
            "at_grasp/gripper_opening_range; at_grasp does not guarantee a held or lifted object."
        )
    if not isinstance(mapping, Mapping):
        raise ValueError("SGS build config must be a mapping")
    if "schema_version" not in mapping:
        raise ValueError("SGS build config: missing keys schema_version")
    version = mapping["schema_version"]
    if type(version) is not int or version not in BUILD_FAMILIES:
        raise ValueError("schema_version must be the integer 3, 4 or 5")
    families = BUILD_FAMILIES[version]
    root_keys = {"schema_version", "object", "validation", *families}
    _keys(mapping, root_keys, "SGS build config")
    _keys(mapping["object"], {"yaw_offset_range_deg"}, "object")
    _keys(mapping["validation"], _VALIDATION_KEYS, "validation")
    steps = mapping["validation"]["physics_steps"]
    if type(steps) is not int or steps <= 0:
        raise ValueError("validation.physics_steps must be a positive integer")
    validation = {"physics_steps": steps}
    for key in ("max_object_displacement_m", "max_contact_force_n"):
        value = _number(mapping["validation"][key], f"validation.{key}")
        if value <= 0:
            raise ValueError(f"validation.{key} must be strictly positive")
        validation[key] = value
    result = {
        "schema_version": version,
        "object": {"yaw_offset_range_deg": _range(
            mapping["object"]["yaw_offset_range_deg"], "object.yaw_offset_range_deg", angle=True)},
        "validation": validation,
    }
    for family, extra_key in _FAMILY_EXTRA.items():
        if family not in families:
            continue
        source = mapping[family]
        extra_keys = _PREGRASP_OFFSET_KEYS if family == "pregrasp" else set()
        _keys(source, {"base", extra_key, *extra_keys}, family)
        result[family] = {"base": _base(source["base"], f"{family}.base")}
        if family == "full_task":
            if source[extra_key] != "default":
                raise ValueError("full_task.arm_pose must be 'default'")
            result[family][extra_key] = "default"
        elif family == "pregrasp":
            result[family][extra_key] = _range(
                source[extra_key], f"{family}.{extra_key}", positive=True
            )
            for key in ("ee_lateral_offset_range_m", "ee_vertical_offset_range_m"):
                result[family][key] = _range(source[key], f"{family}.{key}")
            rotations = source["ee_rotation_offset_range_deg"]
            _keys(rotations, set(_ROTATION_AXES), "pregrasp.ee_rotation_offset_range_deg")
            result[family]["ee_rotation_offset_range_deg"] = {}
            for axis in _ROTATION_AXES:
                path = f"pregrasp.ee_rotation_offset_range_deg.{axis}"
                bounds = _range(rotations[axis], path, angle=True)
                if bounds[0] < -45 or bounds[1] > 45:
                    raise ValueError(f"{path} must stay within [-45, 45] degrees")
                result[family]["ee_rotation_offset_range_deg"][axis] = bounds
            def closest_to_zero(bounds):
                return 0. if bounds[0] <= 0 <= bounds[1] else min(map(abs, bounds))

            lateral = closest_to_zero(result[family]["ee_lateral_offset_range_m"])
            vertical = closest_to_zero(result[family]["ee_vertical_offset_range_m"])
            distance_max = result[family][extra_key][1]
            if math.hypot(lateral, vertical) >= distance_max:
                raise ValueError(
                    "pregrasp lateral/vertical offsets have no feasible combination: "
                    "minimum transverse distance must be smaller than max(distance)"
                )
        else:
            opening = _range(source[extra_key], f"{family}.{extra_key}")
            if opening[0] < 0 or opening[1] > 1:
                raise ValueError(f"{family}.{extra_key} must stay within [0, 1]")
            result[family][extra_key] = opening
    if "near_goal" in families:
        source = mapping["near_goal"]
        _keys(source, {"base", "object_lift_height_range_m", *_NEAR_GOAL_POSITIVE_KEYS},
              "near_goal")
        near_goal = {"base": _base(source["base"], "near_goal.base")}
        near_goal["object_lift_height_range_m"] = _range(
            source["object_lift_height_range_m"], "near_goal.object_lift_height_range_m",
            positive=True)
        threshold = _number(.2 if lifted_threshold is None else lifted_threshold,
                            "lifted_threshold")
        if threshold <= 0:
            raise ValueError("lifted_threshold must be strictly positive")
        if near_goal["object_lift_height_range_m"][1] >= threshold:
            raise ValueError(
                "near_goal.object_lift_height_range_m upper bound must be strictly below "
                "lifted_threshold; near-goal resets must start before task success")
        for key in _NEAR_GOAL_POSITIVE_KEYS:
            value = _number(source[key], f"near_goal.{key}")
            if value <= 0:
                raise ValueError(f"near_goal.{key} must be strictly positive")
            near_goal[key] = value
        if near_goal["hold_time_s"] < .1:
            raise ValueError("near_goal.hold_time_s must be at least 0.1 seconds")
        if near_goal["min_contact_fraction"] > 1:
            raise ValueError("near_goal.min_contact_fraction must be within (0, 1]")
        if near_goal["max_relative_rotation_deg"] > 180:
            raise ValueError("near_goal.max_relative_rotation_deg must be at most 180 degrees")
        result["near_goal"] = near_goal
    return result


def load_build_config(path=None, lifted_threshold=None):
    """Load a UTF-8 YAML file; the default path is independent of working dir."""
    source = DEFAULT_BUILD_CONFIG_PATH if path is None else Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        try:
            mapping = yaml.load(stream, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid SGS build YAML {source}: {error}") from error
    return validate_build_config(mapping, lifted_threshold=lifted_threshold)
