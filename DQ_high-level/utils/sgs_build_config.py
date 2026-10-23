"""Read and validate SGS bank-construction parameters without a simulator.

The builder samples targets from these ranges and checks geometry, collision and
approximate ground contact. It does not require a successful scripted grasp or
stable standing. The accepted bank need not have a uniform distribution.
"""

from collections.abc import Mapping
import math
from numbers import Real
from pathlib import Path

import yaml


DEFAULT_BUILD_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "cfg" / "sgs_bank.yaml"
SINGLE_VALUE_POSITION_TOLERANCE_M = 0.005
SINGLE_VALUE_HEADING_TOLERANCE_DEG = 1.0

_BASE_KEYS = {"x_range_m", "y_range_m", "heading_offset_range_deg"}
_FAMILY_EXTRA = {
    "full_task": "arm_pose",
    "pregrasp": "ee_approach_distance_range_m",
    "at_grasp": "gripper_opening_range",
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


def validate_build_config(mapping, lifted_threshold=None):
    """Return a detached, normalized config, rejecting ambiguous settings.

    ``lifted_threshold`` remains an optional compatibility argument; it has no
    effect because the at-grasp family does not pre-lift the object.
    """
    if isinstance(mapping, Mapping) and (
        mapping.get("schema_version") == 1 or "grasped_before_lift" in mapping
    ):
        raise ValueError(
            "Legacy SGS build config: migrate to schema_version: 2; replace "
            "grasped_before_lift/object_lift_height_range_m with "
            "at_grasp/gripper_opening_range ([0, 1], 0 = closed, 1 = fully open). "
            "The at_grasp family does not guarantee a held or lifted object."
        )
    _keys(mapping, {"schema_version", *_FAMILY_EXTRA}, "SGS build config")
    version = mapping["schema_version"]
    if type(version) is not int or version != 2:
        raise ValueError("schema_version must be the integer 2")
    result = {"schema_version": 2}
    for family, extra_key in _FAMILY_EXTRA.items():
        source = mapping[family]
        _keys(source, {"base", extra_key}, family)
        _keys(source["base"], _BASE_KEYS, f"{family}.base")
        base = {
            key: _range(source["base"][key], f"{family}.base.{key}",
                        angle=key == "heading_offset_range_deg")
            for key in ("x_range_m", "y_range_m", "heading_offset_range_deg")
        }
        if base["x_range_m"] == [0., 0.] and base["y_range_m"] == [0., 0.]:
            raise ValueError(f"{family}.base: x and y cannot both be fixed at zero")
        result[family] = {"base": base}
        if family == "full_task":
            if source[extra_key] != "default":
                raise ValueError("full_task.arm_pose must be 'default'")
            result[family][extra_key] = "default"
        elif family == "pregrasp":
            result[family][extra_key] = _range(
                source[extra_key], f"{family}.{extra_key}", positive=True
            )
        else:
            opening = _range(source[extra_key], f"{family}.{extra_key}")
            if opening[0] < 0 or opening[1] > 1:
                raise ValueError(f"{family}.{extra_key} must stay within [0, 1]")
            result[family][extra_key] = opening
    return result


def load_build_config(path=None):
    """Load a UTF-8 YAML file; the default path is independent of working dir."""
    source = DEFAULT_BUILD_CONFIG_PATH if path is None else Path(path).expanduser()
    with source.open("r", encoding="utf-8") as stream:
        try:
            mapping = yaml.load(stream, Loader=_UniqueKeySafeLoader)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid SGS build YAML {source}: {error}") from error
    return validate_build_config(mapping)
