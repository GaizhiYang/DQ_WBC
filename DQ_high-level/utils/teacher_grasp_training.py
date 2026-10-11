"""Validate and checkpoint the original teacher's grasp-selection experiment."""
from copy import deepcopy
import math
import re
from pathlib import Path


# Reconstruct the same observation/action convention when loading a checkpoint.
ENVIRONMENT_OPTIONS = ("task", "roboinfo", "observe_gait_commands", "no_feature",
                       "last_commands", "pitch_control", "use_tanh", "mask_arm",
                       "control_freq", "near_goal_stop", "obj_move_prob", "rand_control",
                       "arm_delay", "rand_cmd_scale", "rand_depth_clip", "stop_pick",
                       "table_height", "small_value_set_zero", "seed")

GEOMETRIC_OPTIONS = ("switch_margin", "center_weight", "topdown_weight",
                     "height_weight", "table_clearance", "lock_distance")


def resolve_selection(args, cfg, checkpoint=None):
    saved = None
    if checkpoint is not None:
        if "teacher_vision_state" in checkpoint or "asymmetric_teacher_state" in checkpoint:
            raise ValueError("Use a GFM, KARL or geometric privileged-teacher checkpoint for this experiment")
        state = checkpoint.get("grasp_selection_state")
        if state is not None:
            if state.get("version") != 1:
                raise ValueError("Unsupported grasp-selection checkpoint version")
            saved = state["settings"]
            cfg = deepcopy(state["experiment_config"])
            for name, value in state["environment_options"].items():
                setattr(args, name, value)
        else:
            if "query_proj.weight" not in checkpoint.get("policy", {}):
                raise ValueError("Unrecognized teacher checkpoint (missing GFM or grasp-selection metadata)")
            saved = {"mode": "gfm", "switch_margin_deg": 30.0, "orientation_preference": "none"}
    defaults = {"mode": "gfm", "switch_margin_deg": 30.0, "orientation_preference": "none"}
    settings = deepcopy(saved) if saved is not None else dict(defaults, **deepcopy(cfg.get("grasp_selection", {})))
    for arg, key in (("grasp_selector", "mode"), ("karl_switch_margin_deg", "switch_margin_deg"),
                     ("karl_orientation_preference", "orientation_preference")):
        value = getattr(args, arg, None)
        if value is not None:
            if saved is not None and value != saved[key]:
                raise ValueError("--%s conflicts with checkpoint grasp selection; start a new experiment" % arg)
            settings[key] = value
    if settings["mode"] not in ("gfm", "karl", "geometric") or settings["orientation_preference"] not in ("none", "karl"):
        raise ValueError("Invalid grasp-selection configuration")
    saved_drop = settings.get("actor_drop_velocity_obs", False)
    if not isinstance(saved_drop, bool):
        raise ValueError("actor_drop_velocity_obs must be a boolean")
    override_drop = getattr(args, "actor_drop_velocity_obs", None)
    if override_drop is not None:
        if not isinstance(override_drop, bool):
            raise ValueError("actor_drop_velocity_obs must be a boolean")
        if saved is not None and override_drop != saved_drop:
            raise ValueError("Actor velocity observation option conflicts with checkpoint; start a new experiment")
        saved_drop = override_drop
    if saved_drop:
        if settings["mode"] != "gfm":
            raise ValueError("--actor_drop_velocity_obs requires --grasp_selector gfm")
        if args.task != "B1Z1PickMulti" or not args.roboinfo:
            raise ValueError("Actor velocity ablation requires --task B1Z1PickMulti and --roboinfo")
        if args.no_feature or args.last_commands or cfg["env"].get("lastCommands", False) or args.pitch_control:
            raise ValueError("Actor velocity ablation requires feature observations, last actions and 9 controls; omit --no_feature/--last_commands/--pitch_control")
        if cfg.get("sensor", {}).get("enableCamera", False):
            raise ValueError("Actor velocity ablation requires sensor.enableCamera: false")
        settings["actor_drop_velocity_obs"] = True
    else:
        # Preserve old settings dictionaries so existing checkpoints still
        # load exactly. Absence of this key denotes the original Actor.
        settings.pop("actor_drop_velocity_obs", None)
    args.actor_drop_velocity_obs = saved_drop
    if checkpoint is not None and settings["mode"] == "gfm":
        weight = checkpoint.get("policy", {}).get("net.0.weight")
        if weight is not None and (saved_drop and weight.shape[1] != 201 or not saved_drop and weight.shape[1] == 201):
            raise ValueError("Checkpoint Actor input shape conflicts with velocity-observation metadata")
    margin = settings["switch_margin_deg"]
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("--karl_switch_margin_deg must be finite and non-negative")
    if settings["mode"] != "karl" and any(getattr(args, key, None) is not None for key in
                                          ("karl_switch_margin_deg", "karl_orientation_preference")):
        raise ValueError("KARL-specific options require --grasp_selector karl")
    if settings["mode"] == "geometric":
        from modules.geometric_grasp_selector import GEOMETRIC_DEFAULTS, validate_geometric_settings

        if saved is None:
            geometric = deepcopy(GEOMETRIC_DEFAULTS)
            geometric.update(settings.get("geometric", {}))
        else:
            geometric = deepcopy(saved.get("geometric", {}))
            if set(geometric) != set(GEOMETRIC_DEFAULTS):
                raise ValueError("Incomplete or unsupported geometric checkpoint settings")
        for key in GEOMETRIC_OPTIONS:
            arg = "geometric_" + key
            value = getattr(args, arg, None)
            if value is not None:
                if saved is not None and value != geometric[key]:
                    raise ValueError("--%s conflicts with checkpoint grasp selection; start a new experiment" % arg)
                geometric[key] = value
        settings["geometric"] = validate_geometric_settings(geometric)
    elif any(getattr(args, "geometric_" + key, None) is not None for key in GEOMETRIC_OPTIONS):
        raise ValueError("Geometric-specific options require --grasp_selector geometric")
    if settings["mode"] in ("karl", "geometric"):
        label = "KARL" if settings["mode"] == "karl" else "Geometric"
        if not args.roboinfo:
            raise ValueError("%s teacher comparison requires --roboinfo" % label)
        if args.no_feature or args.last_commands or cfg["env"].get("lastCommands", False) or args.pitch_control:
            raise ValueError("%s comparison requires feature observations, last actions and 9 controls; omit --no_feature/--last_commands/--pitch_control" % label)
        if args.task != "B1Z1PickMulti" or cfg.get("sensor", {}).get("enableCamera", False):
            raise ValueError("%s comparison requires --task B1Z1PickMulti and sensor.enableCamera: false" % label)
    if getattr(args, "vis_selected_grasp", False):
        if settings["mode"] not in ("karl", "geometric"):
            raise ValueError("--vis_selected_grasp requires --grasp_selector karl or geometric (or a matching checkpoint)")
        if getattr(args, "headless", False):
            raise ValueError("--vis_selected_grasp needs a viewer; omit --headless")
        if getattr(args, "grasp_vis_envs", 8) <= 0:
            raise ValueError("--grasp_vis_envs must be a positive integer")
    cfg["grasp_selection"] = deepcopy(settings)
    return cfg, settings


def checkpoint_step(checkpoint, path):
    if checkpoint and "grasp_selection_state" in checkpoint:
        return int(checkpoint["grasp_selection_state"]["global_step"])
    match = re.fullmatch(r"agent_(\d+)\.pt", Path(path).name)
    if not match:
        raise ValueError("Legacy teacher checkpoint needs an agent_<step>.pt filename to recover training progress")
    return int(match.group(1))


class TeacherGraspTrainingState:
    def __init__(self, env, settings, cfg, args):
        self.env = env
        self.settings = deepcopy(settings)
        self.cfg = deepcopy(cfg)
        self.options = {key: getattr(args, key) for key in ENVIRONMENT_OPTIONS}

    def state_dict(self):
        return {"version": 1, "settings": deepcopy(self.settings),
                "experiment_config": deepcopy(self.cfg),
                "environment_options": deepcopy(self.options),
                "global_step": int(self.env.global_step_counter)}

    def load_state_dict(self, state):
        if state.get("version") != 1 or state["settings"] != self.settings:
            raise ValueError("Incompatible grasp-selection checkpoint")
        self.env._env.global_step_counter = int(state["global_step"])
        # Physics and rollouts are not serialized: each selector initializes
        # its target again when the fresh episode observations are packed.
