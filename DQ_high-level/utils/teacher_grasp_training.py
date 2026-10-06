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


def resolve_selection(args, cfg, checkpoint=None):
    saved = None
    if checkpoint is not None:
        if "teacher_vision_state" in checkpoint or "asymmetric_teacher_state" in checkpoint:
            raise ValueError("Use an original GFM or KARL privileged-teacher checkpoint for this experiment")
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
    settings = deepcopy(saved or cfg.get("grasp_selection", {
        "mode": "gfm", "switch_margin_deg": 30.0, "orientation_preference": "none"}))
    for arg, key in (("grasp_selector", "mode"), ("karl_switch_margin_deg", "switch_margin_deg"),
                     ("karl_orientation_preference", "orientation_preference")):
        value = getattr(args, arg, None)
        if value is not None:
            if saved is not None and value != saved[key]:
                raise ValueError("--%s conflicts with checkpoint grasp selection; start a new experiment" % arg)
            settings[key] = value
    if settings["mode"] not in ("gfm", "karl") or settings["orientation_preference"] not in ("none", "karl"):
        raise ValueError("Invalid grasp-selection configuration")
    margin = settings["switch_margin_deg"]
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("--karl_switch_margin_deg must be finite and non-negative")
    if settings["mode"] == "gfm" and any(getattr(args, key, None) is not None for key in
                                         ("karl_switch_margin_deg", "karl_orientation_preference")):
        raise ValueError("KARL-specific options require --grasp_selector karl")
    if settings["mode"] == "karl":
        if not args.roboinfo:
            raise ValueError("KARL teacher comparison requires --roboinfo")
        if args.no_feature or args.last_commands or cfg["env"].get("lastCommands", False) or args.pitch_control:
            raise ValueError("KARL comparison requires feature observations, last actions and 9 controls; omit --no_feature/--last_commands/--pitch_control")
        if args.task != "B1Z1PickMulti" or cfg.get("sensor", {}).get("enableCamera", False):
            raise ValueError("KARL comparison requires --task B1Z1PickMulti and sensor.enableCamera: false")
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
        # Physics and rollouts are not serialized: new episodes start at index 0.
