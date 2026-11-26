from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
from copy import deepcopy
import torch
import os
import json
import re
from pathlib import Path
from termcolor import cprint
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from envs import *    #import b1z1_pickmulti.py
from PredictPiontTransform.predictpoint_mul import PredictPoint
from utils.config import (load_cfg, get_params, copy_cfg, configure_teacher_experiment,
                          teacher_run_end, apply_pose_prediction_config, apply_actor_history_config,
                          apply_action_scaling_config, apply_reward_options, apply_margin_config,
                          apply_ee_delta_limits)
import utils.wrapper as wrapper
import time
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from modules.predictattention import PredictAttentionSelector, teacher_gfm_kwargs
from utils.tsil_checkpoint import (TemporalCheckpointMetadata, read_temporal_checkpoint,
                                   restore_temporal_arguments)
# Re-exported: rollout scripts import get_predict_point from this entry point.
from utils.grasp_candidates import get_predict_point


def checkpoint_step(path, required=False):
    match = re.search(r"_(\d+)\.pt$", str(path))
    if match:
        return int(match.group(1))
    if required:
        raise ValueError("Training resume requires a numbered agent_<step>.pt checkpoint")
    return 0


def restore_eval_arguments(args):
    """Use the saved observation/control flags when evaluating a new-format run."""
    run_dir = Path(args.checkpoint).resolve().parent.parent
    metadata_path = run_dir / "teacher_run.json"
    if not metadata_path.is_file():
        return
    with metadata_path.open() as stream:
        metadata = json.load(stream)
    saved = metadata.get("arguments", {})
    for name in ("roboinfo", "observe_gait_commands", "small_value_set_zero", "no_feature",
                 "mask_arm", "pitch_control", "rand_control", "arm_delay", "rand_cmd_scale",
                 "rand_depth_clip", "stop_pick", "use_tanh", "last_commands", "no_velocity_obs",
                 "critic_time_obs", "actor_history_frames", "action_scaling", "critic_reward_state"):
        if name in saved:
            setattr(args, name, saved[name])


def teacher_config_path(args, is_eval=False, strict_eval=False):
    if args.cfg_file:
        return args.cfg_file
    if is_eval and args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent.parent
        metadata_path = run_dir / "teacher_run.json"
        if metadata_path.is_file():
            with metadata_path.open() as stream:
                config_name = json.load(stream).get("config_file", "DQ_teacher.yaml")
            candidate = run_dir / Path(config_name).name
            if candidate.is_file():
                return str(candidate)
        candidate = run_dir / "DQ_teacher.yaml"
        if candidate.is_file():
            return str(candidate)
    if strict_eval:
        raise ValueError("No saved teacher YAML found beside the checkpoint; specify --cfg_file explicitly")
    return "data/cfg/DQ_teacher.yaml"

def create_env(cfg, args, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor ):
    apply_pose_prediction_config(cfg, args)
    history_frames = apply_actor_history_config(cfg, args)
    cfg_terrain = B1Z1RoughCfg()

    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = args.small_value_set_zero
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)
    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-0.85, 0, 0.55)
    # _env is actually an instance of an environment class, the name of which is determined by args.task.
    # _env is the entity of the B1Z1Pickmulti 
    _env = eval(args.task)(cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device, 
                         graphics_device_id=args.graphics_device_id, headless=args.headless, 
                         use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands, no_feature=args.no_feature, mask_arm=args.mask_arm, pitch_control=args.pitch_control,
                         rand_control=args.rand_control, arm_delay=args.arm_delay, robot_start_pose=robot_start_pose,
                         rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip, stop_pick=args.stop_pick, table_height=args.table_height, eval=args.eval,
                         camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor, cfg_terrain = cfg_terrain)
    # the step under is just to add some property to _env , just a small extension of _env(B1Z1PickMulti class)
    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env, actor_history_frames=history_frames)
    return wrapped_env

def margin_token_slice(body_width, notrain_obs_num, num_actions, num_features, margin_index,
                       margin_tokens, margin_token_dim, margin_scalars):
    """Slice of the margin tokens inside the state block, or None without them.

    margin_index counts from the end of the frame (like critic_time_index); the
    block holds margin_scalars values, then margin_tokens x margin_token_dim.
    """
    if margin_index is None:
        return None
    width = margin_scalars + margin_tokens * margin_token_dim
    position = body_width + margin_index + notrain_obs_num + num_actions
    if (num_features <= 0 or margin_tokens <= 0 or margin_token_dim <= 0 or margin_scalars < 0
            or position < 6 or position + width > body_width):
        raise ValueError("margin_index/tokens must point inside the teacher state block")
    return slice(position + margin_scalars, position + width)


def margin_tokens_from(state_block, token_slice, num_tokens, token_dim):
    if token_slice is None:
        return None
    return state_block[..., token_slice].reshape(-1, num_tokens, token_dim)


# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model, PredictAttentionSelector):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor,
                  use_tanh=False, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", deterministic=False,
                 num_future_pose_obs=0, num_tsil_obs=0, critic_time_index=None, actor_history_frames=1,
                 action_scaling="legacy", critic_state_index=None, critic_state_width=0,
                 margin_index=None, margin_tokens=0, margin_token_dim=0, margin_scalars=0):
        Model.__init__(self, observation_space, action_space, device)
        if type(actor_history_frames) is not int or actor_history_frames < 1 or self.num_observations % actor_history_frames:
            raise ValueError("actor_history_frames must evenly divide the observation width")
        if action_scaling not in ("legacy", "normalized"):
            raise ValueError("action_scaling must be legacy or normalized")
        # normalized: the environment executes clip(a, -1, 1) * physical limit, so
        # a standard deviation above 1 would exceed the whole executed range.
        normalized_actions = action_scaling == "normalized"
        if normalized_actions:
            max_log_std = min(max_log_std, 0.0)
        self.action_scaling = action_scaling
        self.actor_history_frames = actor_history_frames
        self.frame_observations = self.num_observations // actor_history_frames
        transform_func = torch.distributions.transforms.TanhTransform() if use_tanh else None
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, transform_func=transform_func, deterministic=deterministic)
        PredictAttentionSelector.__init__(self, d_model=64, obj_feat_dim=encode_dim or 128,
                                          future_pose_dim=num_future_pose_obs - 1 if num_future_pose_obs else 0,
                                          token_dim=margin_token_dim if margin_index is not None else 0,
                                          num_tokens=margin_tokens if margin_index is not None else 0)

        self.to(device)
        self.num_features = num_features # 1024
        self.encode_dim = encode_dim # 128
        self.cube_object_num = 30  # Candidate grasps per object, not the number of object assets.
        self.notrain_obs_num = 3*self.cube_object_num + 3*self.cube_object_num # 30 cube's pos + rpy feature_num
        # Layout: current state, EKF poses + valid, optional TSIL, grasps, last action.
        # GFM consumes only the poses; skip the forecast block's final valid channel.
        future_end = -(self.notrain_obs_num + self.num_actions + num_tsil_obs)
        self.future_pose_slice = slice(future_end - num_future_pose_obs, future_end - 1)
        # Critic-only channels (episode progress, hidden reward state) are removed
        # from the actor's MLP input; store their positions within that input slice.
        body_width = self.frame_observations - self.num_features - self.notrain_obs_num - self.num_actions
        self.critic_time_position = None
        if critic_time_index is not None:
            position = body_width + critic_time_index + self.notrain_obs_num + self.num_actions
            if num_features <= 0 or not 6 <= position < body_width:
                raise ValueError("critic_time_index must point inside the teacher state block")
            self.critic_time_position = position
        self.critic_state_position = None
        if critic_state_index is not None:
            position = body_width + critic_state_index + self.notrain_obs_num + self.num_actions
            if (num_features <= 0 or type(critic_state_width) is not int or critic_state_width <= 0
                    or position < 6 or position + critic_state_width > body_width):
                raise ValueError("critic_state_index/width must point inside the teacher state block")
            self.critic_state_position = position
        # Margin tokens go to the GFM only; the margin scalars stay in the MLP input.
        self.margin_token_slice = margin_token_slice(body_width, self.notrain_obs_num, self.num_actions,
                                                     num_features, margin_index, margin_tokens,
                                                     margin_token_dim, margin_scalars)
        # (position, width), removed from the highest position down so that each
        # cut leaves the earlier positions unchanged.
        blocks = []
        if self.critic_time_position is not None:
            blocks.append((self.critic_time_position, 1))
        if self.critic_state_position is not None:
            blocks.append((self.critic_state_position, critic_state_width))
        if self.margin_token_slice is not None:
            blocks.append((self.margin_token_slice.start, margin_tokens * margin_token_dim))
        blocks.sort(reverse=True)
        if any(low + width > high for (high, _), (low, width) in zip(blocks, blocks[1:])):
            raise ValueError("Critic-only observation channels must not overlap")
        self.critic_only_blocks = tuple(blocks)

        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512), # input : (envs,1024), output : (envs,512)
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),) # input : (envs,512), output : (envs,128)
        critic_only_width = sum(width for _, width in self.critic_only_blocks)
        self.actor_frame_dim = (self.frame_observations - self.num_features + self.encode_dim - self.notrain_obs_num + 6 - critic_only_width
                                if num_features > 0 else self.frame_observations)
        self.net = nn.Sequential(nn.Linear(self.actor_frame_dim * actor_history_frames, 512),
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, self.num_actions)
                            )
        cprint(f"Actor MLP input: {self.actor_frame_dim} x {actor_history_frames} = {self.net[0].in_features} (oldest to current)", "yellow")

        if normalized_actions:
            # Andrychowicz et al. 2021: start near zero mean with a moderate,
            # state-independent std, here half of each dimension's range.
            self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), 0.5).log())
            with torch.no_grad():
                self.net[-1].weight.mul_(0.01)
                self.net[-1].bias.zero_()
        else:
            self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        batch_shape = inputs["states"].shape[:-1]
        # Each PPO sample contains its own raw frames. Re-encode them with the
        # current shared encoder/GFM, without a mutable model-side history.
        inputs = {"states": inputs["states"].reshape(-1, self.frame_observations)}
        grasp_predict_local_pos_flat = inputs["states"][...,-6*self.cube_object_num-self.num_actions:-3*self.cube_object_num-self.num_actions]
        grasp_predict_local_rpy_flat = inputs["states"][...,-3*self.cube_object_num-self.num_actions:-self.num_actions]
        grasp_predict_local_pos = grasp_predict_local_pos_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local_rpy = grasp_predict_local_rpy_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local = torch.cat([grasp_predict_local_pos,grasp_predict_local_rpy],dim=-1)
        cube_local_pos_rpy = inputs["states"][...,self.num_features:self.num_features+6]
        last_action = inputs["states"][...,-self.num_actions:]

        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features]) # input : (envs,1024), output : (envs,128)
            future_pose_obs = inputs["states"][..., self.future_pose_slice] if self.future_pose_dim else None
            state_block = inputs["states"][..., self.num_features:-self.notrain_obs_num-self.num_actions]
            right_predict_point = self.attention_forward(grasps=grasp_predict_local, obj_feat=features_encode,
                                                        pose=cube_local_pos_rpy, future_pose_obs=future_pose_obs,
                                                        tokens=margin_tokens_from(state_block, self.margin_token_slice,
                                                                                  self.num_tokens, self.token_dim))
            for position, width in self.critic_only_blocks:
                state_block = torch.cat((state_block[..., :position],
                                         state_block[..., position + width:]), dim=-1)
            frame_inputs = torch.cat([state_block, last_action, features_encode, right_predict_point], dim=-1)
        else:
            frame_inputs = inputs["states"]
        actions = self.net(frame_inputs.reshape(*batch_shape, self.actor_frame_dim * self.actor_history_frames))
        return actions, self.log_std_parameter, {}

class Value(DeterministicMixin, Model, PredictAttentionSelector):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor,
                 num_future_pose_obs=0, num_tsil_obs=0, critic_time_index=None, actor_history_frames=1,
                 critic_state_index=None, critic_state_width=0,
                 margin_index=None, margin_tokens=0, margin_token_dim=0, margin_scalars=0):
        # The critic-only indices need no handling here: the value MLP consumes
        # the whole state block, including the progress and reward-state channels.
        Model.__init__(self, observation_space, action_space, device)
        if type(actor_history_frames) is not int or actor_history_frames < 1 or self.num_observations % actor_history_frames:
            raise ValueError("actor_history_frames must evenly divide the observation width")
        self.frame_observations = self.num_observations // actor_history_frames
        DeterministicMixin.__init__(self)
        PredictAttentionSelector.__init__(self, d_model=64, obj_feat_dim=encode_dim or 128,
                                          future_pose_dim=num_future_pose_obs - 1 if num_future_pose_obs else 0,
                                          token_dim=margin_token_dim if margin_index is not None else 0,
                                          num_tokens=margin_tokens if margin_index is not None else 0)

        self.to(device)

        self.num_features = num_features # 1024
        self.encode_dim = encode_dim # 128
        self.cube_object_num = 30  # Candidate grasps per object, not the number of object assets.
        self.notrain_obs_num = 3*self.cube_object_num + 3*self.cube_object_num # 30 cube's pos + rpy feature_num
        future_end = -(self.notrain_obs_num + self.num_actions + num_tsil_obs)
        self.future_pose_slice = slice(future_end - num_future_pose_obs, future_end - 1)
        # The critic MLP keeps the whole state block, tokens included; its GFM also attends to them.
        body_width = self.frame_observations - self.num_features - self.notrain_obs_num - self.num_actions
        self.margin_token_slice = margin_token_slice(body_width, self.notrain_obs_num, self.num_actions,
                                                     num_features, margin_index, margin_tokens,
                                                     margin_token_dim, margin_scalars)
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512), # input : (envs,1024), output : (envs,512)
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),) # input : (envs,512), output : (envs,128)
        critic_dim = (self.frame_observations - self.num_features + self.encode_dim - self.notrain_obs_num + 6
                      if num_features > 0 else self.frame_observations)
        self.net = nn.Sequential(nn.Linear(critic_dim, 512),
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, 1))
        

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # Keep the critic's original single-frame information and architecture.
        inputs = {"states": inputs["states"][..., -self.frame_observations:]}
        grasp_predict_local_pos_flat = inputs["states"][...,-6*self.cube_object_num-self.num_actions:-3*self.cube_object_num-self.num_actions]
        grasp_predict_local_rpy_flat = inputs["states"][...,-3*self.cube_object_num-self.num_actions:-self.num_actions]
        grasp_predict_local_pos = grasp_predict_local_pos_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local_rpy = grasp_predict_local_rpy_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local = torch.cat([grasp_predict_local_pos,grasp_predict_local_rpy],dim=-1)
        cube_local_pos_rpy = inputs["states"][...,self.num_features:self.num_features+6]
        last_action = inputs["states"][...,-self.num_actions:]

        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features]) # input : (envs,1024), output : (envs,128)
            future_pose_obs = inputs["states"][..., self.future_pose_slice] if self.future_pose_dim else None
            state_block = inputs["states"][..., self.num_features:-self.notrain_obs_num-self.num_actions]
            right_predict_point = self.attention_forward(grasps=grasp_predict_local, obj_feat=features_encode,
                                                        pose=cube_local_pos_rpy, future_pose_obs=future_pose_obs,
                                                        tokens=margin_tokens_from(state_block, self.margin_token_slice,
                                                                                  self.num_tokens, self.token_dim))
            actions = self.net(torch.cat([inputs["states"][..., self.num_features:-self.notrain_obs_num-self.num_actions],last_action , features_encode,right_predict_point], dim=-1))
            return actions, {}        
        else:
            return self.net(inputs["states"]), {}


def get_trainer(is_eval=False, strict_eval=False, argv=None, enable_camera_sensors=False):
    args = get_params(argv)
    args.eval = is_eval
    if strict_eval:
        if not args.checkpoint or args.resume:
            raise ValueError("Strict evaluation requires --checkpoint and does not accept --resume")
    if is_eval and args.checkpoint:
        restore_eval_arguments(args)
    temporal_metadata = None
    if is_eval and args.checkpoint:
        temporal_metadata = read_temporal_checkpoint(args.checkpoint)
        if temporal_metadata is not None:
            restore_temporal_arguments(args, temporal_metadata)
    set_seed(args.seed)
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    if not is_eval and args.sgs_mode in ("uniform", "sgs") and (args.resume or args.checkpoint):
        raise ValueError("SGS/uniform-bank training starts from scratch; sampler resume is not implemented")
    embedded_config = temporal_metadata is not None and args.cfg_file is None
    file_path = ("data/cfg/DQ_teacher.yaml" if embedded_config else
                 teacher_config_path(args, is_eval=is_eval, strict_eval=strict_eval))
    cfg_file = Path(file_path).name
    
    if args.resume:
        experiment_dir = os.path.join(args.experiment_dir, args.wandb_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        pt_files = os.listdir(checkpoint_dir)
        pt_files = [file for file in pt_files if file.endswith(".pt") and (not file.startswith("best"))]
        # Find the latest checkpoint
        checkpoint_steps = 0
        if len(pt_files) > 0:
            args.checkpoint = os.path.join(checkpoint_dir, sorted(pt_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1])
            checkpoint_steps = int(args.checkpoint.split("_")[-1].split(".")[0])
        cfg_files = os.listdir(experiment_dir)
        cfg_files = [file for file in cfg_files if file.endswith(".yaml")]
        if len(cfg_files) > 0 and args.cfg_file is None:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
    cprint(f"Training script: {os.path.realpath(__file__)}", "yellow")
    cprint(f"Using config file: {os.path.realpath(file_path)}", "red")
        
    cfg = deepcopy(temporal_metadata["config"]) if embedded_config else load_cfg(file_path)
    # Standalone callers pass their own argv; the environment must not reparse
    # the launcher's CLI to recover the reward/online-learning schedule.
    cfg["env"]["totalTimesteps"] = args.timesteps
    if enable_camera_sensors:
        # Offscreen rollout cameras need a graphics context even without a viewer.
        cfg["enableCameraSensors"] = True
    configure_teacher_experiment(cfg, args, is_eval=is_eval, strict_eval=strict_eval)
    if is_eval and cfg["env"]["tsil"]["enabled"] and temporal_metadata is None:
        raise ValueError("TSIL evaluation requires a checkpoint with matching temporal state and normalization")
    prediction_cfg = apply_pose_prediction_config(cfg, args)
    cprint(f"Teacher future pose observations: {prediction_cfg}", "yellow")
    if args.no_velocity_obs:
        cfg["env"]["noVelocityObs"] = True
    cprint(f"Teacher velocity observations: {'disabled' if cfg['env'].get('noVelocityObs', False) else 'enabled'}", "yellow")
    if args.critic_time_obs is not None:
        cfg["env"]["criticTimeObs"] = args.critic_time_obs
    cprint(f"Teacher critic episode-progress input: {'enabled' if cfg['env'].get('criticTimeObs', False) else 'disabled'}", "yellow")
    if args.critic_reward_state is not None:
        cfg["env"]["criticRewardStateObs"] = args.critic_reward_state
    cprint(f"Teacher critic reward-state input: {'enabled' if cfg['env'].get('criticRewardStateObs', False) else 'disabled'}", "yellow")
    reward_options = apply_reward_options(cfg, args)
    cprint(f"Teacher reward options: {reward_options}", "yellow")
    if is_eval and args.checkpoint and args.reach_model is None:
        # A saved run keeps a copy of its T file; use it if the original moved.
        spec = cfg["env"].get("margin", {}).get("reachModel", "")
        saved = Path(args.checkpoint).resolve().parent.parent / "reach_model_initial.pt"
        if (spec and spec != "analytic" and not spec.startswith("new:")
                and not Path(spec).is_file() and saved.is_file()):
            cfg["env"]["margin"]["reachModel"] = str(saved)
    margin_settings = apply_margin_config(cfg, args)
    if margin_settings["reachModel"]:
        cprint("Main line A margin: model={reachModel} reward={reward} tokens={tokens} "
               "curriculum={curriculum} online={online}".format(**margin_settings), "yellow")
    multi_obj = cfg["env"]["asset"]["asset_multi"]
    if args.object_name is not None:
        if args.object_name not in multi_obj:
            raise ValueError(f"Unknown --object_name {args.object_name!r}. Available objects: {', '.join(multi_obj)}")
        multi_obj = {args.object_name: multi_obj[args.object_name]}
        cfg["env"]["asset"]["asset_multi"] = multi_obj
    object_indices = [obj["dict_idx"] for obj in multi_obj.values()]
    cprint(f"Using {len(multi_obj)} object asset(s): {', '.join(multi_obj)}", "yellow")
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh
    action_scaling = apply_action_scaling_config(cfg, args)
    cprint(f"Teacher action scaling: {action_scaling}", "yellow")
    # A saved run's own limits when evaluating its checkpoint (0.02 m / 0.06 rad if it predates them).
    ee_limits = apply_ee_delta_limits(cfg, args.checkpoint if is_eval else None)
    cprint(f"Teacher EE-goal increment limits per step: {ee_limits}", "yellow")
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.debug:
        cfg['env']['numEnvs'] = 34
        
    if args.eval:
        cfg['env']['numEnvs'] = 800 # 34
        cfg["env"]["maxEpisodeLength"] = 100
        if args.checkpoint:
            checkpoint_steps = checkpoint_step(args.checkpoint)
            cfg["env"]["globalStepCounter"] = checkpoint_steps

    # Allow the command line to override both the YAML value and the
    # evaluation default above (800 environments).
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be a positive integer")
        cfg["env"]["numEnvs"] = args.num_envs
    if strict_eval:
        cfg["env"]["numEnvs"] = min(cfg["env"]["numEnvs"], args.eval_episodes)

    """
    To clarify, when capturing images of each object, the images were taken from a car. 
    The z-height of each environment's car was determined by the intervel. 
    To ensure correct camera pose for each cube,
    the intervel needed to be aligned here. This is a legacy issue and will not be used further.
    Since we no longer use the car as a moving base, this intervel has no effect in the subsequent training code; 
    it is only used when initializing the camera pose.
    """

    intervel = 23
    args.intervel = intervel
    cube_predict_info_path = "contact_grasp_info_mul"
    cube_root_states_info_path = "30all_nomove_cube_root_states5.pt"
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(cube_predict_info_path=cube_predict_info_path
                                                                            ,cube_root_states_info_path=cube_root_states_info_path,num_env=cfg['env']['numEnvs'],
                                                                            intervel=intervel,delta_height=0.1,object_indices=object_indices)
    env_num = cfg['env']['numEnvs']
    cprint(f"env_num is {env_num}", "yellow")
    # Save the selected assets and overrides before the environment mutates cfg.
    experiment_cfg = deepcopy(cfg)
    env = create_env(cfg=cfg, args=args, camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor)
    device = env.rl_device
    memory = None if is_eval else RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
    
    num_features = 0 if args.no_feature else 1024
    encode_dim = 0 if args.no_feature else 128
    models_ppo = {}
    gfm_kwargs = teacher_gfm_kwargs(env)
    models_ppo["policy"] = Policy(env.observation_space, env.action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor, use_tanh=args.use_tanh, clip_actions=args.use_tanh, deterministic=args.eval, action_scaling=action_scaling, **gfm_kwargs)
    models_ppo["value"] = Value(env.observation_space, env.action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor, **gfm_kwargs)
    
    cfg_ppo = deepcopy(PPO_DEFAULT_CONFIG)
    cfg_ppo["rollouts"] = 24  # memory_size
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768
    cfg_ppo["discount_factor"] = 0.99
    if ((reward_options["approachShaping"] == "potential" or margin_settings["reward"])
            and reward_options["shapingGamma"] != cfg_ppo["discount_factor"]):
        # Potential shaping preserves the optimal policy only with the agent's discount.
        raise ValueError("reward.shapingGamma must equal the PPO discount factor "
                         f"({cfg_ppo['discount_factor']})")
    cfg_ppo["lambda"] = 0.95 if args.gae_lambda is None else args.gae_lambda
    if not 0 <= cfg_ppo["lambda"] <= 1:
        raise ValueError("--gae_lambda must be in [0, 1]")
    cfg_ppo["learning_rate"] = 4.2e-4 ## my change ##
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True if args.value_clip is None else args.value_clip
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0
    cfg_ppo["rewards_shaper"] = None
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg_ppo["tsil"] = deepcopy(cfg["env"]["tsil"])
    # Log to TensorBoard/W&B every 10 PPO iterations (one rollout per iteration).
    cfg_ppo["experiment"]["write_interval"] = 10 * cfg_ppo["rollouts"]
    cfg_ppo["experiment"]["steps_per_iteration"] = cfg_ppo["rollouts"]
    env._env.sgs_log_interval = cfg_ppo["experiment"]["write_interval"]
    cfg_ppo["experiment"]["checkpoint_interval"] = 5000 ## my change
    if is_eval:
        cfg_ppo["experiment"]["write_interval"] = 0
        cfg_ppo["experiment"]["checkpoint_interval"] = 0
    cfg_ppo["experiment"]["directory"] = args.experiment_dir
    cfg_ppo["experiment"]["experiment_name"] = args.wandb_name
    cfg_ppo["experiment"]["wandb"] = args.wandb
    if args.wandb:
        cfg_ppo["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "sync_tensorboard": True, "name": args.wandb_name}
        # cfg_ppo["experiment"]["wandb_kwargs"]["resume"] = True
        
    agent_class, agent_kwargs = PPO, {}
    if cfg["env"]["tsil"]["enabled"]:
        from utils.tsil_ppo import TSILPPO
        agent_class, agent_kwargs = TSILPPO, {"tsil_env": env._env}
    agent = agent_class(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device, **agent_kwargs)
    if cfg["env"]["tsil"]["enabled"]:
        agent.checkpoint_modules["tsil_metadata"] = TemporalCheckpointMetadata(
            experiment_cfg, args, env.observation_space.shape[0], env.tsil_time_reference)
    if getattr(env._env, "margin_learner", None) is not None:
        # Online T (EMA target included) travels with each checkpoint, so evaluation
        # rebuilds the actor's margin tokens with the T that training used last.
        agent.checkpoint_modules["reach_learner"] = env._env.margin_learner
    
    # Keep the trainer's rendering setting consistent with the environment.
    # When --headless is omitted, SKRL must call env.render() so the Isaac Gym
    # viewer remains visible and responsive during evaluation.
    cfg_trainer = {"timesteps": args.timesteps, "headless": args.headless}
    if args.checkpoint:
        print("Loading checkpoint: ", args.checkpoint)
        agent.load(args.checkpoint)
        checkpoint_steps = checkpoint_step(args.checkpoint, required=not is_eval)
        if args.record_video:
            experiment_dir = args.checkpoint.split("/")[0]
            wandb_name = args.checkpoint.split("/")[1]
            cfg_trainer["video_name"] = wandb_name +"-"+str(checkpoint_steps)
            cfg_trainer["log_dir"] = experiment_dir
            cfg_trainer["record_video"] = True
        if not args.eval:
            cfg_trainer["initial_timestep"] = checkpoint_steps
            agent.set_running_mode("eval")
    if not is_eval:
        cfg_trainer["timesteps"] = teacher_run_end(
            args.timesteps, cfg_trainer.get("initial_timestep", 0), args.stop_after)
    
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    trainer.teacher_args = args
    trainer.teacher_config = experiment_cfg
    if args.wandb:
        import wandb
        wandb.save(file_path, policy="now")
        wandb.save("envs/b1z1_" + args.task[4:].lower() + ".py", policy="now")
        wandb.save("train_multistate_DQ_teacher.py", policy="now")
    if not args.eval:
        if not os.path.exists(os.path.join(args.experiment_dir, args.wandb_name, cfg_file)):
            copy_cfg(file_path, os.path.join(args.experiment_dir, args.wandb_name), cfg=experiment_cfg)
        metadata_path = Path(args.experiment_dir) / args.wandb_name / "teacher_run.json"
        if not metadata_path.exists():
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with metadata_path.open("w") as stream:
                json.dump({"arguments": vars(args), "config_file": cfg_file,
                           "planned_timesteps": args.timesteps,
                           "ee_delta_limits": ee_limits,
                           "checkpoint_kind": ("temporal_teacher_evaluation" if
                                               cfg["env"]["tsil"]["enabled"] else "model_only")},
                          stream, indent=2)
        reach_spec = experiment_cfg["env"].get("margin", {}).get("reachModel", "")
        if reach_spec and Path(reach_spec).is_file():
            import shutil
            # Provenance: the exact T file the rewards, tokens and curriculum used.
            shutil.copy2(reach_spec, metadata_path.parent / "reach_model_initial.pt")
        if experiment_cfg["env"]["sgs"]["mode"] != "disabled":
            bank = env.sgs_bank
            bank_metadata = {"fingerprint": bank.fingerprint, "num_configs": bank.num_configs,
                             "bank_path": str(Path(experiment_cfg["env"]["sgs"]["bank_path"]).resolve()),
                             "task_hold_steps": env.hold_steps,
                             "hold_compatibility": env.sgs_hold_compatibility}
            (metadata_path.parent / "bank_metadata.json").write_text(
                json.dumps(bank_metadata, indent=2) + "\n")
    
    return trainer


def finalize_teacher_logging(trainer):
    """Include final reset outcomes, which arrive after the last agent write."""
    env = trainer.env
    if not getattr(env, "sgs_enabled", False):
        return
    env._record_sgs_metrics()
    env._print_sgs_success_rates(force=True)
    agent = trainer.agents
    if agent.write_interval > 0:
        # The main full_task window is recorded at termination; family
        # diagnostics settle at reset. Flush both latest snapshots together.
        snapshots = dict(env.extras.get("success_rate", {}))
        snapshots.update(env.extras.get("scalar_snapshots", {}))
        for key, value in snapshots.items():
            agent.track_data_latest(key, value)
        agent.write_tracking_data(env.global_step_counter, trainer.timesteps)
        agent.writer.flush()


if __name__ == "__main__":
    trainer = get_trainer()
    try:
        trainer.train()
    except KeyboardInterrupt:
        # Preserve the partial logging window when training is interrupted,
        # then propagate the interrupt instead of saving a completed run.
        finalize_teacher_logging(trainer)
        raise
    finalize_teacher_logging(trainer)
    run_dir = Path(trainer.teacher_args.experiment_dir) / trainer.teacher_args.wandb_name
    if trainer.teacher_config["env"]["sgs"]["mode"] != "disabled":
        # Diagnostic history is not a checkpoint for resuming training.
        trainer.env.save_sgs_diagnostics(str(run_dir / "sgs_diagnostics.pt"))
    learner = getattr(trainer.env._env, "margin_learner", None)
    if learner is not None:
        from modules.reachability import save_reach_model
        # The EMA target that produced rewards/tokens; reusable via --reach_model.
        save_reach_model(run_dir / "reach_model_final.pt", learner.target, {
            "gripper_body_index": trainer.env._env.gripper_idx, "source": "online",
            "updates": learner.updates, "run": str(run_dir)})
    if trainer.teacher_args.stop_after is not None:
        # This is a model checkpoint for smoke evaluation, not a sampler resume.
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.agents.save(str(checkpoint_dir / f"agent_{trainer.timesteps}.pt"))
    
