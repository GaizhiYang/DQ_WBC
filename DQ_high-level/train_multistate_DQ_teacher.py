from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
from copy import deepcopy
import torch
import os
import os
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
from utils.config import load_cfg, get_params, copy_cfg
import utils.wrapper as wrapper
import time
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg
from modules.predictattention import PredictAttentionSelector
from modules.karl_teacher import KarlTeacherPolicy, KarlTeacherValue
from utils.karl_teacher_wrapper import KarlTeacherWrapper
from utils.teacher_grasp_training import resolve_selection, checkpoint_step, TeacherGraspTrainingState


set_seed(43)

def create_env(cfg, args, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor ):
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
    selection = cfg.get("grasp_selection", {})
    if selection.get("mode") == "karl":
        wrapped_env = KarlTeacherWrapper(
            _env, selection["switch_margin_deg"], selection["orientation_preference"],
            visualize_grasp=args.vis_selected_grasp, grasp_vis_envs=args.grasp_vis_envs)
    elif selection.get("mode") == "geometric":
        from utils.geometric_teacher_wrapper import GeometricTeacherWrapper
        wrapped_env = GeometricTeacherWrapper(
            _env, selection["geometric"],
            visualize_grasp=args.vis_selected_grasp, grasp_vis_envs=args.grasp_vis_envs)
    else:
        wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env

# define models (stochastic and deterministic models) using mixins
class Policy(GaussianMixin, Model, PredictAttentionSelector):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor,
                  use_tanh=False, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", deterministic=False,
                 actor_drop_velocity_obs=False):
        Model.__init__(self, observation_space, action_space, device)
        transform_func = torch.distributions.transforms.TanhTransform() if use_tanh else None
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction, transform_func=transform_func, deterministic=deterministic)
        PredictAttentionSelector.__init__(self, d_model=64)

        self.to(device)
        self.num_features = num_features # 1024
        self.encode_dim = encode_dim # 128
        self.actor_drop_velocity_obs = actor_drop_velocity_obs
        if actor_drop_velocity_obs and (self.num_observations != 1276 or self.num_actions != 9
                                        or num_features != 1024 or encode_dim != 128):
            raise ValueError("Actor velocity ablation requires 1276 raw observations, 1024 features and 9 actions")
        # PPO transports the full observation for the unchanged Critic. The
        # Actor removes the five velocity entries before any network branch.
        self.actor_num_observations = self.num_observations - (5 if actor_drop_velocity_obs else 0)
        self.cube_object_num = 30  # Candidate grasps per object, not the number of object assets.
        self.notrain_obs_num = 3*self.cube_object_num + 3*self.cube_object_num # 30 cube's pos + rpy feature_num
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512), # input : (envs,1024), output : (envs,512)
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),) # input : (envs,512), output : (envs,128)
        self.net = nn.Sequential(nn.Linear(self.actor_num_observations - self.num_features + self.encode_dim - self.notrain_obs_num + 6, 512),
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, self.num_actions)
                            )
        cprint(f"self.num_observations is {self.num_observations}", "yellow")

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]
        if self.actor_drop_velocity_obs:
            if states.shape[-1] != 1276:
                raise ValueError("Actor velocity ablation expects the full 1276-dimensional PPO observation")
            # [1082:1085]: robot local linear velocity (3).
            # [1085:1087]: object velocity relative to robot in XY (2).
            states = torch.cat((states[..., :1082], states[..., 1087:]), dim=-1)
        grasp_predict_local_pos_flat = states[...,-6*self.cube_object_num-9:-3*self.cube_object_num-9]
        grasp_predict_local_rpy_flat = states[...,-3*self.cube_object_num-9:-9]
        grasp_predict_local_pos = grasp_predict_local_pos_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local_rpy = grasp_predict_local_rpy_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local = torch.cat([grasp_predict_local_pos,grasp_predict_local_rpy],dim=-1)
        cube_local_pos_rpy = states[...,self.num_features:self.num_features+6]
        last_action = states[...,-9:]

        if self.num_features > 0:
            features_encode = self.feature_encoder(states[..., :self.num_features]) # input : (envs,1024), output : (envs,128)
            right_predict_point = self.attention_forward(grasps=grasp_predict_local,obj_feat=features_encode,pose=cube_local_pos_rpy)
            actions = self.net(torch.cat([states[..., self.num_features:-self.notrain_obs_num-9],last_action , features_encode,right_predict_point], dim=-1))
        else:
            actions = self.net(states)
        return actions, self.log_std_parameter, {}

class Value(DeterministicMixin, Model, PredictAttentionSelector):
    def __init__(self, observation_space, action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        PredictAttentionSelector.__init__(self, d_model=64)

        self.to(device)

        self.num_features = num_features # 1024
        self.encode_dim = encode_dim # 128
        self.cube_object_num = 30  # Candidate grasps per object, not the number of object assets.
        self.notrain_obs_num = 3*self.cube_object_num + 3*self.cube_object_num # 30 cube's pos + rpy feature_num
        
        if num_features > 0:
            self.feature_encoder = nn.Sequential(nn.Linear(self.num_features, 512), # input : (envs,1024), output : (envs,512)
                                                  nn.ELU(),
                                                  nn.Linear(512, self.encode_dim),) # input : (envs,512), output : (envs,128)
        self.net = nn.Sequential(nn.Linear(self.num_observations - self.num_features + self.encode_dim - self.notrain_obs_num + 6, 512), # self.num_observations = 1094
                            nn.ELU(),
                            nn.Linear(512, 256),
                            nn.ELU(),
                            nn.Linear(256, 128),
                            nn.ELU(),
                            nn.Linear(128, 1))
        

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        grasp_predict_local_pos_flat = inputs["states"][...,-6*self.cube_object_num-9:-3*self.cube_object_num-9]
        grasp_predict_local_rpy_flat = inputs["states"][...,-3*self.cube_object_num-9:-9]
        grasp_predict_local_pos = grasp_predict_local_pos_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local_rpy = grasp_predict_local_rpy_flat.view(-1, self.cube_object_num, 3)
        grasp_predict_local = torch.cat([grasp_predict_local_pos,grasp_predict_local_rpy],dim=-1)
        cube_local_pos_rpy = inputs["states"][...,self.num_features:self.num_features+6]
        last_action = inputs["states"][...,-9:]

        if self.num_features > 0:
            features_encode = self.feature_encoder(inputs["states"][..., :self.num_features]) # input : (envs,1024), output : (envs,128)
            right_predict_point = self.attention_forward(grasps=grasp_predict_local,obj_feat=features_encode,pose=cube_local_pos_rpy)
            actions = self.net(torch.cat([inputs["states"][..., self.num_features:-self.notrain_obs_num-9],last_action , features_encode,right_predict_point], dim=-1)) 
            return actions, {}        
        else:
            return self.net(inputs["states"]), {}


def get_predict_point(camera_6p_info=None, cube_predict_info_path=None, cube_root_states_info_path=None, num_env=None, intervel=None, delta_height=None, object_indices=None):
    import json
    import os
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir,"PredictPiontTransform" , "transform_info")
    cube_root_states_info_path = os.path.join(data_dir, cube_root_states_info_path)

    # load cubes's states（position + quaternion + velocity...）
    cube_states = torch.load(cube_root_states_info_path, map_location="cpu")[0]
    if object_indices is None:
        object_indices = list(range(cube_states.shape[0]))
    if not object_indices or any(i < 0 or i >= cube_states.shape[0] for i in object_indices):
        raise ValueError(f"Object dict_idx values must be in [0, {cube_states.shape[0] - 1}]")
    # Keep initial poses and grasp predictions in the same order as asset_multi.
    cube_states = cube_states[object_indices]
    cprint(f"cube_states.shape is {cube_states.shape}", "yellow")

    cube_pos = cube_states[:, :3]
    cube_quat = cube_states[:, 3:7]
    cube_rpy = torch.tensor(R.from_quat(cube_quat.numpy()).as_euler('xyz'))  # turn to RPY
    cube_init_tensor = torch.cat([cube_pos, cube_rpy], dim=1)
    cube_init_tensor = cube_init_tensor.to(torch.float32)

    base_n = cube_init_tensor.shape[0]  
    T_grasp_cv_all = []

    # read base_num sample's all transform
    for i in object_indices:
        sample_dir = os.path.join(data_dir, cube_predict_info_path)
        json_path = os.path.join(sample_dir, f"predictions_image_{(i):02d}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        grasp_matrices = np.array(data['transform'])  
        T_grasp_cv_all.append(grasp_matrices)

    # Shape: [number of selected objects, candidate grasps per object, 4, 4].
    T_grasp_cv_all = np.stack(T_grasp_cv_all, axis=0)
    n_grasps = T_grasp_cv_all.shape[1]  # get n
    T_grasp_cv_all = torch.tensor(T_grasp_cv_all, dtype=torch.float32)

    # extent to num_env
    T_grasp_cv_list = [T_grasp_cv_all[i % base_n] for i in range(num_env)]
    grasp_cv_tensor = torch.stack(T_grasp_cv_list, dim=0)  # 形状 [num_env, n, 4, 4]

    camera_tensor = []
    cube_init_list = []
    cube_init_ori_list = []

    for i in range(num_env):
        base_idx = i % base_n
        height_offset = (delta_height / intervel) * ((i % intervel) + 1)

        # the initial pose of the camera, x,y is related to the initial pose of the float base 
        camera_x = -0.3
        camera_y = 2.0
        camera_z = 0.80 + height_offset
        camera_info = [camera_x, camera_y, camera_z, 0.0, 0.70, 0.0]
        camera_tensor.append([camera_info] * n_grasps)

        cube_init = cube_init_tensor[base_idx].clone()
        cube_init[2] += height_offset 
        cube_init_list.append([cube_init] * n_grasps)
        cube_init_ori_list.append(cube_init)

    camera_tensor = torch.tensor(camera_tensor, dtype=torch.float32)  # [num_env, n, 6]
    cube_init_tensor = torch.stack([torch.stack(cube_init, dim=0) for cube_init in cube_init_list], dim=0)  # [num_env, n, 6]
    return camera_tensor, grasp_cv_tensor, cube_init_tensor

def get_trainer(is_eval=False):
    args = get_params()
    if not args.task:
        args.task = "B1Z1PickMulti"
    args.eval = is_eval
    args.wandb = args.wandb and (not args.eval) and (not args.debug)
    cfg_file = "DQ_teacher.yaml"    
    # cfg_file = "b1z1_" + args.task[4:].lower() + ".yaml"
    file_path = "data/cfg/" + cfg_file
    
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
        if len(cfg_files) > 0:
            cfg_file = cfg_files[0]
            file_path = os.path.join(experiment_dir, cfg_file)
        
        print("Find the latest checkpoint: ", args.checkpoint)
    cprint(f"Using config file: {file_path}","red" )
        
    cfg = load_cfg(file_path)
    checkpoint = torch.load(args.checkpoint, map_location="cpu") if args.checkpoint else None
    cfg, selection = resolve_selection(args, cfg, checkpoint)
    set_seed(args.seed)
    cprint(f"Grasp selection: {selection}", "cyan")
    checkpoint_steps = checkpoint_step(checkpoint, args.checkpoint) if checkpoint is not None else 0
    if checkpoint is not None:
        cfg["env"]["globalStepCounter"] = checkpoint_steps
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
    cfg['env']["near_goal_stop"] = args.near_goal_stop
    cfg['env']["obj_move_prob"] = args.obj_move_prob
    if args.debug:
        cfg['env']['numEnvs'] = 34
        
    if args.eval:
        cfg['env']['numEnvs'] = 800 # 34
        cfg["env"]["maxEpisodeLength"] = 100
        if args.checkpoint:
            cfg["env"]["globalStepCounter"] = checkpoint_steps

    # Allow the command line to override both the YAML value and the
    # evaluation default above (800 environments).
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be a positive integer")
        cfg["env"]["numEnvs"] = args.num_envs

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
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)
    
    num_features = 0 if args.no_feature else 1024
    encode_dim = 0 if args.no_feature else 128
    models_ppo = {}
    if selection["mode"] in ("karl", "geometric"):
        models_ppo["policy"] = KarlTeacherPolicy(env.observation_space, env.action_space, device,
                                                use_tanh=args.use_tanh, clip_actions=args.use_tanh, deterministic=args.eval)
        models_ppo["value"] = KarlTeacherValue(env.observation_space, env.action_space, device)
    else:
        models_ppo["policy"] = Policy(env.observation_space, env.action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor, use_tanh=args.use_tanh, clip_actions=args.use_tanh, deterministic=args.eval,
                                       actor_drop_velocity_obs=selection.get("actor_drop_velocity_obs", False))
        models_ppo["value"] = Value(env.observation_space, env.action_space, device, num_features, encode_dim, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor)
    cprint("Actor observations: %d; Critic observations: %d; Actor velocity ablation: %s" % (
        getattr(models_ppo["policy"], "actor_num_observations", models_ppo["policy"].num_observations),
        models_ppo["value"].num_observations, selection.get("actor_drop_velocity_obs", False)), "cyan")
    
    cfg_ppo = PPO_DEFAULT_CONFIG.copy()
    cfg_ppo["rollouts"] = 24  # memory_size
    cfg_ppo["learning_epochs"] = 5
    cfg_ppo["mini_batches"] = 6  # 24 * 8192 / 32768
    cfg_ppo["discount_factor"] = 0.99
    cfg_ppo["lambda"] = 0.95
    cfg_ppo["learning_rate"] = 4.2e-4 ## my change ##
    cfg_ppo["learning_rate_scheduler"] = KLAdaptiveRL
    cfg_ppo["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg_ppo["random_timesteps"] = 0
    cfg_ppo["learning_starts"] = 0
    cfg_ppo["grad_norm_clip"] = 1.0
    cfg_ppo["ratio_clip"] = 0.2
    cfg_ppo["value_clip"] = 0.2
    cfg_ppo["clip_predicted_values"] = True
    cfg_ppo["value_loss_scale"] = 1.0
    cfg_ppo["kl_threshold"] = 0
    cfg_ppo["rewards_shaper"] = None
    cfg_ppo["state_preprocessor"] = RunningStandardScaler
    cfg_ppo["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg_ppo["value_preprocessor"] = RunningStandardScaler
    cfg_ppo["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints each 120 and 1200 timesteps respectively
    cfg_ppo["experiment"]["write_interval"] = 24
    cfg_ppo["experiment"]["checkpoint_interval"] = 5000 ## my change
    cfg_ppo["experiment"]["directory"] = args.experiment_dir
    cfg_ppo["experiment"]["experiment_name"] = args.wandb_name
    cfg_ppo["experiment"]["wandb"] = args.wandb
    if args.wandb:
        cfg_ppo["experiment"]["wandb_kwargs"] = {"project": args.wandb_project, "tensorboard": False, "name": args.wandb_name}
        # cfg_ppo["experiment"]["wandb_kwargs"]["resume"] = True
        
    agent = PPO(models=models_ppo,
            memory=memory,
            cfg=cfg_ppo,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)
    if selection["mode"] == "geometric":
        env.diagnostics_callback = agent.track_data
    agent.checkpoint_modules["grasp_selection_state"] = TeacherGraspTrainingState(env, selection, experiment_cfg, args)
    agent.checkpoint_modules["scheduler"] = agent.scheduler
    
    # Keep the trainer's rendering setting consistent with the environment.
    # When --headless is omitted, SKRL must call env.render() so the Isaac Gym
    # viewer remains visible and responsive during evaluation.
    cfg_trainer = {"timesteps": args.timesteps, "headless": args.headless}
    if args.eval:
        cfg_trainer["evaluation_steps"] = args.timesteps
    if args.checkpoint:
        print("Resuming from checkpoint: ", args.checkpoint)
        agent.load(args.checkpoint)
        if args.record_video:
            experiment_dir = args.checkpoint.split("/")[0]
            wandb_name = args.checkpoint.split("/")[1]
            cfg_trainer["video_name"] = wandb_name +"-"+str(checkpoint_steps)
            cfg_trainer["log_dir"] = experiment_dir
            cfg_trainer["record_video"] = True
        if not args.eval:
            cfg_trainer["initial_timestep"] = checkpoint_steps
            agent.set_running_mode("eval")
    
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    if args.wandb:
        import wandb
        wandb.save("data/cfg/" + cfg_file, policy="now")
        wandb.save("envs/b1z1_" + args.task[4:].lower() + ".py", policy="now")
        wandb.save(__file__, policy="now")
    if not args.eval:
        if not os.path.exists(os.path.join(args.experiment_dir, args.wandb_name, cfg_file)):
            copy_cfg(file_path, os.path.join(args.experiment_dir, args.wandb_name), cfg=experiment_cfg)
    
    return trainer
    
if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
    
