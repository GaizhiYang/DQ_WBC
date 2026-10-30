import numpy as np
import os
import torch
import cv2
from typing import Dict, Any, Tuple, List, Set
from collections import defaultdict
import wandb
from termcolor import cprint    
from tqdm import tqdm

from .b1z1_base import B1Z1Base, reindex_all, reindex_feet, LIN_VEL_X_CLIP, ANG_VEL_YAW_CLIP, torch_rand_int
from utils.low_level_model import ActorCritic
from utils.config import load_cfg, get_params, copy_cfg
from utils.pick_success import lifted_grasp_mask, advance_pick_hold, RollingEpisodeSuccess
from modules.pose_prediction import RollingSE3Predictor
from utils.object_pose import object_pose_in_teacher_frame
from PredictPiontTransform.predictpoint_mul import PredictPoint
import time

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
from isaacgym.torch_utils import *
from torch import Tensor
import torchvision.transforms as transforms


class B1Z1PickMulti(B1Z1Base,PredictPoint):
    def __init__(self, cfg_terrain=None, table_height=None,camera_6p_tensor=None, grasp_cv_tensor=None, cube_init_tensor=None, eval=False, *args, **kwargs):
        self.cfg_terrain = cfg_terrain
        self.eval = eval
        self.num_actors = 3
        cprint(f"B1Z1PickMulti is called","cyan")

        B1Z1Base.__init__(self, *args, **kwargs)
        PredictPoint.__init__(self, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor, device=self.device)
        # Standalone rollout tools can supply this through cfg without parsing
        # their own command-line options again as training arguments.
        total_timesteps = self.cfg["env"].get("totalTimesteps")
        if total_timesteps is None:
            total_timesteps = get_params().timesteps
        self.near_goal_stop = self.cfg["env"].get("near_goal_stop", False)
        self.obj_move_prob = self.cfg["env"].get("obj_move_prob", 0.0)
        self.table_heights_fix = table_height
        self.total_timesteps = total_timesteps
        self.train_reward_strict = total_timesteps / 2
        self._init_sgs()

    def _init_sgs(self):
        """Load one immutable bank; all online sampler state stays on the GPU."""
        from utils.sgs import (ConfigurationBank, SGSSampler, SUPPORTED_FAMILY_PROFILES,
                               validate_task_hold_steps)
        from utils.episode_statistics import RecentEpisodeStatistics

        cfg = self.cfg["env"].get("sgs", {})
        self.sgs_enabled = cfg.get("mode", "disabled") != "disabled"
        if not self.sgs_enabled:
            return
        if len(self.obj_list) != 1 or self.table_level != "Level02":
            raise ValueError("SGS requires a single object and Level02")
        if self.enable_camera or self.floating_base or self.obj_move_prob:
            raise ValueError("SGS requires the nonvisual walking teacher and obj_move_prob=0")
        self.sgs_bank = ConfigurationBank.load(
            cfg.get("bank_path", ""), device=self.device,
            expected_metadata={
                "object_name": self.obj_list[0],
                "robot_asset": self.cfg["env"]["asset"]["assetFileRobot"],
                "num_dofs": self.num_dofs,
                "num_bodies": self._rigid_body_pos.shape[1],
                "lifted_threshold": self.lifted_success_threshold,
                "table_level": "Level02",
                "physical_validation": {
                    "passed": True,
                },
            })
        self.sgs_hold_compatibility = validate_task_hold_steps(
            self.sgs_bank.metadata, self.hold_steps)
        physical = self.sgs_bank.metadata.get("physical_validation", {})
        profile = physical.get("validation_profile")
        self.sgs_family_names = self.sgs_bank.family_names
        if (profile not in SUPPORTED_FAMILY_PROFILES
                or self.sgs_family_names != SUPPORTED_FAMILY_PROFILES[profile]):
            raise ValueError("SGS requires supported validation profile and family_names")
        if "near_goal" in self.sgs_family_names and not torch.allclose(
                self.init_height, torch.full_like(self.init_height,
                                                 self.sgs_bank.metadata["object_init_height_m"]),
                atol=1e-6, rtol=0):
            raise ValueError("SGS object_init_height_m does not match the task object")
        expected_size = cfg.get("bank_size")
        if expected_size is not None and (type(expected_size) is not int or expected_size <= 0):
            raise ValueError("SGS bank_size must be null (automatic) or a positive integer")
        if expected_size is not None and self.sgs_bank.num_configs != expected_size:
            raise ValueError("SGS bank_size does not match the loaded bank")
        num_families = len(self.sgs_family_names)
        families = torch.bincount(self.sgs_bank.tensors["family_id"], minlength=num_families)
        if (self.sgs_bank.num_configs % num_families
                or not bool(((families > 0) & (families == self.sgs_bank.num_configs // num_families)).all())):
            raise ValueError("SGS requires equal nonempty reset families")
        self.sgs_sampler = SGSSampler(
            self.sgs_bank.num_configs, history_size=cfg.get("history_size", 100),
            target_success_rate=cfg.get("target_success", 0.5),
            concentration=cfg.get("kappa", 1.0), temperature=cfg.get("temperature", 2.0),
            epsilon=cfg.get("epsilon", 1e-4), p_init=cfg.get("p_init", 0.0),
            seed=self.cfg.get("seed", 43), device=self.device, mode=cfg["mode"])
        self.active_config_id = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._sgs_pending_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._sgs_pending_reset = torch.zeros_like(self._sgs_pending_done)
        self._sgs_has_pending_reset = False
        self._sgs_natural_reset = False
        self.sgs_family_episodes = torch.zeros(num_families, dtype=torch.long, device=self.device)
        self.sgs_family_successes = torch.zeros(num_families, dtype=torch.long, device=self.device)
        self.sgs_family_samples = torch.zeros(num_families, dtype=torch.long, device=self.device)
        # Display-only windows, independent of the per-configuration SGS history.
        self.recent_episode_stats = RecentEpisodeStatistics(
            num_families, device=self.device, family_window=100)
        self.sgs_aborted = torch.zeros((), dtype=torch.long, device=self.device)
        print("SGS bank:", self.sgs_bank.num_configs, "configurations;",
              self.sgs_bank.fingerprint, "; device:", self.device,
              "; bank hold_steps:", self.sgs_bank.metadata["hold_steps"],
              "; task hold_steps:", self.hold_steps,
              "; compatibility:", self.sgs_hold_compatibility)

    def reset(self):
        # Trainer reset() follows a real terminal transition. Direct reset_idx()
        # cancels a trial and must not add a fabricated failure to its history.
        if not getattr(self, "sgs_enabled", False):
            return super().reset()
        self._sgs_natural_reset = True
        try:
            return super().reset()
        finally:
            self._sgs_natural_reset = False

    def _reset_sgs_envs(self, env_ids):
        if len(env_ids) == 0:
            return
        old_ids = self.active_config_id[env_ids]
        started = old_ids >= 0
        completed = started & self._sgs_pending_done[env_ids] & self._sgs_natural_reset
        done_envs, done_ids = env_ids[completed], old_ids[completed]
        successes = self.episode_success[done_envs]
        self.sgs_sampler.observe(done_ids, successes, step=self.global_step_counter)
        done_families = self.sgs_bank.tensors["family_id"][done_ids]
        self.sgs_family_episodes.scatter_add_(0, done_families, torch.ones_like(done_ids))
        self.sgs_family_successes.scatter_add_(0, done_families, successes.long())
        # Within each family, same-step completions use env_ids order (normally
        # ascending environment ID) to break ties at the window boundary.
        self.recent_episode_stats.record(done_families, successes)
        self.sgs_aborted += (started & ~completed).sum()
        # Success counters were already updated by check_termination, exactly
        # once at completion, independently of the reset sampler.
        self.episode_counter[env_ids] = self.episode_counter[env_ids].clamp_min(0)
        self.episode_counter[done_envs] += 1
        self.episode_step_sum[done_envs] += self.one_epoch_step[done_envs]
        self.one_epoch_step[env_ids] = 0

        new_ids = self.sgs_sampler.sample(len(env_ids))
        self.active_config_id[env_ids] = new_ids
        self._sgs_pending_done[env_ids] = False
        state = self.sgs_bank.batch(new_ids)
        self.sgs_family_samples.scatter_add_(0, state["family_id"], torch.ones_like(new_ids))

        # Only clear episode/controller history. Do not invoke the old reset's
        # global simulate(): an asynchronous reset must not advance other trials.
        B1Z1Base._reset_env_tensors(self, env_ids)
        self._reset_pick_state(env_ids)
        # A bank can jump directly into a grasp. Keep its controller/cache
        # isolation intact while sharing task progress and success rules.
        for name in ("actions", "gait_indices", "clock_inputs", "gait_wait_timer",
                     "is_walking", "low_obs_history_buf", "action_data_history_buf",
                     "states_history_buf", "timeout_buf", "foot_contacts_from_sensor",
                     "force_sensor_tensor", "_contact_forces"):
            value = getattr(self, name, None)
            if value is not None:
                value[env_ids] = 0
        if "table_change_interval" in state:
            self.rand_change_step[env_ids] = state["table_change_interval"].long()
        else:
            self.rand_change_step[env_ids] = torch.randint(40, 60, (len(env_ids),), device=self.device)
        self.surpass_dist_flag[env_ids] = True

        self._robot_root_states[env_ids] = state["robot_root"]
        self._table_root_states[env_ids] = state["table_root"]
        self._cube_root_states[env_ids] = state["object_root"]
        self._dof_pos[env_ids] = state["dof_pos"]
        self._dof_vel[env_ids] = state["dof_vel"]
        self._last_dof_vel[env_ids] = state["dof_vel"]
        self.curr_ee_goal_cart[env_ids] = state["ee_goal_cart"]
        self.curr_ee_goal_orn_rpy[env_ids] = state["ee_goal_orn_rpy"]
        self.gripper_dof_pos[env_ids] = state["gripper_target"]
        self.gripper_dof_vel[env_ids] = 0
        self.table_heights[env_ids] = state["table_height"]
        self.table_commands[env_ids] = 0
        self.table_commands[env_ids, 4] = state["table_height"] - self.table_dimz / 2
        self.vel_line_forward_y_level02[env_ids] = state["table_vy"]
        self.lin_circle_select[env_ids] = state["rotation_enabled"]
        self.vel_angular_yaw[env_ids] = state["table_yaw_rate"]
        # Isaac Gym GPU permits one call per state setter between simulate()
        # calls. The next table-control upload commits these roots and DOFs in
        # one batch; otherwise its table-only upload can discard this reset.
        self._sgs_pending_reset[env_ids] = True
        self._sgs_has_pending_reset = True
        # Isaac Gym updates forward kinematics at simulate(). Restore validated
        # caches for the first observation and first IK call, without stepping.
        self._rigid_body_state.view(self.num_envs, -1, 13)[env_ids] = state["rigid_body_state"]
        self.ee_j_eef[env_ids] = state["ee_jacobian"]
        self._update_base_yaw_quat()
        self.update_roboinfo()
        self.last_ee_pos[env_ids] = quat_rotate_inverse(
            self._robot_root_states[env_ids, 3:7], self.ee_pos[env_ids] - self.arm_base[env_ids])
        self.ee_goal_global_cart[env_ids] = quat_apply(
            self._robot_root_states[env_ids, 3:7], state["ee_goal_cart"]) + self.arm_base[env_ids]
        local_orn = quat_from_euler_xyz(*state["ee_goal_orn_rpy"].unbind(-1))
        self.ee_goal_global_orn_quat[env_ids] = quat_mul(
            self._robot_root_states[env_ids, 3:7], local_orn)
        # Use the same distance/height history initialization as an ordinary
        # reset's final refresh, without advancing any environment's physics.
        self._update_curr_dist()
        if self.object_pose_predictor is not None:
            # SGS bypasses the ordinary reset path. Seed the restored bank pose
            # before observations, without carrying history across episodes.
            self.object_pose_predictor.reset(env_ids)
            self.object_pose_predictor.push(
                self._cube_root_states[env_ids, :7], self.gym.get_sim_time(self.sim), env_ids)
        self._compute_observations(env_ids)
        self._record_sgs_metrics()

    def _record_sgs_metrics(self):
        metrics = self.sgs_sampler.metrics(
            step=self.global_step_counter, family_id=self.sgs_bank.tensors["family_id"],
            num_families=len(self.sgs_family_names))
        recent = self.recent_episode_stats
        for i, name in enumerate(self.sgs_family_names):
            metrics["success_" + name] = recent.family_successes[i] / recent.family_count[i].clamp_min(1)
            metrics["successes_" + name] = recent.family_successes[i]
            metrics["sample_fraction_" + name] = self.sgs_family_samples[i] / self.sgs_family_samples.sum().clamp_min(1)
            metrics["episodes_" + name] = recent.family_count[i]
        metrics["aborted"] = self.sgs_aborted
        self.extras["SGS"] = metrics
        # Transfer scalar diagnostics together. Float64 preserves large integer
        # episode counters when combining them with floating-point diagnostics.
        values = torch.stack([value.double() for value in metrics.values()]).detach().cpu().tolist()
        snapshots = {"SGS / " + name: value for name, value in zip(metrics, values)}
        # Keep the old SGS tags, and expose discoverable success-rate curves.
        # No completed episodes means unknown, not an observed zero success rate.
        for name in self.sgs_family_names:
            episodes = snapshots["SGS / episodes_" + name]
            if episodes > 0:
                snapshots["SuccessRate / " + name] = snapshots["SGS / successes_" + name] / episodes
        self.extras.setdefault("success_rate", {}).update(snapshots)
        # Window counts and ratios use the latest snapshot, rather than the
        # logger's ordinary mean over the writing window.
        self.extras["scalar_snapshots"] = snapshots
        self._print_sgs_success_rates()

    def _print_sgs_success_rates(self, force=False):
        step = self.global_step_counter
        last_step = getattr(self, "_sgs_last_print_step", None)
        interval = getattr(self, "sgs_log_interval", 24)
        if not force and last_step is not None and step - last_step < interval:
            return
        snapshots = self.extras.get("scalar_snapshots", {})
        if not snapshots:
            return
        parts = []
        for name in self.sgs_family_names:
            successes = int(snapshots["SGS / successes_" + name])
            episodes = int(snapshots["SGS / episodes_" + name])
            rate = f"{100. * successes / episodes:.2f}%" if episodes else "N/A"
            parts.append(f"{name}={rate} ({successes}/{episodes})")
        suffix = " final" if force else ""
        tqdm.write(f"[SGS step={step}{suffix}] SuccessRate (each family last 100): "
                   + " | ".join(parts))
        self._sgs_last_print_step = step

    def save_sgs_diagnostics(self, path):
        """Export per-configuration statistics; this is not a training resume."""
        if not getattr(self, "sgs_enabled", False):
            return
        from pathlib import Path
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "kind": "sgs_diagnostics_not_training_resume",
            "global_step": self.global_step_counter,
            "bank_fingerprint": self.sgs_bank.fingerprint,
            "bank_metadata": self.sgs_bank.metadata,
            "task_hold_steps": self.hold_steps,
            "hold_compatibility": getattr(self, "sgs_hold_compatibility", "exact"),
            "mode": self.sgs_sampler.mode,
            "family_id": self.sgs_bank.tensors["family_id"].cpu(),
            "success_rate": self.sgs_sampler.success_rates().cpu(),
            "history_count": self.sgs_sampler.count.cpu(),
            "visits_total": self.sgs_sampler.visits_total.cpu(),
            "probability": self.sgs_sampler.probabilities().cpu(),
            "family_samples": self.sgs_family_samples.cpu(),
            "family_episodes": self.sgs_family_episodes.cpu(),
            "family_successes": self.sgs_family_successes.cpu(),
            "recent_family_window": self.recent_episode_stats.family_window,
            "recent_family_episodes": self.recent_episode_stats.family_count.cpu(),
            "recent_family_successes": self.recent_episode_stats.family_successes.cpu(),
        }, output)

    def update_roboinfo(self):
        super().update_roboinfo()
        base_obj_dis = self._cube_root_states[:, :2] - self.arm_base[:, :2]
        self.base_obj_dis = torch.norm(base_obj_dis, dim=-1)
        
    def _setup_obs_and_action_info(self):
        self.pose_prediction_cfg = self.cfg["env"].get("posePrediction", {})
        prediction_enabled = self.pose_prediction_cfg.get("enabled", False)
        if prediction_enabled and (self.no_feature or self.cfg["env"].get("lastCommands", False)):
            raise ValueError("Future pose observations require object features and lastCommands=False")
        self.pose_prediction_horizons = self.pose_prediction_cfg.get("horizons", [0.30, 0.60])
        self.num_future_pose_obs = 6 * len(self.pose_prediction_horizons) + 1 if prediction_enabled else 0
        self.num_velocity_obs = 0 if self.cfg["env"].get("noVelocityObs", False) else 5
        removed_velocity_dim = 5 - self.num_velocity_obs
        # Student states already exclude both velocities, so their size stays fixed.
        super()._setup_obs_and_action_info(
            removed_dim=9+30*6+2-removed_velocity_dim+self.num_future_pose_obs,
            num_action=9,
            num_obs=38+30*6+2+self.num_features-1-removed_velocity_dim+self.num_future_pose_obs,
        )
        
        
    def _extra_env_settings(self):
        self.multi_obj = self.cfg["env"]["asset"]["asset_multi"]  #the cfg here refer to the b1z1_pickmulti.yaml
        self.obj_list = list(self.multi_obj.keys())
        self.obj_height = [self.multi_obj[obj]["height"] for obj in self.obj_list]
        self.obj_orn = [self.multi_obj[obj]["orientation"] for obj in self.obj_list]
        self.obj_scale = [self.multi_obj[obj]["scale"] for obj in self.obj_list]
        obj_dir = os.path.join(self.cfg["env"]["asset"]["assetRoot"], self.cfg["env"]["asset"]["assetFileObj"])
        if not self.no_feature:
            features = []
            for obj_name in self.obj_list:
                feature_path = os.path.join(obj_dir, obj_name, "features.npy")
                feature = np.load(feature_path, allow_pickle=True)
                features.append(feature)
            assert len(features) == len(self.obj_list)
            self.features = np.concatenate(features, axis=0) 
            self.num_features = self.features.shape[1]
        else:
            self.num_features = 0

    def _init_tensors(self):
        """Add extra tensors for cube and table
        """
        super()._init_tensors()
        
        self._table_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 1, :]
        self._initial_table_root_states = self._table_root_states.clone()
        self._initial_table_root_states[:, 7:13] = 0.0
        
        self._cube_root_states = self._root_states.view(self.num_envs, self.num_actors, self._actor_root_state.shape[-1])[..., 2, :]
        self._initial_cube_root_states = self._cube_root_states.clone()
        self._initial_cube_root_states[:, 7:13] = 0.0

        self.object_pose_predictor = None
        self.future_object_obs = torch.zeros((self.num_envs, self.num_future_pose_obs), device=self.device)
        if self.num_future_pose_obs:
            self.object_pose_predictor = RollingSE3Predictor(
                self.num_envs, self.device,
                history=self.pose_prediction_cfg.get("history", 5),
                horizons=self.pose_prediction_horizons,
                q_scale=self.pose_prediction_cfg.get("qScale", 1.0),
                r_scale=self.pose_prediction_cfg.get("rScale", 0.01),
            )
        
        self._table_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.long) + 1
        self._cube_actor_ids = self.num_actors * torch.arange(self.num_envs, device=self.device, dtype=torch.long) + 2
        
        self.table_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.table_handles[0], "box", gymapi.DOMAIN_ENV)
        
        self.lifted_success_threshold = self.cfg["env"]["liftedSuccessThreshold"]
        self.lifted_init_threshold = self.cfg["env"]["liftedInitThreshold"]
        self.base_object_distace_threshold = self.cfg["env"]["baseObjectDisThreshold"] # default 0.6
        self.lifted_object = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.pick_success = torch.zeros_like(self.lifted_object)
        self.pick_success_recorded = torch.zeros_like(self.lifted_object)
        self.pick_hold_progress = torch.zeros_like(self.lifted_object)
        self.best_pick_counter = torch.zeros_like(self.pick_counter)
        self.success_window = RollingEpisodeSuccess(window_size=1000)
        self.episode_outcome_recorded = torch.zeros_like(self.lifted_object)
        
        self.highest_object = -torch.ones(self.num_envs, device=self.device, dtype=torch.float)
        self.curr_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        
        self.random_angle = torch.as_tensor(np.array([0, 1.5708, -1.5708]), device=self.device, dtype=torch.float)
        self.lifted_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.pred_success:
            self.predlift_success_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # SGS and evaluation read the same latched outcome as the original task.
        self.episode_success = self.pick_success_recorded

        

    def _create_extra(self, env_i):
        self.env_ptr = self.envs[env_i]
        col_group = env_i
        col_filter = 0
        i = env_i

        obj_idx = i % len(self.obj_list)
        obj_asset = self.ycb_asset_list[obj_idx]
        obj_height = self.obj_height[obj_idx]

        table_pos = [0.0, 2.0, self.table_dims.z / 2]
        self.table_heights[i] = table_pos[-1] + self.table_dims.z / 2

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0, 0, 0, 1)
        self.table_start_pose = table_start_pose
        self.table_handle = self.gym.create_actor(self.env_ptr, self.table_asset, table_start_pose, "table", col_group, col_filter, 1)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 2.0, 0.5)  # initial position
        pose.r = gymapi.Quat(0, 0, 0.7071, -0.7071)  # initial orientation

        self.cube_start_pose = gymapi.Transform()
        self.cube_start_pose.p.x = table_start_pose.p.x 
        self.cube_start_pose.p.y = table_start_pose.p.y 
        self.cube_start_pose.p.z = self.table_heights[i] + obj_height  

        self.cube_handle = self.gym.create_actor(self.env_ptr, obj_asset, self.cube_start_pose, "cube", col_group, col_filter, 2)

        self.cube_indices[i] = self.gym.get_actor_index(self.env_ptr, self.cube_handle, gymapi.DOMAIN_SIM)

        self.cube_handles.append(self.cube_handle)
        self.table_handles.append(self.table_handle)

        if not self.no_feature:
            self.feature_obs[i, :] = self.features[obj_idx, :]
        self.init_height[i] = obj_height
        self.init_quat[i, :] = torch.tensor(self.obj_orn[obj_idx], device=self.device)

        ## create camera fixed in the floating table ## just when self.camera_test is True to debug  or to catch the image for grasp point prediction 
        if self.camera_test:
            fix_camera_handle = self._create_fix_cameras(self.env_ptr,self.table_handle,i)
            self.fix_camera_handles.append(fix_camera_handle)

            


    def _create_envs(self):
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        asset_file_ycb = self.cfg["env"]["asset"]["assetFileObj"]

        self.table_heights = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # table
        self.table_dimz = 0.1
        self.table_dims = gymapi.Vec3(0.2, 0.2, self.table_dimz)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = False
        table_options.density = 50000
        self.table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_options)
        table_rigid_shape_props = self.gym.get_asset_rigid_shape_properties(self.table_asset)
        table_rigid_shape_props[0].friction = 2.0
        self.gym.set_asset_rigid_shape_properties(self.table_asset, table_rigid_shape_props)


        # cube 
        ycb_opts = gymapi.AssetOptions()
        ycb_opts.use_mesh_materials = True
        ycb_opts.vhacd_enabled = True
        ycb_opts.override_inertia = True
        ycb_opts.override_com = False
        ycb_opts.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        ycb_opts.disable_gravity = False  
        ycb_opts.fix_base_link = False    

        self.ycb_asset_list = []
        cprint(f"############## len(self.obj_list) is {len(self.obj_list)} ################" ,"yellow")
        for i in range(len(self.obj_list)):
            file_path = asset_file_ycb + self.obj_list[i] + "/model.urdf"
            ycb_asset = self.gym.load_asset(self.sim, asset_root, file_path, ycb_opts)

            ycb_asset_props = self.gym.get_asset_rigid_shape_properties(ycb_asset)
            ######## important: because that cube's ycb_asset_props's obj(index) don't just have one ,you may look up to the .obj not just .urdf ######
            for i in range(len(ycb_asset_props)):
                ycb_asset_props[i].friction = 2.0

            self.gym.set_asset_rigid_shape_properties(ycb_asset, ycb_asset_props)
            self.ycb_asset_list.append(ycb_asset)

        self.table_handles, self.cube_handles = [], []
        self.cube_positions = np.zeros((self.num_envs, 2))  # store cube position
        self.cube_angles = np.zeros(self.num_envs)  # store cube angle
        self.cube_speeds = np.zeros(self.num_envs)    

        self.cube_indices = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        if not self.no_feature:
            self.feature_obs = torch.zeros(self.num_envs, self.num_features, device=self.device, dtype=torch.float)
            self.features = torch.tensor(self.features, device=self.device, dtype=torch.float)
        
        self.init_height = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.init_quat = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float)
        

        super()._create_envs()
        


            
    def _record_episode_outcomes(self):
        """Publish the last 1000 completed episodes before step returns infos.

        A reset is not itself an outcome. Only actual terminal steps enter the
        window, once per episode, including failures and timeouts. Simultaneous
        completions have no temporal order; ascending env ID breaks that tie.
        """
        finished = (self.reset_buf.bool() & ~self.episode_outcome_recorded
                    & (self.progress_buf > 0))
        env_ids = finished.nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            return
        outcomes = torch.stack((env_ids % len(self.obj_list),
                                self.pick_success_recorded[env_ids].long()), dim=1)
        self.success_window.extend(outcomes.cpu().tolist())
        self.episode_outcome_recorded[env_ids] = True
        # Replace the snapshot so object-specific rates disappear when their
        # last sample leaves the global window. Logging is independent of W&B.
        metrics = self.success_window.metrics(self.obj_list)
        self.extras["success_rate"] = metrics
        print({"success_rate": metrics, "window_size": self.success_window.window_size})

    def compute_reward(self):
        super().compute_reward()
        # Some optional rewards can also terminate an episode. Record the final
        # done state, after all rewards, and before the trainer receives infos.
        self._record_episode_outcomes()

    def _reset_envs(self, env_ids):
        if getattr(self, "sgs_enabled", False):
            self._reset_sgs_envs(env_ids)
            return
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self.episode_step_sum[env_ids] += self.one_epoch_step[env_ids]
            self.one_epoch_step[env_ids] = 0.0
                

                
    def _reset_objs(self, env_ids):
        if len(env_ids)==0:
            return
        
        self._cube_root_states[env_ids] = self._initial_cube_root_states[env_ids]
        self._cube_root_states[env_ids,0] = self.cube_start_pose.p.x
        self._cube_root_states[env_ids,1] = self.cube_start_pose.p.y
        self._cube_root_states[env_ids,3] = self.cube_start_pose.r.x
        self._cube_root_states[env_ids,4] = self.cube_start_pose.r.y
        self._cube_root_states[env_ids,5] = self.cube_start_pose.r.z
        self._cube_root_states[env_ids,6] = self.cube_start_pose.r.w
        self._cube_root_states[env_ids,7:13] = 0.0
        self._cube_root_states[env_ids,2] = self.table_heights[env_ids] + self.init_height[env_ids]
        
        rand_yaw_box = torch_rand_float(-1.2, 1.2, (len(env_ids), 1), device=self.device).squeeze(1) 
        
        if True: 
            self._cube_root_states[env_ids, 3:7] = self.init_quat[env_ids] # Make sure to learn basic grasp
        else:
            rand_r_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            rand_p_box = self.random_angle[torch_rand_int(0, 3, (len(env_ids),1), device=self.device).squeeze(1)]
            self._cube_root_states[env_ids, 3:7] = quat_mul(quat_from_euler_xyz(rand_r_box, rand_p_box, rand_yaw_box), self.init_quat[env_ids])
        self._cube_root_states[env_ids, 7:13] = 0.
        
    def _reset_table(self, env_ids):
        if len(env_ids)==0:
            return
        
        self._table_root_states[env_ids] = self._initial_table_root_states[env_ids]
        self._table_root_states[env_ids,0] = self.table_start_pose.p.x
        self._table_root_states[env_ids,1] = self.table_start_pose.p.y
        self._table_root_states[env_ids,3] = self.table_start_pose.r.x
        self._table_root_states[env_ids,4] = self.table_start_pose.r.y
        self._table_root_states[env_ids,5] = self.table_start_pose.r.z
        self._table_root_states[env_ids,6] = self.table_start_pose.r.w
        # if self.table_heights_fix is None:
        #     rand_heights = torch_rand_float(0.5, 0.8, (len(env_ids), 1), device=self.device)
        # else:
        #     rand_heights = torch.ones((len(env_ids), 1), device=self.device, dtype=torch.float)*self.table_heights_fix - self.table_dimz / 2
        
        self.table_commands[env_ids,4] = torch_rand_float(0.5, 0.6, (len(env_ids), 1), device=self.device).squeeze(1) - self.table_dimz / 2.0
        self._table_root_states[env_ids, 2] = self.table_commands[env_ids,4]

        self.table_heights[env_ids] = self.table_commands[env_ids,4] + self.table_dimz / 2.0

    
    def _reset_actors(self, env_ids):
        self._reset_table(env_ids)
        self._reset_objs(env_ids)
        super()._reset_actors(env_ids)

        return
    
    def _reset_pick_state(self, env_ids):
        """Reset task progress identically for ordinary and bank initial states."""
        self.lifted_object[env_ids] = 0
        self.lifted_now[env_ids] = False
        self.pick_success[env_ids] = False
        self.pick_success_recorded[env_ids] = False
        self.episode_outcome_recorded[env_ids] = False
        self.pick_hold_progress[env_ids] = False
        self.best_pick_counter[env_ids] = 0
        self.curr_height[env_ids] = 0.
        self.highest_object[env_ids] = -1.

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        robot_ids_int64 = self._robot_actor_ids[env_ids]
        table_ids_int64 = self._table_actor_ids[env_ids]
        cube_ids_int64 = self._cube_actor_ids[env_ids]
        multi_ids_int64 = torch.cat([robot_ids_int64, table_ids_int64, cube_ids_int64], dim=0)
        multi_ids_int32 = multi_ids_int64.to(torch.int32)
        robot_ids_int32 = robot_ids_int64.to(torch.int32)
        
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(robot_ids_int32), len(robot_ids_int64))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                        gymtorch.unwrap_tensor(multi_ids_int32), len(multi_ids_int64))
        self.gym.simulate(self.sim)
        self._refresh_sim_tensors()
        self.render()

        self._reset_pick_state(env_ids)
        if self.object_pose_predictor is not None:
            # super()._reset_envs computes observations immediately after this
            # method. Clear old history BEFORE that first new-episode observation.
            self.object_pose_predictor.reset(env_ids)
            self.object_pose_predictor.push(
                self._cube_root_states[env_ids, :7], self.gym.get_sim_time(self.sim), env_ids)
        return
    
    def _refresh_sim_tensors(self):
        super()._refresh_sim_tensors()
        self._update_curr_dist()
    
    def _update_curr_dist(self):
        # d_s = self.ee_pos - self._cube_root_states[:, :3]
        d = torch.norm(self.ee_pos - self._cube_root_states[:, :3], dim=-1)
        self.curr_dist[:] = d
        self.closest_dist = torch.where(self.closest_dist < 0, self.curr_dist, self.closest_dist)
        
        self.curr_height[:] = self._cube_root_states[:, 2] - self.table_heights - self.init_height
        self.highest_object = torch.where(self.highest_object < 0, self.curr_height, self.highest_object)

    def _update_object_pose_prediction(self):
        if self.object_pose_predictor is not None:
            self.object_pose_predictor.push(self._cube_root_states[:, :7], self.gym.get_sim_time(self.sim))
            self.object_pose_predictor.predict()

    def _future_object_observations(self, env_ids, current_object_obs):
        if self.object_pose_predictor is None:
            return self.future_object_obs[env_ids]
        horizons = len(self.pose_prediction_horizons)
        future = self.object_pose_predictor.future_poses[env_ids]
        # Compute directly from current robot state, including on reset when
        # update_roboinfo() has not run yet. All horizons share this reference.
        robot = self._robot_root_states[env_ids]
        offset = torch.tensor([0.3, 0., 0.09], device=self.device).expand(len(env_ids), -1)
        arm_base = quat_apply(robot[:, 3:7], offset) + robot[:, :3]
        encoded = object_pose_in_teacher_frame(
            future[..., :3].reshape(-1, 3), future[..., 3:7].reshape(-1, 4),
            arm_base[:, None, :].expand(-1, horizons, -1).reshape(-1, 3),
            self.base_yaw_quat[env_ids, None, :].expand(-1, horizons, -1).reshape(-1, 4),
        ).reshape(-1, horizons, 6)
        valid = self.object_pose_predictor.valid[env_ids]
        encoded = torch.where(valid[:, None, None], encoded, current_object_obs[:, None, :])
        self.future_object_obs[env_ids] = torch.cat((encoded.flatten(1), valid[:, None].to(encoded.dtype)), dim=-1)
        return self.future_object_obs[env_ids]

    def _compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            
        obs = super()._compute_observations(env_ids)
        
        if not self.no_feature:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                ####### grasp predict #######
                grasp_predict_world_pos,grasp_predict_world_rpy = self.get_grasp_world(obs[...,-6:], env_ids)

                ####### related pose ########
                grasp_global_rot = quat_from_euler_xyz(grasp_predict_world_rpy[...,0],grasp_predict_world_rpy[...,1],grasp_predict_world_rpy[...,2])
                base_quat = self._robot_root_states[env_ids, 3:7]  # [num_envs, 4], base quaternion
                base_quat = base_quat.unsqueeze(1).repeat(1, grasp_global_rot.shape[1], 1)  # [num_env, n, 4] ， base quaternion
                base_quat_conj = quat_conjugate(base_quat)   # Inverse of base quaternion
                #  Compute local quaternion: q_local = q_base^(-1) * q_global
                ee_grasp_local_orn_quat = quat_mul(base_quat_conj, grasp_global_rot)
                grasp_predict_local_rpy = quat_to_euler_zyx(ee_grasp_local_orn_quat)
                ####### related pose ########

                ####### related orientation ########
                # base_quat = self._robot_root_states[env_ids, 3:7]  # [num_envs, 4], base quaternion
                arm_base = self.arm_base[env_ids].unsqueeze(1).repeat(1, grasp_global_rot.shape[1], 1)
                relative_pos = grasp_predict_world_pos - arm_base
                grasp_predict_local_pos = quat_rotate_inverse_batched(base_quat, relative_pos)
                grasp_predict_local_pos[..., 2] = torch.clip(grasp_predict_local_pos[..., 2], -0.6, 0.6)
                ####### related orientation ########

                ####### grasp predict #######

                grasp_predict_world_pos_flat = grasp_predict_local_pos.view(grasp_predict_local_pos.shape[0], -1) ## [num_env,3*object_num]
                grasp_predict_world_rpy_flat = grasp_predict_local_rpy.view(grasp_predict_local_rpy.shape[0], -1) ## [num_env,3*object_num]
                future_obs = self._future_object_observations(env_ids, obs[:, :6])
                obs_new = torch.cat([obs[...,:-6],future_obs,grasp_predict_world_pos_flat,grasp_predict_world_rpy_flat],dim = -1)

                self.obs_buf[env_ids] = torch.cat([self.feature_obs[env_ids, :], obs_new, self.action_history_buf[env_ids, -1]], dim=-1) # self.action_history_buf: [num_env,3,9], obs: [num_env, 61]

        else:
            if self.cfg["env"].get("lastCommands", False):
                self.obs_buf[env_ids] = torch.cat([obs, self.command_history_buf[env_ids, -1]], dim=-1)
            else:
                self.obs_buf[env_ids] = torch.cat([obs, self.action_history_buf[env_ids, -1]], dim=-1)
    
    def _compute_robot_obs(self, env_ids=None):
        if env_ids is None:
            robot_root_state = self._robot_root_states
            table_root_state = self._table_root_states
            cube_root_state = self._cube_root_states
            body_pos = self._rigid_body_pos
            body_rot = self._rigid_body_rot
            body_vel = self._rigid_body_vel
            body_ang_vel = self._rigid_body_ang_vel
            dof_pos = self._dof_pos
            dof_vel = self._dof_vel
            commands = self.commands
            table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(self.num_envs, 1).to(self.device)
            base_quat_yaw = self.base_yaw_quat
            spherical_center = self.get_ee_goal_spherical_center()
            ee_goal_cart = self.curr_ee_goal_cart
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy
        else:
            robot_root_state = self._robot_root_states[env_ids] 
            table_root_state = self._table_root_states[env_ids] 
            cube_root_state = self._cube_root_states[env_ids] 
            body_pos = self._rigid_body_pos[env_ids]
            body_rot = self._rigid_body_rot[env_ids]
            body_vel = self._rigid_body_vel[env_ids]
            body_ang_vel = self._rigid_body_ang_vel[env_ids]
            dof_pos = self._dof_pos[env_ids]
            dof_vel = self._dof_vel[env_ids]
            commands = self.commands[env_ids]
            # table_dim = torch.tensor([0.6, 1.0, self.table_dimz]).repeat(len(env_ids), 1).to(self.device)
            table_dim = torch.tensor([self.table_dims.x, self.table_dims.y, self.table_dims.z]).repeat(len(env_ids), 1).to(self.device) ## my change _table_dim
            base_quat_yaw = self.base_yaw_quat[env_ids]
            spherical_center = self.get_ee_goal_spherical_center()[env_ids]
            ee_goal_cart = self.curr_ee_goal_cart[env_ids]
            ee_goal_orn_rpy = self.curr_ee_goal_orn_rpy[env_ids]
            cube_vel = cube_root_state[:, 7:10]
            cube_angular = cube_root_state[:, 10:]
            cube_vel_local = quat_rotate_inverse(base_quat_yaw, cube_vel)
            cube_angular_local = quat_rotate_inverse(base_quat_yaw, cube_angular)
            ee_goal_global_cart = self.ee_goal_global_cart[env_ids]
            ee_goal_orn_quat = self.ee_goal_global_orn_quat[env_ids]
        
        obs = compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos,
                                         body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, self.gripper_idx, table_dim,
                                         ee_goal_cart, ee_goal_orn_rpy, ee_goal_global_cart, ee_goal_orn_quat, self.use_roboinfo, self.floating_base)
        if self.num_velocity_obs == 0:
            # Tail: robot velocity (3), relative object velocity (2), world pose (6).
            # Keep the world pose for get_grasp_world; remove velocities, not zero them.
            obs = torch.cat([obs[..., :-11], obs[..., -6:]], dim=-1)
        
        return obs
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        cmd_dim = 2
        pred_dim = 1 if self.pred_success else 0
        if self.near_goal_stop:
            self.extras["replaced_action"] = torch.clone(actions)
            ############### when the end-effector and the object 's distance is lower than 0.6m , the action will not to be applied. ############
            self.extras["replaced_action"][self.base_obj_dis < 0.6, -(cmd_dim+pred_dim):-pred_dim] = 0.0 # enforced these cmd to be 0
            # if not self.enable_camera:
            actions = self.extras["replaced_action"]
        
        # Randomly change the object position in a small probability (like 0.1)
        obj_move_prob = torch_rand_float(0, 1, (self.num_envs, 1), device=self.device).squeeze()
        changed_env_ids = torch.range(0, self.num_envs-1, dtype=int, device=self.device)[obj_move_prob < self.obj_move_prob] # This is a tensor with the length of env_nums
        self.now_env_ids = changed_env_ids
        self._reset_objs(changed_env_ids)

        self.extras["lifted_now"] = self.lifted_now.unsqueeze(-1)*2-1 # This is for the lifted results from the last step, exactly what we want. Lifted = 1, unlifted = -1
        res = super().step(actions)
        if getattr(self, "sgs_enabled", False):
            self._sgs_pending_done.copy_(self.reset_buf.bool())
            self._print_sgs_success_rates()

        pred_lift = actions[...,-1] > 0
        if self.pred_success and (actions.shape[-1] == (self.action_space.shape[-1]+1)):
            pred_true = (self.lifted_now == pred_lift)
            self.predlift_success_counter = torch.where(pred_true, self.predlift_success_counter + 1, self.predlift_success_counter)

        ####### obeserve the velocity ###############
        # _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # rb_states = gymtorch.wrap_tensor(_rb_states)
        ####################################################
        
        return res
    
    def check_termination(self):
        super().check_termination()

        # One grasp/lift definition for shaping, holding and evaluation.
        cube_height = self._cube_root_states[:, 2]
        box_pos = self._cube_root_states[:, :3]
        d1 = torch.norm(box_pos - self.ee_pos, dim=-1)
        self.lifted_now[:] = lifted_grasp_mask(
            cube_height, self.table_heights, self.init_height, d1,
            self.actions[:, 6], self.lifted_success_threshold)
        self.lifted_object[:] = self.lifted_now

        z_cube = self._cube_root_states[:, 2]
        # cube_falls = (z_cube < (self.table_heights + 0.03 / 2 - 0.05))
        cube_falls = z_cube+0.015 < self.table_heights # Fall or model glitch
        self._terminate_buf[:] |= cube_falls
        
        if self.enable_camera and self.cfg["env"].get("cameraVisibilityReset", True):
            robot_head_dir = quat_apply(self.base_yaw_quat, torch.tensor([1., 0., 0.], device=self.device).repeat(self.num_envs, 1))
            cube_dir = self._cube_root_states[:, :3] - self._robot_root_states[:, :3]
            cube_dir[:, 2] = 0
            cube_dir = cube_dir / torch.norm(cube_dir, dim=-1).unsqueeze(-1)
            # check if dot product is negative
            deviate_much = torch.sum(robot_head_dir * cube_dir, dim=-1) < 0. 
            fov_camera_pos = self._robot_root_states[:, :3] + quat_apply(self._robot_root_states[:, 3:7], torch.tensor(self.cfg["sensor"]["onboard_camera"]["position"], device=self.device).repeat(self.num_envs, 1))
            too_close_table = (fov_camera_pos[:, 0] > 0.)
            
            self._terminate_buf[:] |= deviate_much | too_close_table

        # Resolve hard failures before awarding success, including the yaw limit.
        self._terminate_buf[:] |= self._yaw_deviation() > np.deg2rad(70.0)
        self.reset_buf[:] |= self._terminate_buf
        count, best_count, progress, success = advance_pick_hold(
            self.lifted_object, self._terminate_buf.bool(), self.pick_counter,
            self.best_pick_counter, self.pick_success_recorded, self.hold_steps)
        self.pick_counter[:] = count
        self.best_pick_counter[:] = best_count
        self.pick_hold_progress[:] = progress
        self.pick_success[:] = success
        self.pick_success_recorded |= success
        self.success_counter[success] += 1
        self.success_onetime_counter[success & self.surpass_dist_flag] += 1
        self.reset_buf[:] |= self.pick_success_recorded
        # The last allowed step may complete the task. Hard failures still win.
        self.timeout_buf[:] &= ~(self.pick_success_recorded | self._terminate_buf.bool())

    # --------------------------------- reward functions ---------------------------------
    
    def _yaw_deviation(self):
        yaw = euler_from_quat(self._robot_root_states[:, 3:7])[2]
        return torch.abs(yaw - self.initial_yaw_z)

    def _reward_limit_yaw_rotation_penalty(self):
        delta_yaw = self._yaw_deviation()
        reward = torch.where(delta_yaw > np.deg2rad(60.0),
                             -torch.tanh(delta_yaw), torch.zeros_like(delta_yaw))
        return reward, reward

    # def _reward_gripper_grasp_expect_reward(self):
    #     reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    #     mask = self.curr_dist < 0.1
    #     torch.abs(self._cube_root_states[:, 8]) > 0.1
    #     return reward,reward

    # def _reward_standpick(self):    # for when the ee approach the cube and encourage the robot to stand statically to grasp, but may not be suitable for aynamic grasping
    #     reward = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
    #     reward[(self.base_obj_dis < self.base_object_distace_threshold) & (self.commands[:, 0] < LIN_VEL_X_CLIP)] = 1.0
    #     if self.global_step_counter < self.train_reward_strict:
    #         reward = 0.
    #     return reward, reward
    
    def _reward_grasp_base_height(self):
        reward, _ = self._reward_base_height()
        # Repeating an interrupted hold must not earn its shaping reward again.
        reward *= self.pick_hold_progress
        
        return reward, reward
    
    def _reward_approaching(self):
        """Reward approaching until the object is grasped at the target height."""
        reward, _ = super()._reward_approaching()
        reward *= ~self.lifted_object
        return reward, reward
    
    def _reward_base_approaching(self):
        rew, _ = super()._reward_base_approaching(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_command_reward(self):
        rew, _ = super()._reward_command_reward(self._cube_root_states[:, :3])
        return rew, rew
    
    # def _reward_command_penalty(self):
    #     rew, _ = super()._reward_command_penalty(self._cube_root_states[:, :3])
    #     return rew, rew
    
    def _reward_ee_orn(self):
        rew, _ = super()._reward_ee_orn(self._cube_root_states[:, :3])
        return rew, rew
    
    def _reward_base_dir(self):
        rew, _ = super()._reward_base_dir(self._cube_root_states[:, :3])
        return rew, rew
    
    # --------------------------------- reward functions ---------------------------------
    


# --------------------------------- jit functions ---------------------------------

@torch.jit.script
def compute_robot_observations(robot_root_state, table_root_state, cube_root_state, body_pos, 
                               body_rot, body_vel, body_ang_vel, dof_pos, dof_vel, base_quat_yaw, spherical_center, commands, gripper_idx, table_dim, 
                               ee_goal_cart, ee_goal_orn_rpy, ee_goal_global_cart, ee_goal_orn_quat, use_roboinfo, floating_base) :
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, Tensor, Tensor, Tensor, Tensor, Tensor,bool, bool) ->  torch.Tensor
    cube_pos = cube_root_state[:, :3]
    cube_orn = cube_root_state[:, 3:7]
    cube_rpy = torch.stack(euler_from_quat(cube_orn), dim=-1) # rot-squen: x-y-z
    cube_vel = cube_root_state[:, 7:10]
    cube_angular = cube_root_state[:, 10:]
    car_mainbody_vel = table_root_state[:, 7:10]
    car_mainbody_angular = table_root_state[:, 10:]
    
    ee_pos = body_pos[..., gripper_idx, :]
    ee_rot = body_rot[..., gripper_idx, :]
    ee_vel = body_vel[..., gripper_idx, :]
    ee_ang_vel = body_ang_vel[..., gripper_idx, :]
    # box pos and orientation  3+4=7
    # dof pos + vel  6+6=12
    # ee state  13
    if use_roboinfo:
        dof_pos = dof_pos[..., :]
        dof_vel = dof_vel[..., :-1] * 0.05
        if not floating_base:
            dof_pos = reindex_all(dof_pos)
            dof_vel = reindex_all(dof_vel)
    else:
        dof_pos = dof_pos[..., 12:-1] # arm_dof_pos
        dof_vel = dof_vel[..., 12:-1] # arm_dof_vel
    
    base_quat = robot_root_state[:, 3:7]
    arm_base_local = torch.tensor([0.3, 0.0, 0.09], device=robot_root_state.device).repeat(robot_root_state.shape[0], 1) ## have to change if change robot model
    arm_base = quat_apply(base_quat, arm_base_local) + robot_root_state[:, :3]
    
    object_pose_obs = object_pose_in_teacher_frame(cube_pos, cube_orn, arm_base, base_quat_yaw)
    cube_pos_local = object_pose_obs[:, :3]
    cube_orn_local_rpy = object_pose_obs[:, 3:]
    cube_vel_local = quat_rotate_inverse(base_quat_yaw, cube_vel)
    cube_angular_local = quat_rotate_inverse(base_quat_yaw, cube_angular) 
    car_mainbody_vel_local = quat_rotate_inverse(base_quat_yaw, car_mainbody_vel)
    car_mainbody_angular_local = quat_rotate_inverse(base_quat_yaw, car_mainbody_angular)
    car_mainbody_vel_local[...,0] = -car_mainbody_vel[...,1]
    car_mainbody_vel_local[...,1] = car_mainbody_vel[...,0]
    car_mainbody_angular_local[...,0] = -car_mainbody_angular[...,1]
    car_mainbody_angular_local[...,0] = car_mainbody_angular[...,0]
    
    table_pos_local = quat_rotate_inverse(base_quat_yaw, table_root_state[:, :3] - spherical_center)
    table_orn_local = quat_mul(quat_conjugate(base_quat_yaw), table_root_state[:, 3:7])
    table_dim_local = quat_rotate_inverse(base_quat_yaw, table_dim)

    ee_pos_local = quat_rotate_inverse(base_quat, ee_pos - arm_base)
    ee_rot_local = quat_mul(quat_conjugate(base_quat), ee_rot)
    ee_rot_local_rpy = torch.stack(euler_from_quat(ee_rot_local), dim=-1)
    
    robot_vel_local = quat_rotate_inverse(base_quat_yaw, robot_root_state[:, 7:10])
    cube_vel_local = cube_vel_local - robot_vel_local
    cube_vel_local = cube_vel_local[...,:2]
    
    obs = torch.cat((cube_pos_local, cube_orn_local_rpy, ee_pos_local, ee_rot_local_rpy, dof_pos, dof_vel,
                     commands, ee_goal_cart, ee_goal_orn_rpy, robot_vel_local ,cube_vel_local ,cube_pos, cube_rpy), dim=-1) ## 61 + 6
    obs_have_cube_vel = torch.cat((cube_pos_local, cube_orn_local_rpy, car_mainbody_vel_local, car_mainbody_angular_local, ee_pos_local, ee_rot_local_rpy, dof_pos, dof_vel,
                     commands, ee_goal_cart, ee_goal_orn_rpy, robot_vel_local), dim=-1) # if have robo_info: have total 67 dim, if not, have 67-24 dim

    return obs
