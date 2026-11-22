"""Main line A inside the pick task: T, interception margin, shaping, tokens, dynamic SGS.

B1Z1PickMulti mixes this class in and calls its hooks. Everything is a no-op
unless env.margin.reachModel is set. Pure torch: tests run these methods on
simulator stand-ins without importing Isaac Gym.

Information flow (doc/主线A_实现说明.md):
  privileged (training only)  true table motion -> M_priv -> margin shaping, curriculum
  real-obtainable (actor)     world-frame EKF forecasts -> per-horizon tokens + M_ekf
The actor never sees M_priv; T itself only uses proprioception and goals.
"""

import torch

from modules.pose_prediction import RollingSE3Predictor
from modules.reachability import (MARGIN_SCALAR_NAMES, MARGIN_TOKEN_DIM, MARGIN_TOKEN_NAMES,
                                  ReachLearner, TimeToReach, grasp_positions_world,
                                  interception_margin, load_reach_model, margin_horizons,
                                  matrix_to_6d, planar_carrier_future, query_frame,
                                  reach_state_dim, reach_state_features,
                                  selected_grasp_rotations, to_query_frame, yaw_from_quat,
                                  yaw_matrix)
from utils.config import validate_margin_config
from utils.episode_statistics import RecentEpisodeStatistics
from utils.margin_curriculum import (choose_frontier, initial_margins, margin_delta,
                                     prior_success_rates)

FRONTIER_FAMILY = "full_task"
# Validation (gate D2) and the curriculum prior refresh every this many online updates.
MARGIN_REFRESH_UPDATES = 80


class InterceptionMarginMixin:
    # ------------------------------------------------------------------ layout
    def _setup_margin_obs(self):
        """Called by _setup_obs_and_action_info, before the observation size is fixed."""
        settings = validate_margin_config(self.cfg["env"], self.cfg.get("reward", {}))
        self.margin_settings = settings
        self.margin_enabled = bool(settings["reachModel"])
        self.margin_horizon_list = margin_horizons(0.0, settings["horizonStep"], settings["horizonCount"])
        tokens = self.margin_enabled and settings["tokens"]
        if tokens:
            if self.no_feature or self.cfg["env"].get("lastCommands", False):
                raise ValueError("Margin tokens require object features and lastCommands=False")
            if settings["horizonCount"] < 2:
                raise ValueError("Margin tokens need env.margin.horizonCount >= 2")
        self.num_margin_tokens = settings["horizonCount"] if tokens else 0
        self.num_margin_obs = (len(MARGIN_SCALAR_NAMES) + self.num_margin_tokens * MARGIN_TOKEN_DIM
                               if tokens else 0)

    # ---------------------------------------------------------- initialization
    def _init_margin(self):
        """After PredictPoint (grasp candidates) and SGS (bank) are ready."""
        if not getattr(self, "margin_enabled", False):
            return
        settings = self.margin_settings
        if getattr(self, "tsil_enabled", False):
            raise ValueError("The interception margin is not combined with TSIL")
        if self.floating_base or self.enable_camera:
            raise ValueError("The interception margin requires the walking teacher without cameras")
        device = self.device
        num = self.num_envs
        self.margin_horizons = torch.tensor(self.margin_horizon_list, dtype=torch.float32, device=device)
        # rand_control executes 7 or 8 low-level cycles per action: 0.15 s on average.
        cycles = self.control_freq_low_init - (0.5 if self.rand_control else 0.0)
        self.margin_step_seconds = float(cycles * self.dt)
        self.margin_clip = float(settings["clip"])
        self.margin_chunk = settings["chunkEnvs"]
        self.margin_grasp_in_object = self.T_grasp_in_obj[:, :settings["candidates"]].to(device)
        self.margin_grasp_correction = (self.Rx_rot @ self.Rz_rot @ self.Rx_rot).to(device)
        self.margin_all_ids = torch.arange(num, device=device)

        state_dim = reach_state_dim(self.num_gripper_dof)
        spec = settings["reachModel"]
        self.margin_model_metadata = {}
        if spec == "analytic":
            model = TimeToReach(state_dim, kind="analytic")
        elif spec.startswith("new:"):
            model = TimeToReach(state_dim, kind=spec[len("new:"):])
        else:
            model, self.margin_model_metadata = load_reach_model(spec, device=device)
            if model.state_dim != state_dim:
                raise ValueError("reach model state size does not match this robot's gripper DOFs")
            saved_index = self.margin_model_metadata.get("gripper_body_index")
            if saved_index is not None and saved_index != self.gripper_idx:
                raise ValueError("reach model was trained with a different EE body index")
        self.margin_model = model.to(device).eval()
        self.margin_learner = None
        if settings["online"]:
            self.margin_learner = ReachLearner(
                self.margin_model, num, device, window=settings["window"],
                max_steps=settings["maxPairSteps"], batch_size=settings["onlineBatch"],
                learning_rate=settings["onlineLearningRate"], expectile=settings["expectile"],
                ema_decay=settings["onlineEmaDecay"], seed=self.cfg.get("seed", 43))
        self.margin_online_updates = 0
        self.margin_freeze_step = settings["onlineFreezeFraction"] * self.total_timesteps

        self.margin_priv = torch.zeros(num, device=device)
        self.margin_priv_horizon = torch.zeros(num, device=device)
        self.margin_ekf = torch.zeros(num, device=device)
        self.margin_obs_buf = torch.zeros((num, self.num_margin_obs), device=device)
        self.margin_phi_prev = torch.zeros(num, device=device)
        self.margin_episode_id = torch.zeros(num, dtype=torch.long, device=device)
        self.margin_episode_time = torch.zeros(num, device=device)
        self.margin_world_predictor = None
        if self.num_margin_tokens:
            # The same filter as the observation EKF, fed with world poses: on
            # hardware, detections composed with odometry over the EKF window.
            prediction = getattr(self, "pose_prediction_cfg", {})
            self.margin_world_predictor = RollingSE3Predictor(
                num, device, history=prediction.get("history", 5),
                horizons=self.margin_horizon_list[1:], q_scale=prediction.get("qScale", 1.0),
                r_scale=prediction.get("rScale", 0.01))

        self.margin_curriculum = "disabled" if self.eval else settings["curriculum"]
        if self.margin_curriculum != "disabled":
            if not getattr(self, "sgs_enabled", False):
                raise ValueError("env.margin.curriculum requires SGS (--sgs_mode sgs)")
            if FRONTIER_FAMILY not in self.sgs_family_names:
                raise ValueError("the margin curriculum needs a full_task SGS family")
        self.margin_frontier_active = torch.zeros(num, dtype=torch.bool, device=device)
        self.margin_frontier_scale = torch.ones(num, device=device)
        self.margin_frontier_stats = RecentEpisodeStatistics(1, device=device, family_window=100)
        self.margin_bank_initial = None
        self.margin_diagnostics = {}
        if self.margin_curriculum != "disabled":
            self._refresh_margin_prior(recompute=True)

    def _margin_query_model(self):
        learner = getattr(self, "margin_learner", None)
        return self.margin_model if learner is None else learner.target

    def _margin_sim_time(self):
        return self.gym.get_sim_time(self.sim)

    # ------------------------------------------------------------ state, futures
    def _margin_reach_state(self, ids):
        g = self.num_gripper_dof
        robot = self._robot_root_states[ids]
        dof_pos, dof_vel = self._dof_pos[ids], self._dof_vel[ids]
        commands = self.commands[ids]
        return reach_state_features(
            robot[:, :3], robot[:, 3:7], robot[:, 10:13], self.ee_pos[ids], self.ee_orn[ids],
            dof_pos[:, -(6 + g):-g], dof_vel[:, -(6 + g):-g], dof_pos[:, -g:],
            self.curr_ee_goal_cart[ids], self.curr_ee_goal_orn_rpy[ids],
            torch.stack((commands[:, 0], commands[:, 2]), dim=-1))

    def _margin_table_twist(self, ids):
        """Planar body velocity [n, 2] and yaw rate [n] of the table (privileged).

        Level00-02 read the episode's motion parameters, which are set before
        the first step (table_commands are still zero right after a reset).
        """
        level = self.table_level
        zeros = torch.zeros(len(ids), device=self.device)
        if level == "Level02":
            vy = self.vel_line_forward_y_level02[ids]
            yaw_rate = torch.where(self.lin_circle_select[ids], self.vel_angular_yaw[ids], zeros)
            return torch.stack((zeros, vy), dim=-1), yaw_rate
        if level == "Level01":
            return torch.stack((zeros, self.vel_line_forward_y[ids]), dim=-1), zeros
        if level == "Level00":
            return torch.stack((zeros, zeros), dim=-1), zeros
        commands = self.table_commands[ids]
        return commands[:, 0:2], commands[:, 5]

    def _margin_privileged_future(self, ids):
        table, obj = self._table_root_states[ids], self._cube_root_states[ids]
        velocity, yaw_rate = self._margin_table_twist(ids)
        return planar_carrier_future(obj[:, :3], obj[:, 3:7], table[:, :3], yaw_from_quat(table[:, 3:7]),
                                     velocity, yaw_rate, self.margin_horizons)

    def _margin_ekf_future(self, ids):
        """World-frame EKF poses [n, H, 3/4] (h = 0 is the measurement) and validity [n]."""
        current = self._cube_root_states[ids, :7]
        predictor = self.margin_world_predictor
        valid = predictor.valid[ids]
        future = torch.where(valid[:, None, None], predictor.future_poses[ids], current[:, None, :])
        poses = torch.cat((current[:, None, :], future), dim=1)
        return poses[..., :3], poses[..., 3:7], valid

    @torch.no_grad()
    def _margin_times(self, model, state, origin, yaw, obj_pos, obj_quat, ids):
        """T [n, H, C] to every candidate at every horizon, and the goals [n, H, C, 3]."""
        goals = to_query_frame(grasp_positions_world(obj_pos, obj_quat, self.margin_grasp_in_object[ids]),
                               origin, yaw)
        n, h, c, _ = goals.shape
        return model(state, goals.reshape(n, h * c, 3)).reshape(n, h, c), goals

    # ------------------------------------------------------------------ margins
    @torch.no_grad()
    def _update_margins(self, env_ids=None, record=False):
        ids = self.margin_all_ids if env_ids is None else env_ids
        robot = self._robot_root_states[ids]
        state = self._margin_reach_state(ids)
        origin, yaw = query_frame(robot[:, :3], robot[:, 3:7])
        learner = self.margin_learner
        if record and learner is not None:
            learner.observe(state, self.ee_pos[ids], origin, yaw, self.margin_episode_time[ids],
                            self.margin_episode_id[ids])
        remaining = ((self.max_episode_length - self.progress_buf[ids]).clamp_min(0).float()
                     * self.margin_step_seconds)
        model = self._margin_query_model()
        priv_pos, priv_quat = self._margin_privileged_future(ids)
        tokens = self.num_margin_tokens > 0
        if tokens:
            ekf_pos, ekf_quat, ekf_valid = self._margin_ekf_future(ids)
        for start in range(0, len(ids), self.margin_chunk):
            part = slice(start, start + self.margin_chunk)
            sub = ids[part]
            times, _ = self._margin_times(model, state[part], origin[part], yaw[part],
                                          priv_pos[part], priv_quat[part], sub)
            result = interception_margin(times, self.margin_horizons, remaining[part])
            self.margin_priv[sub] = result["margin"]
            self.margin_priv_horizon[sub] = self.margin_horizons[result["horizon"]]
            if tokens:
                times, goals = self._margin_times(model, state[part], origin[part], yaw[part],
                                                  ekf_pos[part], ekf_quat[part], sub)
                result = interception_margin(times, self.margin_horizons, remaining[part])
                self.margin_ekf[sub] = result["margin"]
                self.margin_obs_buf[sub] = self._margin_token_block(
                    result, goals, ekf_quat[part], yaw[part], ekf_valid[part], sub)

    def _margin_token_block(self, result, goals, obj_quat, yaw, ekf_valid, ids):
        """[n, 2 + H * 12]: M_ekf, best horizon, then per-horizon best-candidate tokens."""
        clip = self.margin_clip
        candidate = result["candidate"]
        n, h = candidate.shape
        position = goals.gather(2, candidate[..., None, None].expand(-1, -1, 1, 3)).squeeze(2)
        rotation = selected_grasp_rotations(obj_quat, self.margin_grasp_in_object[ids], candidate,
                                            self.margin_grasp_correction)
        rotation = yaw_matrix(yaw).transpose(-1, -2)[:, None] @ rotation
        horizons = self.margin_horizons.expand(n, -1)
        margins = result["per_horizon"].clamp(-clip, clip)  # -inf (h > remaining) -> -clip
        valid = result["valid"] & (ekf_valid[:, None] | (horizons == 0))
        tokens = torch.cat((position, matrix_to_6d(rotation), horizons[..., None], margins[..., None],
                            valid[..., None].to(position.dtype)), dim=-1)
        scalars = torch.stack((result["margin"].clamp(-clip, clip),
                               self.margin_horizons[result["horizon"]]), dim=-1)
        return torch.cat((scalars, tokens.flatten(1)), dim=-1)

    def _margin_observations(self, env_ids, like):
        """The actor-visible block for _compute_observations (zeros before init)."""
        width = getattr(self, "num_margin_obs", 0)
        buffer = getattr(self, "margin_obs_buf", None)
        if not width:
            return like.new_empty((len(env_ids), 0))
        if buffer is None:
            return like.new_zeros((len(env_ids), width))
        return buffer[env_ids].to(like.dtype)

    # ------------------------------------------------------------------- hooks
    def _reset_margin_state(self, env_ids):
        """After a reset wrote the new initial state, before its first observation."""
        if not getattr(self, "margin_enabled", False) or len(env_ids) == 0:
            return
        self.margin_episode_id[env_ids] += 1
        self.margin_episode_time[env_ids] = 0.
        if self.margin_world_predictor is not None:
            self.margin_world_predictor.reset(env_ids)
            self.margin_world_predictor.push(self._cube_root_states[env_ids, :7], self._margin_sim_time(), env_ids)
        self._update_margins(env_ids)
        self.margin_phi_prev[env_ids] = self._margin_potential(env_ids)

    def _margin_step_update(self):
        """Post-physics, before termination and rewards: one update per high-level step."""
        if not getattr(self, "margin_enabled", False):
            return
        self.margin_episode_time += self.control_freq_low * self.dt
        if self.margin_world_predictor is not None:
            self.margin_world_predictor.push(self._cube_root_states[:, :7], self._margin_sim_time())
            self.margin_world_predictor.predict()
        self._update_margins(record=True)
        diagnostics = {
            "Margin / privileged mean (s)": self.margin_priv.mean(),
            "Margin / privileged nonnegative fraction": (self.margin_priv >= 0).float().mean(),
            "Margin / privileged best horizon mean (s)": self.margin_priv_horizon.mean(),
        }
        if self.num_margin_tokens:
            diagnostics["Margin / EKF mean (s)"] = self.margin_ekf.mean()
            diagnostics["Margin / EKF minus privileged abs mean (s)"] = (self.margin_ekf - self.margin_priv).abs().mean()
        if self.margin_curriculum == "frontier":
            diagnostics["Curriculum / frontier active fraction"] = self.margin_frontier_active.float().mean()
            active = self.margin_frontier_active.float()
            diagnostics["Curriculum / frontier speed scale mean"] = (
                (self.margin_frontier_scale * active).sum() / active.sum().clamp_min(1.))
        if self.margin_curriculum != "disabled":
            diagnostics["Curriculum / delta (s)"] = torch.tensor(self._margin_delta(), device=self.device)
        diagnostics.update(self._margin_online_update())
        self.margin_diagnostics = diagnostics
        self.extras["diagnostics"] = diagnostics

    def _margin_online_update(self):
        learner = self.margin_learner
        settings = self.margin_settings
        if (learner is None or self.eval or self.global_step_counter % settings["onlineInterval"]
                or self.global_step_counter >= self.margin_freeze_step):
            return {}
        metrics = learner.update(settings["onlineUpdates"])
        if not metrics:
            return {}
        self.margin_online_updates += 1
        result = {"Reach / " + name: value for name, value in metrics.items()}
        if self.margin_online_updates % MARGIN_REFRESH_UPDATES == 0:
            quality = learner.validate()
            result.update({"Reach / validation " + name: value for name, value in quality.items()})
            if self.margin_curriculum != "disabled":
                self._refresh_margin_prior(recompute=True)
        return result

    # ------------------------------------------------------------------ shaping
    def _margin_potential(self, env_ids=None):
        """Phi in [0, 1]: (clip(M, -c, c) / c + 1) / 2, and 1 once the object is lifted.

        Non-negative, so a termination never pays a shaping bonus.
        """
        ids = self.margin_all_ids if env_ids is None else env_ids
        clip = self.margin_clip
        phi = 0.5 * (self.margin_priv[ids].clamp(-clip, clip) / clip + 1.0)
        return torch.where(self.lifted_object[ids], torch.ones_like(phi), phi)

    def _reward_margin_shaping(self):
        """gamma * Phi(s') - Phi(s) with Phi(terminal) = 0 (Ng et al. 1999)."""
        if not getattr(self, "margin_enabled", False) or not self.margin_settings["reward"]:
            raise ValueError("reward.scales.margin_shaping requires env.margin.reward")
        phi = self._margin_potential()
        terminal = self.reset_buf.bool()
        gamma = self.reward_options["shapingGamma"]
        reward = gamma * torch.where(terminal, torch.zeros_like(phi), phi) - self.margin_phi_prev
        self.margin_phi_prev.copy_(phi)
        return reward, phi

    # --------------------------------------------------------------- curriculum
    def _margin_delta(self):
        settings = self.margin_settings
        return margin_delta(self.global_step_counter, self.total_timesteps,
                            settings["deltaStart"], settings["deltaEnd"])

    def _margin_initial_margins(self, state, speed_scales=None):
        """M0 [n, K] of bank-format initial states (privileged table motion)."""
        return initial_margins(
            self._margin_query_model(), state, self.margin_grasp_in_object[0], self.margin_horizons,
            self.max_episode_length * self.margin_step_seconds, self.gripper_idx, self.num_gripper_dof,
            speed_scales=speed_scales, chunk=self.margin_chunk)

    def _refresh_margin_prior(self, recompute=False):
        if getattr(self, "margin_curriculum", "disabled") == "disabled":
            return
        if recompute or self.margin_bank_initial is None:
            self.margin_bank_initial = self._margin_initial_margins(self.sgs_bank.tensors)[:, 0]
        settings = self.margin_settings
        rates = prior_success_rates(self.margin_bank_initial, self._margin_delta(),
                                    settings["priorTemperature"])
        self.sgs_sampler.set_prior(rates, settings["priorStrength"])

    def _margin_apply_frontier(self, env_ids, state):
        """Speed up some full_task resets to the M0 frontier; flags them in margin_frontier_active.

        Their outcomes are not observations of the bank configuration: SGS
        history, family windows and the full_task success window skip them.
        """
        self.margin_frontier_active[env_ids] = False
        self.margin_frontier_scale[env_ids] = 1.
        if getattr(self, "margin_curriculum", "disabled") != "frontier" or len(env_ids) == 0:
            return state
        settings = self.margin_settings
        full_task = self.sgs_family_names.index(FRONTIER_FAMILY)
        draw = torch.rand(len(env_ids), device=self.device)
        offer = (state["family_id"] == full_task) & (draw < settings["frontierProbability"])
        scales = torch.linspace(1.0, settings["frontierMaxSpeedScale"], settings["frontierCandidates"],
                                device=self.device)
        margins = self._margin_initial_margins(state, scales)
        chosen = scales[choose_frontier(margins, self._margin_delta())]
        chosen = torch.where(offer, chosen, torch.ones_like(chosen))
        state["table_vy"] = state["table_vy"] * chosen
        state["table_yaw_rate"] = state["table_yaw_rate"] * chosen
        self.margin_frontier_active[env_ids] = chosen > 1.
        self.margin_frontier_scale[env_ids] = chosen
        return state

    def _margin_record_frontier(self, env_ids, completed_mask):
        """Record finished frontier episodes; return the mask of those that were frontier."""
        if getattr(self, "margin_curriculum", "disabled") != "frontier":
            return torch.zeros_like(completed_mask)
        frontier = completed_mask & self.margin_frontier_active[env_ids]
        done = env_ids[frontier]
        self.margin_frontier_stats.record(torch.zeros_like(done), self.episode_success[done])
        return frontier

    def _margin_sgs_snapshots(self, snapshots):
        if getattr(self, "margin_curriculum", "disabled") != "frontier":
            return
        stats = self.margin_frontier_stats
        count = float(stats.family_count[0])
        snapshots["SGS / episodes_frontier"] = count
        if count > 0:
            snapshots["SuccessRate / frontier"] = float(stats.family_successes[0]) / count
