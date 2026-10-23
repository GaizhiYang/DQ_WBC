"""Inspect real SGS bank states in Isaac Gym, without a high-level policy.

Run from DQ_high-level. By default an interactive viewer opens, paused after a
single physics tick needed to update the renderer's forward kinematics.
Randomly samples three configurations per family by default.
--headless exports overview.png, grounding_overview.png, per-configuration
images/GIFs, and measurements.json.
"""

import argparse
import json
from pathlib import Path
import random
import time


FAMILIES = ("full_task", "pregrasp", "at_grasp")
LABELS = ("Full task", "Pregrasp", "At grasp pose")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="data/sgs/sugar_box_v7.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--family", choices=FAMILIES, default="full_task")
    parser.add_argument("--samples-per-family", type=int, default=3,
                        help="Number of configurations randomly sampled from each family.")
    parser.add_argument("--sample-index", type=int, default=None,
                        help="Inspect this index within each family instead of random sampling.")
    parser.add_argument("--config-id", type=int, default=None,
                        help="Select an exact global bank ID in interactive mode.")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", default="sgs_preview")
    parser.add_argument("--steps", type=int, default=8,
                        help="High-level hold steps per exported GIF.")
    args = parser.parse_args()
    if (args.sample_index is not None and args.sample_index < 0) or args.steps < 0:
        parser.error("--sample-index and --steps must be nonnegative")
    if args.samples_per_family < 1:
        parser.error("--samples-per-family must be positive")
    return args


class BankPreview:
    def __init__(self, args):
        # Isaac Gym must precede torch, including transitive imports.
        from isaacgym import gymapi, gymtorch
        import torch
        from utils.sgs import ConfigurationBank
        from utils.sgs_bank_builder import create_builder_environment

        self.args, self.gymapi, self.gymtorch, self.torch = args, gymapi, gymtorch, torch
        self.bank = ConfigurationBank.load(
            args.bank, device=args.device,
            expected_metadata={"object_name": "sugar_box", "table_level": "Level02",
                               "physical_validation": {"passed": True,
                                   "validation_profile": "geometry_v1",
                                   "feet_near_ground": True, "collision_checked": True,
                                   "object_attachment": False}})
        self.family_ids = [
            (self.bank.tensors["family_id"] == family).nonzero().flatten().cpu().tolist()
            for family in range(3)]
        if args.sample_index is not None and any(
                args.sample_index >= len(ids) for ids in self.family_ids):
            raise ValueError("--sample-index is outside a bank family")
        rng = random.Random(args.seed)
        self.sample_indices = [
            [args.sample_index] if args.sample_index is not None else
            rng.sample(range(len(ids)), min(args.samples_per_family, len(ids)))
            for ids in self.family_ids]
        if args.config_id is not None and not 0 <= args.config_id < self.bank.num_configs:
            raise ValueError("--config-id is outside the bank")
        self.env = create_builder_environment(num_envs=1, device=args.device, seed=args.seed)
        self.env.rand_control = True
        # The real policy interprets action 0 as fully open, not "hold".
        # Override only this preview instance so playback preserves the saved
        # randomized aperture until the user explicitly presses O.
        self.env.set_gripper = self._apply_preview_gripper
        self.viewer = None
        self.output = Path(args.output)
        self.output.mkdir(parents=True, exist_ok=True)
        camera = gymapi.CameraProperties()
        camera.width, camera.height = 960, 640
        camera.horizontal_fov = 58
        self.camera = self.env.gym.create_camera_sensor(self.env.envs[0], camera)
        if self.camera < 0:
            raise RuntimeError("Isaac Gym could not create the preview camera")
        self.family = FAMILIES.index(args.family)
        self.index = self.sample_indices[self.family][0]
        self.view = "wide"
        self.gripper_mode = "saved"
        self.paused = True
        self.elapsed = 0.0
        self.ended = False

    def load(self, family=None, index=None, config_id=None):
        from envs.b1z1_base import reindex_all
        from utils.sgs_bank_builder import restore_snapshot

        env, torch, gymtorch = self.env, self.torch, self.gymtorch
        if config_id is not None:
            self.family = int(self.bank.tensors["family_id"][config_id].item())
            self.index = self.family_ids[self.family].index(config_id)
        else:
            self.family = self.family if family is None else family
            self.index = (self.index if index is None else index) % len(self.family_ids[self.family])
        self.config_id = self.family_ids[self.family][self.index]
        state = self.bank.batch(torch.tensor([self.config_id], device=env.device))
        restore_snapshot(env, state)
        self.saved_gripper_target = state["gripper_target"].clone()
        self.gripper_mode = "saved"
        self.ended, self.paused = False, True
        # One ordinary controller/physics tick commits pending roots + DOFs.
        # It updates the actual renderer; editing tensor caches alone would
        # display the previous configuration. No object is attached or pinned.
        with torch.no_grad():
            env._compute_low_level_observations()
            env.last_low_actions[:] = reindex_all(
                env.low_level_policy(env.low_obs_buf.detach(), hist_encoding=True))
            env.floating_table_control()
            targets = env.get_all_pos_targets(env.ee_goal_global_cart, env.ee_goal_global_orn_quat)
            env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(targets))
            env.gym.set_dof_actuation_force_tensor(env.sim, gymtorch.unwrap_tensor(env.get_torques()))
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env._refresh_sim_tensors()
            env.update_roboinfo()
            env._compute_observations()
        self.elapsed = env.sim_params.dt
        self.set_camera(self.view)
        return self.measurements()

    def measurements(self):
        from utils.sgs_grounding import (
            grounding_state, approx_grounded_mask,
        )

        env = self.env
        bottom_z, forces = grounding_state(env)
        closed = float(env.dof_limits_upper[-1].item())
        opened = float(env.dof_limits_lower[-1].item())
        joint = float(env._dof_pos[0, -1].item())
        target = float(env.gripper_dof_pos[0, -1].item())
        return {
            "family": FAMILIES[self.family], "config_id": self.config_id,
            "family_sample_index": self.index, "elapsed_sim_seconds": self.elapsed,
            "base_to_object_xy_m": float((env._robot_root_states[0, :2] - env._cube_root_states[0, :2]).norm().item()),
            "ee_to_object_center_m": float((env.ee_pos[0] - env._cube_root_states[0, :3]).norm().item()),
            "lift_above_original_support_m": float(
                (env._cube_root_states[0, 2] - env.table_heights[0] - env.init_height[0]).item()),
            "foot_bottom_z_m": bottom_z[0].detach().cpu().tolist(),
            "foot_contact_force_z_N": forces[0, :, 2].detach().cpu().tolist(),
            "feet_near_ground": bool(approx_grounded_mask(bottom_z)[0].item()),
            "gripper_joint_rad": joint,
            "gripper_opening_normalized": (joint - closed) / (opened - closed),
            "gripper_target_opening_normalized": (target - closed) / (opened - closed),
            "hold_counter": int(env.pick_counter[0].item()),
            "success": bool(env.episode_success[0].item()),
            "terminated": self.ended,
            "gripper_command": self.gripper_mode,
        }

    def _apply_preview_gripper(self):
        if self.gripper_mode == "saved":
            self.env.gripper_dof_pos[:] = self.saved_gripper_target
        else:
            limits = (self.env.dof_limits_upper if self.gripper_mode == "closed"
                      else self.env.dof_limits_lower)
            self.env.gripper_dof_pos[:] = limits[-self.env.num_gripper_dof:]

    def step(self):
        if self.ended:
            return self.measurements()
        torch, env = self.torch, self.env
        with torch.no_grad():
            actions = torch.zeros((1, env.num_actions), device=env.device)
            actions[:, 6] = {"closed": -1, "open": 1, "saved": 0}[self.gripper_mode]
            env.step(actions)
            env.gym.fetch_results(env.sim, True)
        self.elapsed += env.control_freq_low * env.control_freq_inv * env.sim_params.dt
        self.ended = bool(env.reset_buf[0].item())
        if self.ended:
            self.paused = True
        return self.measurements()

    def set_camera(self, view):
        import numpy as np

        self.view = view
        env, gymapi = self.env, self.gymapi
        obj = env._cube_root_states[0, :3].detach().cpu().numpy()
        robot = env._robot_root_states[0, :3].detach().cpu().numpy()
        if view == "close":
            target = obj + np.array([0., 0., .01])
            eye = target + np.array([.35, -.48, .28])
        else:
            target = (obj + robot) * .5
            target[2] = .46
            eye = target + np.array([.25, -3.25, 1.75])
        env.gym.set_camera_location(
            self.camera, env.envs[0], gymapi.Vec3(*eye), gymapi.Vec3(*target))
        if self.viewer is not None:
            env.gym.viewer_camera_look_at(
                self.viewer, env.envs[0], gymapi.Vec3(*eye), gymapi.Vec3(*target))

    def image(self):
        import numpy as np

        env = self.env
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(
            env.sim, env.envs[0], self.camera, self.gymapi.IMAGE_COLOR)
        return np.asarray(rgba).reshape(640, 960, 4)[..., :3].copy()

    def export(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
        from utils.sgs_grounding import (
            FOOT_NAMES, FOOT_RADIUS, APPROX_GROUND_LIMITS,
        )

        initial, traces = [], {}
        rows = max(map(len, self.sample_indices))
        figure, axes = plt.subplots(rows, 3, figsize=(15, 4.8 * rows), squeeze=False)
        ground_figure, ground_axes = plt.subplots(rows, 3, figsize=(15, 4.8 * rows), squeeze=False)
        for axis in list(axes.flat) + list(ground_axes.flat):
            axis.axis("off")
        for family, indices in enumerate(self.sample_indices):
            for row, index in enumerate(indices):
                self.load(family=family, index=index)
                values = self.measurements()
                initial.append(values)
                name = f"{FAMILIES[family]}_id{self.config_id}"
                overview_view = "wide" if family == 0 else "close"
                for view in ("wide", "close"):
                    self.set_camera(view)
                    frame = self.image()
                    Image.fromarray(frame).save(self.output / f"{name}_{view}.png")
                    if view == overview_view:
                        axes[row, family].imshow(frame)
                    if view == "wide":
                        ground_axes[row, family].imshow(frame)
                title = (
                    f"{LABELS[family]} | ID {self.config_id}\n"
                    f"Base-object: {values['base_to_object_xy_m']:.2f} m | "
                    f"Lift: {values['lift_above_original_support_m']*100:.2f} cm\n"
                    f"Feet gap max: {max(values['foot_bottom_z_m'])*100:.2f} cm | "
                    f"Near ground: {'yes' if values['feet_near_ground'] else 'no'}\n"
                    f"Gripper opening: {values['gripper_opening_normalized']:.2f} "
                    f"(target {values['gripper_target_opening_normalized']:.2f})")
                axes[row, family].set_title(title, fontsize=11)
                ground_axes[row, family].set_title(title, fontsize=11)
                self.set_camera(overview_view)
                frames, durations = [Image.fromarray(self.image())], []
                trace = [values]
                for _ in range(self.args.steps):
                    if self.ended:
                        break
                    previous = self.elapsed
                    trace.append(self.step())
                    durations.append(round(1000 * (self.elapsed - previous)))
                    frames.append(Image.fromarray(self.image()))
                durations.append(700)
                frames[0].save(self.output / f"{name}_hold.gif", save_all=True,
                               append_images=frames[1:], duration=durations, loop=0)
                traces[str(self.config_id)] = trace
                print(f"Rendered {name}", flush=True)
        figure.suptitle(
            f"SGS bank sample | Seed {self.args.seed} | Preview after one physics tick "
            f"({self.env.sim_params.dt*1000:g} ms)", fontsize=15)
        figure.tight_layout()
        figure.savefig(self.output / "overview.png", dpi=160)
        plt.close(figure)
        ground_figure.suptitle(
            f"SGS feet geometry | Seed {self.args.seed} | First physics tick "
            f"({self.env.sim_params.dt*1000:g} ms)", fontsize=15)
        ground_figure.tight_layout()
        ground_figure.savefig(self.output / "grounding_overview.png", dpi=160)
        plt.close(ground_figure)
        report = {
            "bank": str(Path(self.args.bank).resolve()), "fingerprint": self.bank.fingerprint,
            "preview_physics_ticks": 1, "seed": self.args.seed,
            "foot_names": list(FOOT_NAMES), "foot_collision_radius_m": FOOT_RADIUS,
            "ground_z_m": 0., "grounding_limits": APPROX_GROUND_LIMITS,
            "grounding_note": "feet_near_ground checks only collision-bottom geometry; "
                              "contact forces are diagnostic, not a standing requirement.",
            "sampling": "uniform without replacement within each family"
                        if self.args.sample_index is None else "explicit family index",
            "selected_config_ids": {
                FAMILIES[family]: [self.family_ids[family][i] for i in indices]
                for family, indices in enumerate(self.sample_indices)},
            "controller": "Low-level balance/IK; zero high-level deltas; saved gripper target; "
                          "Level02 table motion",
            "note": "EE-to-object-center distance is not the distance to the grasp candidate.",
            "initial": initial, "hold_traces": traces,
        }
        (self.output / "measurements.json").write_text(json.dumps(report, indent=2) + "\n")
        print("Saved preview:", (self.output / "overview.png").resolve(), flush=True)

    def interactive(self):
        from utils.sgs_grounding import FOOT_NAMES

        gymapi, env = self.gymapi, self.env
        camera = gymapi.CameraProperties()
        camera.width, camera.height = 1280, 800
        self.viewer = env.gym.create_viewer(env.sim, camera)
        if self.viewer is None:
            raise RuntimeError("Could not open the viewer; use --headless to export images")
        bindings = {
            gymapi.KEY_1: "full_task", gymapi.KEY_2: "pregrasp", gymapi.KEY_3: "at_grasp",
            gymapi.KEY_SPACE: "pause", gymapi.KEY_N: "step", gymapi.KEY_R: "reset",
            gymapi.KEY_RIGHT: "next", gymapi.KEY_LEFT: "previous",
            gymapi.KEY_F: "wide", gymapi.KEY_C: "close", gymapi.KEY_O: "gripper",
            gymapi.KEY_S: "save", gymapi.KEY_ESCAPE: "quit",
        }
        for key, action in bindings.items():
            env.gym.subscribe_viewer_keyboard_event(self.viewer, key, action)
        self.load(config_id=self.args.config_id)
        print("1/2/3: family | Left/Right: sampled configurations | Space: play/pause | N: step | R: reset\n"
              "F/C: wide/close | O: toggle gripper | S: screenshot | Esc: exit\n"
              "Starts paused with the saved gripper target. O explicitly overrides it; R restores it.\n"
              "No high-level teacher policy is running; Level02 motion is preserved.")
        print("Sampled bank IDs:", {
            FAMILIES[family]: [self.family_ids[family][i] for i in indices]
            for family, indices in enumerate(self.sample_indices)}, flush=True)
        print("Foot measurements order:", ", ".join(FOOT_NAMES), flush=True)
        print(json.dumps(self.measurements()), flush=True)
        next_step, next_log = time.monotonic(), time.monotonic()
        while not env.gym.query_viewer_has_closed(self.viewer):
            single_step = False
            for event in env.gym.query_viewer_action_events(self.viewer):
                if event.value <= 0:
                    continue
                action = event.action
                if action == "quit":
                    return
                if action in FAMILIES:
                    family = FAMILIES.index(action)
                    self.load(family=family, index=self.sample_indices[family][0])
                elif action in ("next", "previous"):
                    indices = self.sample_indices[self.family]
                    position = indices.index(self.index) if self.index in indices else -1
                    offset = 1 if action == "next" else -1
                    self.load(index=indices[(position + offset) % len(indices)])
                elif action == "reset":
                    self.load()
                elif action == "pause":
                    self.paused = True if self.ended else not self.paused
                    next_step = time.monotonic()
                elif action == "step":
                    single_step = True
                elif action in ("wide", "close"):
                    self.set_camera(action)
                elif action == "gripper":
                    if self.gripper_mode == "saved":
                        opening = self.measurements()["gripper_target_opening_normalized"]
                        self.gripper_mode = "closed" if opening >= .5 else "open"
                    else:
                        self.gripper_mode = "open" if self.gripper_mode == "closed" else "closed"
                elif action == "save":
                    from PIL import Image
                    name = self.output / f"id_{self.config_id}_t_{self.elapsed:.3f}.png"
                    Image.fromarray(self.image()).save(name)
                    print("Saved", name, flush=True)
                print(json.dumps(self.measurements()), flush=True)
            now = time.monotonic()
            if single_step or (not self.paused and now >= next_step):
                before = self.elapsed
                self.step()
                next_step = time.monotonic() + max(.02, self.elapsed - before)
                if now >= next_log or self.ended:
                    print(json.dumps(self.measurements()), flush=True)
                    next_log = now + .5
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(self.viewer, env.sim, False)
            time.sleep(.01)

    def close(self):
        if self.viewer is not None:
            self.env.gym.destroy_viewer(self.viewer)
        self.env.gym.destroy_sim(self.env.sim)


def main():
    args = parse_args()
    preview = BankPreview(args)
    try:
        if args.headless:
            preview.export()
        else:
            preview.interactive()
    finally:
        preview.close()


if __name__ == "__main__":
    main()
