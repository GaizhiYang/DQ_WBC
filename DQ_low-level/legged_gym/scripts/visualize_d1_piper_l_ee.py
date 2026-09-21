#!/usr/bin/env python3
"""Load D1 + Piper-L in Isaac Gym and draw the end-effector frame.

The three axes are colored using the conventional ROS/Isaac Gym convention:
X=red, Y=green and Z=blue.  The pose is read from the rigid-body state every
frame, so the axes follow the end-effector if the arm is later driven by a
controller or an interactive test.

Example (run on a machine with Isaac Gym installed)::

    cd DQ_low-level/legged_gym/scripts
    python visualize_d1_piper_l_ee.py --sim_device cuda:0

The base is fixed by default so that the URDF can be inspected without the
robot falling under gravity.  Pass ``--free_base`` to simulate a free base.
"""

from __future__ import print_function

import os
import sys

import numpy as np

from isaacgym import gymapi, gymutil


# Make the repository package importable when this file is launched directly
# from ``legged_gym/scripts``.  The visualization must use the same defaults
# as the training task instead of maintaining a second hard-coded joint map.
LOW_LEVEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if LOW_LEVEL_ROOT not in sys.path:
    sys.path.insert(0, LOW_LEVEL_ROOT)

from legged_gym.envs.manip_loco.d1_piper_l_config import D1PiperLRoughCfg


DEFAULT_URDF = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "resources",
        "robots",
        "d1_piper_l",
        "d1_piper_l.urdf",
    )
)
DEFAULT_JOINT_ANGLES = D1PiperLRoughCfg.init_state.default_joint_angles
DEFAULT_BASE_HEIGHT = float(D1PiperLRoughCfg.init_state.pos[2])


def build_sim(gym, args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
    elif args.physics_engine == gymapi.SIM_FLEX:
        sim_params.flex.solver_type = 5
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10

    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        args.physics_engine,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("Isaac Gym failed to create the simulation")
    return sim


def load_robot(gym, sim, urdf_path, args):
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.isfile(urdf_path):
        raise FileNotFoundError("URDF does not exist: {}".format(urdf_path))

    asset_root = os.path.dirname(urdf_path)
    asset_file = os.path.basename(urdf_path)
    asset_options = gymapi.AssetOptions()
    # Keep fixed links such as ``end_effector`` in the rigid-body tree.  This
    # is required for find_actor_rigid_body_handle() and for reading its pose.
    asset_options.collapse_fixed_joints = False
    asset_options.fix_base_link = not args.free_base
    asset_options.use_mesh_materials = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    asset_options.armature = 0.01

    print("Loading URDF: {}".format(urdf_path))
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    if asset is None:
        raise RuntimeError("Isaac Gym failed to load {}".format(urdf_path))

    # Use moderate gains for a stable viewer.  The actual initial position
    # targets are assigned below from D1PiperLRoughCfg.init_state.
    dof_props = gym.get_asset_dof_properties(asset)
    if len(dof_props) > 0:
        dof_props["stiffness"].fill(80.0)
        dof_props["damping"].fill(8.0)

    env_lower = gymapi.Vec3(-1.5, -1.5, 0.0)
    env_upper = gymapi.Vec3(1.5, 1.5, 2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)
    if env is None:
        raise RuntimeError("Isaac Gym failed to create an environment")

    actor_pose = gymapi.Transform(p=gymapi.Vec3(0.0, 0.0, args.base_height))
    actor = gym.create_actor(env, asset, actor_pose, "d1_piper_l", 0, 0)
    gym.set_actor_dof_properties(env, actor, dof_props)

    # Isaac Gym uses the asset's DOF order, so construct the state by name
    # from the training configuration rather than assuming the order in the
    # URDF.  This keeps the visualized neutral pose synchronized with
    # D1PiperLRoughCfg.init_state.default_joint_angles.
    dof_names = list(gym.get_asset_dof_names(asset))
    missing = [name for name in dof_names if name not in DEFAULT_JOINT_ANGLES]
    if missing:
        raise RuntimeError(
            "The D1 default-angle config has no value for DOFs: {}".format(
                missing
            )
        )
    default_dof_pos = np.asarray(
        [DEFAULT_JOINT_ANGLES[name] for name in dof_names], dtype=np.float32
    )
    default_dof_state = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    default_dof_state["pos"] = default_dof_pos
    gym.set_actor_dof_states(env, actor, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, actor, default_dof_pos)
    print("Initial DOF pose loaded from d1_piper_l_config.py:")
    for name, value in zip(dof_names, default_dof_pos):
        print("  {:20s} {:.6f}".format(name, float(value)))

    body_dict = gym.get_actor_rigid_body_dict(env, actor)
    if args.ee_body not in body_dict:
        names = list(body_dict.keys())
        raise RuntimeError(
            "Rigid body {!r} was not found. Available bodies: {}".format(
                args.ee_body, names
            )
        )
    ee_handle = gym.find_actor_rigid_body_handle(env, actor, args.ee_body)
    return env, actor, ee_handle, body_dict


def main():
    custom_parameters = [
        {
            "name": "--urdf",
            "type": str,
            "default": DEFAULT_URDF,
            "help": "Path to the D1 + Piper-L URDF",
        },
        {
            "name": "--ee_body",
            "type": str,
            "default": "end_effector",
            "help": "Rigid-body name at which to draw the frame",
        },
        {
            "name": "--axis_length",
            "type": float,
            "default": 0.18,
            "help": "Length of each displayed axis in meters",
        },
        {
            "name": "--base_height",
            "type": float,
            "default": DEFAULT_BASE_HEIGHT,
            "help": "Initial robot base height in meters",
        },
        {
            "name": "--free_base",
            "action": "store_true",
            "help": "Do not fix the base link",
        },
    ]
    args = gymutil.parse_arguments(
        description="Visualize D1 + Piper-L end-effector coordinates",
        custom_parameters=custom_parameters,
    )
    if args.axis_length <= 0.0:
        raise ValueError("--axis_length must be positive")

    gym = gymapi.acquire_gym()
    sim = None
    viewer = None
    try:
        sim = build_sim(gym, args)

        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane.static_friction = 1.0
        plane.dynamic_friction = 1.0
        gym.add_ground(sim, plane)

        viewer = gym.create_viewer(sim, gymapi.CameraProperties())
        if viewer is None:
            raise RuntimeError("Isaac Gym failed to create a viewer")

        env, actor, ee_handle, body_dict = load_robot(gym, sim, args.urdf, args)
        print("Loaded {} rigid bodies; {} handle = {}".format(
            len(body_dict), args.ee_body, ee_handle
        ))

        # A useful initial camera view for the arm mounted near x=0.2.
        gym.viewer_camera_look_at(
            viewer,
            env,
            gymapi.Vec3(1.25, -1.25, 1.05),
            gymapi.Vec3(0.20, 0.0, 0.65),
        )
        axes_geom = gymutil.AxesGeometry(args.axis_length)

        while not gym.query_viewer_has_closed(viewer):
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            poses = gym.get_actor_rigid_body_states(env, actor, gymapi.STATE_POS)["pose"]
            ee_pose = gymapi.Transform.from_buffer(poses[ee_handle])

            gym.clear_lines(viewer)
            gymutil.draw_lines(axes_geom, gym, viewer, env, ee_pose)

            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)
    finally:
        if viewer is not None:
            gym.destroy_viewer(viewer)
        if sim is not None:
            gym.destroy_sim(sim)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
