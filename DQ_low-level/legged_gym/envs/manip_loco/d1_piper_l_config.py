"""Configuration for the D1 four-wheel-legged robot with a Piper-L arm.

The low-level policy has one output per actuated leg/wheel DOF (16) and one
output per Piper arm joint (6).  The arm outputs are kept in the policy
interface for compatibility with the B1/Z1 policy, but the environment drives
the arm from a task-space IK target.  The three gripper DOFs are not part of
the policy action vector.
"""

from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg, B1Z1RoughCfgPPO


class D1PiperLRoughCfg(B1Z1RoughCfg):
    class env(B1Z1RoughCfg.env):
        num_actions = 16 + 6
        num_torques = 16 + 6
        num_leg_dofs = 16
        num_arm_dofs = 6
        num_gripper_joints = 3

        # The D1 URDF is already ordered FL, FR, RL, RR, with four DOFs per
        # leg (hip, thigh, calf, wheel).  Keep that order for the policy.
        leg_dof_reorder = list(range(16))
        # Wheel angles are unbounded and should not enter joint-position
        # deviation rewards; wheel velocities/efforts remain in the dynamic
        # penalties and action-rate terms.
        leg_position_indices = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
        feet_reorder = [0, 1, 2, 3]
        wheel_action_indices = [3, 7, 11, 15]

        # 2 + 3 + 22 joint pos/vel + 16 actions + 4 contacts + 3 command
        # + 3 position goal + 3 orientation goal.  The four wheel entries
        # in the joint-position block are kept as zero-valued placeholders;
        # wheel velocity is observed in the following velocity block.
        num_proprio = 2 + 3 + 22 + 22 + 16 + 4 + 3 + 3 + 3
        num_priv = 5 + 1 + 16
        history_len = 10
        num_observations = num_proprio * (history_len + 1) + num_priv
        stop_update_goal = False

    class goal_ee(B1Z1RoughCfg.goal_ee):
        # The Piper-L chain in d1_piper_l.urdf has a roughly 0.75 m reach
        # from the shoulder.  The useful IK region is smaller than the
        # mathematical joint-limit envelope: targets should stay in front of
        # the shoulder, above the chassis, and away from wrist/shoulder
        # singularities.  With the center below and the default D1 pose, the
        # sampled target region is approximately x=[0.50, 0.84] m,
        # y=[-0.29, 0.29] m, z=[0.83, 1.15] m in world coordinates.  Relative
        # to piper_base_link (mount at [0.20, 0, 0.09] m), this is roughly
        # x=[0.30, 0.64] m, y=[-0.29, 0.29] m, z=[0.29, 0.61] m.
        class sphere_center(B1Z1RoughCfg.goal_ee.sphere_center):
            x_offset = 0.30
            y_offset = 0.0
            z_invariant_offset = 0.70

        class ranges(B1Z1RoughCfg.goal_ee.ranges):
            # Both initialization poses are inside the robust sampled region.
            init_pos_start = [0.44, 0.55, 0.0]
            init_pos_end = [0.52, 0.35, 0.0]

            # Spherical coordinates are (radius, pitch, yaw) in the arm-base
            # yaw frame.  These bounds keep the arm in its forward upper
            # workspace instead of sampling near the shoulder or behind the
            # robot.  They also leave margin for the URDF joint limits.
            pos_l = [0.38, 0.58]
            pos_p = [0.35, 0.90]
            pos_y = [-0.55, 0.55]

            # Orientation offsets are applied to the nominal tool orientation
            # generated from the spherical target.  Keep them moderate so
            # joint4/5/6 do not spend most of their motion at limits.
            delta_orn_r = [-0.35, 0.35]
            delta_orn_p = [-0.35, 0.35]
            delta_orn_y = [-0.45, 0.45]

    class init_state(B1Z1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.45]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
            "RR_foot_joint": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "gripper": 0.0,
            "gripper_joint1": 0.0,
            "gripper_joint2": 0.0,
        }

    class control(B1Z1RoughCfg.control):
        control_type = "P"
        stiffness = {"joint": 60.0}
        damping = {"joint": 3.0}

        # The wheel actuator in d1_piper_articulation_cfg.py is a pure
        # velocity/viscous actuator (Kp=0, Kd=0.5), so the low-level adapter
        # handles these entries as velocity targets rather than position
        # residuals.
        wheel_stiffness = 0.0
        wheel_damping = 0.5
        leg_effort_limit = 90.0
        wheel_effort_limit = 12.0
        leg_friction = 0.589
        leg_armature = 0.0535
        arm_stiffness = [50.0, 50.0, 80.0, 30.0, 30.0, 20.0]
        arm_damping = [3.0, 2.0, 3.0, 3.0, 2.5, 1.0]
        arm_effort_limit = [20.0, 20.0, 15.0, 7.0, 5.0, 5.0]
        gripper_stiffness = 20.0
        gripper_damping = 1.0
        # IK safety limits.  The task-space target is updated every policy
        # step; limiting both the pose error and the joint increment prevents
        # a transient PhysX/Jacobian anomaly from producing an extreme arm
        # position target and subsequently corrupting the observation tensor.
        ik_dpose_limit = [0.20, 0.20, 0.20, 0.50, 0.50, 0.50]
        ik_delta_q_limit = [0.25] * 6
        ik_joint_limit_margin = 0.05
        action_scale = (
            [0.25, 0.25, 0.25, 5.0] * 4
            + [0.25] * 6
        )
        decimation = 4

    class asset(B1Z1RoughCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/d1_piper_l/d1_piper_l.urdf"
        foot_name = "foot"
        # Keep fixed end-effector frames available for Jacobian lookup.
        gripper_name = "end_effector"
        base_body_name = "F_base_link"
        gripper_mass_body_name = "gripper_base"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = []
        collapse_fixed_joints = True
        flip_visual_attachments = False

    # piper_mount_joint in d1_piper_l.urdf
    class arm(B1Z1RoughCfg.arm):
        base_offset = [0.20, 0.0, 0.09]
        dof_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

    class domain_rand(B1Z1RoughCfg.domain_rand):
        observe_priv = True
        leg_motor_strength_range = [0.7, 1.3]
        arm_motor_strength_range = [0.7, 1.3]

    class rewards(B1Z1RoughCfg.rewards):
        # D1 is substantially lower than B1 and its four wheel contacts are
        # the normal support contacts.  These values are task-specific and
        # should not be inherited unchanged from the B1/Z1 configuration.
        base_height_target = 0.45
        # D1 weighs about 55 kg, so a static four-wheel load is roughly 135 N
        # per wheel.  Penalize impacts above normal support load instead of
        # penalizing every stationary contact.
        max_contact_force = 180.0

        class scales(B1Z1RoughCfg.rewards.scales):
            # These rewards assume feet periodically leave the ground.  A
            # wheel should remain in contact while rolling, and its body
            # translation is not evidence of foot dragging.
            feet_air_time = None
            feet_height = None
            feet_drag = None

            # Gait-command rewards are not used by this task
            # (observe_gait_commands=False); omit them rather than invoking
            # a four-foot gait objective for wheels.
            tracking_contacts_shaped_force = None
            tracking_contacts_shaped_vel = None

            # The B1 penalties sum 12 leg DOFs.  D1 sums 16 leg/wheel DOFs;
            # scale the per-DOF terms by 12/16 to keep their aggregate weight
            # comparable while still applying them to the wheel actuators.
            torques = -2.5e-5 * 0.75
            dof_acc = -7.5e-7 * 0.75
            action_rate = -0.015 * 0.75
            delta_torques = -1.0e-7 * 0.75
            work = -0.003 * 0.75


class D1PiperLRoughCfgPPO(B1Z1RoughCfgPPO):
    class policy(B1Z1RoughCfgPPO.policy):
        init_std = [[0.8, 0.8, 0.8, 0.8] * 4 + [1.0] * 6]
        num_leg_actions = 16
        num_arm_actions = 6

    class algorithm(B1Z1RoughCfgPPO.algorithm):
        min_policy_std = [[0.20] * 16 + [0.20] * 6]

    class runner(B1Z1RoughCfgPPO.runner):
        experiment_name = "d1_piper_l_low"
        max_iterations = 45000
