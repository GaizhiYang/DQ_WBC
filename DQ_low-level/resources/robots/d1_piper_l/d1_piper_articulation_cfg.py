# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#####################这个脚本是用于isaaclab中配置D1和piper_l的关节电机，但是里面的具体参数配置可以参考，通用##########################


import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg , DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

current_dir = os.path.dirname(os.path.abspath(__file__))

# The training spawner consumes USD assets.  The source URDF is converted once
# with Isaac Lab's ``convert_urdf.py`` and kept next to its conversion metadata.
# Do not pass a URDF path to UsdFileCfg: its API only accepts ``usd_path``.
D1_PIPER_USD = os.path.join(current_dir, "generated_usd", "d1_piper_l.usd")
##
# Configuration
##


D1_PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=D1_PIPER_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     collision_enabled=True,
        #     contact_offset=0.02,
        #     rest_offset=0.005 ,
        # ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 0.8,
            "RR_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
            "RL_foot_joint": 0.0,
            "RR_foot_joint": 0.0,
            # Keep the arm and gripper at their neutral URDF pose on reset.
            "joint[1-6]": 0.0,
            "gripper.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*(hip|thigh|calf)_joint"],
            effort_limit=90.0,
            saturation_effort=90.0,
            velocity_limit=20.0,
            stiffness=60.0,
            damping=3.0,
            friction=0.589,
            armature=0.0535
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=12,
            saturation_effort=12.0,
            velocity_limit=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0
        ),
        # DO NOT MOVE IF POSSIBLE
        "joint1": DelayedPDActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit=20.0,
            effort_limit_sim=20.0,
            stiffness=50.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint2": DelayedPDActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit=20.0,
            effort_limit_sim=20.0,
            stiffness=50.0,
            damping=2.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint3": DelayedPDActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit=15.0,
            effort_limit_sim=15.0,
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint4": DelayedPDActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit=7.0,
            effort_limit_sim=7.0,
            stiffness=30.0,
            damping=3.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint5": DelayedPDActuatorCfg(
            joint_names_expr=["joint5"],
            effort_limit=5.0,
            effort_limit_sim=5.0,
            stiffness=30.0,
            damping=2.5,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        "joint6": DelayedPDActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit=5.0,
            effort_limit_sim=5.0,
            stiffness=20.0,
            damping=1.0,
            armature=0.01,
            min_delay=0,
            max_delay=4,
            friction=0.01,
        ),
        # The gripper is not part of the WBC action vector yet, but keeping a
        # low-gain actuator on its three prismatic joints prevents them from
        # floating freely and makes the intentional passive-gripper choice
        # explicit.
        "gripper": DelayedPDActuatorCfg(
            joint_names_expr=["gripper.*"],
            stiffness=20.0,
            damping=1.0,
            armature=0.001,
            min_delay=0,
            max_delay=0,
            friction=0.01,
        ),
    },
)
