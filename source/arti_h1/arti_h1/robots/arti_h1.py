from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

ISAACLAB_ASSETS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "assets")


# Standing pose = first frame of every reference clip. The right leg chain is the left one rotated
# by pi about z in the URDF (right_hip_yaw rpy 0 0 pi), so both legs use the same joint signs.
JOINT_INIT_POS = {
    "left_hip_yaw": 0.0,
    "left_hip_roll": -0.00001,
    "left_hip_pitch": -0.10078,
    "left_knee": 0.20157,
    "left_ankle_pitch": 0.10078,
    "left_ankle_roll": -0.00001,
    "right_hip_yaw": 0.0,
    "right_hip_roll": 0.00001,
    "right_hip_pitch": -0.10078,
    "right_knee": 0.20157,
    "right_ankle_pitch": 0.10079,
    "right_ankle_roll": 0.00001,
}

# Same as ARTI_H1_JOINT_STIFFNESS / DAMPING in arti_h1_env_cfg.py (real gains x identified scale, 10-23).
JOINT_STIFFNESS = {
    "left_hip_yaw": 80.14,
    "left_hip_roll": 72.13,
    "left_hip_pitch": 72.13,
    "left_knee": 100.17,
    "left_ankle_pitch": 30.0,
    "left_ankle_roll": 27.0,
    "right_hip_yaw": 80.14,
    "right_hip_roll": 72.13,
    "right_hip_pitch": 72.13,
    "right_knee": 100.17,
    "right_ankle_pitch": 30.0,
    "right_ankle_roll": 27.0,
}

JOINT_DAMPING = {
    "left_hip_yaw": 1.152,
    "left_hip_roll": 1.302,
    "left_hip_pitch": 1.002,
    "left_knee": 1.302,
    "left_ankle_pitch": 0.3,
    "left_ankle_roll": 0.4,
    "right_hip_yaw": 1.152,
    "right_hip_roll": 1.302,
    "right_hip_pitch": 1.002,
    "right_knee": 1.302,
    "right_ankle_pitch": 0.3,
    "right_ankle_roll": 0.4,
}

JOINT_EFFORT_LIMIT = {
    "left_hip_yaw": 23.7,
    "left_hip_roll": 23.7,
    "left_hip_pitch": 23.7,
    "left_knee": 23.7,
    "left_ankle_pitch": 24.0,
    "left_ankle_roll": 24.0,
    "right_hip_yaw": 23.7,
    "right_hip_roll": 23.7,
    "right_hip_pitch": 23.7,
    "right_knee": 23.7,
    "right_ankle_pitch": 24.0,
    "right_ankle_roll": 24.0,
}
JOINT_VELOCITY_LIMIT = {
    "left_hip_yaw": 30.0,
    "left_hip_roll": 30.0,
    "left_hip_pitch": 30.0,
    "left_knee": 30.0,
    "left_ankle_pitch": 5.0,
    "left_ankle_roll": 5.0,
    "right_hip_yaw": 30.0,
    "right_hip_roll": 30.0,
    "right_hip_pitch": 30.0,
    "right_knee": 30.0,
    "right_ankle_pitch": 5.0,
    "right_ankle_roll": 5.0,
}


ARTI_H1_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DIR}/models/arti_h1/arti_h1/arti_h1.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Reference pelvis origin height with the soles on the ground (root z of every clip).
        pos=(0.0, 0.0, 0.01),
        joint_pos=JOINT_INIT_POS,
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=JOINT_STIFFNESS,
            damping=JOINT_DAMPING,
            effort_limit=JOINT_EFFORT_LIMIT,
            velocity_limit_sim=JOINT_VELOCITY_LIMIT,
        ),
    },
)
