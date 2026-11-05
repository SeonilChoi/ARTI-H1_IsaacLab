"""ARTI-H1 velocity walking from scratch, following the Microduck recipe (pollen-robotics/microduck_rl).

Microduck (a 25 cm / 800 g biped) learns to walk with plain PPO: no reference motions, no gait clock, no nominal gait
table and no style reward. The policy outputs joint targets around the home pose, and the gait comes only from the
mjlab velocity task rewards (velocity tracking, upright, speed-dependent posture, feet air time / clearance / swing
height / slip, action rate, angular momentum). On top of that it uses a domain randomisation and curriculum stack:
action-rate ramp, standing-env ramp, explicit turn-in-place bucket, fixed modest command ranges, pushes. This cfg ports
that recipe to ARTI-H1 (Isaac Lab, PhysX). It keeps the ARTI-H1 sim-to-real models that were measured on the real
robot: actuator delay / armature / Coulomb friction, real joint limits, MTi-3 IMU model, homing zero errors, stale bus
reads and the stage-3 extra joint lag.

Reward semantics follow mjlab: every term is weight * value * step_dt, cost terms return >= 0 and take a negative
weight. Watch TensorBoard Reward/<term>: every penalty must stay <= 0. The weights are Microduck's
(microduck_velocity_env_cfg.py, mjlab 1.3.0 base). Lengths, speeds and times are scaled to ARTI-H1 (about 2.4x taller,
14x heavier): command ranges, tracking stds, foot clearance target and air-time window.

Policy I/O (ARTI_H1_POLICY_JOINT_NAMES order, 45-D obs, mjlab actor layout):
  obs = [gyro (3, IMU body frame), projected gravity (3, IMU), joint_pos - default (12), joint_vel (12),
         previous action (12), command (vx, vy, wz) (3, always last)]
  q_des = clamp(default + action_scale * clamp(action, -1, 1), joint limits)
scripts/pt_to_onnx.py exports this as is: it reads action_*_scale / nominal_gait_scale = 0 / joint limits from
params/env.yaml. humanoid_controller needs a matching 45-D observation mode before a policy can run on the robot.
"""

import math

from isaaclab.utils import configclass

from .arti_h1_env_cfg import ARTI_H1_NUM_JOINTS, ArtiH1AmpEnvCfg

# Max |q_des - default| (rad) at |action| = 1, symmetric. Microduck uses scale 1.0 on its low-kp XL330s; ARTI-H1's
# 72-100 Nm/rad hips and knees track much more stiffly, so the range is per joint: the sagittal joints get room
# for a step, and hip_roll is already capped at +-15 deg (hip_roll_limit). The target is also clamped to the real
# motor.yaml limits.
ARTI_H1_VELOCITY_ACTION_SCALE_BY_TYPE = {
    "hip_yaw": 0.20,
    "hip_roll": 0.25,
    "hip_pitch": 0.60,
    "knee": 0.70,
    "ankle_pitch": 0.50,
    "ankle_roll": 0.30,
}
ARTI_H1_VELOCITY_ACTION_SCALE = {
    f"{side}_{joint}": scale
    for side in ("left", "right")
    for joint, scale in ARTI_H1_VELOCITY_ACTION_SCALE_BY_TYPE.items()
}
# gyro 3 + projected gravity 3 + joint pos / vel 12 + 12 + previous action 12 + command 3
ARTI_H1_VELOCITY_OBSERVATION_SPACE = 3 + 3 + ARTI_H1_NUM_JOINTS * 3 + 3
# Critic (mjlab critic group): the actor terms without noise / IMU model / encoder errors, plus the true base linear
# velocity (3) and per-foot height (2), air time (2) and contact (2).
ARTI_H1_VELOCITY_STATE_SPACE = ARTI_H1_VELOCITY_OBSERVATION_SPACE + 3 + 2 + 2 + 2

# Microduck counts curriculum stages in PPO iterations of 24 env steps (common_step_counter = iteration * 24).
NUM_STEPS_PER_ENV = 24


@configclass
class ArtiH1VelocityEnvCfg(ArtiH1AmpEnvCfg):
    """Microduck velocity recipe on ARTI-H1, trained from scratch with PPO (Template-Arti-H1-Velocity-Direct-v0).

    Inherits the ARTI-H1 robot, scene, physics, IMU, actuator-model and domain-randomisation fields from
    ArtiH1AmpEnvCfg. None of the reference-motion / gait-clock / AMP fields there are used by ArtiH1VelocityEnv.
    """

    episode_length_s = 20.0  # mjlab velocity task
    observation_space = ARTI_H1_VELOCITY_OBSERVATION_SPACE
    state_space = ARTI_H1_VELOCITY_STATE_SPACE

    # --- Actions (see ARTI_H1_VELOCITY_ACTION_SCALE). Field names match what scripts/pt_to_onnx.py reads. ---
    action_negative_scale = ARTI_H1_VELOCITY_ACTION_SCALE
    action_positive_scale = ARTI_H1_VELOCITY_ACTION_SCALE
    uniform_action_scale = 0.0
    action_scale_min = 0.0
    hip_yaw_action_scale_min = 0.0
    # No nominal gait table: plain actions around the home pose (Microduck). Must stay 0 for the ONNX export.
    nominal_gait_scale = 0.0
    # No training-wheels torque by default (Microduck has none). Set > 0 (e.g. 40) only if no foot ever lifts.
    balance_support_kp = 0.0
    balance_support_kd = 0.0
    balance_support_fade_steps = (0, 0)

    # --- Commands (mjlab UniformVelocityCommand, Microduck settings) ---
    # Fixed, modest ranges: Microduck found that a widening range curriculum outpaced what the robot could do. Microduck walks
    # +-0.4 m/s at 25 cm tall. ARTI-H1's reference gait walked only 0.05 m/s. These are a first guess at a safe real-robot
    # range: widen them once the gait is established.
    command_linear_x_range = (-0.15, 0.15)
    command_linear_y_range = (-0.08, 0.08)
    command_yaw_range = (-0.4, 0.4)
    command_resample_time_s = (3.0, 8.0)
    # Fraction of envs commanded to spin on the spot (lin = 0, |wz| in [0.4, 1] * max). Microduck: independent uniform
    # sampling made turning in place ~2 % of the data, so it never trained.
    turn_in_place_fraction = 0.15
    # All-zero command fraction (the deployment idle state), ramped by step (env steps = iterations * 24).
    standing_env_stages = (
        (0, 0.02),
        (500 * NUM_STEPS_PER_ENV, 0.05),
        (750 * NUM_STEPS_PER_ENV, 0.10),
        (1000 * NUM_STEPS_PER_ENV, 0.15),
        (1500 * NUM_STEPS_PER_ENV, 0.20),
        (2000 * NUM_STEPS_PER_ENV, 0.25),
    )
    # Commands in the obs are raw (m/s, rad/s): the PPO observation preprocessor normalises them.

    # --- Rewards: name -> weight, value * weight * step_dt (mjlab). Cost terms are >= 0 with a negative weight. ---
    rew_track_linear_velocity = 2.0
    rew_track_angular_velocity = 2.0
    # Microduck: std^2 0.1 for +-0.4 m/s and 0.5 for +-1.0 rad/s. Scaled with the command range (std / range kept).
    # Trial 400 iterations (11-04) with 0.015 / 0.08: wobbling at 1.2 rad/s yaw error got exp(-18) = no gradient. The
    # errors the policy must escape early are physical (wobble), not command-sized, so the angular std is Microduck's 0.5
    # and the linear one only slightly tighter than Microduck's 0.1 (standing under a 0.1 m/s command still earns 0.72).
    track_linear_velocity_std = math.sqrt(0.03)
    track_angular_velocity_std = math.sqrt(0.5)
    # Fine tracking terms on top (commanded components only, no roll / pitch / vz). The coarse stds above leave the 480k
    # run (11-05) at 36 % of a lateral command (vy 0.03 for 0.08: exp(-0.0025 / 0.03) = 0.92, almost no loss) and with a
    # constant -0.03 rad/s yaw drift. The coarse terms keep the early gradient, these sharpen the converged gait.
    rew_track_linear_velocity_fine = 1.0
    track_linear_velocity_fine_std = math.sqrt(0.004)  # 0.05 m/s error -> 0.54
    rew_track_angular_velocity_fine = 1.0
    track_angular_velocity_fine_std = math.sqrt(0.03)  # 0.1 rad/s error -> 0.72
    rew_upright = 2.0
    upright_std = math.sqrt(0.05)
    rew_pose = 1.0
    # variable_posture: exp(-mean((q - q_default)^2 / std^2)), std per joint type, standing below walking_threshold
    # (|v_xy cmd| + |wz cmd|). Microduck keeps hip_roll tight (0.05). Here the roll joints get 0.1 walking, because
    # ARTI-H1 must shift its weight over the stance foot through hip_roll / ankle_roll (11-03 robot logs: it tipped
    # toward the swing side).
    pose_std_standing = {"hip_yaw": 0.1, "hip_roll": 0.05, "hip_pitch": 0.15, "knee": 0.15, "ankle_pitch": 0.1,
                         "ankle_roll": 0.05}
    pose_std_walking = {"hip_yaw": 0.15, "hip_roll": 0.1, "hip_pitch": 0.4, "knee": 0.4, "ankle_pitch": 0.25,
                        "ankle_roll": 0.1}
    pose_walking_threshold = 0.01
    rew_body_ang_vel = -0.05  # pelvis roll / pitch rate^2
    rew_angular_momentum = -0.02  # whole-body angular momentum about the CoM, |L|^2
    rew_dof_pos_limits = -1.0  # sum of joint excursions beyond joint_soft_limit_factor of the range
    joint_soft_limit_factor = 0.9
    # action_rate_l2 weight schedule (Microduck: -0.1 while the gait bootstraps, -1.0 by iteration 1500).
    action_rate_stages = (
        (0, -0.1),
        (500 * NUM_STEPS_PER_ENV, -0.2),
        (750 * NUM_STEPS_PER_ENV, -0.4),
        (1000 * NUM_STEPS_PER_ENV, -0.6),
        (1250 * NUM_STEPS_PER_ENV, -0.8),
        (1500 * NUM_STEPS_PER_ENV, -1.0),
    )
    # Feet air time: per foot, 1 while the current air time is inside the window (mjlab pays every step, not at touchdown).
    # Microduck window 0.125-0.300 s at weight 3. ARTI-H1 legs are ~2.4x longer (pendulum time x ~1.6): 0.2-0.5 s.
    rew_air_time = 3.0
    # Deviation from mjlab: count a foot's air time only while the other foot is in contact (Isaac Lab biped air time).
    air_time_single_support_only = True
    # Also require alternation: a foot's swing pays only if the other foot lifted off since this foot's last lift-off.
    # Trial 11-05 with swing_foot_speed: the policy stood on the right leg and tapped the left foot (air ratio
    # 0.89 / 0.01), farming the window with one leg.
    air_time_require_alternation = True
    # Min 0.1 s (not 0.2): with 0.2 s no foot ever left the ground in a 600-iteration trial (the policy skated).
    # With 0.1 s it stepped (single support 0.98, air time 0.22 s, peak 4 cm by iteration 600).
    air_time_min_s = 0.1
    air_time_max_s = 0.5
    # Foot clearance: |sole height - target| * |foot v_xy|. Swing height: (peak / target - 1)^2 at each landing.
    rew_foot_clearance = -2.0
    rew_foot_swing_height = -0.25
    foot_target_height = 0.03  # Microduck 0.02 m. The old ARTI-H1 reference swing peaked at 0.03 m.
    # Foot slip: sum over feet in contact of |v_xy|^2. Microduck uses -0.1 for pivot turning. At -0.1 ARTI-H1 learned to
    # skate on flat soles instead of stepping, so it is -1.0 here.
    rew_foot_slip = -1.0
    # Swing foot speed relative to the trunk, squared, above a free band: sum over feet in the air of
    # relu(|v_foot_xy - v_trunk_xy| - free)^2. The 480k policy flicked each swing leg ~10 cm forward and back at
    # 1.0-1.7 m/s (also when stepping in place): nothing penalised swing-foot motion. A 0.15 m/s stride needs ~0.5 m/s.
    rew_swing_foot_speed = -1.0
    swing_foot_free_speed = 0.5
    # Lowest sole-corner height (contact patch +-foot_sole_half_length / width) below which a foot counts as in contact
    # (no contact sensor). Also the foot height used by foot_clearance / foot_swing_height.
    foot_contact_height = 0.008
    # Gates on |v_xy cmd| + |wz cmd| for the gait terms (air time, clearance, swing height, slip).
    gait_command_threshold = 0.01
    # Microduck's self-collision penalty (trunk vs legs). This asset has self-collisions off, and the ARTI-H1 hazard is the
    # soles colliding, so the cost is the shortfall of the lateral sole-centre distance below foot_min_width (inherited,
    # 0.115 m) in units of 2 cm.
    rew_feet_too_close = -1.0
    # Not in Microduck: sum over feet in contact of sin^2(sole tilt). Trial 400 iterations stood on 25 deg tilted soles;
    # on the robot a stance foot rolling onto its edge preceded the 11-02 falls. 25 deg on both feet costs ~-0.06/step.
    rew_foot_flat = -10.0
    # Raw policy output beyond +-1 (clamped before the PD target): sum(relu(|a| - 1)), capped. Keeps the means in range.
    rew_action_out_of_range = -0.5

    # --- Terminations: inherited (trunk below 0.2 m or tilted > 0.8 rad). ---

    # --- Domain randomisation ---
    # Inherited from ArtiH1AmpEnvCfg: friction 0.7-1.2, pelvis mass +-0.5 kg and CoM +-2 cm, kp x0.85-1.15 / kd x0.5-1.2,
    # actuator delay jitter, armature x0.3-1.3, Coulomb x0.3-1.5, joint zero error +-0.02 rad, stale hip / knee bus
    # reads 5 %, joint-vel noise 0.3 rad/s, MTi-3 noise / bias / delay / mount tilt / tilt drift.
    # Stage-3 findings carried over: roll-joint zero errors up to +-0.06 rad (11-02 robot logs).
    joint_obs_offset_range_by_type = {"hip_roll": 0.06, "ankle_roll": 0.06}
    # Stage-3 extra lag (real walk logs lag 50-70 ms more on hip_roll and ~33 ms on ankle_pitch than the base model),
    # ramped in after the gait exists: iterations 1000 -> 3000. Microduck: no new difficulty before the skill.
    actuator_delay_extra_ms = {"hip_roll": 50.0, "ankle_pitch": 25.0}
    actuator_delay_extra_curriculum_steps = (1000 * NUM_STEPS_PER_ENV, 3000 * NUM_STEPS_PER_ENV)
    # Not a warm start: every schedule starts at step 0.
    curriculum_step_offset = 0

    # Reset: home pose on the ground, random yaw, small drop.
    reset_root_height_range = (0.0, 0.01)
    reset_yaw_range = (-math.pi, math.pi)

    def __post_init__(self):
        super().__post_init__()
        # Microduck pushes: +-0.3 m/s planar velocity kicks every 3-6 s from step 0 (its 2026-07 audit: larger kicks
        # trained a permanently nervous gait). Kept ARTI-H1's small yaw / roll / pitch rate kicks.
        self.events.push_robot.interval_range_s = (3.0, 6.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.30, 0.30), "y": (-0.20, 0.20), "yaw": (-0.2, 0.2), "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3),
        }


@configclass
class ArtiH1VelocityEnvCfg_PLAY(ArtiH1VelocityEnvCfg):
    """Viewer / eval variant: final curriculum stages, no extra-lag ramp, fewer envs."""

    curriculum_step_offset = 10_000 * NUM_STEPS_PER_ENV

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
