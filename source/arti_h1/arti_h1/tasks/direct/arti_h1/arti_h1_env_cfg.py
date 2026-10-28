import os

from arti_h1.robots import ISAACLAB_ASSETS_DIR, ARTI_H1_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


ARTI_H1_NUM_JOINTS = 12
ARTI_H1_AMP_OBSERVATION_SPACE = 1 + 6 + 3 + 3 + ARTI_H1_NUM_JOINTS * 6 + ARTI_H1_NUM_JOINTS + 3 + 3
# Trailing `+ ARTI_H1_NUM_JOINTS` is the previous (clamped) action, fed back to the policy.
ARTI_H1_POLICY_OBSERVATION_SPACE = 6 + 3 + ARTI_H1_NUM_JOINTS * 6 + ARTI_H1_NUM_JOINTS + 3 + 3 + ARTI_H1_NUM_JOINTS
# root height (1) + true root lin/ang velocity in the heading frame (3 + 3)
# + averaged planar velocity and yaw rate used by the reward (3) + command tracking error (3).
ARTI_H1_CRITIC_PRIVILEGED_SPACE = 1 + 3 + 3 + 3 + 3
ARTI_H1_ACTION_SCALE_MARGIN = 1.2
# The reference only uses +-0.027 rad of hip yaw, so a 1.2x margin leaves a PD target of
# +-0.033 rad. With kp=40 a stance-foot twisting torque of ~0.5 Nm already costs ~0.0125 rad
# of steady-state error, so the policy could not reach the yaw the turning clips need.
ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN = 2.0
# Distance from the default pose (first reference frame) to the min / max of every reference
# clip (max over the arti_h1 and arti_h1_zmp0 motion sets; zmp0 rolls reach -0.27/+0.20). All scaled targets stay inside the +-0.7854 rad URDF limits, so none need capping.
ARTI_H1_ACTION_NEGATIVE_SCALE = {
    "left_hip_yaw": 0.02715 * ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN,
    "left_hip_roll": 0.26629 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_hip_pitch": 0.25550 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_knee": 0.55637 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_ankle_pitch": 0.42506 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_ankle_roll": 0.26629 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_hip_yaw": 0.02715 * ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN,
    "right_hip_roll": 0.26986 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_hip_pitch": 0.25556 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_knee": 0.55694 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_ankle_pitch": 0.47669 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_ankle_roll": 0.26986 * ARTI_H1_ACTION_SCALE_MARGIN,
}
ARTI_H1_ACTION_POSITIVE_SCALE = {
    "left_hip_yaw": 0.02715 * ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN,
    "left_hip_roll": 0.16975 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_hip_pitch": 0.44417 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_knee": 0.08326 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_ankle_pitch": 0.27726 * ARTI_H1_ACTION_SCALE_MARGIN,
    "left_ankle_roll": 0.16975 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_hip_yaw": 0.02715 * ARTI_H1_HIP_YAW_ACTION_SCALE_MARGIN,
    "right_hip_roll": 0.19614 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_hip_pitch": 0.49493 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_knee": 0.10446 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_ankle_pitch": 0.27731 * ARTI_H1_ACTION_SCALE_MARGIN,
    "right_ankle_roll": 0.19614 * ARTI_H1_ACTION_SCALE_MARGIN,
}
# Real robot gains (humanoid_controller motor.yaml motor_command_ kp / kd, 10-23): Unitree hip 2.0 / 0.025, knee 2.5 / 0.025
# (rotor side, x 6.33^2 = 40.07 -> joint 80.14 / 1.00 and 100.17 / 1.00), CubeMars ankles 30 / 0.5 (output side), times the
# effective kp / kd scale identified from the real step / chirp / sweep logs (logs/sysid_real2/fits2, JOURNAL 10-23).
ARTI_H1_JOINT_STIFFNESS = {
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
ARTI_H1_JOINT_DAMPING = {
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
ARTI_H1_JOINT_EFFORT_LIMIT = {
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
ARTI_H1_JOINT_VELOCITY_LIMIT = {
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
# trunk_frame in arti_h1.urdf. The reference walk_com_height (0.29725) is this plus the 0.01 m pelvis origin height.
ARTI_H1_TRUNK_OFFSET_FROM_PELVIS = (0.0, 0.0, 0.28725)
# MTi-3 mount relative to the trunk. Mirrored in x compared to Koala (-0.04688, 0.0, 0.023).
ARTI_H1_IMU_OFFSET_FROM_TRUNK = (0.04688, 0.0, 0.023)
ARTI_H1_IMU_OFFSET_FROM_PELVIS = (
    ARTI_H1_TRUNK_OFFSET_FROM_PELVIS[0] + ARTI_H1_IMU_OFFSET_FROM_TRUNK[0],
    ARTI_H1_TRUNK_OFFSET_FROM_PELVIS[1] + ARTI_H1_IMU_OFFSET_FROM_TRUNK[1],
    ARTI_H1_TRUNK_OFFSET_FROM_PELVIS[2] + ARTI_H1_IMU_OFFSET_FROM_TRUNK[2],
)
# left_foot_frame / right_foot_frame in arti_h1.urdf (sole centre). The right leg is the left chain rotated by
# pi about z, so both offsets are identical in their own foot-link frames (and mirrored in x in the world).
ARTI_H1_LEFT_FOOT_OFFSET_FROM_FOOT_LINK = (-0.026, 0.0, -0.051)
ARTI_H1_RIGHT_FOOT_OFFSET_FROM_FOOT_LINK = (-0.026, 0.0, -0.051)


@configclass
class ArtiH1EventCfg:
    """Domain randomization. Sampled once per environment at startup so the implicit-actuator
    and physics-material writes (CPU-side) never run inside the reset path."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Combined with the plane's 1.0 via "average" -> effective 0.7..1.2.
            "static_friction_range": (0.4, 1.4),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            # Sampled independently, so dynamic can otherwise exceed static (unphysical).
            "make_consistent": True,
        },
    )

    add_pelvis_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # Pelvis carries 5.63 kg of the 11.17 kg total (updated 09-30); +-0.5 kg covers battery/payload drift.
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    # The ankle roll holds ~6 Nm all the time (CoM 57 mm off the stance foot), so a CoM error matters on the real
    # robot. Pelvis is half the mass: +-2 cm on its CoM moves the whole-robot CoM by ~+-1 cm.
    pelvis_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.5, 1.2),  # run32: was (0.8, 1.2)
            "operation": "scale",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            # Walking speed is only ~0.07 m/s, so the push has to stay small or it
            # swamps the velocity-tracking reward for a full averaging window.
            "velocity_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.15, 0.15)},
        },
    )


@configclass
class ArtiH1AmpEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 15.0

    action_space = ARTI_H1_NUM_JOINTS
    # + gait clock (sin, cos) + commands (3, always last)
    observation_space = ARTI_H1_POLICY_OBSERVATION_SPACE + 2 + 3
    # Asymmetric critic: policy observation + privileged terms (see ArtiH1AmpEnv._get_critic_state).
    # The tracking reward is computed from a 0.9 s average of the true root velocity, which the
    # policy cannot observe (it only has the IMU), so without these the value function cannot
    # predict the task reward and the advantages for tracking are mostly noise.
    state_space = ARTI_H1_POLICY_OBSERVATION_SPACE + 2 + 3 + ARTI_H1_CRITIC_PRIVILEGED_SPACE
    num_amp_observations = 2
    amp_observation_space = ARTI_H1_AMP_OBSERVATION_SPACE
    action_negative_scale = ARTI_H1_ACTION_NEGATIVE_SCALE
    action_positive_scale = ARTI_H1_ACTION_POSITIVE_SCALE
    # If set (rad), overrides the per-joint reference-derived scales above with the same +-range on every
    # joint (bdx uses 0.6). The reference-derived range (hip yaw +-0.054, hip roll 0.11-0.20, knee 0.08 on
    # one side) leaves no room for balance corrections outside the demo. The target is still clamped to
    # the joint limits in _apply_action.
    # Uniform +-range was the best previous-robot setting (reference-derived scales skated / staggered).
    # The new reference flexes the knee 0.557 rad and the hip pitch 0.495 rad from the default pose, so
    # +-0.5 could not reproduce it; 0.6 (= bdx) covers every clip. 0 = reference-derived scales.
    # run3 (uniform 0.6, fresh): 1.07M steps of skating, air ratio 0.0002 both feet (the old run17 also failed the
    # lifting gate with the wide range). Fresh runs use the reference-derived scales (old run9 recipe that first
    # stepped at 35k); switch to 0.6 only on a warm start once the gait exists.
    uniform_action_scale = 0.0
    # Hard +-limit (rad) on both hip_roll joints in sim; 0 keeps the URDF +-0.7854. USER (10-14): the real hip_roll
    # range is <= 15 deg, so keep its travel small.
    hip_roll_limit = 0.2618
    # run35: hard sim joint limits = the real motor.yaml min / max after homing (rad, both sides); False = URDF +-45 deg.
    real_joint_limits = True
    real_joint_limit_table = {
        "hip_yaw": (-0.7056, 0.7056), "hip_roll": (-0.2854, 0.4), "hip_pitch": (-0.7404, 0.5),
        "knee": (-0.5, 1.5258), "ankle_pitch": (-0.5, 0.7453), "ankle_roll": (-1.5216, 0.5),
    }
    # Measured joint model (humanoid_controller example_sysid + scripts/tuning/sysid_sim.py fit, see JOURNAL 10-23),
    # applied every physics step in ArtiH1AmpEnv._apply_actuator_model. Per joint type (hip_yaw, hip_roll, hip_pitch,
    # knee, ankle_pitch, ankle_roll); 0 disables a term.
    actuator_model = True
    # Command transport delay (ms): the PD target the joint sees is the policy target of delay_ms ago.
    # run34: ankle_roll 15 -> 10 ms. In the 10-24 walk log the CubeMars ankle_roll torque follows q_des one control step
    # later (like ankle_pitch at 10 ms), the 15 ms sim model two steps later; hips / knees match (two steps) in both.
    actuator_delay_ms = {"hip_yaw": 17.0, "hip_roll": 15.0, "hip_pitch": 15.0, "knee": 15.0, "ankle_pitch": 10.0, "ankle_roll": 10.0}
    # Reflected rotor inertia (kg*m^2) added to the joint.
    actuator_armature = {"hip_yaw": 0.005, "hip_roll": 0.005, "hip_pitch": 0.0, "knee": 0.01, "ankle_pitch": 0.02, "ankle_roll": 0.02}
    # Coulomb friction with stiction (Nm), modelled implicitly: the PD target is shifted by coulomb / kp against the
    # motion, and frozen where the joint stuck while |kp * error| <= coulomb.
    actuator_coulomb = {"hip_yaw": 0.2, "hip_roll": 0.15, "hip_pitch": 0.1, "knee": 0.1, "ankle_pitch": 0.3, "ankle_roll": 0.2}
    actuator_stick_velocity = 0.02  # rad/s
    # Per-env randomisation at start-up: delay + uniform(lo, hi) ms, armature and coulomb x uniform(lo, hi).
    # run32: wider towards a faster / less damped joint (the real ankles reacted ~2x faster than the sim on the
    # first steps of the 10-24 robot walk).
    actuator_delay_jitter_ms = (-8.0, 5.0)
    actuator_armature_scale_range = (0.3, 1.3)
    actuator_coulomb_scale_range = (0.3, 1.5)
    # Gaussian noise (rad/s) on the policy joint velocity obs [81:93]; 0 = off (run31 and before).
    policy_joint_vel_noise_std = 0.3
    # Per control step probability that the hip / knee (Unitree bus) joint obs repeat the previous step; 0 = off.
    joint_obs_stale_prob = 0.05
    # Per-episode constant offset (rad, +-) on the policy joint position obs: homing / encoder calibration error.
    joint_obs_offset_range = 0.02
    # Minimum +-range per side on top of the reference-derived scales (used when uniform_action_scale = 0).
    # The reference knee only extends 0.070 rad past the default, but in single support the stance knee sags
    # ~0.16 rad at kp 35 (replay with pelvis support: stance knee 0.146 vs 0.214 even at 4x gains), so with the
    # plain reference scales (+0.084) the policy could not hold the stance leg and never lifted the other foot
    # (runs 4-5).
    action_scale_min = 0.25
    hip_yaw_action_scale_min = 0.10

    # Min / max of the recomputed (0.9 s averaged) reference commands plus ~3% margin
    # (scripts/tuning/reference_stats.py: x +-0.0793, y +-0.0286, yaw +-0.0790).
    command_linear_x_range = (-0.046, 0.046)
    command_linear_y_range = (-0.0172, 0.0172)
    command_yaw_range = (-0.0456, 0.0456)
    command_linear_velocity_scale = 0.5
    command_angular_velocity_scale = 0.25
    sample_commands_from_reference = True
    recompute_motion_commands = True
    use_average_velocities = True
    # One gait period is 0.8925 s; a shorter window leaves gait-frequency ripple in the
    # tracked velocity, which caps the attainable tracking reward at ~0.77 even for a
    # perfect replay of the reference motion (0.97 at 0.9 s).
    average_velocity_window_s = 1.5
    # The sigma must be small relative to the command magnitude (~0.04-0.08). At 0.0025 a
    # robot standing still under a 0.043 m/s backward command still earned 48% of the linear
    # term, and under a pure-turn command 83% of the whole tracking reward, so the unconditional
    # AMP style reward decided the gait. 0.001 drops standing still to ~16% while a perfect
    # replay still keeps ~0.93 (ripple error ~7.6e-5 with the 0.9 s window).
    linear_velocity_tracking_sigma = 0.001
    angular_velocity_tracking_sigma = 0.001
    linear_velocity_xy_reward_scale = 0.5
    # Equal to the linear term: at half weight turning was not worth deviating from the
    # easiest gait the discriminator accepts.
    angular_velocity_z_reward_scale = 0.5
    # Regularization terms are NOT scaled by step_dt (see _get_rewards); these are the
    # per-step magnitudes directly.
    #
    # Size these against the EXPLORATION NOISE FLOOR, not against a converged gait. With
    # initial_log_std = -2.0 the sampled actions alone contribute
    #   E[sum_j (a_{t-1,j} - a_{t,j})^2] = 12 * 2 * exp(-2)^2 = 0.44
    # to the action-rate term no matter what the policy does. At scale -0.25 that is -0.11
    # per step, which cancels the alive bonus and drives log_std to min_log_std (the
    # gradient on sigma^2 is ~18x the entropy bonus), so the policy stops exploring and
    # never learns to step at all. Keep the floor well under the alive bonus and raise
    # these only after the gait works, reading Reward/action_rate and Reward/torque.
    # Torque economy (second training goal after reference-like walking). The torque term is
    # -scale * sum_j (tau_j / effort_limit_j)^2, the power term -scale * sum_j |tau_j * qd_j| (W).
    # Both are ramped in by the torque curriculum below so the penalty cannot stop the policy from
    # learning to step first (see the exploration-floor note above).
    # Reference PD replay: 1.26 Nm rms, 7.6 W -> about -0.07 (torque) and -0.08 (power) per step at full weight.
    # run13 tried -6: torque only 3.47 -> 3.24 Nm, swing tilt 0.9 -> 3-4 deg, roll speeds and power up, style .34 -> .30.
    # The ~6 Nm ankle_roll torque is structural (stance ankle holds 11.2 kg x 9.81 x 0.057 m = 6.25 Nm because the
    # reference CoM stays 57 mm off the stance sole), so back to -2.
    torque_reward_scale = -2.0
    joint_power_reward_scale = -0.015
    joint_effort_limit = ARTI_H1_JOINT_EFFORT_LIMIT
    # Smoothness (USER, run9 150k video: too stiff/jerky). -scale * sum qd^2 (rad/s): the reference joint speeds
    # (vel rms hip_pitch 79, knee 140, ankle_pitch 78, rolls 24 deg/s) give sum ~21 -> about -0.1 per step.
    # run12: normalised form -scale * sum_j (qd_j / ref_rms_j)^2 (ref rms per joint from the clips, floored at
    # joint_vel_rms_floor): exactly-reference speeds give 12 -> -0.1; run11 (ankle_roll 5x ref) gave ~-0.6.
    # run14: -0.008 -> -0.015 (run12 best still moves ankle_roll 2.8x the reference speed).
    joint_vel_reward_scale = -0.015
    joint_vel_rms_floor = 0.17  # rad/s (~10 deg/s), keeps hip yaw (ref 3 deg/s) from dominating
    # -scale * sum qdd^2 (rad/s^2): ~100 rad/s^2 per joint -> about -0.1 per step.
    joint_acc_reward_scale = -2.0e-6
    # Added to common_step_counter by every curriculum (support fade, torque, regularization), so a warm start
    # continues the schedules instead of restarting them at step 0 (set to the checkpoint step).
    curriculum_step_offset = 0
    # Fraction of the torque/power scales applied from step 0, ramped linearly to 1.0 between the two
    # env-step counts (common_step_counter = policy steps per env, the same as skrl timesteps).
    # 0.2 from step 0 (run1) paid -0.11 per step on the exploration noise alone (28 W, 2.6 Nm rms): std fell
    # 0.13 -> 0.08 in 5k steps with both feet still on the ground. Off until the clock gait has formed.
    torque_curriculum_initial_fraction = 0.0
    torque_curriculum_steps = (100_000, 500_000)
    # Same ramp for action_rate, action-energy symmetry, base_flat / base_ang_vel / base_sway and the
    # stagger / step / knee / swing symmetry penalties (see _get_rewards).
    regularization_curriculum_initial_fraction = 0.0
    regularization_curriculum_steps = (60_000, 300_000)
    # Balance "training wheels": pelvis torque kp * sin(tilt) - kd * roll/pitch rate (Nm), full strength until the
    # first step count, then faded linearly to 0. Tipping moment of the whole robot at 0.2 rad is ~5.4 Nm.
    # Set kp = 0 to disable (e.g. for play / eval / warm-started fine-tuning).
    # Residual action on a nominal gait: PD target = default + nominal(clock phase) + scale * action, nominal = mean
    # reference joint offset per phase bin over all clips (stepping in place with the reference foot lift).
    # Runs 2-7 (plain actions) never lifted a foot in any setting: lifting needs a large coordinated knee action
    # (lift_test.py) that exploration never found. Deployment must add the same table (pt_to_onnx.py).
    nominal_gait_scale = 1.0
    nominal_gait_bins = 64
    # Sag/lag-compensated command table (bins x joints, robot joint order) from scripts/tuning/ilc_nominal.py.
    # With the plain reference table the stance knee sags and the swing sole reaches only 2-5 mm (run8: no lift).
    nominal_gait_table_file = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "arti_h1_nominal_gait_command.npy")
    balance_support_kp = 40.0
    balance_support_kd = 2.0
    balance_support_fade_steps = (150_000, 600_000)
    # run31: -0.01 -> -0.1. run30 (std at floor 0.03, so ~all of it is policy jitter) moved hip_roll at 107 deg/s rms
    # vs reference 23 deg/s and looked stiff; the noise floor now costs only ~-0.002.
    # run32: -0.1 -> -0.25 (run31 joint speeds crept up again: fwd knee 94/110 -> 113/125 deg/s by 195k).
    action_rate_reward_scale = -0.75
    alive_reward_scale = 0.1
    motion_time_scale = 1.0
    cycle_reference_motions = False
    # Penalize only the rolling left/right action-energy imbalance. Squared action
    # energy permits the intended half-cycle phase offset of a walking gait.
    symmetry_reward_scale = -0.05
    symmetry_window_s = 1.5
    # Non-saturating reward for moving along each commanded planar axis: clamp(v_a / cmd_a, -1, 1)
    # averaged over the axes with |cmd_a| > min_command. Run 2026-09-23_15-24-01 tracked forward,
    # left and turning commands but stood still under backward commands (46% of the clips) and
    # reached only ~40% of right-side commands; standing still scores 0 here instead of the ~0.2
    # the exp kernel still pays, so the policy cannot settle on marking time.
    # 0.1 with the gait clock (run 2026-09-24_01-37-52): fwd .84 / side .36-.43 but backward stuck at .03 by 190k.
    # run3: 0.3 from a fresh start paid for skating (fwd progress .73 with both feet on the ground). 0.1 = old run9 (first
    # stepping policy); raise to 0.3 on a warm start once both feet lift (old run10).
    signed_progress_reward_scale = 0.1
    signed_progress_min_command = 0.01
    # Seconds between mid-episode command switches (uniform range); (0, 0) disables. With one
    # command per 15 s episode the policy only ever walked in the direction it was reset with and
    # could not follow the 5 s command sequence in play.py.
    command_resample_interval_s = (4.0, 8.0)
    # Run 2026-09-23_16-30-36 reached every commanded velocity by skating: in forward walking the
    # right sole never rose above 1.3 mm (left 28 mm, reference swing height 27 mm), and sideways /
    # turning both feet stayed below ~7 mm. The two terms below make that exploit expensive.
    # Sole height below which a foot counts as in contact (the asset has no contact sensor).
    foot_contact_height = 0.008
    # Paid per foot at touchdown: max(clamp(air_time, max) - target, 0), only under a moving command.
    # Reference single support is 0.34 s; a dragged foot never touches down and earns nothing.
    # With target 0.15 s the learned swings (< 0.15 s) earned ~0, so the term pays from 0 now.
    # run5: back on next to the clock (runs 2-3 never lifted a foot). ~1.7 per touchdown for a 0.34 s swing.
    feet_air_time_reward_scale = 5.0
    feet_air_time_target_s = 0.0
    feet_air_time_max_s = 0.5
    # (stance_fraction_left - stance_fraction_right)^2 over a rolling window, under a moving command.
    # Run 2026-09-23_23-32-24 (L/R air 0.13/0.035, stance 0.87/0.97) pays ~-0.05 per step.
    # run28 tried -5: the 1 s window is dominated by gait-phase noise and did not balance L/R -> off.
    contact_symmetry_reward_scale = 0.0
    contact_symmetry_window_s = 1.5
    # Gait clock. Every run without it settled on one pivot foot and one swing foot (the side depended
    # on the run) and slid the pivot foot. The clock (sin/cos in the policy observation) follows the
    # reference gait: phase_rate 1.12045 /s (0.8925 s cycle), right foot swings over phase [0, 0.375),
    # left over [0.5, 0.875), half-sine swing height peaking at walk_foot_height 0.030 m. It is reset
    # to the reference gait_phase of the RSI frame, so reference state and clock agree at reset.
    gait_phase_rate = 0.66667
    # run35 standing: under a zero command (max |vx|, |vy|, |wz| <= stand_command_threshold) the clock runs on only to the
    # next rest phase (mid double support) and stays there, so the nominal stepping holds a two-feet stance; it starts
    # again from there with the next moving command. Empty = the clock always runs (stepping in place at zero command).
    gait_rest_phases = (0.427, 0.927)
    stand_command_threshold = 0.003
    # Standing (zero command) rewards: both soles down, and -scale * sum(joint_vel^2).
    stand_feet_contact_reward_scale = 0.5
    stand_joint_vel_reward_scale = 0.01
    # run36: -scale * (|yaw rate| + |planar base speed|) while standing; run35 turned in place at -0.017 rad/s.
    stand_base_motion_reward_scale = 5.0
    gait_swing_start_phase = (0.5107, 0.0107)  # (left, right)
    gait_swing_duration_phase = 0.3333
    gait_swing_height = 0.030
    # 0.01 gave ~0 reward below ~15 mm, so a dragged foot saw no gradient toward lifting (runs 2-3). 0.02: 5 mm -> 0.21.
    gait_swing_height_sigma = 0.02
    # Fraction of feet whose contact state matches the clock (stance foot down, swing foot up).
    # 0.3 warm-started from a pivot-foot policy (run 2026-09-24_01-05-30) never started using the clock.
    gait_contact_reward_scale = 1.0  # 0.5 was skated around with the uniform action range (run 13-09-44)
    # exp(-(z - z_ref)^2 / sigma^2) averaged over the feet that should be swinging.
    gait_swing_height_reward_scale = 1.0
    # Share of the swing-height term paid linearly in sole height (clamped at gait_swing_height), rest = exp kernel.
    gait_swing_lift_weight = 0.5
    # -scale per foot that is still on the ground while the clock says it should swing (moving commands only).
    # The contact-match reward alone paid standing/skating 0.62 of its 1.0 and was never escaped (runs 2-3).
    gait_swing_contact_penalty_scale = 1.0
    # |cycle-averaged x_right - x_left| in the heading frame (L1). Quadratic -100/-400 plateaued at 2 / 1.2 cm
    # (runs 05-45-58 / 07-15-59); L1 -8 costs -0.08 at 1 cm and keeps a constant pull down to 0.
    foot_stagger_reward_scale = -8.0
    # -scale * max(0, foot_min_width - lateral sole-centre distance) (m), heading frame, every step (not curriculum
    # scaled). Self-collisions are off, so run 2026-10-27_06-44-49 300k walked with the soles overlapping 41 % of the
    # time (distance mean 90 / p5 2 / min -22 mm, soles are 90 mm wide); the reference stays at 146 mean / 115 min mm.
    foot_min_width = 0.115
    foot_width_reward_scale = -20.0
    # Per touchdown: -scale * |step_k - step_(k-1)| (m). A 2 cm alternating gap costs ~-0.045 per control step.
    # -60 left a 0.75 cm forward gap (run 2026-09-24_09-41-10 plateau); -150.
    step_symmetry_reward_scale = -150.0
    # Sole contact patch half-size (m) from the foot collision mesh (x +-0.057, y +-0.045 about the sole frame).
    foot_sole_half_length = 0.057
    foot_sole_half_width = 0.045
    # -scale * sum_feet sin^2(sole tilt). Reference swing and stance soles are flat; 25 deg costs ~-0.18 per foot.
    # run11: -1 -> -5 (USER: soles tilt briefly while lifting; at -1 a 10 deg tilt cost only -0.03 per foot).
    foot_flat_reward_scale = -30.0
    # -scale * sum over feet off the ground of (roll rate^2 + pitch rate^2) (rad/s); a 3 rad/s flick costs ~-0.18.
    swing_foot_ang_vel_reward_scale = -0.05
    # Knee = shin link origin. Reference knee lift (swing peak - stance) is 18.0 / 18.2 mm (scripts/tuning/reference_knee_lift.py).
    left_knee_body = "left_shin"
    right_knee_body = "right_shin"
    knee_lift_target = 0.0180
    # Per touchdown: scale * |swing_k - swing_(k-1)| (s); a 0.07 s gap costs ~-0.35.
    swing_symmetry_reward_scale = -20.0
    knee_lift_sigma = 0.008
    # Per touchdown: -scale * |lift_k - lift_(k-1)| (m); a 1 cm gap costs ~-0.05 per control step.
    # -150 let the knee gap oscillate 0.5-3 mm after convergence (run 2026-09-24_17-16-10); -300 for fine-tuning.
    knee_symmetry_reward_scale = -300.0
    # Per touchdown: -scale * |ankle pitch range of this stride - other leg's last stride range| (rad). Like the knee
    # term, an 8 deg (0.14 rad) gap costs ~2.8 per touchdown at -20. Run26 had 31 / 23 deg (reference 26 / 26).
    ankle_pitch_symmetry_reward_scale = -20.0
    # Per touchdown: scale * exp(-((lift - target) / sigma)^2).
    # 1.0 let the knee lift sink to 15-21 mm once the pelvis was held level (run 2026-09-25_09-25-22).
    knee_lift_reward_scale = 2.0
    # Pelvis level: -scale * |projected gravity xy|^2 (= sin^2 tilt); 4.5 deg costs ~-0.12. Reference roll/pitch = 0.
    # -20 halved the pelvis roll (9 -> 5 deg, run 2026-09-25_08-36-15) but stalled above the 3 deg target.
    base_flat_reward_scale = -60.0
    # -scale * (roll_rate^2 + pitch_rate^2) in rad/s; 25 deg/s costs ~-0.04.
    base_ang_vel_xy_reward_scale = -0.5
    # -scale * (lateral vel - its 0.9 s mean)^2 in m/s; 180 mm/s std costs ~-0.1.
    # -3 only took the lateral vel oscillation from ~180 to ~165 mm/s (run 2026-09-25_10-14-37; reference 69).
    # run31: -50 -> -10. -50 (run27-30) bought ~60 mm/s sway with jittery hip_roll corrections (user: too stiff).
    base_lateral_sway_reward_scale = -10.0
    # Sum over stance feet of |v_xy|^2 at the sole. Skating both feet at ~0.045 m/s costs ~-0.03 per
    # step; a planted stance foot costs ~0. At -20 (run 2026-09-23_22-56-52) standing still won.
    # run5: -8 -> -20 (run3 skated 1M steps at -8). run7: -20 cost -3.5 per step on the initial action noise and
    # drove std .35 -> .14 in 10k; with the nominal gait lifting the feet, -8 again.
    foot_slip_reward_scale = -8.0

    early_termination = True
    termination_height = 0.2
    # Trunk tilt from vertical, in radians. Height alone lets the robot survive in
    # heavily tipped poses that it can never recover from.
    termination_max_tilt_rad = 0.8
    reference_body = "pelvis"
    left_toe_body = "left_foot_link"
    right_toe_body = "right_foot_link"
    left_toe_body_offset = ARTI_H1_LEFT_FOOT_OFFSET_FROM_FOOT_LINK
    right_toe_body_offset = ARTI_H1_RIGHT_FOOT_OFFSET_FROM_FOOT_LINK
    # The USD exposes pelvis as the rigid prim; trunk is a fixed point above it.
    termination_body = "pelvis"
    termination_body_offset = ARTI_H1_TRUNK_OFFSET_FROM_PELVIS
    # "random" enables reference state initialization (RSI): episodes start at a random
    # time in the clip. "random_start" pins every reset to t=0 of the clip.
    reset_strategy = "random"
    # Fraction of RSI resets forced to t=0 of the clip (standstill start).
    reset_standstill_probability = 0.2
    use_isaac_imu_sensor = True
    imu_backend = "isaaclab"
    imu_body = "pelvis"
    imu_sensor_name = "mti3_imu"
    imu_prim_path = "/World/envs/env_.*/Robot/pelvis"
    imu_parent_path_template = "/World/envs/env_{env_id}/Robot/{body}"
    imu_sensor_period_s = 0.0
    imu_linear_acceleration_filter_size = 1
    imu_angular_velocity_filter_size = 1
    imu_orientation_filter_size = 1
    imu_translation = ARTI_H1_IMU_OFFSET_FROM_PELVIS
    imu_orientation_wxyz = (1.0, 0.0, 0.0, 0.0)
    imu_read_gravity = True
    # White noise and orientation noise already default to the MTi-3 datasheet values in
    # Mti3ImuModel; only the in-run bias random walk and the transport delay were disabled.
    imu_latency_s = 0.01  # -> 1 control step at 60 Hz
    mti3_gyro_bias_walk_std_rad_s = 0.0005
    mti3_accel_bias_walk_std_m_s2 = 0.002
    # run33: matched to the real MTi-3 on the robot (10-24 observe / walk logs): gyro noise .003-.011 rad/s (incl. the
    # standing motor vibration), bias up to .0064 rad/s, gyro ~1 control step behind the orientation.
    mti3_gyro_noise_std_rad_s = 0.01
    mti3_gyro_bias_range_rad_s = (-0.01, 0.01)
    imu_gyro_extra_delay_steps = 1
    # run34: per-episode constant IMU roll / pitch mounting error (rad, +-); 0 = off.
    # 10-28 robot logs: standing IMU pitch differs from the leg FK pitch by -6.8 .. +4.7 deg between runs (was 0.02).
    # The robot hung on a rope with the feet on the ground, so part of that is real tilt; kept as a moderate margin.
    mti3_mount_tilt_range_rad = 0.05
    # Bounded random walk (rad/sqrt(s)) on top of the mount error; 0 = off. Off: the 10-28 IMU - FK drift came from the
    # rope tilting the robot (feet not flat), and the MTi-3 roll / pitch is gravity-referenced, so it doesn't drift.
    mti3_tilt_walk_std_rad_s = 0.0
    mti3_tilt_max_rad = 0.08
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            bounce_threshold_velocity=0.2,
        ),
    )

    robot: ArticulationCfg = ARTI_H1_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=ARTI_H1_JOINT_STIFFNESS,
                damping=ARTI_H1_JOINT_DAMPING,
                effort_limit=ARTI_H1_JOINT_EFFORT_LIMIT,
                velocity_limit_sim=ARTI_H1_JOINT_VELOCITY_LIMIT,
            ),
        },
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=1.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    events: ArtiH1EventCfg = ArtiH1EventCfg()

    dt = 1 / 240 * 4
    motion_folder_path = os.path.join(ISAACLAB_ASSETS_DIR, "motions", "arti_h1")
ArtiH1EnvCfg = ArtiH1AmpEnvCfg
