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
    # turn-in-place |wz| drawn from [min_frac, 1] * 0.4. Was 0.4 (|wz| >= 0.16): the policy then stood still for
    # |wz| = 0.1 and for wz = -0.2 (but turned for +0.2).
    turn_in_place_min_frac = 0.25
    # pure forward / backward commands (|vx| in [0.3, 1] * 0.15, vy = wz = 0). Uniform sampling gives
    # |vy| < 0.02 and |wz| < 0.05 only ~3 % of the time, too rare for gait_symmetry to act.
    straight_walk_fraction = 0.15
    # pure sideways commands (|vy| in [0.3, 1] * 0.08, vx = wz = 0), sampled after the straight bucket. Off by default
    # (RSI r04 turns it on): without it a pure max-|vy| command is rare (vy is uniform and mixed with vx / wz), and the
    # 11-09 eval tracked only ~35 % of a 0.08 m/s sideways command.
    lateral_walk_fraction = 0.0
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
    # Was 1.0 / sqrt(0.03): standing still under a 0.2 rad/s turn command kept 0.26 of it, and the 11-06 policies
    # stood for |wz| <= 0.1 and for one sign of |wz| = 0.2. sqrt(0.01): standing under 0.2 keeps 0.02, under 0.1 0.37.
    rew_track_angular_velocity_fine = 1.5
    track_angular_velocity_fine_std = math.sqrt(0.01)  # 0.1 rad/s error -> 0.37
    # Gait start. Both sharpened yaw and the symmetry policies still stood still when a |wz| <= 0.2 turn (or 0.1) was
    # commanded from standstill, although they tracked it once walking. Penalty per step while a motion is commanded
    # and both feet have been down longer than max_double_support_s (walking double support is ~0.05 s per step).
    rew_stand_still_commanded = -2.0
    max_double_support_s = 0.3
    # Lateral-only tracking (RSI). Policies reach ~33 % of a 0.08 m/s lateral command (fine term: a 0.05 m/s miss still
    # keeps 0.54). Off by default; RSI rounds set it by override.
    rew_track_lateral_velocity = 0.0
    track_lateral_velocity_std = 0.03
    lateral_cmd_min = 0.02
    # Yaw-only tracking (RSI r16). r15 (joint zero-error randomisation) reached only 17-30 % of a +0.1 rad/s turn
    # (start_ok 0.22-0.31) while -0.1 tracked 98 %: the fine term (std 0.1) still pays 0.37 for standing under 0.1.
    # Gated on |wz cmd| > yaw_cmd_min, std 0.04 (standing under 0.1 -> 0.002). Off by default.
    rew_track_yaw_velocity = 0.0
    track_yaw_velocity_std = 0.04
    yaw_cmd_min = 0.03
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
    # 11-05 full run: the gate above with "other foot lifted at all" was beaten from iteration ~100 (air ratio 0.91 /
    # 0.06: right foot blips, left foot swings). The other foot must now complete a real swing (>= air_time_min_s),
    # and the ~1 s average air ratios of the two feet are pushed together (|ema_L - ema_R|, gated on a moving command).
    rew_air_asymmetry = -3.0
    air_asymmetry_tau_s = 1.0
    # cadence ceiling (11-13). On the robot the gait speeds up inside a walking bout (1.4 -> 2.0-2.5 Hz per foot, short
    # steps, with a left yaw); in sim r21 steps at ~1.35 Hz per foot (1.5 Hz lateral). Per lift-off: cost =
    # relu(cadence_min_period_s - time since the same foot's previous lift-off) / cadence_min_period_s, gated on a moving
    # command. 0.55 s = 1.8 Hz leaves normal sim gaits untouched (unlike air_time_min_s 0.2, which collapsed kb03).
    rew_cadence = 0.0
    cadence_min_period_s = 0.55
    # anti-shudder (11-14): r21 / kb05 walk with a body roll / pitch rate 10x standing, 60-90 % of it above 3 Hz (also
    # without any DR), i.e. a per-step shudder, not the gait sway. action_acc = sum (a_t - 2 a_t-1 + a_t-2)^2 of the
    # executed action; foot_impact = per touchdown, (downward foot speed the step before contact)^2. 0 = off.
    rew_action_acc = 0.0
    rew_foot_impact = 0.0
    # yaw shudder (11-14): straight walking wobbles in yaw at 0.11-0.20 rad/s rms, >80 % above 3 Hz, with almost no net
    # heading drift; body_ang_vel covers roll / pitch only. L2 penalty on (wz cmd - wz body)^2. 0 = off.
    rew_yaw_rate_err = 0.0
    # gait symmetry. With the legs not mirror images (right leg = left leg rotated pi about z), the knee-fine-tune policy
    # lifted the right foot 0.3-1.3 cm lower than the left (2.7-3.1 vs 3.3-4.0 cm), landed it 0.5 cm further forward
    # and 1.1 cm wider when walking forward, and stepped the left foot forward / the right foot back in place (yaw drift).
    # Penalty on per-foot landing averages (alpha per landing, ~4-swing window), left vs mirrored right:
    # (|peak_L - peak_R| / peak_scale + |x_L - x_R| / pos_scale + |y_L - y_R| / pos_scale) for pure forward / backward
    # commands only (lateral / turn commands move the feet asymmetrically by design).
    rew_gait_symmetry = -1.0
    gait_symmetry_alpha = 0.25
    gait_symmetry_peak_scale = 0.01
    gait_symmetry_pos_scale = 0.02
    gait_symmetry_max_vy = 0.02
    gait_symmetry_max_wz = 0.05
    gait_symmetry_nominal_y = 0.065
    # mirror_peak: per landing, |swing peak - population mean peak of the mirror cell (other foot, negated vy / wz command
    # signs)| / mirror_peak_scale, for every walking command. Off by default (RSI r05 turns it on): the 11-09/10 evals
    # showed the leading leg lifting ~0.3-0.4 cm higher on one side for sideways / large-turn commands, which
    # gait_symmetry (pure forward / backward only) never sees. Table EMA rate per step with landings in the cell.
    rew_mirror_peak = 0.0
    mirror_peak_scale = 0.01
    mirror_peak_alpha = 0.05
    # Open Duck Mini actor history: also observe the two older actions (t-1, t-2) and the commanded joint target
    # (after rate limit / clamps, minus default) before the command block -> 45 + 36 = 81-D. Commands stay last.
    obs_action_history = False
    # Joint target slew limit (rad/s), applied once per control step to q_des before the actuator model (Open Duck
    # max_motor_velocity 5.24). The real controller must apply the same limit. 0 = off.
    target_rate_limit = 0.0
    # First-order low-pass on the joint target each control step, before the rate limit: target = prev + alpha * (new - prev).
    # 1 (or 0) = off. 11-11 real r12 run: 30 Hz period-2 chatter (the real joints follow the target faster than the sim).
    target_lowpass_alpha = 1.0
    # Knee flexion floor: sum over knees of relu(knee_extension_min - knee)^2 (absolute joint angle, default 0.2 rad).
    # NOTE the knee sign: q = 0 is a 90 deg bend and +q EXTENDS the leg (hip-ankle 10.5 cm at 0, 11.5 at 0.2, 8.2 at
    # -0.4), the same in both legs (right leg = left rotated pi about z, so it bends the other way in the world). So
    # this caps swing FLEXION (foot lift), not hyperextension. The 11-11 real r12 walk swung the knee to -0.4 rad
    # (sim the same); a cap here trades foot clearance for a smaller gait. 0 = off.
    rew_knee_extension = 0.0
    knee_extension_min = -0.87266  # old-zero 0.0
    # Reference swing imitation (Open Duck Mini / Disney BDX style, but phase-free for the policy: the reference is
    # anchored to each foot's own lift-off, s = air_time / ref_swing_time). Swing-foot cost per step, capped:
    # (z - h sin(pi s))^2 / sz^2 + (x - x_ref)^2 / sx^2 + (y - y_ref)^2 / sy^2 in the heading frame relative to the
    # root (y mirrored for the right foot). x_ref = -Ax cos(pi s), Ax = vx * ref_stance_time / 2 (Raibert touchdown);
    # y_ref = nominal_y - side * Ay cos(pi s), Ay = vy * ref_stance_time / 2; z_ref = 0 once s > 1 (no long swings).
    # r05 swings: 0.30-0.40 s, peak 3.1 cm, touchdown x +1.5 cm at vx 0.15. Off by default (RSI r10 turns it on).
    rew_ref_swing = 0.0
    ref_swing_time = 0.35
    ref_stance_time = 0.40
    ref_sigma_z = 0.01
    ref_sigma_x = 0.03
    ref_sigma_y = 0.02
    ref_cost_cap = 10.0
    # RSI r17: the plain reference lands the trailing foot Ay (1.6 cm at 0.08 m/s) inside the nominal 6.5 cm, i.e. soles
    # ~10 cm apart while feet_too_close penalises < foot_min_width (0.12): r12-r16 reached only 0.015-0.02 m/s of a
    # 0.08 m/s sideways command. True: both feet's lateral reference is shifted out by |Ay|.
    ref_lateral_widen = False
    # Straight hips: sum over hip_yaw of (q - default)^2 / std^2 while no turn is commanded (|wz cmd| < max_wz).
    # r05 walked with hip_yaw L +3.8 / R -6.7 deg off default for straight commands (feet toed differently).
    rew_hip_yaw_straight = 0.0
    hip_yaw_straight_std = 0.05
    hip_yaw_straight_max_wz = 0.05
    # Stance-knee crouch: sum over feet in contact of relu(knee - default - free)^2. The 11-05 local run walked and
    # stood on a 0.85 rad stance knee (default 0.2, 480k server run 0.38). In that squat the standing pose reward is
    # exp(-5.8) ~ 0, so it never learned to stand still (stepped in place under a zero command, fell when pushed).
    rew_stance_knee_flex = -5.0
    stance_knee_free_flex = 0.2
    # Min 0.1 s (not 0.2): with 0.2 s no foot ever left the ground in a 600-iteration trial (the policy skated).
    # With 0.1 s it stepped (single support 0.98, air time 0.22 s, peak 4 cm by iteration 600).
    air_time_min_s = 0.1
    air_time_max_s = 0.5
    # Foot clearance: |sole height - target| * |foot v_xy|. Swing height: (peak / target - 1)^2 at each landing.
    rew_foot_clearance = -2.0
    # -0.25 (Microduck) left the per-foot peaks free: the left foot rose 3.2-3.7 cm, the right 2.8-3.0 cm, for every
    # command. A 0.4 cm miss costs (0.4 / 3)^2 = 0.018 per landing, so -10 makes it ~0.2 / s per foot. Per landing
    # (not a running average), so standing still does not avoid it for free.
    rew_foot_swing_height = -10.0
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
    # Per-episode constant yaw torque on the pelvis (Nm, world z), uniform in this range; (0, 0) = off.
    # Real KB sessions drift +0.09 / -0.02 rad/s in yaw with no sim joint-error explanation (11-13 real2sim).
    yaw_torque_bias_range = (0.0, 0.0)
    # DR widening (11-14, kb05): start real-matched / narrow, widen linearly to the usual cfg + event ranges over
    # dr_widen_steps (absolute curriculum steps = common_step_counter + curriculum_step_offset; (0, 0) = off, final ranges
    # from the start). Startup-type draws are redrawn for all envs at dr_widen_levels evenly spaced levels.
    dr_widen_steps = (0, 0)
    dr_widen_levels = 5
    dr_start_static_friction = (0.7, 1.0)
    dr_start_dynamic_friction = (0.6, 0.9)
    dr_start_pelvis_mass = (-0.2, 0.2)
    dr_start_pelvis_com = 0.005
    dr_start_kp = (1.05, 1.2)
    dr_start_kd = (0.9, 1.1)
    dr_start_push_x = (-0.2, 0.2)
    dr_start_push_y = (-0.1, 0.1)
    dr_start_coulomb_scale = (0.4, 0.6)
    dr_start_armature_scale = (0.9, 1.1)
    dr_start_delay_jitter_ms = (-2.0, 2.0)
    dr_start_joint_offset_frac = 0.15
    dr_start_yaw_bias = (-0.6, 0.6)
    # Stability gate (11-14, nr04e): dr_gate_min_steps > 0 replaces the time schedule; the level goes up one step when it
    # has been held dr_gate_min_steps env steps and the EMA (dr_gate_ema_steps) stand-still ratio < dr_gate_max_stand_still,
    # fall rate (falls / env / s) <= dr_gate_fall_ratio x max(level-0 reference, dr_gate_fall_floor) and linear tracking
    # error <= dr_gate_track_ratio x level-0 reference. It is a collapse detector, not a quality bar (wider DR raises
    # falls / error a bit by itself).
    # A level that never passes stays (the log says which signal failed).
    dr_gate_min_steps = 0
    dr_gate_ema_steps = 2000.0
    dr_gate_max_stand_still = 0.08
    dr_gate_fall_ratio = 3.0
    dr_gate_fall_floor = 0.003
    dr_gate_track_ratio = 1.3
    # Level the gate starts at (a resumed run that already adapted to some levels; the reference is taken there) and the
    # max hold (env steps) after which a level whose only failing check is tracking goes up anyway. 0 = no max hold.
    dr_gate_start_level = 0
    dr_gate_max_hold_steps = 0

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


# ---------------------------------------------------------------------------------------------------------------------
# Knee-bent (KB) line (11-13, user): r21's crouched stance on the patched USD, kept separate from the straight-leg
# q=0 line. Training / eval / real2sim for this line use Template-Arti-H1-Velocity-KB(-Play)-Direct-v0 and logs/rsi_kb.
@configclass
class ArtiH1VelocityKBEnvCfg(ArtiH1VelocityEnvCfg):
    # the patched (5.1-import) USD keeps the links at Robot/<link>
    imu_prim_path = "/World/envs/env_.*/Robot/pelvis"
    imu_parent_path_template = "/World/envs/env_{env_id}/Robot/{body}"

    def __post_init__(self):
        super().__post_init__()
        from arti_h1.robots import ARTI_H1_KB_CFG

        self.scene.robot = self.scene.robot.replace(spawn=ARTI_H1_KB_CFG.spawn, init_state=ARTI_H1_KB_CFG.init_state)


@configclass
class ArtiH1VelocityKBEnvCfg_PLAY(ArtiH1VelocityKBEnvCfg):
    curriculum_step_offset = 10_000 * NUM_STEPS_PER_ENV

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
