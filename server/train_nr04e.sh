#!/bin/bash
# nr04e (11-14): straight-leg (q=0) line, fine-tune nr01 with a real-matched start and stability-gated DR widening.
# nr03b_repro on the server reproduced the local nr03b (curves identical, no stand-still collapse), so the nr04c / nr04d
# collapse comes from the recipe (push 0.5 / joint zero-error DR / cmd resample switched on at full range at once).
# nr04e = nr03b recipe +
#  - real2sim centres (11-13 KB real logs, same hardware): actuator delay 4/4/4/4/3/3 ms (was 17/15/15/15/10/10),
#    gyro extra delay 1 step (was 2), extra lag hip_roll 25 / ankle_pitch 12 ms (was 50 / 25), yaw torque bias
#  - DR starts narrow around the real values (cfg.dr_start_*: friction 0.7-1.0, pelvis mass +-0.2 kg, CoM +-0.5 cm,
#    kp x1.05-1.2, kd x0.9-1.1, push +-0.2 / +-0.1, Coulomb x0.4-0.6, armature x0.9-1.1, jitter +-2 ms, 15 % of the
#    joint zero-error ranges, yaw bias +-0.6 Nm, extra lag 0) and widens in 5 levels to the final ranges below
#  - a level goes up only when it has been held 24k steps (~1000 iterations) AND the gait is stable (EMA stand-still
#    ratio < 0.08, fall rate <= 3x max(level 0, 0.003/s), linear tracking error <= 1.3x level 0). Log lines: [DR] gate ...
#  - final ranges: push x +-0.5 / y +-0.3, joint zero error nr04d (ankle_pitch +-0.20), friction 0.4-1.4,
#    pelvis mass +-0.5 kg, kp x0.85-1.3, kd x0.5-1.2, Coulomb x0.3-1.2, armature x0.5-1.3, jitter +-6 ms, yaw +-0.8 Nm
#  - command resample stays at the default 3-8 s (not real2sim; nr04c/d changed it)
# Read the result as: final level reached? (grep '\[DR\]' train.log) + stability gate on the final-range eval:
#   falls 0, push <= ~4.7%, tilt <= 3.6 deg, angvel_xy <= 0.45, |drift_wz| <= 0.02, start_ok 1; compare with nr01.
# usage (repo root, Isaac Lab python as $PY): server/train_nr04e.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/vel_nr01.pt env.curriculum_step_offset=720000 agent.trainer.timesteps=240000 \
  env.obs_action_history=true env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.4 env.rew_track_yaw_velocity=2.0 env.rew_stand_still_commanded=-3.0 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.target_rate_limit=5.24 env.target_lowpass_alpha=0.6 \
  env.imu_gyro_extra_delay_steps=1 \
  'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' \
  'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
  'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' \
  'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
  'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' \
  'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
  'env.yaw_torque_bias_range=[-0.8,0.8]' \
  'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
  'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
  'env.dr_widen_steps=[720000,720000]' env.dr_gate_min_steps=24000
