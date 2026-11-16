#!/bin/bash
# kbS01 (11-16): knee-bent (KB) line, NORMAL battery (deployable), on the server. Fine-tune kb10 (local, deployed
# to the robot 11-16, humanoid_controller 8f81450).
# kb10 (sim, real2sim conditions): falls 0, push 0.8 %, tilt 3.03, angvel .396, vx .64, yaw .80, straight |vy|/vx 9.5 %,
# gyro -> action gain 1.16 (r21 1.06), but lateral walking only ~25 % of a 0.08 m/s command (r21 33 %), and the
# kb10 curves were flat from 30k on (training longer with the same rewards does not help).
# kbS01 = kb10 recipe + overall improvements (same as local kb11, A/B on one term):
#  - lateral: pure-sideways bucket 0.15 -> 0.25, vy command range +-0.08 -> +-0.12, lateral std 0.05 -> 0.03
#    (standing under 0.08 earned 0.24 of the term, now 0.02), weight 4 -> 5
#  - yaw-only tracking weight 2 -> 3 (kb10 small-turn tracking 0.80 vs r21 0.95)
#  - stand-still-under-command -3 -> -5 (nr04g / nr04h collapsed to standing)
#  - straight_drift kept at -1500 here (local kb11: -1000, to see whether it costs forward speed)
#  - 480k steps (new targets need time), checkpoints every 120k sent back in mid/ to pick the best point
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1),
# gyro_gain_sum <= ~1.2, then straight_score / lateral tracking; compare with kb10.
# usage (repo root of the server worktree, Isaac Lab python as $PY): server/train_kbS01.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/vel_kb10_240k.pt env.curriculum_step_offset=4866000 agent.trainer.timesteps=480000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.target_lowpass_alpha=0.6' 'env.mti3_gyro_noise_std_rad_s=0.04' 'env.mti3_gyro_bias_range_rad_s=[-0.03,0.03]' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.imu_gyro_extra_delay_steps=1' \
 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.25' 'env.command_linear_y_range=[-0.12,0.12]' 'env.track_lateral_velocity_std=0.03' 'env.rew_track_lateral_velocity=5.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.rew_lean_roll=-150.0' 'env.rew_straight_drift=-1500.0' 'env.rew_track_yaw_velocity=3.0' 'env.rew_stand_still_commanded=-5.0' \
 'env.command_resample_time_s=[1.5,5.0]' \
 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
 'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' 'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
 'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' 'env.actuator_delay_extra_curriculum_steps=[4866000,4866000]' \
 'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
 'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' 'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
 'env.yaw_torque_bias_range=[-0.8,0.8]'
