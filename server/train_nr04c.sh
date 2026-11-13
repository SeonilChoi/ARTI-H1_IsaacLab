#!/bin/bash
# nr04c (11-13): straight-leg (q=0) line, fine-tune nr01 (best stability-gate round: falls 0, push 3.1%, tilt 3.8 deg,
# vx .61 / vy .47 / wz .83-.93) toward stable walking with a low foot lift.
#  - foot_target_height 0.018 (lift the foot only a little), body_ang_vel -0.5 (nr01 -0.4), angular_momentum -0.02
#  - joint zero-error randomisation + command resample 1.5-5 s (as r21)
#  - nr04a (upright 3 / std 0.1, body_ang_vel -0.8, angmom -0.05) collapsed into standing still (vx .05, start_ok .2), so
#    the stability weights stay near nr01.
# Accept on the stability gate: falls 0, push <= ~4.7%, tilt <= 3.6 deg, angvel_xy <= 0.45, |drift_wz| <= 0.02, start_ok 1.
# usage (repo root, Isaac Lab python as $PY): server/train_nr04c.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/vel_nr01.pt env.curriculum_step_offset=720000 agent.trainer.timesteps=120000 \
  env.obs_action_history=true env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.events.push_robot.params.velocity_range.y=[-0.3,0.3] env.rew_track_yaw_velocity=2.0 env.rew_stand_still_commanded=-3.0 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.imu_gyro_extra_delay_steps=2 env.target_rate_limit=5.24 env.target_lowpass_alpha=0.6 env.events.push_robot.params.velocity_range.x=[-0.5,0.5] env.foot_target_height=0.018 env.rew_body_ang_vel=-0.5 env.rew_angular_momentum=-0.02 \
  'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
  'env.command_resample_time_s=[1.5,5.0]'
