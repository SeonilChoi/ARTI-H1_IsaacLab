#!/bin/bash
# nr03b_repro (11-14): server check. nr03b re-run unchanged (same init nr01, offset, timesteps, overrides) to tell apart
# "the server collapses straight-leg runs" from "one of nr04d's additions collapses them".
#  - local nr03b (11-12): start_ok 1, vx .43, wz .92/.97, falls 1, push 14%; it walked.
#  - server nr04c / nr04d (11-13) collapsed into standing still (stand_still_commanded_ratio .03 -> .89, start_ok ~.02).
#    nr04d = nr03b + push x 0.5 + joint zero DR (all joints) + command resample 1.5-5 s; its env.yaml differs from local
#    nr03b's only in those three items.
# Reading: repro walks -> one of the three items is the cause; repro stands still -> the server stack is the cause.
# usage (repo root, Isaac Lab python as $PY): server/train_nr03b_repro.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/vel_nr01.pt env.curriculum_step_offset=720000 agent.trainer.timesteps=120000 \
  env.obs_action_history=true env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.4 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' env.rew_track_yaw_velocity=2.0 'env.events.push_robot.params.velocity_range.x=[-0.4,0.4]' env.rew_stand_still_commanded=-3.0 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.imu_gyro_extra_delay_steps=2 env.target_rate_limit=5.24 env.target_lowpass_alpha=0.6
