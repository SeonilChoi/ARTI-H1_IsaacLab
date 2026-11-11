#!/bin/bash
# RSI round r15 (11-11): fine-tune the deployed r14 velocity policy (humanoid_controller agent_vel_r14_240000.onnx)
# for real-robot robustness:
#  - per-joint homing / encoder zero errors drawn per episode (ankle_pitch +-0.15 rad; the real right ankle_pitch
#    reads ~10-16 deg too high and the robot spins left ~16 deg/s; sim r14 with -15 deg spins +28 deg/s)
#  - commands resampled every 1.5-5 s (was 3-8 s) for more walk <-> stop transitions (real robot staggers 1-4 s on stop)
# Everything else = r14 (81-D obs, slew limit 5.24 rad/s, target low-pass 0.6, actuator / IMU delay jitter).
# usage (repo root, Isaac Lab python on PATH as $PY): server/train_r15.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/vel_r14_240000.pt env.curriculum_step_offset=2832000 agent.trainer.timesteps=240000 \
  env.obs_action_history=true env.target_rate_limit=5.24 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.imu_gyro_extra_delay_steps=2 env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=3.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.4 env.events.push_robot.params.velocity_range.x=[-0.4,0.4] env.events.push_robot.params.velocity_range.y=[-0.3,0.3] env.target_lowpass_alpha=0.6 \
  'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.15,ankle_roll:0.06}' \
  'env.command_resample_time_s=[1.5,5.0]'
