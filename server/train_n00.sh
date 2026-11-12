#!/bin/bash
# n00 (11-12): FRESH velocity training (no checkpoint) on the new robot model:
#  - new pelvis (hip_yaw mount 317.07 mm, trunk 0.32367), legs re-zeroed to -20/40/-20 deg (hip_pitch / knee / ankle_pitch)
#  - USD imported from that URDF with Isaac Sim 6.2 (assets/models/arti_h1/arti_h1_new/arti_h1.usd/arti_h1/arti_h1.usda)
#  - default stance q = 0 = -20/40/-20 deg physical (the old stance was hip -39 / knee 78 / ankle -39 deg), init z -0.002
# Recipe = RSI r21 overrides (81-D obs, slew 5.24 rad/s, low-pass 0.6, delays, zero-error randomisation, pushes),
# curriculum from step 0, 1152000 timesteps (= the original vel_start base run). No demos / reference clips.
# usage (repo root, Isaac Lab python as $PY): server/train_n00.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  env.curriculum_step_offset=0 agent.trainer.timesteps=1152000 \
  env.obs_action_history=true env.target_rate_limit=5.24 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.imu_gyro_extra_delay_steps=2 env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.4 env.events.push_robot.params.velocity_range.x=[-0.5,0.5] env.events.push_robot.params.velocity_range.y=[-0.3,0.3] env.target_lowpass_alpha=0.6 \
  'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
  'env.command_resample_time_s=[1.5,5.0]' env.rew_track_yaw_velocity=2.0 env.rew_stand_still_commanded=-3.0
