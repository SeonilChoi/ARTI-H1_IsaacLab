#!/bin/bash
# kb14s (11-17): server copy of the LOCAL round kb14 = kb11 recipe (exact) + the Olaf real-robot runtime pipeline
# (Disney, arXiv 2512.16705, Sec. VII-C): the control-rate joint target is upsampled by first-order hold (linear ramp
# from the previous to the new target over the 4 physics substeps of a control step) and low-passed at 37.5 Hz at the
# physics rate (env.target_foh / env.target_substep_lowpass_hz). ~12 ms more target lag than kb11, so it needs this
# fine-tune; on the robot set policy.target_foh: true / target_substep_lowpass_hz: 37.5 (humanoid_controller b774a94)
# ONLY for models trained with it. Init kb11 (server/vel_kb11_480k.pt), 240k.
# Accept on the stability gate, gyro_gain_sum <= ~1.2, compare with kb11 (shudder / angvel should drop).
# Run in the second folder (branch server-b, see train_kb13s.sh) or after kbS03 in the first.
# usage (repo root, Isaac Lab python as $PY): server/train_kb14s.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/vel_kb11_480k.pt env.curriculum_step_offset=5346000 agent.trainer.timesteps=240000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.target_lowpass_alpha=0.6' 'env.target_foh=true' 'env.target_substep_lowpass_hz=37.5' 'env.mti3_gyro_noise_std_rad_s=0.04' 'env.mti3_gyro_bias_range_rad_s=[-0.03,0.03]' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.imu_gyro_extra_delay_steps=1' \
 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.25' 'env.command_linear_y_range=[-0.12,0.12]' 'env.track_lateral_velocity_std=0.03' 'env.rew_track_lateral_velocity=5.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.rew_lean_roll=-150.0' 'env.rew_straight_drift=-1000.0' 'env.rew_track_yaw_velocity=3.0' 'env.rew_stand_still_commanded=-5.0' \
 'env.command_resample_time_s=[1.5,5.0]' \
 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
 'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' 'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
 'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' 'env.actuator_delay_extra_curriculum_steps=[5346000,5346000]' \
 'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
 'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' 'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
 'env.yaw_torque_bias_range=[-0.8,0.8]'
