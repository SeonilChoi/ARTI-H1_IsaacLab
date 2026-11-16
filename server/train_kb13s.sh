#!/bin/bash
# kb13s (11-17): server copy of the LOCAL round kb13 (same overrides, init, offset and length), to run in a SECOND
# server folder next to kbS03 (see "two folders" below). kb13 = kb11 recipe + gentle heel strike on forward walking
# only (heel_strike -0.5, target 0.05 rad, excess above 1.4x, vx cmd > 0.03 and |vy| < vx), foot_impact -1.0,
# foot_flat -20, straight_drift -1000 (kbS03: -1500, 360k), 240k steps from kb11 (server/vel_kb11_480k.pt).
# Accept on the stability gate + heel_first_frac on forward steps; compare with kb11 / kbS03.
# Two folders: run_round.sh puts this folder's source/arti_h1 first on PYTHONPATH (arti_h1 is an editable install of
# the first folder), and logs/skrl + server/results are per folder; push from branch server-b (results land there).
# usage (repo root of the second worktree, Isaac Lab python as $PY): server/train_kb13s.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/vel_kb11_480k.pt env.curriculum_step_offset=5346000 agent.trainer.timesteps=240000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.target_lowpass_alpha=0.6' 'env.mti3_gyro_noise_std_rad_s=0.04' 'env.mti3_gyro_bias_range_rad_s=[-0.03,0.03]' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.imu_gyro_extra_delay_steps=1' \
 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.25' 'env.command_linear_y_range=[-0.12,0.12]' 'env.track_lateral_velocity_std=0.03' 'env.rew_track_lateral_velocity=5.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.rew_lean_roll=-150.0' 'env.rew_straight_drift=-1000.0' 'env.rew_track_yaw_velocity=3.0' 'env.rew_stand_still_commanded=-5.0' 'env.rew_heel_strike=-0.5' 'env.heel_strike_target_rad=0.05' 'env.heel_strike_max_ratio=1.4' 'env.heel_strike_min_vx=0.03' 'env.rew_foot_impact=-1.0' 'env.rew_foot_flat=-20.0' \
 'env.command_resample_time_s=[1.5,5.0]' \
 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
 'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' 'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
 'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' 'env.actuator_delay_extra_curriculum_steps=[5346000,5346000]' \
 'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
 'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' 'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
 'env.yaw_torque_bias_range=[-0.8,0.8]'
