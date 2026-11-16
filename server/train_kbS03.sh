#!/bin/bash
# kbS03 (11-17): knee-bent (KB) line, NORMAL battery (deployable). Replaces kbS02 (do not run kbS02).
# Init kb11 (local, server/vel_kb11_480k.pt): kb10 + lateral / yaw / anti-standing tuning. Sim eval (real2sim conditions):
# falls 0, push 2.3 %, tilt 2.92, angvel .406, vx .65, lateral .29 (kb10 .25), yaw .85 / .86, gyro gain 1.16.
# kbS02 problem (seen locally in kb12 = same terms): heel_strike -1.0 / target 0.09 rad / excess above 2x + foot_impact
# -2.0 landed heel first 96 % at 7.5 deg and touched down at 0.21 m/s (0.33 before), but then stood on its heels (stance
# sole tilt 2 -> 5.5 deg) and fell (eval falls 6-22 from 60k on).
# kbS03 = kb11 recipe + gentle heel strike, FORWARD WALKING ONLY (user: sideways steps / turning in place keep the old gait):
#  - heel_strike -0.5, target 0.05 rad toes up at touchdown, excess penalty above 1.4x (0.07 rad), only when
#    vx cmd > 0.03 and |vy cmd| < vx cmd (Metrics/touchdown_pitch_deg, heel_first_frac; other steps:
#    Metrics/touchdown_pitch_other_deg should stay at the old value)
#  - foot_impact -1.0 (all steps: slower touchdown), foot_flat -10 -> -20 (stance sole flat, no standing on the heel)
#  - straight_drift -1500 (local kb13: -1000, otherwise identical), 360k steps, 120k checkpoints sent back in mid/
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1),
# gyro_gain_sum <= ~1.2, heel_first_frac up on forward steps; compare with kb11.
# usage (repo root of the server worktree, Isaac Lab python as $PY): server/train_kbS03.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/vel_kb11_480k.pt env.curriculum_step_offset=5346000 agent.trainer.timesteps=360000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.target_lowpass_alpha=0.6' 'env.mti3_gyro_noise_std_rad_s=0.04' 'env.mti3_gyro_bias_range_rad_s=[-0.03,0.03]' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.imu_gyro_extra_delay_steps=1' \
 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.25' 'env.command_linear_y_range=[-0.12,0.12]' 'env.track_lateral_velocity_std=0.03' 'env.rew_track_lateral_velocity=5.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.rew_lean_roll=-150.0' 'env.rew_straight_drift=-1500.0' 'env.rew_track_yaw_velocity=3.0' 'env.rew_stand_still_commanded=-5.0' 'env.rew_heel_strike=-0.5' 'env.heel_strike_target_rad=0.05' 'env.heel_strike_max_ratio=1.4' 'env.heel_strike_min_vx=0.03' 'env.rew_foot_impact=-1.0' 'env.rew_foot_flat=-20.0' \
 'env.command_resample_time_s=[1.5,5.0]' \
 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
 'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' 'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
 'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' 'env.actuator_delay_extra_curriculum_steps=[5346000,5346000]' \
 'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
 'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' 'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
 'env.yaw_torque_bias_range=[-0.8,0.8]'
