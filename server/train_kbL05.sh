#!/bin/bash
# kbL05 (11-16): knee-bent (KB) line with the 0.8 kg lighter battery (not on the robot yet), fine-tune kbL04.
# kbL04 trained at the final DR ranges: falls 0, push 2.3 %, tilt 3.38, angvel .432, start_ok 1, vx .63, but
# drift_wz -.023 (limit .02) and straight-walk sideways drift 10.5 % (heading error 9.7 deg).
# kbL05 = kbL04 recipe (final ranges from the start: gate start level 4, rollback at stand-still 0.25) +
#  - joint target low-pass 0.6 -> 0.35 (local kb08, deployed: body rate -27 %, push falls 3.1 -> 0.8 % vs r21);
#    the robot yaml must then use policy.target_lowpass_alpha 0.35
#  - lean_roll -150 (1 s mean of sin(roll)^2: the real robot stands 1-2 deg right side down, sim policies < 0.5 deg)
#  - straight_drift -1500 (pure forward / backward commands: 1 s mean of body vy^2 + (0.3 x yaw-rate error)^2)
#    Metrics/lean_roll_deg, Metrics/straight_vy_drift.
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1)
# plus straight_score / straight_lat_ratio; compare with kbL04.
# usage (repo root of the server-light worktree, Isaac Lab python as $PY): server/train_kbL05.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/results/kbL04/final.pt env.curriculum_step_offset=5106000 agent.trainer.timesteps=240000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.15' 'env.track_lateral_velocity_std=0.05' 'env.rew_track_lateral_velocity=4.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.rew_lean_roll=-150.0' 'env.rew_straight_drift=-1500.0' 'env.target_lowpass_alpha=0.35' 'env.command_resample_time_s=[1.5,5.0]' 'env.rew_track_yaw_velocity=2.0' 'env.rew_stand_still_commanded=-3.0' \
 'env.imu_gyro_extra_delay_steps=1' \
 'env.actuator_delay_ms={hip_yaw:4.0,hip_roll:4.0,hip_pitch:4.0,knee:4.0,ankle_pitch:3.0,ankle_roll:3.0}' 'env.actuator_delay_jitter_ms=[-6.0,6.0]' \
 'env.actuator_delay_extra_ms={hip_roll:25.0,ankle_pitch:12.0}' \
 'env.actuator_coulomb_scale_range=[0.3,1.2]' 'env.actuator_armature_scale_range=[0.5,1.3]' \
 'env.events.actuator_gains.params.stiffness_distribution_params=[0.85,1.3]' 'env.events.actuator_gains.params.damping_distribution_params=[0.5,1.2]' \
 'env.yaw_torque_bias_range=[-0.8,0.8]' \
 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' \
 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' \
 'env.events.add_pelvis_mass.params.mass_distribution_params=[-1.3,-0.3]' \
 'env.dr_start_pelvis_mass=[-1.0,-0.6]' \
 'env.dr_widen_steps=[5106000,5106000]' 'env.dr_gate_min_steps=24000' 'env.dr_gate_start_level=4' 'env.dr_gate_max_hold_steps=36000' 'env.dr_gate_rollback_stand=0.25' 'env.dr_gate_max_stand_still=0.12'
