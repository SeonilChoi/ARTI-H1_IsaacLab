#!/bin/bash
# nr04g (11-15): straight-leg (q=0) line, fine-tune nr04e again (nr04f discarded).
# nr04f (nr04e + gate resumed at level 3 + anti-shudder rewards) reached the final DR level after 24k steps, then stopped
# stepping ~70k steps later: stand-still 0.05 -> 0.86, air ratio 0.23 -> 0.005; eval vx .006, start_ok .016, falls 3.
# Locally (kb06 / kb07) and on kbL03 the anti-shudder rewards only scaled the motion down (vx -15..-30 %) and left the
# 3-10 Hz share of the body rate unchanged (>80 %), so they are dropped here: nr04g = nr04e rewards unchanged +
#  - gate resumes at level 3, max hold 36k (as nr04f): the final ranges are trained
#  - collapse guard env.dr_gate_rollback_stand=0.25: EMA stand-still above 0.25 drops one DR level (log: [DR] gate rollback)
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1),
# compare with nr04e (vx .42 at the final ranges after training at level 3 only).
# usage (repo root, Isaac Lab python as $PY): server/train_nr04g.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/results/nr04e/final.pt env.curriculum_step_offset=960000 agent.trainer.timesteps=240000 \
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
  'env.dr_widen_steps=[960000,960000]' env.dr_gate_min_steps=24000 env.dr_gate_start_level=3 env.dr_gate_max_hold_steps=36000 env.dr_gate_rollback_stand=0.25
