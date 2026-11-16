#!/bin/bash
# nr04h (11-16): straight-leg (q=0) line, fine-tune nr04g's agent_150000 (before its stand-still collapse).
# nr04g trained ~150k steps at the final DR level with a normal gait (stand-still 0.015-0.024, air ratio 0.24), then
# stopped stepping (stand-still 0.26, rollback to level 1 did not recover it; eval falls 151). When it collapsed the
# linear tracking reward went UP (1.10 -> 1.55): with commands <= 0.15 m/s standing earns more tracking than walking.
# NEEDS the nr04g run folder on the server: logs/skrl/arti_h1_velocity/2026-11-15_00-56-32_ppo_torch/checkpoints/agent_150000.pt
# nr04h = nr04g recipe +
#  - init agent_150000, gate at the final level from the start (start level 4), rollback at stand-still 0.25 kept
#  - stand_still_commanded -3 -> -6 (the walking gait must out-earn standing)
#  - lean_roll -150 (1 s mean of sin(roll)^2: the real robot stands 1-2 deg right side down; sim per-env lean 1-1.2 deg, p90 2-2.4, from joint zero errors)
#  - straight_drift -1500 (pure forward / backward commands: 1 s mean of body vy^2 + (0.3 x yaw-rate error)^2; every
#    policy crabs 8-13 % sideways per env). Metrics/lean_roll_deg, Metrics/straight_vy_drift.
#  - 11-16 robot test: kb08 (low-pass 0.35) stood in a self-sustained 8 Hz ankle-pitch oscillation after walking (not
#    reproduced in sim); its gyro -> joint target gain was 2-7x r21. So the low-pass stays at 0.6 and the gyro is made
#    less trustworthy: MTi-3 gyro white noise 0.01 -> 0.04 rad/s, per-episode bias +-0.01 -> +-0.03 rad/s
#    (eval: gyro_gain_sum / gyro_y_gain_sagittal, compare with r21)
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1)
# plus straight_score / straight_lat_ratio; check [DR] for rollbacks.
# usage (repo root, Isaac Lab python as $PY): server/train_nr04h.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint logs/skrl/arti_h1_velocity/2026-11-15_00-56-32_ppo_torch/checkpoints/agent_150000.pt env.curriculum_step_offset=1110000 agent.trainer.timesteps=240000 \
  env.obs_action_history=true env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.4 env.rew_track_yaw_velocity=2.0 env.rew_stand_still_commanded=-6.0 env.rew_lean_roll=-150.0 env.rew_straight_drift=-1500.0 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.target_rate_limit=5.24 env.target_lowpass_alpha=0.6 env.mti3_gyro_noise_std_rad_s=0.04 'env.mti3_gyro_bias_range_rad_s=[-0.03,0.03]' \
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
  'env.dr_widen_steps=[1110000,1110000]' env.dr_gate_min_steps=24000 env.dr_gate_start_level=4 env.dr_gate_max_hold_steps=36000 env.dr_gate_rollback_stand=0.25
