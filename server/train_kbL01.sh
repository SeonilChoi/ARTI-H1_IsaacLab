#!/bin/bash
# kbL01 (11-13): kb03 with a battery 0.8 kg lighter. Pelvis mass offset -0.8 kg +-0.5 (training range -1.3..-0.3 kg,
# was -0.5..+0.5); everything else = kb03. Branch server-light, run from its own worktree (see server/run_round.sh).
# kb03 (11-13): knee-bent (KB) line = r21 crouched stance on the patched USD (deployed on humanoid_controller
# arti-h1-new-pelvis). Fine-tune kb01 (r21 + real latency: base delay 5-8 ms, jitter +-8 ms) with the 11-13 real2sim items:
#  - yaw_torque_bias_range [-0.8, 0.8] Nm: per-episode constant pelvis yaw torque. Real sessions drift in yaw by a different
#    amount each time (09:20 +0.09 rad/s, 12:42 a stalled left turn, 13:15 +0.2..0.3 rad/s walking forward); a replay of
#    the 09:20 log reproduces every segment with +0.4..0.6 Nm, the 12:42 one with about -0.5 Nm.
#  - air_time_min_s 0.2 (was 0.1): on the robot the cadence runs away inside a walking bout (1.4 -> 2.0-2.5 Hz, steps get
#    short: knee amplitude 14 -> 8 deg) together with an unwanted left yaw; at 2.3 Hz each swing lasts ~0.17 s, so with a
#    0.2 s floor those stutter steps earn no air-time reward. Sim replays (yaw bias, low friction) do not show the runaway.
# usage (repo root, Isaac Lab python as $PY): server/train_kbL01.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/vel_kb01_114k.pt env.curriculum_step_offset=4386000 agent.trainer.timesteps=240000 \
  'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.imu_gyro_extra_delay_steps=2' 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.15' 'env.track_lateral_velocity_std=0.05' 'env.rew_track_lateral_velocity=4.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.events.push_robot.params.velocity_range.x=[-0.5,0.5]' 'env.events.push_robot.params.velocity_range.y=[-0.3,0.3]' 'env.target_lowpass_alpha=0.6' 'env.joint_obs_offset_range_by_type={hip_yaw:0.03,hip_roll:0.06,hip_pitch:0.08,knee:0.08,ankle_pitch:0.20,ankle_roll:0.06}' 'env.command_resample_time_s=[1.5,5.0]' 'env.rew_track_yaw_velocity=2.0' 'env.rew_stand_still_commanded=-3.0' 'env.actuator_delay_ms={hip_yaw:8.0,hip_roll:7.0,hip_pitch:7.0,knee:7.0,ankle_pitch:5.0,ankle_roll:5.0}' 'env.actuator_delay_jitter_ms=[-8.0,8.0]' \
  'env.yaw_torque_bias_range=[-0.8,0.8]' env.air_time_min_s=0.2 \
  'env.events.add_pelvis_mass.params.mass_distribution_params=[-1.3,-0.3]'
