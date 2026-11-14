#!/bin/bash
# kbL02 (11-14): knee-bent (KB) line with the 0.8 kg lighter battery, round 2. Branch server-light, own worktree.
# kbL01b (kb03 recipe, air_time_min_s 0.1, pelvis mass -1.3..-0.3 kg) passed the stability gate on its own eval:
#   start_ok 1, vx .60, wz .80/.81, falls 0, push 3.9 %, tilt 3.55, angvel .427, drift_wz -.011 (flat curves, no collapse);
#   the lighter battery itself is fine. Its training kept the old delay model (8/7/7/7/5/5 ms, jitter +-8, gyro +2 steps).
# kbL02 = fine-tune kbL01b with the kb05 / nr04e strategy:
#  - real2sim centres (11-13 real logs): actuator delay 4/4/4/4/3/3 ms, gyro extra delay 1 step, extra lag hip_roll 25 /
#    ankle_pitch 12 ms, kp x1.05-1.2 at the start, low Coulomb (x0.4-0.6 at the start)
#  - DR starts narrow around the real values and widens in 5 levels, each level only after a 24k-step hold AND a stable
#    gait (EMA stand-still < 0.12 [kbL01b sits at ~0.06, a collapse ~0.9],
#    falls <= 3x max(level 0, 0.003/s), lin. tracking error <= 1.3x level 0); [DR] gate logs
#  - battery: pelvis mass starts at -0.8 +-0.2 kg and widens to -0.8 +-0.5 kg (-1.3..-0.3, as kbL01b)
#  - final ranges as kbL01b otherwise: push +-0.5 / +-0.3, joint zero error (ankle_pitch +-0.20), yaw bias +-0.8 Nm,
#    cmd resample 1.5-5 s; kp x0.85-1.3, kd x0.5-1.2, Coulomb x0.3-1.2, armature x0.5-1.3, jitter +-6 ms
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1)
# and compare with kbL01b / kb05 / r21.
# usage (repo root of the server-light worktree, Isaac Lab python as $PY): server/train_kbL02.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/results/kbL01b/final.pt env.curriculum_step_offset=4626000 agent.trainer.timesteps=240000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.15' 'env.track_lateral_velocity_std=0.05' 'env.rew_track_lateral_velocity=4.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.4' 'env.target_lowpass_alpha=0.6' 'env.command_resample_time_s=[1.5,5.0]' 'env.rew_track_yaw_velocity=2.0' 'env.rew_stand_still_commanded=-3.0' \
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
 'env.dr_widen_steps=[4626000,4626000]' 'env.dr_gate_min_steps=24000' 'env.dr_gate_max_stand_still=0.12'
