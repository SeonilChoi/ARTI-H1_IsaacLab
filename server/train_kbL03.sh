#!/bin/bash
# kbL03 (11-14): knee-bent (KB) line with the 0.8 kg lighter battery, round 3. Branch server-light, own worktree.
# kbL02 result (best KB policy so far on the final-range eval): start_ok 1, vx .73 (kbL01b .59, kb05 .64), vy .34,
#   wz .81/.86, push 1.6 %, tilt 3.25, angvel .417, drift_wz 0.000, but falls_nopush 1 -> one fall short of the gate.
#   The gate passed levels 0-2 in 72k steps, then held level 3 (75 %) for ~165k steps on linear tracking error alone
#   (0.102 vs 1.3 x 0.077 = 0.100; falls 0.0008/s, stand-still 0.04), so it never trained at the final ranges.
# kbL03 = kbL02 recipe +
#  - gate resumes at level 3 (env.dr_gate_start_level=3) and env.dr_gate_max_hold_steps=36000 (a level whose only failing
#    check is tracking goes up after 36k steps; stand-still / falls must still pass) -> trains at the final ranges
#  - anti-shudder (local kb06 / kb07, 11-14): KB policies shudder at 3-10 Hz in roll / pitch AND yaw (straight walking:
#    yaw wobble 4-8 deg p-p, >80 % above 3 Hz, almost no heading drift). body_ang_vel -0.4 -> -0.8 (kb06's -1.5 cost 18 %
#    forward tracking), new yaw_rate_err -2.0, action_acc -0.6, foot_impact -2.0 (Metrics/body_rollpitch_rate,
#    yaw_rate_err_rms, action_acc_rms, touchdown_vz)
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1)
# and compare with kbL02 (vx .73 is the bar to keep).
# usage (repo root of the server-light worktree, Isaac Lab python as $PY): server/train_kbL03.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-KB-Direct-v0 physics=isaacsim_physx --num_envs $N \
 --checkpoint server/results/kbL02/final.pt env.curriculum_step_offset=4866000 agent.trainer.timesteps=240000 \
 'env.obs_action_history=true' 'env.target_rate_limit=5.24' 'env.actuator_delay_step_jitter=1' 'env.imu_gyro_delay_per_step=true' 'env.rew_ref_swing=-0.5' 'env.rew_hip_yaw_straight=-0.25' 'env.rew_mirror_peak=-2.0' 'env.lateral_walk_fraction=0.15' 'env.track_lateral_velocity_std=0.05' 'env.rew_track_lateral_velocity=4.0' 'env.foot_min_width=0.12' 'env.rew_feet_too_close=-4.0' 'env.rew_body_ang_vel=-0.8' 'env.rew_yaw_rate_err=-2.0' 'env.rew_action_acc=-0.6' 'env.rew_foot_impact=-2.0' 'env.target_lowpass_alpha=0.6' 'env.command_resample_time_s=[1.5,5.0]' 'env.rew_track_yaw_velocity=2.0' 'env.rew_stand_still_commanded=-3.0' \
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
 'env.dr_widen_steps=[4866000,4866000]' 'env.dr_gate_min_steps=24000' 'env.dr_gate_start_level=3' 'env.dr_gate_max_hold_steps=36000' 'env.dr_gate_max_stand_still=0.12'
