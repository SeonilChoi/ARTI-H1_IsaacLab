#!/bin/bash
# nr04f (11-14): straight-leg (q=0) line, fine-tune nr04e.
# nr04e result: the gate passed levels 0-2 in 75k steps, then held level 3 (75 %) for the remaining ~165k steps on
# linear tracking error alone (0.116-0.122 vs the 1.3 x 0.082 = 0.107 limit; falls 0.003-0.004/s within 3 x 0.003, stand-
# still 0.01). The final-range eval was therefore on ranges it never trained at: falls 3, push 8.6 %, tilt 4.2, angvel
# .44, vx .42 -> not accepted. Tracking error grows with the DR width itself, so a fixed 1.3 x level-0 bar is too tight.
# nr04f = nr04e recipe +
#  - gate resumes at level 3 (env.dr_gate_start_level=3; reference re-taken there), so level 4 (final ranges) comes after
#    the 24k hold; env.dr_gate_max_hold_steps=36000: a level held 36k steps whose only failing check is tracking goes
#    up anyway (stand-still and falls must still pass). Log: [DR] gate pass / force (max hold)
#  - anti-shudder (local kb06 / kb07, 11-14): r21 / kb05 / nr policies shudder at 3-10 Hz in roll / pitch AND yaw (straight
#    walking: yaw wobble 4-8 deg p-p, 0.15-0.20 rad/s rms, >80 % above 3 Hz, but almost no heading drift), also without DR.
#    body_ang_vel -0.4 -> -0.8 (kb06's -1.5 cost 18 % forward tracking), new yaw_rate_err -2.0 ((wz cmd - wz)^2; body_ang_vel
#    is roll / pitch only), action_acc -0.6 (sum (a_t - 2 a_t-1 + a_t-2)^2), foot_impact -2.0 (touchdown vz^2)
#    New metrics: Metrics/body_rollpitch_rate, yaw_rate_err_rms, action_acc_rms, touchdown_vz.
# Accept on the stability gate (falls 0, push <= ~4.7 %, tilt <= 3.6, angvel <= 0.45, |drift_wz| <= 0.02, start_ok 1),
# compare with nr04e; check the [DR] log reached level 4.
# usage (repo root, Isaac Lab python as $PY): server/train_nr04f.sh [num_envs]
set -eu
PY=${PY:-python}
N=${1:-4096}
$PY scripts/skrl/train.py --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs $N \
  --checkpoint server/results/nr04e/final.pt env.curriculum_step_offset=960000 agent.trainer.timesteps=240000 \
  env.obs_action_history=true env.rew_ref_swing=-0.5 env.rew_hip_yaw_straight=-0.25 env.rew_mirror_peak=-2.0 env.lateral_walk_fraction=0.15 env.track_lateral_velocity_std=0.05 env.rew_track_lateral_velocity=4.0 env.foot_min_width=0.12 env.rew_feet_too_close=-4.0 env.rew_body_ang_vel=-0.8 env.rew_yaw_rate_err=-2.0 env.rew_action_acc=-0.6 env.rew_foot_impact=-2.0 env.rew_track_yaw_velocity=2.0 env.rew_stand_still_commanded=-3.0 env.actuator_delay_step_jitter=1 env.imu_gyro_delay_per_step=true env.target_rate_limit=5.24 env.target_lowpass_alpha=0.6 \
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
  'env.dr_widen_steps=[960000,960000]' env.dr_gate_min_steps=24000 env.dr_gate_start_level=3 env.dr_gate_max_hold_steps=36000
