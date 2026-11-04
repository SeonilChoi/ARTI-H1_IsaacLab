# arti_h1

AMP walking policy for the ARTI-H1 biped (Isaac Lab + skrl). The cfg defaults are the current training setup,
so training needs no extra arguments. Trained policies are exported to ONNX and deployed with
[humanoid_controller](https://github.com/SeonilChoi/humanoid_controller) (`config/robot/ARTI-H1/agent.onnx`).

## Requirements

Isaac Lab 3.x (main @ 0ca25c56), Isaac Sim 6.2 (PhysX) and/or Newton 1.8 (MuJoCo-Warp 3.15), Warp 1.17,
Python 3.12, PyTorch 2.11 (cu130), skrl 2.1.1; onnxruntime / onnx for export.
The Isaac Sim 5.1 / Isaac Lab 2.3 version of this project is on `main`; see `PORTING_NOTES.md` for the port.

## Install

Linux (Ubuntu 22.04 / 24.04). The robot USD files are stored with Git LFS, so install it first.

```bash
git lfs install
git clone <this repo> arti_h1 && cd arti_h1
source <IsaacLab>/.venv/bin/activate          # Isaac Lab 3.x uv environment (uv sync --extra isaacsim --extra skrl)
uv pip install -e source/arti_h1              # registers the tasks for `isaaclab train|play|list_envs`
uv pip install onnxruntime onnx               # only for ONNX export
```

## Train

```bash
export OMNI_KIT_ACCEPT_EULA=yes
isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Direct-v0 physics=isaacsim_physx
# same run plus the AMP discriminator diagnostics in TensorBoard:
python scripts/skrl/train.py --task Template-Arti-H1-Amp-Direct-v0 physics=isaacsim_physx
```

### Server (Isaac Sim PhysX, tmux)

Two training modes on the same environment and rewards:

| mode | task | agent cfg | log dir |
|---|---|---|---|
| imitation + RL (AMP: style reward from the reference motions) | `Template-Arti-H1-Amp-Direct-v0` | `skrl_amp_cfg.yaml` | `logs/skrl/arti_h1_amp_walk/<date>_amp_torch/` |
| RL only (PPO: task rewards only, no discriminator) | `Template-Arti-H1-Direct-v0` | `skrl_ppo_cfg.yaml` | `logs/skrl/arti_h1_ppo_walk/<date>_ppo_torch/` |

Both use the same actor / asymmetric critic and hyperparameters; the policy is a residual on the nominal gait table in
both. Each terminal (tmux session) starts with:

```bash
source /opt/isaac/IsaacLab-main/.venv/bin/activate
cd /opt/isaac/projects/arti_h1
git pull origin server
export OMNI_KIT_ACCEPT_EULA=yes
```

Imitation + RL (AMP), terminal 1:

```bash
tmux new -s h1-amp
isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Direct-v0 physics=isaacsim_physx --num_envs 8192
# stage 2 (walk at the commanded speed), warm start from a converged stage-1 checkpoint:
isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Finetune-Direct-v0 physics=isaacsim_physx --num_envs 8192 \
    --checkpoint <stage-1 agent_N.pt>
# stage 3 (sim-to-real: the real robot's joint lag, ramped in over the first 200k steps), warm start from stage 2:
isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Sim2Real-Direct-v0 physics=isaacsim_physx --num_envs 8192 \
    --checkpoint <stage-2 agent_N.pt>
# continue a stage-3 run on the current stage-3 cfg (lag ramp already done):
isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Sim2Real-Direct-v0 physics=isaacsim_physx --num_envs 8192 \
    --checkpoint <stage-3 agent_N.pt> env.curriculum_step_offset=1200000
```

RL only (PPO), terminal 2:

```bash
tmux new -s h1-ppo
isaaclab train --rl_library skrl --task Template-Arti-H1-Direct-v0 physics=isaacsim_physx --num_envs 8192
# stage 3 (sim-to-real like the AMP stage 3, plus the stance CoM term), warm start from a PPO checkpoint:
isaaclab train --rl_library skrl --task Template-Arti-H1-Sim2Real-Direct-v0 physics=isaacsim_physx --num_envs 8192 \
    --checkpoint server/ppo_agent_1360000.pt
```

Velocity walking from scratch with the Microduck recipe (PPO only), terminal 3:

```bash
tmux new -s h1-vel
isaaclab train --rl_library skrl --task Template-Arti-H1-Velocity-Direct-v0 physics=isaacsim_physx --num_envs 8192
# resume: add --checkpoint <agent_N.pt> env.curriculum_step_offset=N
```

This task is a separate line of work, ported from [microduck_rl](https://github.com/pollen-robotics/microduck_rl)
(`Mjlab-Velocity-Flat-MicroDuck`, mjlab 1.3.0). See `arti_h1_velocity_env_cfg.py` for the details.
- It uses no reference motions, gait clock, ILC nominal table or AMP. q_des = clamp(default + scale * clamp(a, -1, 1)),
  with scale per joint (hip_pitch 0.6, knee 0.7, ankle_pitch 0.5, ankle_roll 0.3, hip_roll 0.25, hip_yaw 0.2 rad).
- The obs is 45-D (mjlab actor layout): IMU gyro, IMU projected gravity, joint pos - default, joint vel, previous
  action, command (vx, vy, wz, raw, last).
- The asymmetric critic adds the true base linear velocity and the per-foot height / air time / contact.
- Rewards are mjlab terms with Microduck weights, each weight * value * step_dt:
  - track_linear_velocity / track_angular_velocity 2 / 2, upright 2, pose 1 (variable_posture)
  - body_ang_vel -0.05, angular_momentum -0.02, dof_pos_limits -1
  - action_rate -0.1 -> -1.0 by iteration 1500
  - air_time 3 (window 0.1-0.5 s, counted only in single support), foot_clearance -2 / foot_swing_height -0.25
    (target 3 cm), foot_slip -1.0 (Microduck -0.1; at -0.1 ARTI-H1 skated instead of stepping)
  - feet_too_close -1 (in place of Microduck's self-collision term), foot_flat -10 (sin^2 sole tilt in contact)
  - Contact comes from the lowest sole corner (< 8 mm), not the sole centre. With the sole centre, standing on
    tilted soles counted as "in air" and the policy farmed air time.
  - Every `Reward/<penalty>` in TensorBoard must stay <= 0.
- Commands are fixed ranges vx +-0.15, vy +-0.08, wz +-0.4. 15 % of envs turn in place, and the standing (all-zero)
  fraction ramps 2 -> 25 % by iteration 2000.
- The ARTI-H1 sim-to-real models are kept: actuator delay / armature / Coulomb DR, real joint limits, hip_roll +-15
  deg, MTi-3 IMU model, joint zero errors (roll joints +-0.06 rad), stale bus reads, joint-vel noise. Pushes are
  +-0.3 / +-0.2 m/s every 3-6 s. The stage-3 extra lag (hip_roll 0-50 ms, ankle_pitch 0-25 ms) ramps in over
  iterations 1000 -> 3000.
- PPO uses the Microduck hyperparameters (`agents/skrl_velocity_ppo_cfg.yaml`): 512-256-128 ELU, 24 steps/env,
  5 epochs x 4 mini-batches, lr 1e-3 KL-adaptive 0.01, entropy 0.01. One iteration = 24 timesteps. The run is 20000
  iterations (480k timesteps), with a checkpoint every 250 iterations. All curricula end by iteration 3000, so stop
  early once tracking plateaus.
- Logs go to `logs/skrl/arti_h1_velocity/`. TensorBoard: `Reward/*`, `Episode_Reward/*`, `Metrics/air_time_mean`,
  `Metrics/peak_height_mean`, `Metrics/air_ratio_*`, `Tracking/*`, `Curriculum/*`.
- Export with `scripts/pt_to_onnx.py` (unchanged, 45 -> 12). humanoid_controller still needs a matching 45-D obs mode
  (no gait clock / nominal table) before the policy can run on the robot.
- Viewer: `--task Template-Arti-H1-Velocity-Play-Direct-v0` (final curriculum stages, 16 envs).
- Local check (2048 envs, 600 iterations): alternating steps appear by iteration ~300 (single support 0.98, air ratio
  0.5 / 0.5, air time 0.22 s, peak 4 cm, stance sole tilt 3 deg, episodes reach 20 s). Yaw tracking is still poor at
  that point (error 0.8 rad/s), so a long run is needed.
- Option: `env.balance_support_kp=40.0 env.balance_support_kd=2.0 "env.balance_support_fade_steps=[24000,72000]"` adds
  the old trunk-support crutch, faded out over iterations 1000 -> 3000. It is not needed for stepping, but it lowered
  slip (0.10 vs 0.21 m/s) and yaw error (0.47 vs 0.83 rad/s) at iteration 600.

Play over WebRTC livestream (Kit viewer; connect with the Isaac Sim WebRTC Streaming Client to the server IP).
`--checkpoint latest` loads the newest checkpoint of that task's log dir; pass a path (`.../checkpoints/agent_N.pt`)
for a specific one. Stop the viewer with Ctrl+C.

```bash
# imitation + RL (AMP)
isaaclab play --rl_library skrl --task Template-Arti-H1-Amp-Direct-v0 physics=isaacsim_physx \
    --num_envs 16 --checkpoint latest --viz kit --livestream 2
# RL only (PPO)
isaaclab play --rl_library skrl --task Template-Arti-H1-Direct-v0 physics=isaacsim_physx \
    --num_envs 16 --checkpoint latest --viz kit --livestream 2
```

- Physics: `physics=isaacsim_physx` (default, the original PhysX settings), `physics=newton_mjwarp`
  (Newton / MuJoCo-Warp), `physics=newton_kamino` (untuned stub). Add `--viz newton_gl` (or `--viz kit`) to watch.

- Output: `logs/skrl/arti_h1_amp_walk/<date>_amp_torch/` (PPO: `arti_h1_ppo_walk/<date>_ppo_torch/`; checkpoints, `params/env.yaml`, `params/agent.yaml`).
- Resume: add `--checkpoint <agent_N.pt> env.curriculum_step_offset=N` (or `--checkpoint latest`).
- Stage 2 (walk at the commanded speed): warm start a converged stage-1 checkpoint on the fine-tune task
  (`ArtiH1AmpFinetuneEnvCfg`: signed progress 0.3, linear tracking sigma 0.0005, curricula already finished, 600k steps):

  ```bash
  isaaclab train --rl_library skrl --task Template-Arti-H1-Amp-Finetune-Direct-v0 physics=isaacsim_physx \
      --num_envs 8192 --checkpoint server/agent_1365000.pt
  ```
- Stage 3 (`ArtiH1AmpSim2RealEnvCfg`): stage 2 plus a per-env extra joint lag uniform(0, 50 ms) on hip_roll and
  uniform(0, 25 ms) on ankle_pitch, resampled at every reset and ramped in over 200k steps (the real walk logs lag
  that much more than the sim model; TensorBoard `Sim2Real/extra_delay_ms_mean`), sagittal training pushes +-0.30 m/s
  (other tasks +-0.20), and hip_roll / ankle_roll homing zero errors up to +-0.06 rad (other joints +-0.02; the
  joint obs and targets are both in the encoder frame, like the robot). 600k steps.
- Stance CoM (stage 3, AMP and PPO, 11-03): `stance_com_reward_scale` -10 per m beyond `stance_com_margin` 30 mm of
  the lateral distance of the whole-body CoM from the stance sole centre toward the swing foot while the gait clock
  swings one foot, and `base_lateral_sway` -10 -> -5. Both PPO 1360k and AMP 600k tipped toward the swing side on the
  robot; the reference clips keep the pelvis 44 mm from the stance sole (motion_generator `foot_zmp_target_y` -0.03,
  sole half-width 45 mm), AMP 600k CoM 44 mm, PPO 1360k 55 mm. TensorBoard `Gait/stance_com_toward_swing_m`.
  PPO stage 3: `Template-Arti-H1-Sim2Real-Direct-v0` (`ArtiH1PpoSim2RealEnvCfg`, warm start from a PPO checkpoint).
- Progress summary: `PY=<isaac lab python> bash scripts/tuning/check.sh`.

Setup:
- Reference motions `assets/motions/arti_h1`: 1000 slow gait clips from motion_generator (1.5 s cycle) plus a standing clip.
- Command (0, 0, 0) means stand still and balance: the gait clock stops at a double-support rest phase (`gait_rest_phases`).
- Residual policy on an ILC nominal gait table, real joint limits, hip_roll +-15 deg.
- Sim-to-real: identified actuator model, IMU model, and observation / latency randomization.

## Evaluate, export, deploy

```bash
M=source/arti_h1/arti_h1/assets/motions
python scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --checkpoint latest --num_envs 4 \
    physics=newton_mjwarp --viz newton_gl                                # command sequence with pushes
python -u scripts/tuning/eval_gait.py --mode sequence --checkpoint <pt> --num_envs 32 --uniform_action_scale 0.0 \
    --motion_folder $M/arti_h1 --nominal_table $M/arti_h1_nominal_gait_command.npy --hip_roll_limit 0.2618
PY=<isaac lab python> bash scripts/tuning/record_video.sh <pt> <name>    # 40 s video with pushes (AMP checkpoint)
TASK=Template-Arti-H1-Direct-v0 ALGORITHM=PPO PY=<isaac lab python> bash scripts/tuning/record_video.sh <pt> <name>  # PPO
python scripts/pt_to_onnx.py --checkpoint <pt> --output policy.onnx
python -u scripts/tuning/verify_onnx.py <pt> policy.onnx
```

humanoid_controller `robot.yaml` policy settings for this setup: `gait_phase_rate: 0.66667`, `initial_gait_phase: 0.927`,
`gait_rest_phases: [0.427, 0.927]`, `command_max: [0.046, 0.0172, 0.0456]`.

## Regenerate reference motions

```bash
python scripts/tuning/gen_motions.py arti_h1 1000 10          # MOTION_GENERATOR=<motion_generator path>
python scripts/tuning/make_stand_clip.py $M/arti_h1/0.json $M/arti_h1/1000.json --phase 0.927 --weight 176
python -u scripts/tuning/ilc_nominal.py --iters 3 --motion_dir $M/arti_h1 --hip_roll_limit 0.2618 \
    --gait_phase_rate 0.66667 --out $M/arti_h1_nominal_gait_command.npy
python scripts/replay_demo.py --motion_set arti_h1 --index 292    # --video is not available in the 3.x port
```

## Real robot logs

`python scripts/tuning/analyze_robot_log.py <log.csv> --onnx <agent.onnx>` checks a humanoid_controller runtime log
against the training environment.
