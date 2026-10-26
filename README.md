# arti_h1

AMP walking policy for the ARTI-H1 biped (Isaac Lab + skrl). The cfg defaults are the current training setup,
so training needs no extra arguments. Trained policies are exported to ONNX and deployed with
[humanoid_controller](https://github.com/SeonilChoi/humanoid_controller) (`config/robot/ARTI-H1/agent.onnx`).

## Requirements

Isaac Sim 5.1.0, Isaac Lab 0.54.3, skrl 2.0.0, PyTorch 2.7.0 (cu128); onnxruntime / onnx for export.

## Install

Linux (Ubuntu 22.04 / 24.04) and Windows both work. The robot USD files are stored with Git LFS, so install it first.

```bash
git lfs install
git clone <this repo> arti_h1 && cd arti_h1
python -m pip install -e source/arti_h1      # python with Isaac Lab installed
python -m pip install onnxruntime onnx        # only for ONNX export
```

## Train

```bash
python scripts/skrl/train.py --task Template-Arti-H1-Amp-Direct-v0 --algorithm AMP --headless
```

- Output: `logs/skrl/arti_h1_amp_walk/<date>_amp_torch/` (checkpoints, `params/env.yaml`, `params/agent.yaml`).
- Resume: add `--checkpoint <agent_N.pt> env.curriculum_step_offset=N`.
- Progress summary: `PY=<isaac lab python> bash scripts/tuning/check.sh`.

Setup:
- Reference motions `assets/motions/arti_h1`: 1000 slow gait clips from motion_generator (1.5 s cycle) plus a standing clip.
- Command (0, 0, 0) means stand still and balance: the gait clock stops at a double-support rest phase (`gait_rest_phases`).
- Residual policy on an ILC nominal gait table, real joint limits, hip_roll +-15 deg.
- Sim-to-real: identified actuator model, IMU model, and observation / latency randomization.

## Evaluate, export, deploy

```bash
M=source/arti_h1/arti_h1/assets/motions
python -u scripts/tuning/eval_gait.py --mode sequence --checkpoint <pt> --headless --num_envs 32 --uniform_action_scale 0.0 \
    --motion_folder $M/arti_h1 --nominal_table $M/arti_h1_nominal_gait_command.npy --hip_roll_limit 0.2618
PY=<isaac lab python> bash scripts/tuning/record_video.sh <pt> <name>    # 40 s video with pushes
python scripts/pt_to_onnx.py --checkpoint <pt> --output policy.onnx
python -u scripts/tuning/verify_onnx.py <pt> policy.onnx
```

humanoid_controller `robot.yaml` policy settings for this setup: `gait_phase_rate: 0.66667`, `initial_gait_phase: 0.927`,
`gait_rest_phases: [0.427, 0.927]`, `command_max: [0.046, 0.0172, 0.0456]`.

## Regenerate reference motions

```bash
python scripts/tuning/gen_motions.py arti_h1 1000 10          # MOTION_GENERATOR=<motion_generator path>
python scripts/tuning/make_stand_clip.py $M/arti_h1/0.json $M/arti_h1/1000.json --phase 0.927 --weight 176
python -u scripts/tuning/ilc_nominal.py --headless --iters 3 --motion_dir $M/arti_h1 --hip_roll_limit 0.2618 \
    --gait_phase_rate 0.66667 --out $M/arti_h1_nominal_gait_command.npy
python scripts/replay_demo.py --motion_set arti_h1 --index 292 --video replay.mp4 --headless --enable_cameras
```

## Real robot logs

`python scripts/tuning/analyze_robot_log.py <log.csv> --onnx <agent.onnx>` checks a humanoid_controller runtime log
against the training environment.
