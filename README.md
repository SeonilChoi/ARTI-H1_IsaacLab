# arti_h1: AMP 보행 정책 학습 (Isaac Lab)

ARTI-H1 이족 로봇의 AMP(Adversarial Motion Priors) 보행 정책을 학습한다. 기본 설정(cfg 기본값)이 곧 현재 학습(run35)
설정이라, 추가 인자 없이 학습 명령 한 줄로 같은 학습이 재현된다. 학습한 정책은 ONNX로 내보내
[humanoid_controller](https://github.com/SeonilChoi/humanoid_controller) 의 `config/robot/ARTI-H1/agent.onnx` 로 배포한다.

## 환경

| 항목 | 버전 |
|---|---|
| Isaac Sim | 5.1.0 |
| Isaac Lab | 0.54.3 |
| skrl | 2.0.0 |
| PyTorch | 2.7.0 (cu128) |
| onnxruntime | 1.31.0 (ONNX 내보내기 / 검증용) |

## 설치

```bash
# Isaac Lab 설치: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
git clone <this repo> arti_h1
cd arti_h1
python -m pip install -e source/arti_h1          # Isaac Lab 이 설치된 python
python -m pip install onnxruntime onnx            # ONNX 내보내기에만 필요
```

## 학습

```bash
python scripts/skrl/train.py --task Template-Arti-H1-Amp-Direct-v0 --algorithm AMP --headless
```

- 결과: `logs/skrl/arti_h1_amp_walk/<날짜>_amp_torch/` (체크포인트 `checkpoints/agent_<step>.pt`, 설정 `params/env.yaml`, `params/agent.yaml`)
- 이어서 학습: `--checkpoint <agent_xxx.pt> env.curriculum_step_offset=<step>` (보조 토크 등 커리큘럼이 이어진 step 부터 진행)
- 진행 확인: `PY=<isaac lab python> bash scripts/tuning/check.sh` (TensorBoard 요약)

### 현재 학습 설정 (run35)

- 기준 모션 `assets/motions/arti_h1_slow`: motion_generator 로 만든 느리고 조심스러운 보행 1000개
  (걸음 주기 1.5 s, 한 걸음 0.75 s, 이중지지 0.25 s, 발 높이 30 mm) + 정지 클립 `1000.json` (샘플링 15 %).
- 명령 (0, 0, 0) = 제자리 정지·균형: 게이트 시계가 다음 rest phase(0.427 / 0.927, 이중지지 중앙)까지만 가서 멈추고
  (`gait_rest_phases`), 정지 보상(두 발 접지, 관절 속도)으로 학습한다. 같은 규칙이 ONNX 와 humanoid_controller 에 있다.
- 잔차 정책: q_des = 기본 자세 + 정책 action + ILC 공칭 보행 테이블(`arti_h1_slow_nominal_gait_command.npy`).
- 실물 관절 범위(`real_joint_limits`, humanoid_controller motor.yaml), hip_roll ±15°.
- sim2real: 실측 액추에이터 모델(지연·관성·쿨롱 마찰, 실물 게인), 관절 속도 관측 잡음, Unitree 피드백 누락,
  관절 영점·IMU 장착 오차, MTi-3 IMU 잡음·바이어스·지연.

## 평가 / 영상 / 배포

```bash
M=source/arti_h1/arti_h1/assets/motions
# 명령 시퀀스 평가 (전진/정지/옆/후진/대각/회전, 32 env)
python -u scripts/tuning/eval_gait.py --mode sequence --checkpoint <pt> --headless --num_envs 32 --uniform_action_scale 0.0 \
    --motion_folder $M/arti_h1_slow --nominal_table $M/arti_h1_slow_nominal_gait_command.npy --hip_roll_limit 0.2618
# 영상 (40 s, 중간 push 포함): logs/tuning/videos/<name>.mp4
PY=<isaac lab python> bash scripts/tuning/record_video.sh <pt> <name>
# ONNX 내보내기 + 검증
python scripts/pt_to_onnx.py --checkpoint <pt> --output policy.onnx
python -u scripts/tuning/verify_onnx.py <pt> policy.onnx
python -u scripts/tuning/check_onnx_in_env.py --onnx policy.onnx --headless \
    --motion_folder $M/arti_h1_slow --nominal_table $M/arti_h1_slow_nominal_gait_command.npy --hip_roll_limit 0.2618
```

배포 시 humanoid_controller `robot.yaml` policy: `gait_phase_rate: 0.66667`, `initial_gait_phase: 0.927`,
`gait_rest_phases: [0.427, 0.927]`, `command_max` = 학습 명령 범위 (0.046, 0.0172, 0.0456).

## 기준 모션 다시 만들기

```bash
# motion_generator (env_placo) 의 config/arti_h1_slow 로 1000개 생성 (MOTION_GENERATOR=<motion_generator 경로>)
python scripts/tuning/gen_motions.py arti_h1_slow 1000 10
python scripts/tuning/make_stand_clip.py $M/arti_h1_slow/0.json $M/arti_h1_slow/1000.json --phase 0.927 --weight 176
python scripts/tuning/reference_stats.py $M/arti_h1_slow
python -u scripts/tuning/ilc_nominal.py --headless --iters 3 --motion_dir $M/arti_h1_slow --hip_roll_limit 0.2618 \
    --gait_phase_rate 0.66667 --out $M/arti_h1_slow_nominal_gait_command.npy
# 모션 재생 확인 (humanoid_controller example_action 용 yaml + 영상)
python scripts/replay_demo.py --motion_set arti_h1_slow --index 292 --output logs/replay_demo/slow_292.yaml \
    --no_realtime --video logs/tuning/videos/replay_slow_292.mp4 --headless --enable_cameras
```

## 실물 로그 분석

humanoid_controller 런타임 로그(`robot.yaml log_file`)를 학습 환경과 비교한다:
`python scripts/tuning/analyze_robot_log.py <log.csv> --onnx <agent.onnx>`
(observation 계산, IMU, 관절, ONNX 재현 일관성 확인).
