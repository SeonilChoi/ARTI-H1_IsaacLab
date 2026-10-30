# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay a recorded motion clip on the Arti H1 robot and save it as a humanoid_controller action log.

The YAML has the same layout as humanoid_controller/config/*.yaml: joint_targets[i] (rad, Isaac joint order) is what
example_action sends to the motors at dt; observations[i] is the 116-dim policy observation of that step.

    python scripts/replay_demo.py --motion_set arti_h1 --index 292 --output logs/replay_demo/292.yaml
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


TASK_NAME = "Template-Arti-H1-Amp-Direct-v0"
NUM_ENVS = 1

parser = argparse.ArgumentParser(description="Replay a recorded Arti H1 demo motion.")
parser.add_argument("--motion_set", type=str, default="arti_h1", help="Folder under assets/motions.")
parser.add_argument("--index", type=int, default=0, help="Motion JSON index to replay, e.g. 0 for 0.json.")
parser.add_argument("--loop", action="store_true", default=False, help="Loop the selected motion.")
parser.add_argument("--output", type=str, default=None, help="YAML log path. Default: logs/replay_demo/<index>.yaml.")
parser.add_argument("--hip_roll_limit", type=float, default=0.2618, help="Hard +-limit (rad) on hip_roll, as in training.")
parser.add_argument("--no_realtime", action="store_true", help="Do not sleep to real time.")
parser.add_argument("--video", type=str, default=None, help="Also save an mp4 of the replay (play.py camera), e.g. logs/tuning/videos/x.mp4")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.video:
    # Isaac Lab 3.x: render_mode="rgb_array" frames were removed (env.render() returns None); recording goes through
    # VideoRecorderCfg sources resolved by isaaclab.app.launch_simulation, which this AppLauncher script does not use.
    raise SystemExit("--video is not available in the Isaac Lab 3.x port of replay_demo.py (see PORTING_NOTES.md)")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import arti_h1.tasks  # noqa: F401
from arti_h1.robots import ISAACLAB_ASSETS_DIR


def _resolve_motion_path(index: int) -> Path:
    motion_path = Path(ISAACLAB_ASSETS_DIR) / "motions" / args_cli.motion_set / f"{index}.json"
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    return motion_path


def _configure_replay_env(motion_folder_path: str):
    env_cfg = parse_env_cfg(
        TASK_NAME,
        num_envs=NUM_ENVS,
        use_fabric=True,
    )
    env_cfg.motion_folder_path = motion_folder_path
    env_cfg.reset_strategy = "default"
    env_cfg.early_termination = False
    env_cfg.use_isaac_imu_sensor = False
    env_cfg.episode_length_s = 24.0 * 60.0 * 60.0
    # Open-loop replay: the joint targets are the reference joint positions themselves, so no nominal gait offset
    # (the policy's residual base) and no training support torque.
    env_cfg.nominal_gait_scale = 0.0
    env_cfg.balance_support_kp = 0.0
    env_cfg.uniform_action_scale = 0.0
    env_cfg.hip_roll_limit = args_cli.hip_roll_limit
    if args_cli.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = (0.3, -2.8, 0.82)  # play.py VIDEO_CAMERA_EYE / LOOKAT
        env_cfg.viewer.lookat = (0.3, 0.0, 0.08)
        env_cfg.viewer.resolution = (1280, 720)
    return env_cfg


def _write_motion_state(replay_env, motion_time: float, env_ids: torch.Tensor):
    motion_ids = np.zeros(replay_env.num_envs, dtype=np.int64)
    motion_times = np.full(replay_env.num_envs, motion_time, dtype=np.float64)
    (
        root_position,
        root_orientation_quat_xyzw,
        root_linear_velocity,
        root_angular_velocity,
        joint_pos,
        joint_vel,
        _,
        _,
    ) = replay_env.motion_loader.sample(
        replay_env.num_envs,
        motion_ids=motion_ids,
        times=motion_times,
        return_full_state=True,
    )

    root_state = replay_env._default_root_state(env_ids)
    root_state[:, :3] = root_position + replay_env.scene.env_origins[env_ids]
    root_state[:, 3:7] = root_orientation_quat_xyzw  # Isaac Lab 3.x quaternions are (x, y, z, w)
    root_state[:, 7:10] = root_linear_velocity
    root_state[:, 10:13] = root_angular_velocity

    replay_env.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
    replay_env.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
    replay_env.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    replay_env.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
    return joint_pos


def _joint_pos_to_actions(replay_env, joint_pos: torch.Tensor) -> torch.Tensor:
    """Invert the env's action mapping: target = default_joint_pos + scale * action."""
    delta = joint_pos - replay_env.robot.data.default_joint_pos.torch
    action_scale = torch.where(delta >= 0.0, replay_env.action_positive_scale, replay_env.action_negative_scale)
    return delta / action_scale


def _to_list(values: torch.Tensor) -> list[float]:
    return [round(float(v), 6) for v in values.cpu().tolist()]


def _policy_observation_layout(replay_env) -> list[tuple[str, list[str]]]:
    """Name each policy observation index, following ArtiH1AmpEnv._get_observations / compute_policy_obs."""
    joint_names = list(replay_env.robot.data.joint_names)
    tan_norm = ["tan_x", "tan_y", "tan_z", "norm_x", "norm_y", "norm_z"]
    xyz = ["x", "y", "z"]
    return [
        # Root orientation with heading removed, as rotated x-axis (tan) and z-axis (norm).
        ("root_orientation_tan_norm", [f"root_orientation/{c}" for c in tan_norm]),
        # Root angular velocity rotated into the heading frame.
        ("root_angular_velocity", [f"root_angular_velocity/{c}" for c in xyz]),
        # Each joint angle as an axis-angle rotation encoded in tan-norm form.
        (
            "joint_positions_tan_norm",
            [f"joint_position/{name}/{c}" for name in joint_names for c in tan_norm],
        ),
        ("joint_velocities", [f"joint_velocity/{name}" for name in joint_names]),
        # Toe positions in the root frame.
        ("left_toe_position", [f"left_toe_position/{c}" for c in xyz]),
        ("right_toe_position", [f"right_toe_position/{c}" for c in xyz]),
        # Previous action clamped to [-1, 1].
        ("previous_actions", [f"previous_action/{name}" for name in joint_names]),
        # Gait clock: phase advances by gait_phase_rate per second, 0 = left single support start.
        ("gait_phase", ["gait_phase/sin", "gait_phase/cos"]),
        # Velocity command multiplied by command_scale.
        ("commands", ["command/linear_velocity_x", "command/linear_velocity_y", "command/angular_velocity_z"]),
    ]


def _save_replay_data(
    output_path: Path,
    replay_env,
    motion_path: Path,
    actions: list[list[float]],
    joint_targets: list[list[float]],
    observations: list[list[float]],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_steps = min(len(actions), len(joint_targets), len(observations))
    observation_layout = _policy_observation_layout(replay_env)
    observation_names = [name for _, names in observation_layout for name in names]
    observation_groups = []
    start = 0
    for group_name, names in observation_layout:
        observation_groups.append({"name": group_name, "start": start, "end": start + len(names)})
        start += len(names)
    if observations and len(observation_names) != len(observations[0]):
        print(
            f"[WARN] Observation layout has {len(observation_names)} names but observations have "
            f"{len(observations[0])} values. Update _policy_observation_layout to match the env."
        )
    data = {
        "motion_file": str(motion_path),
        "dt": float(replay_env.step_dt),
        "num_steps": num_steps,
        "joint_names": list(replay_env.robot.data.joint_names),
        "action_dim": len(actions[0]) if actions else 0,
        "observation_dim": len(observations[0]) if observations else 0,
        "action_names": list(replay_env.robot.data.joint_names),
        # joint_targets[i] is the position target (rad) the env sent to the joints for actions[i]:
        # clamp(default_joint_pos + scale * clamp(action, -1, 1), joint limits incl. hip_roll_limit).
        "joint_target_names": list(replay_env.robot.data.joint_names),
        "joint_target_unit": "rad",
        # observation_groups ranges are [start, end) indices into each observation vector.
        "observation_groups": observation_groups,
        "observation_names": observation_names,
        # observations[i] is the policy observation returned by env.step(actions[i]).
        "actions": actions[:num_steps],
        "joint_targets": joint_targets[:num_steps],
        "observations": observations[:num_steps],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None)
    print(f"[INFO] Saved {num_steps} action/joint target/observation steps to: {output_path}")


def main():
    """Replay the selected demo motion."""
    motion_path = _resolve_motion_path(args_cli.index)
    print(f"[INFO] Replaying motion: {motion_path}")
    output_path = Path(args_cli.output) if args_cli.output else Path.cwd() / "logs" / "replay_demo" / f"{args_cli.index}.yaml"

    env = None
    replay_env = None
    recorded_actions: list[list[float]] = []
    recorded_joint_targets: list[list[float]] = []
    recorded_observations: list[list[float]] = []
    with tempfile.TemporaryDirectory(prefix="arti_h1_replay_") as motion_dir:
        replay_motion_path = Path(motion_dir) / motion_path.name
        shutil.copy2(motion_path, replay_motion_path)

        try:
            env_cfg = _configure_replay_env(motion_dir)
            env = gym.make(TASK_NAME, cfg=env_cfg)
            replay_env = env.unwrapped
            replay_env.motion_loader._cycle_motion = args_cli.loop

            env.reset()
            env_ids = torch.arange(replay_env.num_envs, dtype=torch.long, device=replay_env.device)
            duration = max(float(replay_env.motion_loader._motion_durations[0]), replay_env.step_dt)
            motion_time = 0.0

            print(
                "[INFO] Duration: "
                f"{duration:.3f}s | dt: {replay_env.step_dt:.4f}s | loop: {args_cli.loop}"
            )

            while simulation_app.is_running():
                frame_start = time.time()
                if not args_cli.loop and motion_time > duration:
                    break

                with torch.inference_mode():
                    joint_pos = _write_motion_state(replay_env, motion_time, env_ids)
                    actions = _joint_pos_to_actions(replay_env, joint_pos)
                    if hasattr(replay_env.sim, "render"):
                        replay_env.sim.render()
                    # Step with the recovered actions so the joint position targets follow the motion.
                    obs, _, _, _, _ = env.step(actions)
                    recorded_actions.append(_to_list(actions[0]))
                    recorded_joint_targets.append(_to_list(replay_env.robot.actuators.target_command.position.torch[0]))
                    recorded_observations.append(_to_list(obs["policy"][0]))

                motion_time += replay_env.step_dt
                if args_cli.loop:
                    motion_time = motion_time % duration

                sleep_time = replay_env.step_dt - (time.time() - frame_start)
                if sleep_time > 0.0 and not args_cli.no_realtime:
                    time.sleep(sleep_time)
        finally:
            if replay_env is not None and recorded_actions and not args_cli.loop:
                _save_replay_data(
                    output_path,
                    replay_env,
                    motion_path,
                    recorded_actions,
                    recorded_joint_targets,
                    recorded_observations,
                )
            if env is not None:
                env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
