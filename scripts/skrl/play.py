# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Play an ARTI-H1 skrl checkpoint through the fixed command sequence with scheduled pushes.

Isaac Lab 3.x port of the 2.x script, following ``isaaclab_rl.entrypoints.backends.play_skrl``. Arguments are those of
``isaaclab play --rl_library skrl`` plus --no_push / --push_scale / --video_width / --video_height, for example::

    python scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --checkpoint latest --num_envs 4 \
        physics=newton_mjwarp --viz newton_gl
    python scripts/skrl/play.py --task Template-Arti-H1-Amp-Direct-v0 --checkpoint <run>/checkpoints/agent_N.pt \
        --num_envs 4 --video --video_length 2400

Like the 2.x script it plays the training env config (no play_mode() caps), with balance support and the random
push event off, and a 40 s episode.
"""

import argparse
import contextlib
import logging
import os
import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

from isaaclab_rl.entrypoints.backends import cli_args_skrl  # noqa: E402
from isaaclab_rl.entrypoints.common import (  # noqa: E402
    CHECKPOINT_SELECTORS,
    add_common_play_args,
    apply_env_overrides,
    apply_video_recording,
    close_env,
    create_isaaclab_env,
    enable_cameras_for_video,
    normalize_task_name,
    pre_launch_video_config,
    resolve_checkpoint_selector,
    resolve_seed,
    run_playback,
    set_hydra_args,
)
from isaaclab_rl.skrl import (  # noqa: E402
    SkrlVecEnvWrapper,
    check_skrl_version,
    import_skrl_runner,
    resolve_skrl_agent_cfg_entry_point,
    resolve_skrl_algorithm,
)

from isaaclab_tasks.utils import get_checkpoint_path, resolve_task_config, setup_preset_cli  # noqa: E402

import arti_h1.tasks  # noqa: F401, E402

logger = logging.getLogger(__name__)

PLAY_COMMAND_DURATION_S = 5.0
VIDEO_CAMERA_EYE = (0.3, -2.8, 0.82)
VIDEO_CAMERA_LOOKAT = (0.3, 0.0, 0.08)
# Commands inside the training range (x .046 y .017 yaw .046) plus standing (zero command) segments.
PLAY_COMMAND_SEQUENCE = (
    ("forward", (0.040, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("side_left", (0.0, 0.015, 0.0)),
    ("backward", (-0.035, 0.0, 0.0)),
    ("stand", (0.0, 0.0, 0.0)),
    ("diagonal_right", (0.035, -0.012, 0.0)),
    ("turn_left", (0.0, 0.0, 0.040)),
    ("stand", (0.0, 0.0, 0.0)),
)

# Pushes to watch the recovery: (episode time s, delta root velocity vx, vy m/s, yaw rate rad/s), one in the middle of
# every 5 s command segment. Training pushes are only +-0.04 m/s / +-0.15 rad/s, so these are 2-4x stronger.
PLAY_PUSH_SCHEDULE = (
    (2.5, 0.10, 0.0, 0.0),    # forward: from behind
    (7.5, 0.0, -0.10, 0.0),   # side_left: from the left
    (12.5, 0.0, 0.10, 0.0),   # backward: from the right
    (17.5, -0.10, 0.0, 0.0),  # diagonal_right: from the front
    (22.5, 0.0, 0.0, 0.5),    # turn_left: yaw kick
    (27.5, 0.0, 0.15, 0.0),   # forward: stronger, from the right
)


def _apply_play_pushes(env, scale: float):
    """Add the scheduled velocity kick to every env's root when env 0's episode time crosses a push time."""
    t = float(env.episode_length_buf[0].item()) * env.step_dt
    last = getattr(env, "_last_play_push_t", -1.0)
    env._last_play_push_t = t
    for push_t, vx, vy, wz in PLAY_PUSH_SCHEDULE:
        if last < push_t <= t:
            vel = env.robot.data.root_vel_w.torch.clone()
            vel[:, 0] += scale * vx
            vel[:, 1] += scale * vy
            vel[:, 5] += scale * wz
            env.robot.write_root_velocity_to_sim_index(root_velocity=vel)
            print(f"[INFO] push at t={t:.2f} s: dvx {scale * vx:+.2f} dvy {scale * vy:+.2f} m/s, dwz {scale * wz:+.2f} rad/s", flush=True)


def _overwrite_observation_commands(obs, command_obs: torch.Tensor):
    """Keep policy observations consistent after overriding environment commands."""
    if isinstance(obs, torch.Tensor):
        obs = obs.clone()
        obs[..., -command_obs.shape[-1] :] = command_obs
        return obs
    if isinstance(obs, dict):
        return {key: _overwrite_observation_commands(value, command_obs) for key, value in obs.items()}
    return obs


def _apply_play_command_sequence(env, obs):
    required_attrs = ("commands", "command_scale", "episode_length_buf", "max_episode_length")
    if not all(hasattr(env, attr) for attr in required_attrs):
        return obs

    command_values = torch.tensor(
        [command for _, command in PLAY_COMMAND_SEQUENCE],
        dtype=torch.float32,
        device=env.device,
    )
    episode_progress = env.episode_length_buf.float() / max(float(env.max_episode_length), 1.0)
    command_ids = torch.clamp((episode_progress * len(PLAY_COMMAND_SEQUENCE)).long(), max=len(PLAY_COMMAND_SEQUENCE) - 1)
    env.commands[:] = command_values.index_select(0, command_ids)
    current_command_id = int(command_ids[0].item())
    if getattr(env, "_last_play_command_id", None) != current_command_id:
        command_name, command = PLAY_COMMAND_SEQUENCE[current_command_id]
        print(
            f"[INFO] Current command: {command_name} "
            f"(x={command[0]:.4f}, y={command[1]:.4f}, yaw={command[2]:.4f})",
            flush=True,
        )
        env._last_play_command_id = current_command_id
    return _overwrite_observation_commands(obs, env.commands * env.command_scale)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play an ARTI-H1 skrl checkpoint with the play command sequence.")
    add_common_play_args(parser, agent_default=None, agent_help="Agent configuration entry point.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path, or latest/best.")
    parser.add_argument("--video_width", type=int, default=1920, help="Recorded video width in pixels.")
    parser.add_argument("--video_height", type=int, default=1080, help="Recorded video height in pixels.")
    parser.add_argument("--no_push", action="store_true", default=False, help="Disable the scheduled play pushes.")
    parser.add_argument("--push_scale", type=float, default=1.0, help="Scale of the scheduled play pushes.")
    cli_args_skrl.add_skrl_args(parser)
    add_launcher_args(parser)
    args_cli, hydra_args = setup_preset_cli(parser, argv)
    enable_cameras_for_video(args_cli)
    set_hydra_args(hydra_args)
    return args_cli


def _resolve_checkpoint(args_cli, log_root_path: str, algorithm: str, metadata: dict[str, str]) -> str:
    if args_cli.checkpoint in CHECKPOINT_SELECTORS:
        return resolve_checkpoint_selector(
            log_root_path,
            args_cli.checkpoint,
            library="skrl",
            task=normalize_task_name(args_cli.task),
            checkpoint_pattern=r".*",
            other_dirs=["checkpoints"],
            latest_checkpoint_pattern=r"agent_.*",
            metadata=metadata,
        )
    if args_cli.checkpoint:
        return os.path.abspath(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"])


def main(argv: list[str] | None = None) -> None:
    args_cli = _parse_args(sys.argv[1:] if argv is None else argv)
    check_skrl_version()
    agent_cfg_entry_point = resolve_skrl_agent_cfg_entry_point(args_cli.agent, args_cli.algorithm)
    # the 2.x script played the training config (no play_mode() overrides); keep that
    env_cfg, agent_cfg = resolve_task_config(args_cli.task, agent_cfg_entry_point, play_mode=False)
    algorithm = resolve_skrl_algorithm(agent_cfg, args_cli.algorithm)

    env_cfg.episode_length_s = PLAY_COMMAND_DURATION_S * len(PLAY_COMMAND_SEQUENCE)
    # Training-only balance support torque must never act during play.
    if hasattr(env_cfg, "balance_support_kp"):
        env_cfg.balance_support_kp = 0.0
    # The random training push event is off; play uses the fixed PLAY_PUSH_SCHEDULE instead (--no_push to disable).
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.video:
        # ViewerCfg is deprecated in 3.x but DirectRLEnv still forwards it to the visualizers (incl. env-origin tracking)
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = VIDEO_CAMERA_EYE
        env_cfg.viewer.lookat = VIDEO_CAMERA_LOOKAT
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)
    pre_launch_video_config(env_cfg, args_cli)

    with launch_simulation(env_cfg, args_cli), contextlib.ExitStack() as cleanup:
        runner_cls = import_skrl_runner(args_cli.ml_framework)
        apply_env_overrides(args_cli, env_cfg)
        args_cli.seed = resolve_seed(args_cli.seed)
        if args_cli.seed is not None:
            agent_cfg["seed"] = args_cli.seed
        env_cfg.seed = agent_cfg["seed"]

        experiment_cfg = agent_cfg["agent"]["experiment"]
        log_root_path = os.path.abspath(os.path.join("logs", "skrl", experiment_cfg["directory"]))
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        metadata = {"agent": agent_cfg_entry_point, "algorithm": algorithm, "ml_framework": args_cli.ml_framework}
        resume_path = _resolve_checkpoint(args_cli, log_root_path, algorithm, metadata)
        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env_cfg.log_dir = log_dir
        apply_video_recording(env_cfg, log_dir, args_cli, subdir="play")

        env = create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        cleanup.callback(lambda: close_env(env))
        command_env = env.unwrapped
        dt = command_env.step_dt
        print("[INFO] Play command sequence:")
        for index, (name, command) in enumerate(PLAY_COMMAND_SEQUENCE, start=1):
            print(f"  {index}. {name}: {command}")

        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
        agent_cfg["trainer"]["close_environment_at_exit"] = False
        experiment_cfg["write_interval"] = 0  # don't log to TensorBoard
        experiment_cfg["checkpoint_interval"] = 0  # don't generate checkpoints
        runner = runner_cls(env, agent_cfg)
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)
        runner.agent.enable_training_mode(False, apply_to_models=True)

        obs, _ = env.reset()
        obs = _apply_play_command_sequence(command_env, obs)
        states = env.state()

        def step() -> None:
            nonlocal obs, states
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
            obs = _apply_play_command_sequence(command_env, obs)
            if not args_cli.no_push:
                _apply_play_pushes(command_env, args_cli.push_scale)
            states = env.state()

        run_playback(step, dt=dt, args_cli=args_cli, env_cfg=env_cfg)


if __name__ == "__main__":
    main()
