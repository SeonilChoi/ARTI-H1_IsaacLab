# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_width", type=int, default=1920, help="Recorded video width in pixels.")
parser.add_argument("--video_height", type=int, default=1080, help="Recorded video height in pixels.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help=(
        "Name of the RL algorithm to use (e.g. AMP, DDPG, IPPO, MAPPO, PPO, SAC, TD3, etc.) "
        "when several algorithms exist for the same task. For a more specific selection, use the argument --agent."
    ),
)
parser.add_argument("--no_push", action="store_true", default=False, help="Disable the scheduled play pushes.")
parser.add_argument("--push_scale", type=float, default=1.0, help="Scale of the scheduled play pushes.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import arti_h1.tasks  # noqa: F401

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
            vel = env.robot.data.root_vel_w.clone()
            vel[:, 0] += scale * vx
            vel[:, 1] += scale * vy
            vel[:, 5] += scale * wz
            env.robot.write_root_velocity_to_sim(vel)
            print(f"[INFO] push at t={t:.2f} s: dvx {scale * vx:+.2f} dvy {scale * vy:+.2f} m/s, dwz {scale * wz:+.2f} rad/s", flush=True)


class _BlankFrameGuard(gym.Wrapper):
    """The headless RTX renderer sometimes hands back a blank (uniformly white) frame when the annotator has not
    received the new image yet - ~55% of the frames in run10_50000.mp4. Re-render until the frame has content,
    and fall back to the last good frame instead of writing a white one."""

    def __init__(self, env, retries: int = 4):
        super().__init__(env)
        self._retries = retries
        self._last_good = None
        self.blank_frames = 0

    def render(self):
        frame = self.env.render()
        for _ in range(self._retries):
            if frame is not None and frame.size and frame.std() > 1.0:
                self._last_good = frame
                return frame
            self.env.unwrapped.sim.render()
            frame = self.env.render()
        self.blank_frames += 1
        return self._last_good if self._last_good is not None else frame


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

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = PLAY_COMMAND_DURATION_S * len(PLAY_COMMAND_SEQUENCE)
    # Training-only balance support torque must never act during play.
    if hasattr(env_cfg, "balance_support_kp"):
        env_cfg.balance_support_kp = 0.0
    # The random training push event is off; play uses the fixed PLAY_PUSH_SCHEDULE instead (--no_push to disable).
    if getattr(env_cfg, "events", None) is not None and hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if args_cli.video:
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.eye = VIDEO_CAMERA_EYE
        env_cfg.viewer.lookat = VIDEO_CAMERA_LOOKAT
        env_cfg.viewer.resolution = (args_cli.video_width, args_cli.video_height)

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)
    command_env = env.unwrapped
    print("[INFO] Play command sequence:")
    for index, (name, command) in enumerate(PLAY_COMMAND_SEQUENCE, start=1):
        print(f"  {index}. {name}: {command}")

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during play.")
        print(f"        video_folder: {video_kwargs['video_folder']}")
        print(f"        video_length: {video_kwargs['video_length']}")
        print(f"        resolution: {args_cli.video_width}x{args_cli.video_height}")
        env = gym.wrappers.RecordVideo(_BlankFrameGuard(env), **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.enable_training_mode(False, apply_to_models=True)

    # reset environment
    obs, _ = env.reset()
    obs = _apply_play_command_sequence(command_env, obs)
    states = env.state()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, states, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)
            obs = _apply_play_command_sequence(command_env, obs)
            if not args_cli.no_push:
                _apply_play_pushes(command_env, args_cli.push_scale)
            states = env.state()
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
