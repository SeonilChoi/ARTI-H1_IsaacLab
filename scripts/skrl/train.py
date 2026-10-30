# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Train ARTI-H1 with skrl through Isaac Lab's unified training backend, plus the AMP diagnostics.

Accepts the same arguments as ``isaaclab train --rl_library skrl`` (which trains the same task without the extra
AMP TensorBoard scalars), for example::

    python scripts/skrl/train.py --task Template-Arti-H1-Amp-Direct-v0 physics=isaacsim_physx --num_envs 4096
    python scripts/skrl/train.py --task Template-Arti-H1-Amp-Direct-v0 physics=newton_mjwarp --viz newton_gl

Runs are written to logs/skrl/<agent.experiment.directory>/<timestamp>_amp_torch as before.
"""

import sys

import warp as wp

wp.config.enable_backward = False

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from isaaclab_rl.entrypoints.backends import train_skrl  # noqa: E402

import arti_h1.tasks  # noqa: F401, E402


def _install_amp_diagnostics(runner) -> None:
    """Log AMP reward and discriminator separation metrics missing from skrl's default logger."""
    agent = runner.agent
    required_attributes = (
        "discriminator",
        "motion_dataset",
        "_amp_observation_preprocessor",
        "memory",
    )
    if any(getattr(agent, attribute, None) is None for attribute in required_attributes):
        return

    original_update = agent.update

    def update_with_diagnostics(*, timestep: int, timesteps: int) -> None:
        with torch.no_grad():
            agent_observations = agent.memory.get_tensor_by_name("amp_observations").reshape(
                -1, agent.amp_observation_space.shape[0]
            )
            motion_observations = agent.motion_dataset.get_tensor_by_name("observations").reshape(
                -1, agent.amp_observation_space.shape[0]
            )
            diagnostic_batch_size = min(4096, len(agent_observations), len(motion_observations))
            if diagnostic_batch_size:
                agent_observations = agent_observations[:diagnostic_batch_size]
                motion_observations = motion_observations[:diagnostic_batch_size]
                if agent_observations.shape[1] == 206:
                    feature_groups = {
                        "root_height": (0, 1),
                        "root_orientation": (1, 7),
                        "root_linear_velocity": (7, 10),
                        "root_angular_velocity": (10, 13),
                        "joint_rotation": (13, 85),
                        "joint_velocity": (85, 97),
                        "toe_position": (97, 103),
                    }
                    for name, (start, end) in feature_groups.items():
                        agent_features = torch.cat(
                            (agent_observations[:, start:end], agent_observations[:, 103 + start : 103 + end]),
                            dim=-1,
                        )
                        motion_features = torch.cat(
                            (motion_observations[:, start:end], motion_observations[:, 103 + start : 103 + end]),
                            dim=-1,
                        )
                        standardized_gap = torch.abs(
                            (agent_features.mean(dim=0) - motion_features.mean(dim=0))
                            / motion_features.std(dim=0).clamp_min(1.0e-4)
                        ).mean()
                        agent.track_data(f"AMP Feature Gap / {name}", standardized_gap.item())

                agent_observations = agent._amp_observation_preprocessor(agent_observations)
                motion_observations = agent._amp_observation_preprocessor(motion_observations)
                agent_logits, _ = agent.discriminator.act(
                    {"observations": agent_observations}, role="discriminator"
                )
                motion_logits, _ = agent.discriminator.act(
                    {"observations": motion_observations}, role="discriminator"
                )
                agent_probability = torch.sigmoid(agent_logits)
                motion_probability = torch.sigmoid(motion_logits)
                agent.track_data("Reward / AMP style reward (mean)", F.softplus(agent_logits).mean().item())
                agent.track_data("AMP / Agent real probability", agent_probability.mean().item())
                agent.track_data("AMP / Motion real probability", motion_probability.mean().item())
                agent.track_data("AMP / Agent classification accuracy", (agent_probability < 0.5).float().mean().item())
                agent.track_data(
                    "AMP / Motion classification accuracy",
                    (motion_probability >= 0.5).float().mean().item(),
                )

        original_update(timestep=timestep, timesteps=timesteps)

    agent.update = update_with_diagnostics


def _import_runner_with_diagnostics(import_skrl_runner):
    """Wrap the backend's runner import so every torch Runner gets the AMP diagnostics hook."""

    def import_runner(ml_framework: str) -> type:
        runner_cls = import_skrl_runner(ml_framework)
        if ml_framework != "torch":
            return runner_cls

        class Runner(runner_cls):
            def __init__(self, env, cfg):
                super().__init__(env, cfg)
                _install_amp_diagnostics(self)

        return Runner

    return import_runner


def main(argv: list[str] | None = None) -> None:
    train_skrl.import_skrl_runner = _import_runner_with_diagnostics(train_skrl.import_skrl_runner)
    train_skrl.run(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
