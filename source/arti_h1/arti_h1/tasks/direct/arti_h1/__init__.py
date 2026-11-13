# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


_ARTI_H1_AMP_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.arti_h1_env_cfg:ArtiH1AmpEnvCfg",
    # `isaaclab train|play --rl_library` picks this library when none is given
    "default_agent": "skrl",
    "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
}


gym.register(
    id="Template-Arti-H1-Amp-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    # Without --algorithm the skrl CLI loads skrl_cfg_entry_point: for this task that is the AMP recipe
    # (same as Isaac Lab's humanoid_amp), so `isaaclab train --rl_library skrl --task <this id>` trains AMP.
    kwargs={**_ARTI_H1_AMP_KWARGS, "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml"},
)

# Reinforcement learning only (PPO, no AMP style reward): same env cfg and rewards, agents/skrl_ppo_cfg.yaml.
gym.register(
    id="Template-Arti-H1-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_AMP_KWARGS,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# Stage 2: warm start from a stage-1 checkpoint to track the commanded speed (ArtiH1AmpFinetuneEnvCfg).
gym.register(
    id="Template-Arti-H1-Amp-Finetune-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_AMP_KWARGS,
        "env_cfg_entry_point": f"{__name__}.arti_h1_env_cfg:ArtiH1AmpFinetuneEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_finetune_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_finetune_cfg.yaml",
    },
)

# Stage 3: warm start from a stage-2 checkpoint, real-robot joint lag randomisation (ArtiH1AmpSim2RealEnvCfg).
gym.register(
    id="Template-Arti-H1-Amp-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_AMP_KWARGS,
        "env_cfg_entry_point": f"{__name__}.arti_h1_env_cfg:ArtiH1AmpSim2RealEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_finetune_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_finetune_cfg.yaml",
    },
)

# Stage 3 for RL only (PPO): warm start from a PPO checkpoint; AMP stage-3 cfg + stance CoM term (ArtiH1PpoSim2RealEnvCfg).
gym.register(
    id="Template-Arti-H1-Sim2Real-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_AMP_KWARGS,
        "env_cfg_entry_point": f"{__name__}.arti_h1_env_cfg:ArtiH1PpoSim2RealEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_finetune_cfg.yaml",
    },
)

# Velocity walking from scratch with the Microduck recipe (pollen-robotics/microduck_rl): plain PPO, no reference
# motions / gait clock / nominal table / AMP, 45-D obs (ArtiH1VelocityEnvCfg, agents/skrl_velocity_ppo_cfg.yaml).
_ARTI_H1_VELOCITY_KWARGS = {
    "env_cfg_entry_point": f"{__name__}.arti_h1_velocity_env_cfg:ArtiH1VelocityEnvCfg",
    "default_agent": "skrl",
    "skrl_cfg_entry_point": f"{agents.__name__}:skrl_velocity_ppo_cfg.yaml",
}

gym.register(
    id="Template-Arti-H1-Velocity-Direct-v0",
    entry_point=f"{__name__}.arti_h1_velocity_env:ArtiH1VelocityEnv",
    disable_env_checker=True,
    kwargs=_ARTI_H1_VELOCITY_KWARGS,
)

gym.register(
    id="Template-Arti-H1-Velocity-Play-Direct-v0",
    entry_point=f"{__name__}.arti_h1_velocity_env:ArtiH1VelocityEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_VELOCITY_KWARGS,
        "env_cfg_entry_point": f"{__name__}.arti_h1_velocity_env_cfg:ArtiH1VelocityEnvCfg_PLAY",
    },
)

# Knee-bent line (crouched r21 stance, patched USD), separate from the straight-leg q=0 line above.
gym.register(
    id="Template-Arti-H1-Velocity-KB-Direct-v0",
    entry_point=f"{__name__}.arti_h1_velocity_env:ArtiH1VelocityEnv",
    disable_env_checker=True,
    kwargs={**_ARTI_H1_VELOCITY_KWARGS, "env_cfg_entry_point": f"{__name__}.arti_h1_velocity_env_cfg:ArtiH1VelocityKBEnvCfg"},
)

gym.register(
    id="Template-Arti-H1-Velocity-KB-Play-Direct-v0",
    entry_point=f"{__name__}.arti_h1_velocity_env:ArtiH1VelocityEnv",
    disable_env_checker=True,
    kwargs={**_ARTI_H1_VELOCITY_KWARGS, "env_cfg_entry_point": f"{__name__}.arti_h1_velocity_env_cfg:ArtiH1VelocityKBEnvCfg_PLAY"},
)
