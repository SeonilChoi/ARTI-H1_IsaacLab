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

gym.register(
    id="Template-Arti-H1-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs={
        **_ARTI_H1_AMP_KWARGS,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
