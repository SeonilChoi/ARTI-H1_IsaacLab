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
    "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
}


gym.register(
    id="Template-Arti-H1-Amp-Direct-v0",
    entry_point=f"{__name__}.arti_h1_env:ArtiH1AmpEnv",
    disable_env_checker=True,
    kwargs=_ARTI_H1_AMP_KWARGS,
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
