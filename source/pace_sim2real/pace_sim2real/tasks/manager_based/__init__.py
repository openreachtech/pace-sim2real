# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import gymnasium as gym  # noqa: F401

gym.register(
    id="Isaac-Pace-Anymal-D-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.anymal_pace_env_cfg:AnymalDPaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-Anaguma-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.anaguma_pace_env_cfg:AnagumaPaceEnvCfg"
    },
)

gym.register(
    id="Isaac-Pace-Anaguma-Mujoco-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace.anaguma_pace_env_cfg:AnagumaMujocoPaceEnvCfg"
    },
)
