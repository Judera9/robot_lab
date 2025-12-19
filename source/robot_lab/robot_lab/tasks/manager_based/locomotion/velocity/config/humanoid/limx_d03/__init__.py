import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-D03-Flat-LimX-v0",
    entry_point="robot_lab.envs:CustomBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:LimxHumanoidEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# gym.register(
#     id="Isaac-D03-AMP-Flat-LimX-v0",
#     entry_point="robot_lab.envs:CustomBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.amp_loco_flat_env_cfg.LimxHumanoidAMPEnvCfg",
#         "skrl_cfg_entry_point": f"{agents.__name__}.limx_rl_amp_ppo_cfg:LimxRLAMPPPORunnerCfg",
#     },
# )

# gym.register(
#     id="Isaac-D03-AMP-Flat-LimX-v0-play",
#     entry_point="robot_lab.envs:CustomBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.amp_loco_flat_env_cfg.LimxHumanoidAMPEnvCfg_PLAY",
#         "skrl_cfg_entry_point": f"{agents.__name__}:LimxRLAMPPPORunnerCfg",
#     },
# )
