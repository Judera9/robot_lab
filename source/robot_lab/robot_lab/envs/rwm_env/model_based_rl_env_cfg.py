# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from robot_lab.envs.custom_based_rl_env_cfg import CustomBasedRLEnvCfg
from isaaclab.utils import configclass
from skrl.resources.preprocessors import EmpiricalNormalization
from skrl.models import SystemDynamicsEnsemble


@configclass
class ModelBasedRLEnvCfg(CustomBasedRLEnvCfg):
    """Configuration for the custom RL environment with additional reward managers."""
    
    num_imagination_envs: int = MISSING
    """Number of imagination environments."""

    num_imagination_steps: int = MISSING
    """Number of imagination steps."""

    rwm_state_normalizer: EmpiricalNormalization | None = None
    """State normalizer for the reward-weighted model."""

    rwm_action_normalizer: EmpiricalNormalization | None = None
    """Action normalizer for the reward-weighted model."""

    world_model: SystemDynamicsEnsemble | None = None
    """System dynamics (world model) for the reward-weighted model."""