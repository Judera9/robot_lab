# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import Any, Sequence

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from robot_lab.envs import CustomBasedRLEnv
from isaaclab.managers import RewardManager

from .model_based_rl_env_cfg import ModelBasedRLEnvCfg
from skrl.models import SystemDynamicsEnsemble
from skrl.resources.preprocessors import EmpiricalNormalization
from isaaclab.managers import SceneEntityCfg

class ModelBasedRLEnv(CustomBasedRLEnv):
    """Custom RL environment that extends ManagerBasedRLEnv with additional reward managers."""

    cfg: ModelBasedRLEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """Initialize the custom environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None.
        """
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self.reward_term_names = self.reward_manager.active_terms
        self.reward_term_names.append("uncertainty")
        self._init_additional_attributes()

        # termination flags
        self.termination_flags = None

    def init(self,
        num_imagination_envs: int,
        num_imagination_steps: int,
        rwm_state_normalizer: EmpiricalNormalization,
        rwm_action_normalizer: EmpiricalNormalization,
        world_model: SystemDynamicsEnsemble,
        uncertainty_penalty_weight: float = 0.0,
    ):
        # assigned in runner
        self.num_imagination_envs = num_imagination_envs
        self.num_imagination_steps = num_imagination_steps
        self.rwm_state_normalizer = rwm_state_normalizer
        self.rwm_action_normalizer = rwm_action_normalizer
        self.world_model = world_model
        self.uncertainty_penalty_weight = uncertainty_penalty_weight

    def load_managers(self):
        """Load the managers including the additional reward manager."""
        # call the parent class to load the base managers
        super().load_managers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        super().step(action)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        super()._reset_idx(env_ids)

    def prepare_imagination(self):
        """Prepare the environment for imagination."""

        self.world_model.reset()
        self.system_dynamics_model_ids = torch.randint(0, self.world_model.ensemble_size, (1, self.num_imagination_envs, 1), device=self.device)
        self._reset_imagination_reward_buffer()
        self._prepare_additional_imagination_attributes()
        self.imagination_step_counter = 0
        self.last_obs = None
    
    def sample_imagination_command(self):
        for name, term in self.command_manager._terms.items():
            setattr(self, name, term.sample_command(self.num_imagination_envs))

    def _reset_imagination_reward_buffer(self):
        self.imagination_reward_buffer = {
            term: torch.zeros(
                self.num_imagination_envs,
                self.num_imagination_steps,
                device=self.device
                ) for term in self.reward_term_names
            }
        self.imagination_reward_per_step = {
            term: torch.zeros(
                self.num_imagination_envs,
                device=self.device
                ) for term in self.reward_term_names
            }

    def get_imagination_reward_per_step(self):
        per_step_reward_imagination = {term: torch.mean(value) for term, value in self.imagination_reward_buffer.items()}
        return per_step_reward_imagination

    def imagination_step(self, rollout_action, state_history, action_history):
        """Imagination step."""

        action_history = torch.cat([action_history[:, 1:].clone(), self.rwm_action_normalizer(rollout_action).unsqueeze(1)], dim=1)
        imagination_states, aleatoric_uncertainty, self.epistemic_uncertainty, extensions, contacts, terminations = self.world_model.forward(state_history, action_history, self.system_dynamics_model_ids)
        imagination_states_denormalized = self.rwm_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        parsed_extensions = self._parse_extensions(extensions)
        parsed_contacts = self._parse_contacts(contacts)
        self.termination_flags = self._parse_terminations(terminations)
        self._compute_imagination_reward_terms(parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts)
        rewards, dones, extras = self._post_imagination_step()
        state_history = torch.cat([state_history[:, 1:].clone(), imagination_states.unsqueeze(1)], dim=1)
        return self.last_obs, rewards, dones, extras, state_history, action_history, self.epistemic_uncertainty
    
    def _post_imagination_step(self):
        for term, value in self.imagination_reward_buffer.items():
            if term == "uncertainty":
                value[:, self.imagination_step_counter] = self.uncertainty_penalty_weight * self.epistemic_uncertainty
            else:
                term_cfg = self.reward_manager.get_term_cfg(term)
                term_value = self.imagination_reward_per_step[term]
                value[:, self.imagination_step_counter] = term_cfg.weight * term_value
        
        rewards = torch.sum(
            torch.stack(
                [
                    value[:, self.imagination_step_counter]
                    for value in self.imagination_reward_buffer.values()
                    ]
                ),
            dim=0) * self.step_dt
        
        if self.imagination_step_counter == self.num_imagination_steps - 1:
            dones = torch.ones(self.num_imagination_envs, dtype=torch.int, device=self.device)
            time_outs = torch.ones(self.num_imagination_envs, dtype=torch.int, device=self.device)
        else:
            dones = self.termination_flags if self.termination_flags is not None else torch.zeros(self.num_imagination_envs, dtype=torch.int, device=self.device)
            time_outs = torch.zeros(self.num_imagination_envs, dtype=torch.int, device=self.device)
        
        infos = {"time_outs": time_outs}
        self.imagination_step_counter += 1
        return rewards, dones, infos


    # TODO: not implemented methods, currently just use it

    def _init_additional_attributes(self):
        asset_cfg = SceneEntityCfg("robot", self.cfg.joint_names, preserve_order=True)
        asset_cfg.resolve(self.scene)
        self.default_joint_pos = self.scene[asset_cfg.name].data.default_joint_pos[0, asset_cfg.joint_ids]
        self.default_joint_vel = self.scene[asset_cfg.name].data.default_joint_vel[0, asset_cfg.joint_ids]
        self.base_velocity = None

    def _prepare_additional_imagination_attributes(self):
        raise NotImplementedError

    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        raise NotImplementedError
    
    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:32]
        joint_vel = imagination_states_denormalized[:, 32:55]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
        }
        return parsed_imagination_states

    def _parse_extensions(self, extensions):
        raise NotImplementedError

    def _parse_contacts(self, contacts):
        raise NotImplementedError

    def _parse_terminations(self, terminations):
        raise NotImplementedError

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        raise NotImplementedError
