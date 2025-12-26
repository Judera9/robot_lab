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
from .model_based_rl_env import ModelBasedRLEnv
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.events import (
    reset_joints_to_specified, 
    reset_root_velocity_to_specified,
    reset_root_state_to_specified
)

USE_MROS = False
if USE_MROS:
    import skrl.utils.log_utils.mros_utils as mros_utils
    mros_utils = mros_utils.MrosUtils("model_based_vis_env")

class ModelBasedVisEnv(ModelBasedRLEnv):
    """Custom RL environment that extends ManagerBasedRLEnv with additional reward managers."""

    cfg: ModelBasedRLEnvCfg

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        """Initialize the custom environment.

        Args:
            cfg: The configuration for the environment.
            render_mode: The render mode for the environment. Defaults to None.
        """
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.env_ids_real = torch.arange(0, self.num_envs, 2, device=self.device)
        self.env_ids_imagination = torch.arange(1, self.num_envs, 2, device=self.device)
        self.noise_scale = 0.0

    def set_noise_scale(self, noise_scale):
        self.noise_scale = noise_scale

    def init_imagination_history(self, history_horizon):
        self.imagination_state_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_state"][0], device=self.device)
        self.imagination_action_history = torch.zeros(self.num_envs // 2, history_horizon, self.observation_manager.group_obs_dim["system_action"][0], device=self.device)
        # self.world_model.reset()
        
    def _sync_imagination_history(self, env_ids_real):
        self.imagination_state_history[env_ids_real // 2] = 0.0
        self.imagination_action_history[env_ids_real // 2] = 0.0
        self.imagination_state_history[env_ids_real // 2, -1] = self.rwm_state_normalizer(self.observation_manager.compute()["system_state"])[env_ids_real]
        self.world_model.reset()

    def load_managers(self):
        """Load the managers including the additional reward manager."""
        # call the parent class to load the base managers
        super().load_managers()

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # process actions
        self.action_manager.process_action(action.to(self.device))

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for i in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # write imagination states
            if i == self.cfg.decimation - 1:
                self._update_imagination_envs(action)
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            uniques, counts = torch.cat([reset_env_ids, self.env_ids_real]).unique(return_counts=True)
            env_ids_real = uniques[counts > 1]
            env_ids_imagination = env_ids_real + 1
            env_ids = torch.vstack([env_ids_real, env_ids_imagination]).T.flatten()
            if len(env_ids) > 0:
                self._reset_idx(env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        if len(reset_env_ids) > 0 and len(env_ids_real) > 0:
            self._sync_imagination_history(env_ids_real)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _update_imagination_envs(self, action):
        self.num_imagination_envs = len(self.env_ids_imagination)
        rollout_action = action[self.env_ids_imagination]
        self.imagination_action_history = torch.cat([
            self.imagination_action_history[:, 1:].clone(), 
            self.rwm_action_normalizer(rollout_action).unsqueeze(1) + self.noise_scale * torch.randn_like(rollout_action)
        ], dim=1)
        if self.world_model.architecture_config["type"] in ["rnn", "rssm"]:
            self.imagination_state_history = self.imagination_state_history[:, -1].unsqueeze(1)
            self.imagination_action_history = self.imagination_action_history[:, -1].unsqueeze(1)
        inputs = {
            "x_state_batch": self.imagination_state_history,
            "x_action_batch": self.imagination_action_history
        }

        true_system_state = self.obs_buf['system_state'][self.env_ids_real]
        true_system_action = self.obs_buf['system_action'][self.env_ids_real]
        true_system_contact = self.obs_buf['system_contact'][self.env_ids_real]
        true_system_termination = self.obs_buf['system_termination'][self.env_ids_real]

        imagination_states, *_ = self.world_model.compute(inputs)
        imagination_states = imagination_states + self.noise_scale * torch.randn_like(imagination_states)
        imagination_states_denormalized = self.rwm_state_normalizer.inverse(imagination_states)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        self._reset_imagination_sim(parsed_imagination_states)
        self.imagination_state_history = torch.cat([self.imagination_state_history[:, 1:].clone(), imagination_states.unsqueeze(1)], dim=1)

    def _reset_imagination_sim(self, parsed_imagination_states):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_pos += self.default_joint_pos
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_vel += self.default_joint_vel

        root_quat_w = self.scene["robot"].data.root_quat_w[self.env_ids_imagination]

        base_lin_vel_w = math_utils.quat_apply(root_quat_w, base_lin_vel)
        base_ang_vel_w = math_utils.quat_apply(root_quat_w, base_ang_vel)

        velocities = torch.cat([base_lin_vel_w, base_ang_vel_w], dim=1)

        reset_joints_to_specified(
            self, self.env_ids_imagination, joint_pos, joint_vel, 
            asset_cfg=SceneEntityCfg("robot", joint_names=self.cfg.joint_names, preserve_order=True)
        )
        reset_root_velocity_to_specified(self, self.env_ids_imagination, velocities)

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        super()._reset_idx(env_ids)