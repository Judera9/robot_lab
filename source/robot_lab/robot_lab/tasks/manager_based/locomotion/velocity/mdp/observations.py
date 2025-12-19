# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def body_contact(env: ManagerBasedEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Contact status of the body of the asset."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return is_contact

""" AMP Locomotion for Limx D03 """

from robot_lab.tasks.manager_based.locomotion.velocity.mdp.extensions.gait import GaitHandler

def gait_observation(env: ManagerBasedRLEnv, frequency_range, offset_range, height_range):
    # # init gait
    # if gait_handler.get_init_flag() == False:
    #     gait_handler.init(
    #         kappa=0.05,
    #         num_envs=env.num_envs,
    #         device=env.device,
    #         frequency_range=frequency_range,
    #         offset_range=offset_range,
    #         height_range=height_range,
    #     )
    assert env.gait_handler is not None and env.gait_handler.get_init_flag() == True, "gait_handler is not initialized or not exist"
    gait_handler:GaitHandler = env.gait_handler
    gait_phase = gait_handler.get_phase()
    gait_info = gait_handler.get_gait_info()

    cmd = env.command_manager.get_command("base_velocity")
    stand_still_flag = gait_handler.get_stand_still_flag()
    # set the zero velocity to zero and set to standing
    stand_still_flag[:] = cmd[:, 3].bool()
    # set the nearly zero velocity to zero and but not set to standing
    # step_flag = cmd[:, 4].bool()

    dt = env.cfg.sim.dt * env.cfg.decimation
    gait_phase[:] = torch.remainder(gait_phase + dt * (~stand_still_flag) * gait_info[:, 0], 1.0)

    # reset gait parameter ranges
    if hasattr(env, "reset_buf"):
        reset_idx = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        gait_phase[reset_idx] = (torch.rand(len(reset_idx), device=env.device) > 0.5) * 0.5
        gait_info[reset_idx, 0] = (
            torch.rand(len(reset_idx), device=env.device) * (frequency_range[1] - frequency_range[0])
            + frequency_range[0]
        )
        gait_info[reset_idx, 1] = (
            torch.rand(len(reset_idx), device=env.device) * (offset_range[1] - offset_range[0]) + offset_range[0]
        )
        gait_info[reset_idx, 2] = stand_still_flag[reset_idx].float()

    stand_still_idx = stand_still_flag.nonzero(as_tuple=False).flatten()
    gait_phase[stand_still_idx] = (gait_phase[stand_still_idx] >= 0.5) * 0.5
    gait_info[:, 3] = torch.sin(2.0 * torch.pi * gait_phase)
    gait_info[:, 4] = torch.cos(2.0 * torch.pi * gait_phase)
    gait_handler.reference_compute()

    return gait_info

def joint_acc(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_acc[:, asset_cfg.joint_ids]

def joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids].reshape(env.num_envs, -1)

def keypoints_position_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    heading_align_quat = math_utils.yaw_quat(base_quat)
    ee_position_b = math_utils.quat_rotate_inverse(
        heading_align_quat.unsqueeze(1), 
        asset.data.body_pos_w[:, asset_cfg.body_ids] - asset.data.root_pos_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    )
    return ee_position_b.reshape(env.num_envs, -1)

def keypoints_velocity_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    heading_align_quat = math_utils.yaw_quat(base_quat)
    ee_velocity_b = math_utils.quat_rotate_inverse(
        heading_align_quat.unsqueeze(1), 
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids] - asset.data.root_lin_vel_w.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1)
    )
    return ee_velocity_b.reshape(env.num_envs, -1)

def keypoints_rotmat_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    base_quat = asset.data.root_quat_w
    heading_align_quat = math_utils.yaw_quat(base_quat)
    ee_quat_heading = math_utils.quat_mul(
        math_utils.quat_inv(heading_align_quat).unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1), asset.data.body_quat_w[:, asset_cfg.body_ids]
    )
    keypoints_rotmat_heading = math_utils.matrix_from_quat(ee_quat_heading).reshape(env.num_envs, len(asset_cfg.body_ids), 3, 3)
    return keypoints_rotmat_heading[:, :, 0:2, :].reshape(env.num_envs, -1)

def gait_info(env: ManagerBasedRLEnv) -> torch.Tensor:
    assert env.gait_handler is not None and env.gait_handler.get_init_flag() == True, "gait_handler is not initialized or not exist"
    gait_handler:GaitHandler = env.gait_handler
    return gait_handler.get_full_gait_info()