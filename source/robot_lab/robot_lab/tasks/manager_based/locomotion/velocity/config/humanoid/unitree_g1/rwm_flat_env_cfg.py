# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch
import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeG1RoughEnvCfg

@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class SystemStateCfg(ObsGroup):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)})
        # joint_torque = ObsTerm(func=mdp.joint_effort)
        
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        # observation terms (order preserved)
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemExtensionCfg(ObsGroup):
        pass

    @configclass
    class SystemContactCfg(ObsGroup):
        # observation terms (order preserved)
        # thigh_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0})
        foot_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):  # TODO: fix termination
        # observation terms (order preserved)
        base_contact = ObsTerm(func=mdp.body_contact, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis"), "threshold": 1.0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    # system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()

class SampleUniformVelocityCommand(UniformVelocityCommand):
    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        if hasattr(self._env, "env_ids_real"):
            self.vel_command_b[self._env.env_ids_imagination] = self.vel_command_b[self._env.env_ids_real]
            self.heading_target[self._env.env_ids_imagination] = self.heading_target[self._env.env_ids_real]
            self.is_heading_env[self._env.env_ids_imagination] = self.is_heading_env[self._env.env_ids_real]
            self.is_standing_env[self._env.env_ids_imagination] = self.is_standing_env[self._env.env_ids_real]

    def sample_command(self, num_envs: int):
        # sample velocity commands
        r = torch.empty(num_envs, device=self.device)
        vel_command_b = torch.zeros(num_envs, 3, device=self.device)
        # -- linear velocity - x direction
        vel_command_b[:, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        vel_command_b[:, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        vel_command_b[:, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        return vel_command_b

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:  # TODO: fix this, should not be in HU
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                # -- imagination current
                imagination_vel_visualizer_cfg : VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
                    prim_path="/Visuals/Command/velocity_imagination"
                )
                imagination_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.imagination_vel_visualizer = VisualizationMarkers(imagination_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.imagination_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
                self.imagination_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not hasattr(self._env, "env_ids_real"):
            self.imagination_vel_visualizer.set_visibility(False)
            super()._debug_vis_callback(event)
            return
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w[self._env.env_ids_real], vel_arrow_quat[self._env.env_ids_real], vel_arrow_scale[self._env.env_ids_real])
        self.imagination_vel_visualizer.visualize(base_pos_w[self._env.env_ids_imagination], vel_arrow_quat[self._env.env_ids_imagination], vel_arrow_scale[self._env.env_ids_imagination])

@configclass
class RWMUnitreeG1FlatEnvCfg(UnitreeG1RoughEnvCfg):

    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.system_state.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.system_state.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # events modified
        self.events.randomize_push_robot = EventTerm(
            func=mdp.push_by_setting_velocity_mod,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )
        self.events.randomize_apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque_mod,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=""),
                "force_range": (-10.0, 10.0),
                "torque_range": (-10.0, 10.0),
            },
        )
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_apply_external_force_torque.mode = "interval"
        # self.events.randomize_apply_external_force_torque.interval_range_s = (2.0, 2.0)
        # self.events.randomize_apply_external_force_torque.params['force_range'] = (15.0, 15.0)
        # self.events.randomize_apply_external_force_torque.params['torque_range'] = (15.0, 15.0)

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Terminations
        self.terminations.bad_orientation.params["limit_angle"] = 0.6
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.1

        # Commands
        # override commands
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand
        # self.commands.base_velocity.ranges = mdp.UniformThresholdVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        # )

        # Rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.joint_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time_positive_biped.weight = 0.75
        self.rewards.feet_air_time_positive_biped.params["threshold"] = 0.4
        self.rewards.feet_air_time_positive_biped.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [".*_hip_.*", ".*_knee_joint"]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RWMUnitreeG1FlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class RWMUnitreeG1FlatEnvVisCfg(RWMUnitreeG1FlatEnvCfg):

    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Commands
        # self.commands.base_velocity.ranges = mdp.UniformThresholdVelocityCommandCfg.Ranges(
        #     lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        # )
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)
        # Events
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base = None
        self.events.randomize_push_robot = None
        # self.events.randomize_push_robot.interval_range_s = (2.0, 2.0)
        # self.events.randomize_push_robot.params['velocity_range']['x'] = (-2.0, 2.0)
        # self.events.randomize_push_robot.params['velocity_range']['y'] = (-2.0, 2.0)
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        # self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = ['head_link']
        # self.events.randomize_apply_external_force_torque.mode = "interval"
        # self.events.randomize_apply_external_force_torque.interval_range_s = (2.0, 2.0)
        # self.events.randomize_apply_external_force_torque.params['force_range'] = (10.0, 10.0)
        # self.events.randomize_apply_external_force_torque.params['torque_range'] = (10.0, 10.0)
        self.events.randomize_reset_joints = None
         
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RWMUnitreeG1FlatEnvVisCfg":
            self.disable_zero_weight_rewards()