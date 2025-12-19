# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch
import math

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand

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

        # no terrain curriculum
        self.curriculum.terrain_levels = None

        # Terminations
        self.terminations.bad_orientation = None
        self.terminations.root_height_below_minimum = None

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
        self.commands.base_velocity.ranges = mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 0.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(-math.pi, math.pi)
        )

        # Events
        self.events.randomize_actuator_gains = None
        self.events.randomize_reset_base = None
        self.events.randomize_push_robot = None
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_com_positions = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_reset_joints = None
         
        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "RWMUnitreeG1FlatEnvVisCfg":
            self.disable_zero_weight_rewards()