# # Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause
# from locomotion.tasks.locomotion.velocity.velocity_amp_env_cfg import MotionTrackingAMPEnvCfg


# from isaaclab.utils import configclass

# ##
# # Pre-defined configs
# ##
# # from locomotion.assets.limx import LIMX_HUD02_CFG  # isort: skip
# from locomotion.assets.HU_D03_03 import LIMX_HUD03_03_CFG


# @configclass
# class LimxHumanoidAMPEnvCfg(MotionTrackingAMPEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         self.scene.robot = LIMX_HUD03_03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#         self.scene.num_envs = 4096
#         self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"
#         # change terrain to flat
#         # self.scene.terrain.terrain_type = "plane"
#         # self.scene.terrain.terrain_generator = None
#         # self.curriculum.terrain_levels = None

#         # reduce action scale
#         # self.actions.joint_pos.scale = 0.25

#         # event
#         # self.events.push_robot = None
#         self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
#         self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
#         # self.events.base_external_force_torque.params["asset_cfg"].body_names = (
#         #     "base_link"
#         # )
#         # if hasattr(self.events, "reset_robot_joints") and self.events.reset_robot_joints is not None:
#         #     self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
#         # if hasattr(self.events, "reset_base") and self.events.reset_base is not None:
#         #     self.events.reset_base.params = {
#         #         "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
#         #         "velocity_range": {
#         #             "x": (0.0, 0.0),
#         #             "y": (0.0, 0.0),
#         #             "z": (0.0, 0.0),
#         #             "roll": (0.0, 0.0),
#         #             "pitch": (0.0, 0.0),
#         #             "yaw": (0.0, 0.0),
#         #         },
#         #     }

#         # rewards
#         # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
#         # self.rewards.feet_air_time.weight = 0.01
#         # self.rewards.undesired_contacts = None
#         # self.rewards.dof_torques_l2.weight = -0.0002
#         # self.rewards.track_lin_vel_xy_exp.weight = 1.5
#         # self.rewards.track_ang_vel_z_exp.weight = 0.75
#         # self.rewards.dof_acc_l2.weight = -2.5e-7


# @configclass
# class LimxHumanoidAMPEnvCfg_PLAY(LimxHumanoidAMPEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()

#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # spawn the robot randomly in the grid (instead of their terrain levels)
#         self.scene.terrain.max_init_terrain_level = None
#         # reduce the number of terrains to save memory
#         if self.scene.terrain.terrain_generator is not None:
#             self.scene.terrain.terrain_generator.num_rows = 5
#             self.scene.terrain.terrain_generator.num_cols = 5
#             self.scene.terrain.terrain_generator.curriculum = False

#         # disable randomization for play
#         self.observations.policy.enable_corruption = False
#         # remove random pushing event
#         # self.events.base_external_force_torque = None
#         self.events.push_robot = None
