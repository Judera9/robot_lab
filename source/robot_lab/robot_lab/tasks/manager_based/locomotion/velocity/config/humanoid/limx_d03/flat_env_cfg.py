# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gausnoise

from robot_lab.assets.limx import LIMX_HUD03_03_CFG, LIMX_HUD03_03_CFG_JOINT_NAMES, JOINT_UPBODY_NAMES, JOINT_WEIGHTS


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES,
        scale={
            ".*hip_pitch.*": 0.25,
            ".*hip_roll.*": 0.25,
            ".*hip_yaw.*": 0.5,
            ".*knee.*": 0.25,
            # ".*ankle.*": 0.25,
            ".*A_achilles.*": 0.25,
            ".*B_achilles.*": 0.25,
            # ".*waist.*": 0.1,
            ".*waist_yaw.*": 0.1,
            ".*A_joint.*": 0.1,
            ".*B_joint.*": 0.1,
            ".*head.*": 0.1,
            ".*shoulder_pitch.*": 0.1,
            ".*shoulder_roll.*": 0.1,
            ".*shoulder_yaw.*": 0.1,
            ".*elbow.*": 0.1,
            ".*hand.*": 0.1,
            ".*wrist.*": 0.1,
        },
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Gausnoise(mean=0.0, std=0.1), scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Gausnoise(mean=0.0, std=0.03), scale=1.0)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            noise=Gausnoise(mean=0.0, std=0.0075),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            noise=Gausnoise(mean=0.0, std=0.25),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        gait = ObsTerm(
            func=mdp.gait_observation,
            params={
                "frequency_range": (0.8, 1.15),
                "offset_range": (0.5, 0.5),
                "height_range": (0.10, 0.10),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, scale=1.0)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0)
        base_height = ObsTerm(func=mdp.base_pos_z, scale=1.0)
        joint_acc = ObsTerm(
            func=mdp.joint_acc,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            scale=0.0025,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_torques,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)},
            scale=0.025,
        )
        ee_pos = ObsTerm(
            func=mdp.keypoints_position_heading,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    body_names=[
                        "left_shoulder_pitch_link",
                        "right_shoulder_pitch_link",
                        "left_hip_roll_link",
                        "right_hip_roll_link",
                        "left_knee_link",
                        "right_knee_link",
                        "left_ankle_roll_link", 
                        "right_ankle_roll_link",
                    ], 
                    preserve_order=True
                )
            },
            scale=1.0,
        )
        ee_vel = ObsTerm(
            func=mdp.keypoints_velocity_heading, 
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    body_names=[
                        "left_hip_roll_link",
                        "right_hip_roll_link",
                        "left_knee_link",
                        "right_knee_link",
                        "left_ankle_roll_link", 
                        "right_ankle_roll_link",
                    ], 
                    preserve_order=True
                )
            },
            scale = 0.05, 
        )
        ee_quat = ObsTerm(
            func=mdp.keypoints_rotmat_heading,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", 
                    body_names=[
                        "waist_pitch_link",
                        "left_hip_roll_link",
                        "right_hip_roll_link",
                        "left_knee_link",
                        "right_knee_link",
                        "left_ankle_roll_link", 
                        "right_ankle_roll_link",
                    ], 
                    preserve_order=True
                )
            },
            scale=1.0,
        )
        body_incoming_wrench = ObsTerm(
            func=mdp.body_incoming_wrench,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base_link", ".*hand_yaw.*", ".*ankle_roll.*"])},
            scale=0.01,
        )
        # contact_forces = ObsTerm(
        #     func=mdp.contact_forces_state,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*")},
        #     scale=0.01,
        # )
        gait_info = ObsTerm(
            func=mdp.gait_info,
            params={"frequency_range": (0.8, 1.15), "offset_range": (0.5, 0.5), "height_range": (0.10, 0.10)},
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-3.0, 3.0),
            "operation": "add",
        },
    )
    add_waist_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "mass_distribution_params": (-2.0, 6.0),
            "operation": "add",
        },
    )
    random_centroid = EventTerm(
        func= mdp.randomize_rigid_body_centroid,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "x_range": (-0.1, 0.1),
            "y_range": (-0.08, 0.08),
            "z_range": (-0.1, 0.1),
        },
    )
    random_inertias = EventTerm(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "waist_pitch_link"]),
            "scale": (0.9, 1.2),
        },
    )

    # reset
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.0),
            "dynamic_friction_range": (0.3, 0.9),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (1.0, 0.15),
            "damping_distribution_params": (1.0, 0.15),
            "operation": "scale",
            "distribution": "gaussian",
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_robot_joints_scale = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}, 
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link", "waist_pitch_link"]),
            "force_range": (-5.0, 5.0),
            "torque_range": (-5.0, 5.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task specific -- #

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, 
        weight=8.0, 
        params={
            "command_name": "base_velocity", 
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "std": math.sqrt(0.25), 
        }
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, 
        weight=4.0, 
        params={
            "command_name": "base_velocity", 
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_pitch_link"),
            "std": math.sqrt(0.25)
        }
    )

    # -- gait -- #
    tracking_contacts_shaped_contacts = RewTerm(  # feet_contact_force
        func=mdp.tracking_contacts_shaped_contacts,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "std_force": math.sqrt(10000.0),
            "std_vel": math.sqrt(0.05),
        },
    )
    natural_swing_arm = RewTerm( # Could be altered with ee_ref
        func=mdp.natural_swing_arm,
        weight=4.0,
        params={
            "command_name": "base_velocity",
            "std_pos": math.sqrt(0.01),
            "std_vel": math.sqrt(1.0),
        },
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    straight_knee = RewTerm(
        func=mdp.straight_knee, 
        weight=1.0, 
        params={
            "std": 0.1, 
            "asset_cfg": SceneEntityCfg("robot", joint_names=["left_knee_joint", "right_knee_joint"], preserve_order=True),        
        }
    )
    stand_still = RewTerm(
        func=mdp.stand_still, 
        weight=1.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_UPBODY_NAMES, preserve_order=True),
        }
    )

    # -- termination -- #
    is_alive = RewTerm(func=mdp.is_alive, weight=10.0)
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # -- torso control -- #
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_pitch_link", "base_link"]),
        },
    )
    body_orientation = RewTerm(
        func=mdp.body_projected_gravity_l2, 
        weight=-1.0, 
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_pitch_link", "base_link"]), 
            "std": math.sqrt(0.01)
        }
    )
    body_lin_acc = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["waist_pitch_link", "base_link"]),
        },
    )

    # -- penalties -- #
    feet_slide = RewTerm(
        func=mdp.feet_slide, 
        weight=-0.1, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"), 
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")
        }
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base_link", ".*shoulder.*", ".*elbow.*", ".*hip.*", ".*knee.*", ".*waist.*", ".*hand.*", ".*wrist.*"],
            ),
            "threshold": 0.1,
        },
    )
    joint_position_normalization = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*head.*", ".*hand.*", ".*wrist.*"], preserve_order=True),
        },
    )

    contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.01,
        params={
            "threshold": 550,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # -- energy
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_weights_l2,
        params={
            "joint_weights": JOINT_WEIGHTS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True),
        },
        weight=-5.0e-7,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_weights_l2,
        params={
            "joint_weights": JOINT_WEIGHTS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True),
        },
        weight=-7.5e-3,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_weights_l2,
        params={
            "joint_weights": JOINT_WEIGHTS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True),
        },
        weight=-3.0e-6,
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_weights,
        params={
            "joint_weights": JOINT_WEIGHTS,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True),
        },
        weight=-1.0,
    )
    applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits, 
        weight=-5.0e-2
    )
    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2, 
        weight=-1.2e-1, 
    )



@configclass
class LimxHumanoidEnvCfg(LocomotionVelocityRoughEnvCfg):

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        self.scene.num_envs = 16
        self.scene.robot = LIMX_HUD03_03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # commands
        self.commands.base_velocity.resampling_time_range = (2.0, 10.0)
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # gait
        self.use_gait = True
        self.gait_handler_cfg = {
            "frequency_range": (0.8, 1.15),
            "offset_range": (0.5, 0.5),
            "height_range": (0.10, 0.10),
        }

        # observations
        

        # curriculums
        self.curriculum.command_levels = None


        # event
        self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 3.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "LimxHumanoidAMPEnvCfg":
            self.disable_zero_weight_rewards()
