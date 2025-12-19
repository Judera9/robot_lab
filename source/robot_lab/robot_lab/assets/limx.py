import isaaclab.sim as sim_utils
from robot_lab.assets.actuators import DelayedImplicitActuatorCfg
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

ARMATURE_LEG = 0.15257125
ARMATURE_ANKLE_WAIST = 0.094889232
ARMATURE_HEAD_WRIST = 0.015556062
ARMATURE_SHOULDER_ELBOW = 0.045760625 

NATURAL_FREQ = 5.0 * 2.0 * 3.1415926535 # 5Hz
DAMPING_RATIO = 2.0

STIFFNESS_LEG = ARMATURE_LEG * NATURAL_FREQ ** 2 
STIFFNESS_ANKLE_WAIST = ARMATURE_ANKLE_WAIST * NATURAL_FREQ ** 2
STIFFNESS_HEAD_WRIST = ARMATURE_HEAD_WRIST * NATURAL_FREQ ** 2
STIFFNESS_SHOULDER_ELBOW = ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ ** 2

DAMPING_LEG = 2.0 * DAMPING_RATIO * ARMATURE_LEG * NATURAL_FREQ
DAMPING_ANKLE_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_WAIST * NATURAL_FREQ
DAMPING_HEAD_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_HEAD_WRIST * NATURAL_FREQ
DAMPING_SHOULDER_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ

##
# Configuration
##

# LIMX_ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))
LIMX_HUD03_03_CFG_JOINT_NAMES = [
    # leg
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    # "left_ankle_pitch_joint",
    # "left_ankle_roll_joint",
    "left_A_achilles_joint",
    "left_B_achilles_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    # "right_ankle_pitch_joint",
    # "right_ankle_roll_joint",
    "right_A_achilles_joint",
    "right_B_achilles_joint",
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",
    # "waist_pitch_joint",
    "waist_B_joint",
    "waist_A_joint",
    # head
    "head_yaw_joint",
    "head_pitch_joint",
    # arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_hand_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_hand_yaw_joint",
]

JOINT_UPBODY_NAMES = [
    # waist
    "waist_yaw_joint",
    # "waist_roll_joint",
    # "waist_pitch_joint",
    "waist_A_joint",
    "waist_B_joint",
    # head
    "head_pitch_joint",
    "head_yaw_joint",
    # arm
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_hand_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_hand_yaw_joint",
]

JOINT_WEIGHTS = [
    0.5,  # left hip pitch joint
    1.2,  # left hip roll joint
    1.0,  # left hip yaw joint
    0.5,  # left knee joint
    1.0,  # left ankle pitch joint
    1.5,  # left ankle roll joint
    0.5,  # right hip pitch joint
    1.2,  # right hip roll joint
    1.0,  # right hip yaw joint
    0.5,  # right knee joint
    1.0,  # right ankle pitch joint
    1.5,  # right ankle roll joint
    1.0,  # waist yaw joint
    1.2,  # waist roll joint
    1.2,  # waist pitch joint
    1.5,  # head pitch joint
    1.5,  # head yaw joint
    1.2,  # left shoulder pitch joint
    1.8,  # left shoulder roll joint
    1.2,  # left shoulder yaw joint
    1.5,  # left elbow joint
    1.5,  # left hand yaw joint
    1.5,  # left hand roll joint
    1.5,  # left hand pitch joint
    1.2,  # right shoulder pitch joint
    1.8,  # right shoulder roll joint
    1.2,  # right shoulder yaw joint
    1.5,  # right elbow joint
    1.5,  # right hand yaw joint
    1.5,  # right hand roll joint
    1.5,  # right hand pitch joint
]

LIMX_HUD03_03_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/limx/HU_D03/usd/HU_D03_03.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=3.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=10,
            solver_velocity_iteration_count=1,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(
        #     contact_offset=0.01, rest_offset=0.005
        # ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force")
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.92),
        joint_pos={
            'left_hip_pitch_joint': -0.15,
            'left_hip_roll_joint': -0.00,
            'left_hip_yaw_joint': -0.05,
            'left_knee_joint': 0.30,
            'left_ankle_pitch_joint': -0.16,
            'left_ankle_roll_joint': 0.0,
            'left_A_achilles_joint': -0.15, 
            'left_B_achilles_joint': 0.15, 

            'right_hip_pitch_joint': -0.15,
            'right_hip_roll_joint': 0.00,
            'right_hip_yaw_joint': 0.05,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.16,
            'right_ankle_roll_joint': 0.0,
            'right_A_achilles_joint': 0.15, 
            'right_B_achilles_joint': -0.15, 

            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            'waist_A_joint': 0.0, 
            'waist_B_joint': 0.0, 

            'head_pitch_joint': 0.0,
            'head_yaw_joint': 0.0,

            'left_shoulder_pitch_joint': 0.1,
            'left_shoulder_roll_joint': 0.1,
            'left_shoulder_yaw_joint': -0.2,
            'left_elbow_joint': -0.2,
            'left_wrist_yaw_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_hand_yaw_joint': 0.0,

            'right_shoulder_pitch_joint': 0.1,
            'right_shoulder_roll_joint': -0.1,
            'right_shoulder_yaw_joint': 0.2,
            'right_elbow_joint': -0.2,
            'right_wrist_yaw_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_hand_yaw_joint': 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.00,
    actuators={
        "robot": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                '.*_hip_pitch_joint',
                '.*_hip_roll_joint',
                '.*_hip_yaw_joint',
                '.*_knee_joint',

                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",

                '.*_A_achilles_joint',
                '.*_B_achilles_joint',

                'waist_yaw_joint',
                'waist_roll_joint',
                'waist_pitch_joint',
                'waist_A_joint',
                'waist_B_joint',

                'head_pitch_joint',
                'head_yaw_joint',

                '.*_shoulder_pitch_joint',
                '.*_shoulder_roll_joint',
                '.*_shoulder_yaw_joint',
                '.*_elbow_joint',

                '.*_wrist_yaw_joint',
                '.*_wrist_pitch_joint',
                '.*_hand_yaw_joint',
            ],
            stiffness={
                '.*_hip_pitch_joint': STIFFNESS_LEG,
                '.*_hip_roll_joint': STIFFNESS_LEG,
                '.*_hip_yaw_joint': STIFFNESS_LEG,
                '.*_knee_joint': STIFFNESS_LEG,

                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,

                '.*_A_achilles_joint': STIFFNESS_ANKLE_WAIST/2,
                '.*_B_achilles_joint': STIFFNESS_ANKLE_WAIST/2,

                'waist_yaw_joint': STIFFNESS_ANKLE_WAIST,
                'waist_roll_joint': 0.0,
                'waist_pitch_joint': 0.0,
                'waist_A_joint': STIFFNESS_ANKLE_WAIST,
                'waist_B_joint': STIFFNESS_ANKLE_WAIST,

                'head_pitch_joint': STIFFNESS_HEAD_WRIST,
                'head_yaw_joint': STIFFNESS_HEAD_WRIST,

                '.*_shoulder_pitch_joint': STIFFNESS_SHOULDER_ELBOW,
                '.*_shoulder_roll_joint': STIFFNESS_SHOULDER_ELBOW,
                '.*_shoulder_yaw_joint': STIFFNESS_SHOULDER_ELBOW,
                '.*_elbow_joint': STIFFNESS_SHOULDER_ELBOW,

                '.*_wrist_yaw_joint': STIFFNESS_HEAD_WRIST,
                '.*_wrist_pitch_joint': STIFFNESS_HEAD_WRIST,
                '.*_hand_yaw_joint': STIFFNESS_HEAD_WRIST,
            },
            damping={
                '.*_hip_pitch_joint': DAMPING_LEG,
                '.*_hip_roll_joint': DAMPING_LEG,
                '.*_hip_yaw_joint': DAMPING_LEG,
                '.*_knee_joint': DAMPING_LEG,

                ".*_ankle_pitch_joint": 0.003,
                ".*_ankle_roll_joint": 0.003,

                '.*_A_achilles_joint': DAMPING_ANKLE_WAIST,
                '.*_B_achilles_joint': DAMPING_ANKLE_WAIST,

                'waist_yaw_joint': DAMPING_ANKLE_WAIST,
                'waist_roll_joint': 0.003,
                'waist_pitch_joint': 0.003,

                'waist_A_joint': DAMPING_ANKLE_WAIST,
                'waist_B_joint': DAMPING_ANKLE_WAIST,

                'head_pitch_joint': DAMPING_HEAD_WRIST,
                'head_yaw_joint': DAMPING_HEAD_WRIST,

                '.*_shoulder_pitch_joint': DAMPING_SHOULDER_ELBOW,
                '.*_shoulder_roll_joint': DAMPING_SHOULDER_ELBOW,
                '.*_shoulder_yaw_joint': DAMPING_SHOULDER_ELBOW,
                '.*_elbow_joint': DAMPING_SHOULDER_ELBOW,

                '.*_wrist_yaw_joint': DAMPING_HEAD_WRIST,
                '.*_wrist_pitch_joint': DAMPING_HEAD_WRIST,
                '.*_hand_yaw_joint': DAMPING_HEAD_WRIST,
            },
             effort_limit={
                '.*_hip_pitch_joint': 120.,
                '.*_hip_roll_joint': 120.,
                '.*_hip_yaw_joint': 120.,
                '.*_knee_joint': 120.,

                ".*_ankle_pitch_joint": 30,
                ".*_ankle_roll_joint": 30,

                '.*_A_achilles_joint': 40,
                '.*_B_achilles_joint': 40,

                'waist_yaw_joint': 45,
                'waist_roll_joint': 45,
                'waist_pitch_joint': 45,

                'waist_A_joint': 35,
                'waist_B_joint': 35,

                'head_pitch_joint': 18.0,
                'head_yaw_joint': 18.0,

                '.*_shoulder_pitch_joint': 30,
                '.*_shoulder_roll_joint': 30,
                '.*_shoulder_yaw_joint': 30,
                '.*_elbow_joint': 30,

                '.*_wrist_yaw_joint': 18,
                '.*_wrist_pitch_joint': 18,
                '.*_hand_yaw_joint': 18,
            },
            velocity_limit={
                '.*_hip_pitch_joint': 12,
                '.*_hip_roll_joint': 12,
                '.*_hip_yaw_joint': 12,
                '.*_knee_joint': 12,

                ".*_ankle_pitch_joint": 12,
                ".*_ankle_roll_joint": 12,

                '.*_A_achilles_joint': 16,
                '.*_B_achilles_joint': 16,

                'waist_yaw_joint': 16.0,
                'waist_roll_joint': 16.0,
                'waist_pitch_joint': 16.0,

                'waist_A_joint': 16,
                'waist_B_joint': 16,

                'head_pitch_joint': 15,
                'head_yaw_joint': 15,

                '.*_shoulder_pitch_joint': 16,
                '.*_shoulder_roll_joint': 16,
                '.*_shoulder_yaw_joint': 16,
                '.*_elbow_joint': 16,

                '.*_wrist_yaw_joint': 15,
                '.*_wrist_pitch_joint': 15,
                '.*_hand_yaw_joint': 15,
            },
            armature={
                '.*_hip_pitch_joint': ARMATURE_LEG,
                '.*_hip_roll_joint': ARMATURE_LEG,
                '.*_hip_yaw_joint': ARMATURE_LEG,
                '.*_knee_joint': ARMATURE_LEG,

                ".*_ankle_pitch_joint": 0.001,
                ".*_ankle_roll_joint": 0.001,

                '.*_A_achilles_joint': ARMATURE_ANKLE_WAIST,
                '.*_B_achilles_joint': ARMATURE_ANKLE_WAIST,

                'waist_yaw_joint': ARMATURE_ANKLE_WAIST,
                'waist_roll_joint': 0.001,
                'waist_pitch_joint': 0.001,

                'waist_A_joint': ARMATURE_ANKLE_WAIST,
                'waist_B_joint': ARMATURE_ANKLE_WAIST,

                'head_pitch_joint': ARMATURE_HEAD_WRIST,
                'head_yaw_joint': ARMATURE_HEAD_WRIST,

                '.*_shoulder_pitch_joint': ARMATURE_SHOULDER_ELBOW,
                '.*_shoulder_roll_joint': ARMATURE_SHOULDER_ELBOW,
                '.*_shoulder_yaw_joint': ARMATURE_SHOULDER_ELBOW,
                '.*_elbow_joint': ARMATURE_SHOULDER_ELBOW,

                '.*_wrist_yaw_joint': ARMATURE_HEAD_WRIST,
                '.*_wrist_pitch_joint': ARMATURE_HEAD_WRIST,
                '.*_hand_yaw_joint': ARMATURE_HEAD_WRIST,
            },
            friction={
                '.*_hip_pitch_joint': 0.0,
                '.*_hip_roll_joint': 0.0,
                '.*_hip_yaw_joint': 0.0,
                '.*_knee_joint': 0.0,

                ".*_ankle_pitch_joint": 0.0,
                ".*_ankle_roll_joint": 0.0,

                '.*_A_achilles_joint': 0.0,
                '.*_B_achilles_joint': 0.0,

                'waist_yaw_joint': 0.0,
                'waist_roll_joint': 0.0,
                'waist_pitch_joint': 0.0,

                'waist_A_joint': 0.0,
                'waist_B_joint': 0.0,

                'head_pitch_joint': 0.0,
                'head_yaw_joint': 0.0,

                '.*_shoulder_pitch_joint': 0.0,
                '.*_shoulder_roll_joint': 0.0,
                '.*_shoulder_yaw_joint': 0.0,
                '.*_elbow_joint': 0.0,

                '.*_wrist_yaw_joint': 0.0,
                '.*_wrist_pitch_joint': 0.0,
                '.*_hand_yaw_joint': 0.0,
            },
            min_delay=0,  # physics time steps (min: 1.0*0=0.0ms)
            max_delay=0,  # physics time steps (max: 1.0*5=5.0ms)
        ),
    },
)