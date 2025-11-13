# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py -f path_to_motion.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import torch
from isaaclab.app import AppLauncher
import os
from typing import Sequence
import isaaclab.utils.math as math_utils
import tqdm

class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], fps: int, device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        self.device = device
        self._body_indexes = body_indexes
        self.fps = fps
        if motion_file.endswith(".npz"):
            self.load_beyondmimic_data(motion_file)
        elif motion_file.endswith(".npy"):
            self.load_amp_npy_data(motion_file)
        else:
            raise ValueError(f"Unsupported motion file format: {motion_file}")

    def load_amp_npy_data(self, motion_file: str):
        data = np.load(motion_file, allow_pickle=True).item()
        self.joint_pos = data["joint_pos"].to(self.device)
        self.joint_vel = data["joint_vel"].to(self.device)
        self._body_pos_w = data["root_pos"].unsqueeze(1).to(self.device)
        self._body_quat_w = math_utils.quat_unique(data["root_rot"].unsqueeze(1).to(self.device))
        # self._body_quat_w = math_utils.quat_unique(torch.cat([self._body_quat_w[..., 3].unsqueeze(-1), self._body_quat_w[..., :3]], dim=-1))
        self.time_step_total = self.joint_pos.shape[0]

    def load_beyondmimic_data(self, motion_file: str):
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=self.device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=self.device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=self.device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=self.device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=self.device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=self.device)
        self.time_step_total = self.joint_pos.shape[0]

    def resample_data(self, target_fps: int):
        original_length = 1.0 / self.fps * (self.time_step_total - 1)
        target_num_frames = int(original_length * target_fps)
        resampled_duration = 1.0 / target_fps
        resampled_joint_pos = torch.zeros((target_num_frames, self.joint_pos.shape[1]), dtype=self.joint_pos.dtype, device=self.device)
        resampled_joint_vel = torch.zeros((target_num_frames, self.joint_vel.shape[1]), dtype=self.joint_vel.dtype, device=self.device)
        resampled_body_pos_w = torch.zeros((target_num_frames, *self._body_pos_w.shape[1:]), dtype=self._body_pos_w.dtype, device=self.device)
        resampled_body_quat_w = torch.zeros((target_num_frames, *self._body_quat_w.shape[1:]), dtype=self._body_quat_w.dtype, device=self.device)
        
        for i in range(target_num_frames):
            # Current time in the resampled motion
            current_time = i * resampled_duration
            # Position in the original motion (normalized to [0, 1])
            p = current_time / original_length
            # Make sure p doesn't exceed 1.0
            p = min(p, 1.0)
            
            n = self.time_step_total
            # Convert to tensor for torch operations
            p_tensor = torch.tensor(p, device=self.device)
            idx_low = torch.floor(p_tensor * (n - 1)).int()
            idx_high = torch.min(idx_low + 1, torch.tensor(n - 1, device=self.device)).int()
            blend = p_tensor * (n - 1) - idx_low

            frame_start = self.joint_pos[idx_low]
            frame_end = self.joint_pos[idx_high]
            resampled_joint_pos[i] = torch.lerp(frame_start, frame_end, blend)

            frame_start = self.joint_vel[idx_low]
            frame_end = self.joint_vel[idx_high]
            resampled_joint_vel[i] = torch.lerp(frame_start, frame_end, blend)

            frame_start = self._body_pos_w[idx_low]
            frame_end = self._body_pos_w[idx_high]
            resampled_body_pos_w[i] = torch.lerp(frame_start, frame_end, blend)

            frame_start = self._body_quat_w[idx_low]
            frame_end = self._body_quat_w[idx_high]
            resampled_body_quat_w[i] = math_utils.quat_slerp(frame_start.squeeze(0), frame_end.squeeze(0), blend).unsqueeze(0)

        # update the motion data
        self.fps = target_fps
        self.joint_pos = resampled_joint_pos
        self.joint_vel = resampled_joint_vel
        self._body_pos_w = resampled_body_pos_w
        self._body_quat_w = resampled_body_quat_w
        self.time_step_total = target_num_frames

    def save_amp_npy_data(self, save_path: str):
        saved_dict = {
            "joint_pos": self.joint_pos,
            "joint_vel": self.joint_vel,
            "root_pos": self._body_pos_w.squeeze(1),
            "root_rot": self._body_quat_w.squeeze(1),
        }
        np.save(save_path, saved_dict)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--file", "-f", type=str, required=False)
parser.add_argument("--files", "-fs", type=str, nargs='+', required=False, help="List of motion files to replay")
parser.add_argument("--dir", "-d", type=str, required=False)
parser.add_argument("--articulation_cfg", "-ac", type=str, required=False)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_G1_29DOF_CFG
from robot_lab.assets.limx import LIMX_HUD03_03_CFG, LIMX_HUD03_03_CFG_JOINT_NAMES

VIS_CONFIG = {
    "multi_envs": True if args_cli.dir is not None or args_cli.files is not None else False,
    "num_envs": (len(os.listdir(args_cli.dir)) if args_cli.dir is not None else (len(args_cli.files) if args_cli.files is not None else 1)),
    "data_fps": 50,
    "sim_dt": 0.02,  # correspond to motion fps
    "save_data": False,
    "save_path": "./motion_data.npy",
    "origin_offset": 3.0,
    "fix_camera": False,
    "look_at_robot_index": 0,
}

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    if args_cli.articulation_cfg == "UNITREE_G1_29DOF_CFG":
        robot: ArticulationCfg = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.articulation_cfg == "LIMX_HUD03_03_CFG":
        robot: ArticulationCfg = LIMX_HUD03_03_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Unknown articulation cfg: {args_cli.articulation_cfg}")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> tqdm.tqdm:
    # Extract scene entities
    robot: Articulation = scene["robot"]
    if args_cli.articulation_cfg == "UNITREE_G1_29DOF_CFG":
        robot_cfg = SceneEntityCfg("robot", preserve_order=True)
    elif args_cli.articulation_cfg == "LIMX_HUD03_03_CFG":
        robot_cfg = SceneEntityCfg("robot", joint_names=LIMX_HUD03_03_CFG_JOINT_NAMES, preserve_order=True)
    robot_cfg.resolve(scene)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    assert args_cli.file is not None or args_cli.files is not None or args_cli.dir is not None, "Please provide a file or a directory of files."
    
    if args_cli.file is not None:
        # Single file mode - all environments use the same motion
        motion = MotionLoader(
            args_cli.file,
            torch.tensor([0], dtype=torch.long, device=sim.device),
            VIS_CONFIG["data_fps"],
            sim.device,
        )
        if VIS_CONFIG["data_fps"] != 1/VIS_CONFIG["sim_dt"]:
            motion.resample_data(int(1 / VIS_CONFIG["sim_dt"]))
        if VIS_CONFIG["save_data"]:
            motion.save_amp_npy_data(VIS_CONFIG["save_path"])
        motions = [motion] * scene.num_envs
    elif args_cli.files is not None:
        # Multi-file mode - each environment uses a different motion
        if len(args_cli.files) != scene.num_envs:
            raise ValueError(f"Number of files ({len(args_cli.files)}) must match number of environments ({scene.num_envs})")
        motions = []
        for file_path in args_cli.files:
            motion = MotionLoader(
                file_path,
                torch.tensor([0], dtype=torch.long, device=sim.device),
                VIS_CONFIG["data_fps"],
                sim.device,
            )
            if VIS_CONFIG["data_fps"] != 1/VIS_CONFIG["sim_dt"]:
                motion.resample_data(int(1 / VIS_CONFIG["sim_dt"]))
            motions.append(motion)
    else:
        raise NotImplementedError("Directory mode not implemented yet.")

    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Initialize progress bar (use first motion's length for progress tracking)
    max_frames = max(m.time_step_total for m in motions)
    progress_bar = tqdm.tqdm(
        total=max_frames,
        desc="Playing motions",
        unit="frame",
        dynamic_ncols=True
    )

    # Simulation loop
    while simulation_app.is_running():
        if progress_bar.total is None:
            progress_bar.total = max_frames
        if progress_bar.n >= progress_bar.total:
            break
        # Update progress bar with info from all environments
        postfix_info = {f"env {i} frame": time_steps[i].item() for i in range(scene.num_envs)}
        progress_bar.set_postfix(postfix_info)
        progress_bar.update(1)
        
        time_steps += 1
        
        # Handle motion resets for each environment
        for i in range(scene.num_envs):
            if time_steps[i] >= motions[i].time_step_total:
                time_steps[i] = 0
        
        # Prepare joint states for all environments
        all_joint_pos = []
        all_joint_vel = []
        all_root_states = []
        
        for i in range(scene.num_envs):
            motion = motions[i]
            current_step = time_steps[i]
            
            # Root states for environment i
            root_state = robot.data.default_root_state.clone()[0, ...]
            root_state[:3] = motion.body_pos_w[current_step] + \
                scene.env_origins[i, None, :] + torch.tensor([0, i * VIS_CONFIG["origin_offset"], 0], dtype=torch.float32, device=sim.device)
            root_state[3:7] = motion.body_quat_w[current_step]
            all_root_states.append(root_state.unsqueeze(0))
            
            # Joint states for environment i
            joint_pos = robot.data.default_joint_pos.clone()[0, robot_cfg.joint_ids]
            joint_vel = robot.data.default_joint_vel.clone()[0, robot_cfg.joint_ids]
            joint_pos = motion.joint_pos[current_step]
            joint_vel = motion.joint_vel[current_step]
            all_joint_pos.append(joint_pos.unsqueeze(0))
            all_joint_vel.append(joint_vel.unsqueeze(0))
        
        # Stack all environment states
        root_states = torch.cat(all_root_states, dim=0)
        joint_pos = torch.cat(all_joint_pos, dim=0)
        joint_vel = torch.cat(all_joint_vel, dim=0)
        
        robot.write_root_state_to_sim(root_states)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, robot_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        if VIS_CONFIG["fix_camera"]:
            # set camera to follow specified robot
            pos_lookat = root_states[VIS_CONFIG["look_at_robot_index"], :3].cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)
    return progress_bar

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = VIS_CONFIG["sim_dt"]
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=VIS_CONFIG["num_envs"], env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    progress_bar = run_simulator(sim, scene)
    # Close progress bar
    progress_bar.close()
    tqdm.tqdm.write("Simulation completed.")

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
