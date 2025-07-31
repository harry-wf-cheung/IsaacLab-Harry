# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the RMPflow controller with a dynamically draggable goal
in Isaac Lab's simulation environment.

The robot will attempt to reach the position of a red sphere that can be dragged using the mouse.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/05_controllers/run_rmpflow_draggable_goal.py
    # (assuming you save this as 'run_rmpflow_draggable_goal.py' in a similar path)

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the RMPflow controller with a draggable goal.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn (set to 1 for dragging).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# IMPORTANT: Ensure UI is enabled for dragging functionality
args_cli.headless = False
args_cli.ui_enabled = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from dataclasses import MISSING

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import SingleArticulation
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from rmp_flow import RmpFlowController, RmpFlowControllerCfg # This is the change!# Changed from DifferentialIKController
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg# For kinematic object
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # Added ISAACLAB_ASSETS_DIR
from isaaclab.utils.math import subtract_frame_transforms # Not strictly needed with RMPFlow, but kept for context
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.robot_motion.lula")
enable_extension("isaacsim.robot_motion.motion_generation")
from isaacsim.robot_motion.motion_generation import ArticulationMotionPolicy
from isaacsim.robot_motion.motion_generation.lula.motion_policies import RmpFlow, RmpFlowSmoothed

##
# Pre-defined configs
##
# RMPflow typically works best with Franka's default or slightly adjusted PD gains,
# as it handles its own internal impedance. FRANKA_PANDA_HIGH_PD_CFG might be too stiff.
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG
from isaaclab_assets.robots.franka import RMP_CONFIG_PATH, RMP_JOINT_CONFIG_PATH # RMPflow configs



@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with a robot and a draggable target."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        # For RMPflow, it's generally better to use the default Franka config
        # or one with default/lower PD gains, as RMPflow handles its own impedance.
        # FRANKA_PANDA_HIGH_PD_CFG might make it too stiff.
        # However, for simplicity of modification, we'll keep it for now.
        # If RMPflow behaves poorly, consider a simpler ArticulationCfg for Franka.
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")

    # Draggable Target Object (a simple sphere)
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
           # height=0.2,
            scale=(0.05, 0.05, 0.05), # Small sphere
            rigid_props=RigidBodyPropertiesCfg(
                rigid_body_enabled=False, # Make it non-physical
                kinematic_enabled=True,   # Make it kinematic so we can drag it
                disable_gravity=True,
            ),
            visual_material_path="/World/Looks/Red", # Make it red for visibility
            collision_props=None,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.5), # Initial position for the target
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        collision_group=1, # Different group from robot if you want to filter
        debug_vis=True,
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["robot"]
    target_object = scene["target"] # Get the target object

    # Create RMPflow controller
    rmpflow_cfg = RMPFlowControllerCfg(
        prim_path=robot.prim_path, # Path to the robot prim
        robot_description_path=str(RMP_CONFIG_PATH),
        joint_internal_config_path=str(RMP_JOINT_CONFIG_PATH),
        end_effector_frame_name="panda_hand" if args_cli.robot == "franka_panda" else "ee_link", # EE frame
        nominal_grasp_quat=(0.707, 0.707, 0.0, 0.0), # Example: gripper pointing down
        use_target_pose_for_end_effector=True, # We will command pose (pos + quat)
        use_state_estimation=True, # Use Isaac Lab's state estimation
        num_envs=scene.num_envs,
        dt=sim.get_physics_dt(),
        command_type="pose",
        asset_name="robot",
    )
    rmpflow_controller = RMPFlowController(rmpflow_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal")) # This will track draggable sphere

    # Specify robot-specific parameters for reading joint states (needed by RMPflow internally)
    if args_cli.robot == "franka_panda":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    elif args_cli.robot == "ur10":
        robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Initial target pose for the robot (based on the initial target sphere position)
    initial_target_pos = target_object.data.pos[0].clone() # Get initial position of the sphere
    # Use the nominal grasp quaternion for orientation
    initial_target_quat = torch.tensor(rmpflow_cfg.nominal_grasp_quat, device=sim.device).repeat(scene.num_envs, 1)
    # The command for RMPflow is pos + quat (7-DOF)
    rmpflow_command = torch.cat([initial_target_pos, initial_target_quat], dim=-1)

    # Reset the controller at the start
    rmpflow_controller.reset()
    rmpflow_controller.set_command(rmpflow_command)

    # Simulation loop
    while simulation_app.is_running():
        # Reset the robot every now and then (or on request)
        if count % 150 == 0:
            count = 0
            # Reset joint state to default
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos,
                robot.data.default_joint_vel
            )
            robot.reset()
            # Reset the controller state
            rmpflow_controller.reset()
            # IMPORTANT: Re-set the RMPflow command to the *current* target sphere position
            # after a full environment reset to ensure it tracks correctly.
            current_target_pos = target_object.data.pos[0].clone()
            current_target_quat = torch.tensor(rmpflow_cfg.nominal_grasp_quat, device=sim.device).repeat(scene.num_envs, 1)
            rmpflow_command = torch.cat([current_target_pos, current_target_quat], dim=-1)
            rmpflow_controller.set_command(rmpflow_command)

        # --- Update RMPflow command based on draggable target ---
        # Get the current position of the draggable target sphere
        # target_object.data.pos gives (num_envs, 3)
        current_target_pos_w = target_object.data.pos[0, 0:3].clone() # Get position for the first (and only) env

        # Construct the RMPflow command: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        # For this example, we keep the orientation constant (nominal_grasp_quat).
        # If your task requires, you could set target_quat based on another object's orientation.
        current_rmpflow_command = torch.cat([current_target_pos_w, initial_target_quat[0]], dim=-1)
        rmpflow_controller.set_command(current_rmpflow_command)

        # --- Compute actions from RMPflow ---
        # RMPflow computes joint position targets.
        joint_pos_des = rmpflow_controller.compute_actions()

        # --- Apply actions to the robot ---
        # RMPflow outputs joint position targets, which are directly applied by Isaac Lab's actuators.
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # --- Perform simulation step ---
        scene.write_data_to_sim() # Write commands to sim
        sim.step() # Advance physics
        count += 1
        scene.update(sim_dt) # Update scene data from sim

        # --- Visualization ---
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        # Visualize the actual goal RMPflow is tracking (which is the draggable sphere's position)
        goal_marker.visualize(target_object.data.pos[0:1], target_object.data.quat[0:1]) # Use target's actual pos/quat for marker


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device) # dt is usually smaller for RMPflow
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator to load the scene
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete. Drag the red sphere to move the robot's goal.")
    print("[INFO]: Press 'ESC' to quit.")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()