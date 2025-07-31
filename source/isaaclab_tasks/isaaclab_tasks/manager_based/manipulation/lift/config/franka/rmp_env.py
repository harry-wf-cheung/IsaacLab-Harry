# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# We will replace DifferentialIKControllerCfg and DifferentialInverseKinematicsActionCfg
# with RMPFlow-related configurations.
# from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
# from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from isaaclab.controllers.rmpflow_controller_cfg import RMPFlowControllerCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionActionCfg # RMPFlow outputs joint positions
from isaaclab.utils import configclass
from isaaclab.assets.articulation import ArticulationCfg # Import for robot details

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
# We might need to adjust the robot's PD gains for RMPflow if they are too stiff
# The RMPflow config often handles its own internal impedance/compliance.
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG # This cfg uses high PD gains, might need adjusting or replacing if RMPflow struggles.
from isaaclab_assets.robots.franka import RMP_CONFIG_PATH, RMP_JOINT_CONFIG_PATH # Paths to RMPflow config files

@configclass
class FrankaCubeLiftEnvCfg(joint_pos_env_cfg.FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # --- Robot Configuration ---
        # Set Franka as robot
        # We will use the standard Franka config. RMPflow will handle the joint-level control.
        # It's often better to have the robot's inherent PD gains set to a more compliant
        # value when using RMPflow's internal impedance control, but FRANKA_PANDA_HIGH_PD_CFG
        # might still work. If you encounter issues, consider replacing it with a cfg

        # that has lower PD gains or a custom ArticulationCfg with different ImplicitActuatorCfg settings.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # --- RMPflow Controller Setup ---
        # Add the RMPflow controller to the environment's controllers dictionary
        # This controller will generate joint position commands.
        self.controllers["rmpflow_controller"] = RMPFlowControllerCfg(
            prim_path=self.scene.robot.prim_path.replace("Robot", "robot"), # Match the actual prim path pattern
            robot_description_path=str(RMP_CONFIG_PATH),
            joint_internal_config_path=str(RMP_JOINT_CONFIG_PATH),
            end_effector_frame_name="panda_hand", # The end-effector link for RMPflow
            # Example nominal grasp orientation (optional, depends on task needs)
            # This is often used for aligning the gripper along a certain axis.
            # (0.707, 0.707, 0.0, 0.0) corresponds to a 90-degree rotation around X, making gripper point down.
            nominal_grasp_quat=(0.707, 0.707, 0.0, 0.0),
            use_target_pose_for_end_effector=True, # We'll send target poses (position + orientation)
            use_state_estimation=True, # Use Isaac Lab's internal state estimation
            num_envs=self.scene.num_envs, # Important for multi-environment setups
            dt=self.sim.dt,
            command_type="pose", # RMPflow expects a pose command (x,y,z,qx,qy,qz,qw)
            asset_name="robot", # Matches the asset name in the scene config
        )

        # --- Actions Configuration ---
        # Instead of DifferentialInverseKinematicsActionCfg, we now use
        # a JointPositionActionCfg. This action simply applies the joint positions
        # computed by the RMPflow controller.
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            # Since RMPflow directly outputs joint positions, this action type is suitable.
            # The values from RMPflow will be directly used as target positions for the actuators.
            # If RMPflow were to output joint velocities or efforts, you'd choose a different ActionCfg.
        )

        # --- Commands Configuration ---
        # The DifferentialIKController took a "pose" command directly.
        # Now, our `rmpflow_controller` is the one consuming the pose command.
        # We need to explicitly define the command that feeds into RMPflow.
        # This command will specify the desired end-effector pose for the robot.
        
            # IMPORTANT: We need to connect this command to the RMPflow controller.
            # Isaac Lab's system automatically handles the flow if the command
            # name matches what the controller expects.
            # RMPFlowControllerCfg doesn't have a direct "command_name" parameter
            # like actions often do. The standard way is that the RMPFlowController
            # internally generates actions based on an internal target/goal.
            # So, instead of a direct command, you would typically *set* the RMPflow's
            # target in the `_pre_physics_step` of your environment, similar to the previous example.
            # For a cube-lifting task, the target pose command would usually come from the task logic
            # (e.g., target cube position + desired gripper orientation).
        

      
        self.observations.policy.rmpflow_target = ObsTerm(func=generated_commands, params={"command_name": "pose_command"})


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # For play mode, also ensure the RMPflow controller is configured for num_envs
        self.controllers["rmpflow_controller"].num_envs = self.scene.num_envs
        self.controllers["rmpflow_controller"].dt = self.sim.dt

        # Ensure debug visualization for commands is active in play mode if desired
        self.commands["pose_command"].debug_vis = True