# Copyright (c) 2024-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
A collection of classes used to represent waypoints and trajectories.
"""
import asyncio
import inspect
import torch
from copy import deepcopy

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv
from isaaclab.managers import TerminationTermCfg


class Waypoint:
    """
    Represents a single desired 6-DoF waypoint, along with corresponding gripper actuation for this point.
    """

    def __init__(self, pose, gripper_action, noise=None):
        """
        Args:
            pose (torch.Tensor): 4x4 pose target for robot controller
            gripper_action (torch.Tensor): gripper action for robot controller
            noise (float or None): action noise amplitude to apply during execution at this timestep
                (for arm actions, not gripper actions)
        """
        self.pose = pose
        self.gripper_action = gripper_action
        self.noise = noise

    def __str__(self):
        """String representation of the waypoint."""
        return f"Waypoint:\n  Pose:\n{self.pose}\n"


class WaypointSequence:
    """
    Represents a sequence of Waypoint objects.
    """

    def __init__(self, sequence=None):
        """
        Args:
            sequence (list or None): if provided, should be a list of Waypoint objects
        """
        if sequence is None:
            self.sequence = []
        else:
            for waypoint in sequence:
                assert isinstance(waypoint, Waypoint)
            self.sequence = deepcopy(sequence)

    @classmethod
    def from_poses(cls, poses, gripper_actions, action_noise):
        """
        Instantiate a WaypointSequence object given a sequence of poses,
        gripper actions, and action noise.

        Args:
            poses (torch.Tensor): sequence of pose matrices of shape (T, 4, 4)
            gripper_actions (torch.Tensor): sequence of gripper actions
                that should be applied at each timestep of shape (T, D).
            action_noise (float or torch.Tensor): sequence of action noise
                magnitudes that should be applied at each timestep. If a
                single float is provided, the noise magnitude will be
                constant over the trajectory.
        """
        assert isinstance(action_noise, (float, torch.Tensor))

        # handle scalar to tensor conversion
        num_timesteps = poses.shape[0]
        if isinstance(action_noise, float):
            action_noise = action_noise * torch.ones((num_timesteps, 1), dtype=torch.float32)
        action_noise = action_noise.reshape(-1, 1)

        # make WaypointSequence instance
        sequence = [
            Waypoint(
                pose=poses[t],
                gripper_action=gripper_actions[t],
                noise=action_noise[t, 0],
            )
            for t in range(num_timesteps)
        ]
        return cls(sequence=sequence)

    def get_poses(self):
        poses = []
        for waypoint in self.sequence:
            poses.append(waypoint.pose[:2, 3])
        return poses

    def __len__(self):
        # length of sequence
        return len(self.sequence)

    def __getitem__(self, ind):
        """
        Returns waypoint at index.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.sequence[ind]

    def __add__(self, other):
        """
        Defines addition (concatenation) of sequences
        """
        return WaypointSequence(sequence=(self.sequence + other.sequence))

    def __str__(self):
        """Prints all waypoints in the sequence."""
        output = []
        for idx, waypoint in enumerate(self.sequence):
            output.append(f"Waypoint {idx}: {waypoint}")
        return "\n".join(output)

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return deepcopy(self.sequence[-1])

    def split(self, ind):
        """
        Splits this sequence into 2 pieces, the part up to time index @ind, and the
        rest. Returns 2 WaypointSequence objects.
        """
        seq_1 = self.sequence[:ind]
        seq_2 = self.sequence[ind:]
        return WaypointSequence(sequence=seq_1), WaypointSequence(sequence=seq_2)


class WaypointTrajectory:
    """
    A sequence of WaypointSequence objects that corresponds to a full 6-DoF trajectory.
    """

    def __init__(self):
        self.waypoint_sequences = []

    def __len__(self):
        # sum up length of all waypoint sequences
        return sum(len(s) for s in self.waypoint_sequences)

    def __getitem__(self, ind):
        """
        Returns waypoint at time index.

        Returns:
            waypoint (Waypoint instance)
        """
        assert len(self.waypoint_sequences) > 0
        assert (ind >= 0) and (ind < len(self))

        # find correct waypoint sequence we should index
        end_ind = 0
        for seq_ind in range(len(self.waypoint_sequences)):
            start_ind = end_ind
            end_ind += len(self.waypoint_sequences[seq_ind])
            if (ind >= start_ind) and (ind < end_ind):
                break

        # index within waypoint sequence
        return self.waypoint_sequences[seq_ind][ind - start_ind]

    @property
    def last_waypoint(self):
        """
        Return last waypoint in sequence.

        Returns:
            waypoint (Waypoint instance)
        """
        return self.waypoint_sequences[-1].last_waypoint

    def get_poses(self):
        poses = []
        for waypoint_sequence in self.waypoint_sequences:
            for waypoint in waypoint_sequence:
                poses.append(waypoint.pose[:2, 3])
        return poses

    def add_waypoint_sequence(self, sequence):
        """
        Directly append sequence to list (no interpolation).

        Args:
            sequence (WaypointSequence instance): sequence to add
        """
        assert isinstance(sequence, WaypointSequence)
        self.waypoint_sequences.append(sequence)

    def add_waypoint_sequence_for_target_pose(
        self,
        pose,
        gripper_action,
        num_steps,
        skip_interpolation=False,
        action_noise=0.0,
    ):
        """
        Adds a new waypoint sequence corresponding to a desired target pose. A new WaypointSequence
        will be constructed consisting of @num_steps intermediate Waypoint objects. These can either
        be constructed with linear interpolation from the last waypoint (default) or be a
        constant set of target poses (set @skip_interpolation to True).

        Args:
            pose (torch.Tensor): 4x4 target pose

            gripper_action (torch.Tensor): value for gripper action

            num_steps (int): number of action steps when trying to reach this waypoint. Will
                add intermediate linearly interpolated points between the last pose on this trajectory
                and the target pose, so that the total number of steps is @num_steps.

            skip_interpolation (bool): if True, keep the target pose fixed and repeat it @num_steps
                times instead of using linearly interpolated targets.

            action_noise (float): scale of random gaussian noise to add during action execution (e.g.
                when @execute is called)
        """
        if len(self.waypoint_sequences) == 0:
            assert skip_interpolation, "cannot interpolate since this is the first waypoint sequence"

        if skip_interpolation:
            # repeat the target @num_steps times
            assert num_steps is not None
            poses = pose.unsqueeze(0).repeat((num_steps, 1, 1))
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps, 1))
        else:
            # linearly interpolate between the last pose and the new waypoint
            last_waypoint = self.last_waypoint
            poses, num_steps_2 = PoseUtils.interpolate_poses(
                pose_1=last_waypoint.pose,
                pose_2=pose,
                num_steps=num_steps,
            )
            assert num_steps == num_steps_2
            gripper_actions = gripper_action.unsqueeze(0).repeat((num_steps + 2, 1))
            # make sure to skip the first element of the new path, which already exists on the current trajectory path
            poses = poses[1:]
            gripper_actions = gripper_actions[1:]

        # add waypoint sequence for this set of poses
        sequence = WaypointSequence.from_poses(
            poses=poses,
            gripper_actions=gripper_actions,
            action_noise=action_noise,
        )
        self.add_waypoint_sequence(sequence)

    def pop_first(self):
        """
        Removes first waypoint in first waypoint sequence and returns it. If the first waypoint
        sequence is now empty, it is also removed.

        Returns:
            waypoint (Waypoint instance)
        """
        first, rest = self.waypoint_sequences[0].split(1)
        if len(rest) == 0:
            # remove empty waypoint sequence
            self.waypoint_sequences = self.waypoint_sequences[1:]
        else:
            # update first waypoint sequence
            self.waypoint_sequences[0] = rest
        return first

    def merge(
        self,
        other,
        num_steps_interp=None,
        num_steps_fixed=None,
        action_noise=0.0,
    ):
        """
        Merge this trajectory with another (@other).

        Args:
            other (WaypointTrajectory object): the other trajectory to merge into this one

            num_steps_interp (int or None): if not None, add a waypoint sequence that interpolates
                between the end of the current trajectory and the start of @other

            num_steps_fixed (int or None): if not None, add a waypoint sequence that has constant
                target poses corresponding to the first target pose in @other

            action_noise (float): noise to use during the interpolation segment
        """
        need_interp = (num_steps_interp is not None) and (num_steps_interp > 0)
        need_fixed = (num_steps_fixed is not None) and (num_steps_fixed > 0)
        use_interpolation_segment = need_interp or need_fixed

        if use_interpolation_segment:
            # pop first element of other trajectory
            other_first = other.pop_first()

            # Get first target pose of other trajectory.
            # The interpolated segment will include this first element as its last point.
            target_for_interpolation = other_first[0]

            if need_interp:
                # interpolation segment
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_interp,
                    action_noise=action_noise,
                    skip_interpolation=False,
                )

            if need_fixed:
                # segment of constant target poses equal to @other's first target pose

                # account for the fact that we pop'd the first element of @other in anticipation of an interpolation segment
                num_steps_fixed_to_use = num_steps_fixed if need_interp else (num_steps_fixed + 1)
                self.add_waypoint_sequence_for_target_pose(
                    pose=target_for_interpolation.pose,
                    gripper_action=target_for_interpolation.gripper_action,
                    num_steps=num_steps_fixed_to_use,
                    action_noise=action_noise,
                    skip_interpolation=True,
                )

            # make sure to preserve noise from first element of other trajectory
            self.waypoint_sequences[-1][-1].noise = target_for_interpolation.noise

        # concatenate the trajectories
        self.waypoint_sequences += other.waypoint_sequences

    def get_full_sequence(self):
        """
        Returns the full sequence of waypoints in the trajectory.

        Returns:
            sequence (WaypointSequence instance)
        """
        return WaypointSequence(sequence=[waypoint for seq in self.waypoint_sequences for waypoint in seq.sequence])


class MultiWaypoint:
    """
    A collection of Waypoint objects for multiple end effectors in the environment.
    """

    def __init__(self, waypoints: dict[str, Waypoint]):
        """
        Args:
            waypoints (dict): a dictionary of waypionts of end effectors
        """
        self.waypoints = waypoints

    async def execute(
        self,
        env: ManagerBasedRLMimicEnv,
        success_term: TerminationTermCfg,
        env_id: int = 0,
        env_action_queue: asyncio.Queue | None = None,
    ):
        """
        Executes the multi-waypoint eef actions in the environment.

        Args:
            env: The environment to execute the multi-waypoint actions in.
            success_term: The termination term to check for task success.
            env_id: The environment ID to execute the multi-waypoint actions in.
            env_action_queue: The asyncio queue to put the action into.

        Returns:
            A dictionary containing the state, observation, action, and success of the multi-waypoint actions.
        """
        # current state
        state = env.scene.get_state(is_relative=True)

        # construct action from target poses and gripper actions
        target_eef_pose_dict = {eef_name: waypoint.pose for eef_name, waypoint in self.waypoints.items()}
        gripper_action_dict = {eef_name: waypoint.gripper_action for eef_name, waypoint in self.waypoints.items()}
        if "action_noise_dict" in inspect.signature(env.target_eef_pose_to_action).parameters:
            action_noise_dict = {eef_name: waypoint.noise for eef_name, waypoint in self.waypoints.items()}
            play_action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                action_noise_dict=action_noise_dict,
                env_id=env_id,
            )
        else:
            # calling user-defined env.target_eef_pose_to_action() with noise parameter is deprecated
            # (replaced by action_noise_dict)
            play_action = env.target_eef_pose_to_action(
                target_eef_pose_dict=target_eef_pose_dict,
                gripper_action_dict=gripper_action_dict,
                noise=max([waypoint.noise for waypoint in self.waypoints.values()]),
                env_id=env_id,
            )

        if play_action.dim() == 1:
            play_action = play_action.unsqueeze(0)  # Reshape with additional env dimension

        # step environment
        if env_action_queue is None:
            obs, _, _, _, _ = env.step(play_action)
        else:
            await env_action_queue.put((env_id, play_action[0]))
            await env_action_queue.join()
            obs = env.obs_buf

        success = bool(success_term.func(env, **success_term.params)[env_id])

        result = dict(
            states=[state],
            observations=[obs],
            actions=[play_action],
            success=success,
        )
        return result
