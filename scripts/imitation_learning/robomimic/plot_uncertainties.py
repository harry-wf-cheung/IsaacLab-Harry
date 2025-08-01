import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from isaaclab.utils.logging_helper import LoggingHelper

def get_subtask_timesteps():
    """
    for each rollout, return the timesteps that each subtask was completed
    """
    subtask_codes = {
        'APPR': '0',
        'GRASP': '1',
        'LIFT': '2',
        'FINISH': '4'
    }
    with open(LoggingHelper().namefile, 'r') as file:
        lines = file.readlines()

    rollouts = [] 
    rollout_number = 0
    current_timestep = -1
    for line in lines:
        line = line.strip().split(':')
        
        if line[0] == 'X':
            rollout_number += 1
            current_timestep = -1

        if line[0] == 'S':
            current_timestep += 1
    
        # APPR subtask was completed successfully
        if line[0] == subtask_codes['APPR'] and line[2] == 'TRUE' and 'APPR' not in [subtask_name for rol_num, subtask_name, _ in rollouts if rol_num == rollout_number]:
            rollouts.append([rollout_number, 'APPR', current_timestep])

        if line[0] == subtask_codes['GRASP'] and line[2] == 'TRUE' and 'GRASP' not in [subtask_name for rol_num, subtask_name, _ in rollouts if rol_num == rollout_number]:
            rollouts.append([rollout_number, 'GRASP', current_timestep])
        
        if line[0] == subtask_codes['LIFT'] and line[2] == 'TRUE' and 'LIFT' not in [subtask_name for rol_num, subtask_name, _ in rollouts if rol_num == rollout_number]:
            rollouts.append([rollout_number, 'LIFT', current_timestep])

        if line[0] == subtask_codes['FINISH'] and line[2] == 'TRUE' and 'FINISH' not in [subtask_name for rol_num, subtask_name, _ in rollouts if rol_num == rollout_number]:
            rollouts.append([rollout_number, 'FINISH', current_timestep])

    # fix incorrect rollout numbers
    prev_rollout_number = 1
    rollout_number = 1
    for i, rollout in enumerate(rollouts):
        og_rollout_num = rollout[0]
        if rollout[0] != prev_rollout_number:
            rollout_number += 1
        rollout[0] = rollout_number
        prev_rollout_number = og_rollout_num
    

    subtask_timesteps = {(rol_num, subtask) : timestep for rol_num, subtask, timestep in rollouts}
    print(subtask_timesteps)

    return subtask_timesteps
    


def load_uncertainties_file(file):
    rollouts = []
    rollout = []
    labels = []
    all_uncertainties = []
    all_timesteps = []

    with open(file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            timestep = int(line[0])
            uncertainties = [float(f) for f in line[1:8]]
            label = line[-1]

            if timestep == 0: 
                if rollout:
                    rollouts.append(rollout)
                    rollout = []
                labels.append(True if label == 'True' else False)

            if timestep % 1 == 0:
                rollout.append((timestep, uncertainties))
                all_timesteps.append(timestep)
                all_uncertainties.append(uncertainties)

    # Add final rollout
    if rollout:
        rollouts.append(rollout)

    return rollouts, labels, all_uncertainties, all_timesteps


def plot_rollouts(rollouts, labels, results_path, parameters, duration=400, joints_to_plot=[_ for _ in range(7)]):
    # define a window for each joint
    windows = {joint_num: deque(maxlen=parameters['window_size']) for joint_num in joints_to_plot} 
  

    successful_rollout_means, failed_rollout_means = [], []
    num_joints = 7
    for i, rollout in enumerate(rollouts):
        rollout_timesteps = [t for t, _ in rollout]
        rollout_uncertainties = [unc for _, unc in rollout]

        # define a set of unsafe timesteps for the current rollout
        unsafe_timesteps = []

        # transpose the list of uncertainties so each sublist is for one joint
        rollout_uncertainties_per_joint = list(zip(*rollout_uncertainties))

        plt.figure(figsize=(10, 6))
        plt.ylim(bottom=0)
        plt.ylim(top=1.0)
        for joint_num in range(num_joints):
            if joint_num in joints_to_plot:
                rollout_timesteps_to_plot = list(rollout_timesteps[:duration])
                rollout_uncertainties_per_joint_to_plot = list(rollout_uncertainties_per_joint[joint_num][:duration])
                
                # pad timesteps
                j = len(rollout_timesteps_to_plot)
                while len(rollout_timesteps_to_plot) < duration:
                    rollout_timesteps_to_plot.append(j + 1)
                    j += 1
                # pad uncertainties
                while len(rollout_uncertainties_per_joint_to_plot) < duration:
                    rollout_uncertainties_per_joint_to_plot.append(0)

                plotted_line = plt.plot(rollout_timesteps_to_plot, rollout_uncertainties_per_joint_to_plot, alpha=0.3, label=f"Joint {joint_num}")
                mean = np.mean(rollout_uncertainties_per_joint_to_plot)
                mean = [mean for _ in range(len(rollout_timesteps_to_plot))]
                plt.plot(rollout_timesteps_to_plot, mean, label=f"Joint {joint_num} mean uncertainty", linestyle='--', alpha=0.4, color=plotted_line[0].get_color()) 

                if labels[i] == True:
                    successful_rollout_means.append(mean)
                else:
                    failed_rollout_means.append(mean)

               
                # simulate the real-time window (uncertainty values come in one timestep at a time...)
                for timestep, unc in zip(rollout_timesteps_to_plot, rollout_uncertainties_per_joint_to_plot):
                    #windows[joint_num].append((timestep, unc))
                    windows[joint_num].append((timestep, unc))
                    unc_threshold = parameters['unc_threshold']
                    peaks = sum(1 for _, curr_unc in windows[joint_num] if curr_unc > unc_threshold)
                    if peaks > parameters['max_peaks']:
                        #print(f"number of peaks in window: {peaks} for rollout {i}")
                        # backtrack to get the timesteps for the entire window - so it says something like "there is danger in this window"
                        for unsafe_timestep, unsafe_uncertainty in windows[joint_num]:
                            unsafe_timesteps.append((unsafe_timestep, unsafe_uncertainty))
            

                # plot the unsafe windows 
                # Extract x and y
                unsafe_x = [t for t, _ in unsafe_timesteps]
                unsafe_y = [u for _, u in unsafe_timesteps]

                # sort by timestep to ensure proper line order
                unsafe_sorted = sorted(zip(unsafe_x, unsafe_y))
                unsafe_x, unsafe_y = zip(*unsafe_sorted) if unsafe_sorted else ([], [])
                detected_unsafe_windows = int(len(unsafe_x)/parameters['window_size'])
                # plt.plot(unsafe_x, unsafe_y, color='red', linewidth=1, linestyle='-', zorder=2, label=f"Detected Unsafe Window (Joint {joint_num}). Total unsafe windows detected: {detected_unsafe_windows}")
                
                for window_size in [10, 50, 100]:
                    # Plot average uncertainty over each window
                    window_avg_timesteps = []
                    window_avg_values = []
                    # temp_window = deque(maxlen=parameters[window_size])
                    temp_window = deque(maxlen=window_size)
                    for timestep, unc in zip(rollout_timesteps_to_plot, rollout_uncertainties_per_joint_to_plot):
                        temp_window.append((timestep, unc))
                        # if len(temp_window) == parameters[window_size]:
                        if len(temp_window) == window_size:
                            # avg_unc = sum(u for _, u in temp_window) / parameters[window_size]
                            avg_unc = sum(u for _, u in temp_window) / window_size
                            end_timestep = temp_window[-1][0]
                            window_avg_timesteps.append(end_timestep)
                            window_avg_values.append(avg_unc)

                    plt.plot(window_avg_timesteps, window_avg_values, zorder=3, label=f"Sliding Window Avg (Joint {joint_num}) for Window Size {window_size}")

                # Reset the original window buffer
                windows[joint_num].clear()

        title = f"Uncertainty for joint {joint_num} for: Rollout {i}, Model: {model_arch}, Task: {task}, {'Success' if labels[i] == True else 'Failure'}"
        plt.title(title)
        plt.xlabel('Timestep')
        plt.ylabel('Uncertainty Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{results_path}{title}.png")
        plt.close()

    for joint_num in range(num_joints):

        if joint_num in joints_to_plot:
            success_current_joint_means = [mean[joint_num] for mean in successful_rollout_means]
            failed_current_joint_means = [mean[joint_num] for mean in failed_rollout_means]
            
            current_joint_successful_mean = sum(success_current_joint_means) / len(success_current_joint_means)
            current_joint_failed_mean = sum(failed_current_joint_means) / len(failed_current_joint_means)

            print("======================")
            print(f"Average mean over all successful rollouts for joint {joint_num}: {current_joint_successful_mean}")
            print(f"Average mean over all failed rollouts for joint {joint_num}: {current_joint_failed_mean}")
            print("======================")
            
            plt.figure(figsize=(10, 6))
            plt.ylim(bottom=0)
            plt.ylim(top=0.5)

            plt.bar(['Success', 'Failure'], [current_joint_successful_mean, current_joint_failed_mean], 
                    label=["{:.4f}".format(current_joint_successful_mean), "{:.4f}".format(current_joint_failed_mean)], color=['green', 'red'])
            title = f"Average Mean Over All Successful Rollouts for Joint {joint_num}"
            plt.title(title)
            plt.xlabel('Task Result')
            plt.ylabel('Average Uncertainty Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{results_path}{title}.png")
            plt.close()



def plot_joint_uncertainty(all_timesteps, all_uncertainties, labels):
    num_joints = 7
    label_idx = 0
    success_uncertainties = [[] for _ in range(num_joints)]
    failure_uncertainties = [[] for _ in range(num_joints)]
    used_timesteps = []

    for i, (timestep, uncertainties) in enumerate(zip(all_timesteps, all_uncertainties)):
        # Use the current rollout label
        if i > 0 and all_timesteps[i] < all_timesteps[i - 1]:
            label_idx += 1  # New rollout started

        label = labels[label_idx]
        used_timesteps.append(timestep)

        for joint_num in range(num_joints):
            if label:
                success_uncertainties[joint_num].append(uncertainties[joint_num])
                failure_uncertainties[joint_num].append(np.nan)
            else:
                success_uncertainties[joint_num].append(np.nan)
                failure_uncertainties[joint_num].append(uncertainties[joint_num])

    # Plot
    for joint_num in range(num_joints):
        plt.figure(figsize=(10, 6))
        plt.plot(used_timesteps, success_uncertainties[joint_num], label="Task Success", color='green')
        plt.plot(used_timesteps, failure_uncertainties[joint_num], label="Task Failure", color='red')

        plt.title(f"Uncertainties for Joint {joint_num}")
        plt.xlabel('Timestep')
        plt.ylabel('Uncertainty Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"./docs/training_data/uncertainty_joint_{joint_num}_plot_stack_cube.png")
        plt.close()


# model_arch = "BC_RNN_GMM"
# task = "stack_cube" # stack_cube or pick_place
# model_name = f"model2"

# uncertainties_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/{model_name}/uncertainties.txt"

model_arch = "BC_RNN_GMM"
task = "pick_place" # stack_cube or pick_place
model_name = f"ensemble"
#uncertainties_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/ensemble/Isaac-Stack-Cube-Franka-IK-Rel-v0/{model_name}/uncertainties.txt"

uncertainties_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/ensemble/uncertainties3.txt"
results_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/{model_name}/" 

# uncertainties_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/ensemble/Isaac-Stack-Cube-Franka-IK-Rel-v0/video_plots/uncertainties.txt"
# results_path = f"./docs/training_data/{task}/uncertainty_rollout_{task}/{model_name}/Isaac-Stack-Cube-Franka-IK-Rel-v0/video_plots" 



parameters = {
    'stack_cube': {
        'unc_threshold': 0.1,
        'max_peaks': 20,
        'window_size': 40
    }, 
    'pick_place': {
        'unc_threshold': 0.05,
        'max_peaks': 10,
        'window_size': 60
    }
}


subtask_timesteps = get_subtask_timesteps()

rollouts, labels, all_uncertainties, all_timesteps = load_uncertainties_file(uncertainties_path)
plot_rollouts(rollouts, labels, results_path, parameters[task], duration=800, joints_to_plot=[6])





# plot another graph that says something like - "for the window at timestep x, the average uncertainty inside the window was y"
