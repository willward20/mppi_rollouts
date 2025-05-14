from bagpy import bagreader
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

def process_csvs(odom_array, cmdvel_array):

    # Unwrap the yaw meaurements.
    odom_array[:, 3] = np.unwrap(odom_array[:,3])
    # Convert numpy array to torch tensor. 
    odom_tensor = torch.tensor(odom_array).to(torch.float32)
    # Convert the time stamps column to changes in time. 
    odom_tensor[:-1, 0] = odom_tensor[1:, 0] - odom_tensor[:-1, 0]

    # Get the change in position/heading at each time expressed in the inertial frame. 
    del_pose_I = odom_tensor[1:, 1:4] - odom_tensor[:-1, 1:4]
    # Rotate the position/heading data from the inertial
    # frame and into the initial body frame.  
    del_pose_Bi = inertial_to_body(
        odom_tensor[:-1, 3], # don't need the final data point
        del_pose_I
    )

    # Get the velcoities at the next state in the final body frame.
    # For now, I'm not transforming because that's complicated and
    # may not be neccessary.  
    vel_next_Bf = odom_tensor[1:,4:]

    # Rotate the velocities of the next body frame into the inertial frame. 
    vel_next_I = vel_body_to_inertial(
        odom_tensor[1:, 3],  # yaw angles of the next body frame relative to I 
        vel_next_Bf, # velocities of the next body frame
    )
    # Rotate the velocities from the inertial frame to the initial body frame. 
    vel_next_Bi = inertial_to_body(
        odom_tensor[:-1, 3],  # yaw angles of the initial frame relative to I
        vel_next_I, # velocities of the next body frame relative to I, expressed in I
    )
    # These two rotations gives me the velocity of the next body frame
    # relative to the inertial frame but express in the initial body frame. 
    # Now, take the difference between the velocities in Bi. 
    del_vel_Bi = vel_next_Bi - odom_tensor[:-1,4:]


    # Zero out the initial pose to put it into the body frame.
    odom_tensor[:,1:4] = torch.zeros(odom_tensor.shape[0], 3)
    # Remove the bottom row from the data. 
    odom_tensor = odom_tensor[:-1,:]


    # Find the closest cmd_vel timestamps to the odom timestamps. DON'T NEED THIS. HAPPENS BEFORE PROCESSING.
    indices = np.abs(cmdvel_array[:, 0, None] - odom_array[:, 0]).argmin(axis=0)
    # Filter the cmd_vel data to only keep closest values. DON'T NEED THIS. HAPPENS BEFORE PROCESSING.
    cmdvel_array = cmdvel_array[indices,:]
    # Convert numpy array to torch tensor. 
    cmdvel_tensor = torch.tensor(cmdvel_array).to(torch.float32)
    # Remove the time stamps from the cmd_vel data.  
    cmdvel_tensor = cmdvel_tensor[:, [1,2]]
    # Remove the bottom row from the data. 
    cmdvel_tensor = cmdvel_tensor[:-1,:]

    # Append states, inputs, and changes in states to one tensor. 
    return torch.cat((odom_tensor, cmdvel_tensor, del_pose_Bi, del_vel_Bi), dim=1)

def inertial_to_body(
        yaws,    # (T x 1) rotation of the body frame w.r.t inertial frame
        xIMat,    # (T x 3) matrix of vectors in the inertial frame
):
    """ Transforms inertial frame vectors into the body frame. """

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0])
    ones = torch.ones(yaws.shape[0])

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, sin_yaw, zeros], dim=1),
        torch.stack([-sin_yaw, cos_yaw, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (N, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.bmm(R, (xIMat).unsqueeze(-1)).squeeze(-1)  # Shape: (T, 3)
    return xBMat

def vel_body_to_inertial(
        yaws,    # (T x 1) rotation of the body frame w.r.t. inertial frame. 
        xIMat,    # (T x 3) matrix of vectors in the body frame
):
    """ Transforms the velocity of the body frame vectors into the inertial frame. """

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0])
    ones = torch.ones(yaws.shape[0])

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, zeros], dim=1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (N, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.bmm(R, (xIMat).unsqueeze(-1)).squeeze(-1)  # Shape: (T, 3)
    return xBMat

# Set the torch seed
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Set the random seed.
set_seed(42)

# Set the flag for plotting
FLAG = True

# Set the path to the CSV files.
home = os.path.expanduser('~')
name = '2025-05-14-12-29-42'
data_path = f"{home}/catkin_ws/src/mppi_rollouts/data/{name}"
odom_path = f"{data_path}/warty-odom_processed_full2D.csv"
cmdvel_path = f"{data_path}/warty-cmd_vel.csv"

# Read the bag files and convert to CSV. 
b = bagreader(f'{data_path}.bag')
bmesg = b.message_by_topic('/warty/odom_processed_full2D')
bmesg = b.message_by_topic('/warty/cmd_vel')

# Load odom CSV into a pd data frame. 
odom_df = pd.read_csv(odom_path)
cmdvel_df = pd.read_csv(cmdvel_path)

# Remove the unecessary command velocity columns.
cmdvel_df = cmdvel_df.drop(columns=['linear.y', 'linear.z', 'angular.x', 'angular.y'])

# Remove the time column added by rosbag record. 
odom_df = odom_df.drop(columns=['Time'])

# Sample the odom data at 10 Hz/
sampling_interval = 0.1  # 100 ms for 10 Hz

# Keep the first row, and then only rows at least 0.05s after the last kept row
filtered_rows = [odom_df.iloc[0]]
last_time = odom_df.iloc[0]['time']
for _, row in odom_df.iloc[1:].iterrows():
    if row['time'] - last_time >= sampling_interval:
        filtered_rows.append(row)
        last_time = row['time']

odom_df = pd.DataFrame(filtered_rows)

# Perform additional conversions
data = process_csvs(odom_df.to_numpy(), cmdvel_df.to_numpy())

# Shuffle indices.
num_samples = data.shape[0]
indices = torch.randperm(num_samples)  # Generate a random permutation of indices

# Split indices into 90% train and 10% test.
split = round(num_samples * 0.9)
train_indices = indices[:split]
test_indices = indices[split:]

# Sample the train/test data.
train_data = data[train_indices, :]
test_data = data[test_indices, :]

# Save the train and test data.
np.savetxt(f"{data_path}/train_data.csv", train_data.numpy(), delimiter=",")
np.savetxt(f"{data_path}/test_data.csv", test_data.numpy(), delimiter=",")

# Scatter plot 1: del_y vs del_x
# plt.figure(figsize=(6, 5))
# plt.scatter(train_data[:, 9], train_data[:, 10])
# plt.xlabel("del_x")
# plt.ylabel("del_y")
# plt.title("Scatter Plot: del_y vs del_x")
# plt.grid(True)
# plt.axis("equal")
# plt.show()

# Use merge_asof to find closest cmd_vel row for each odom time
# cmdvel_df = pd.merge_asof(
#     odom_df[['time']],                       # Just time column
#     cmdvel_df.rename(columns={'Time': 'time'}),
#     on='time',
#     direction='nearest'
# )

# # Shuffle the rows of the dataset. 
# odom_shuffled = odom_df.sample(frac=1, random_state=42).reset_index(drop=True)
# cmdvel_shuffled = cmdvel_df.sample(frac=1, random_state=42).reset_index(drop=True)

# # Split 90% train, 10% test
# split_idx = int(0.9 * len(odom_shuffled))
# odom_train = odom_shuffled[:split_idx]
# odom_test = odom_shuffled[split_idx:]
# cmdvel_train = cmdvel_shuffled[:split_idx]
# cmdvel_test = cmdvel_shuffled[split_idx:]

# Save the dataframes to CSV files without the column titles.
# odom_train.to_csv(f"{data_path}/warty-odom_processed_full2D-PROCESSED-TRAIN.csv", index=False, header=False)
# odom_test.to_csv(f"{data_path}/warty-odom_processed_full2D-PROCESSED-TEST.csv", index=False, header=False)
# cmdvel_train.to_csv(f"{data_path}/warty-cmd_vel-PROCESSED-TRAIN.csv", index=False, header=False)
# cmdvel_test.to_csv(f"{data_path}/warty-cmd_vel-PROCESSED-TEST.csv", index=False, header=False)







if FLAG == True:

    # Plot the cmd_vel angular velocity as a function of the linear velocity.
    plt.figure()
    plt.scatter(cmdvel_df['linear.x'], cmdvel_df['angular.z'], s=1)
    plt.xlabel('Linear X Velocity')
    plt.ylabel('Angular Z Velocity')
    plt.title('Commanded Velocities')
    plt.grid()
    plt.savefig(f'{data_path}/cmd_vel.png')
    plt.close()

    # Plot the odom y velocity as a function of the x velocity.
    plt.figure()
    plt.scatter(odom_df['xVel'], odom_df['yVel'], s=1)
    plt.xlabel('X Velocity')
    plt.ylabel('Y Velocity')
    plt.title('Odom Linear Y Velocity vs Linear X Velocity')
    plt.grid()
    plt.savefig(f'{data_path}/odom_x_vel_y_vel.png')
    plt.close()

    # Plot the odom angular velocity as a function of time.
    plt.figure()
    plt.plot(odom_df['time']/60, odom_df['zAngVel'], linewidth=1)
    plt.xlabel('Time [mins]')
    plt.ylabel('Z Angular Velocity')
    plt.title('Odom Angular Velocity vs Time')
    plt.grid()
    plt.savefig(f'{data_path}/odom_ang_vel_time.png')
    plt.close()

    # Plot the odom angular velocity as a function of the linear velocity.
    plt.figure()
    plt.scatter(odom_df['xVel'], odom_df['zAngVel'], s=1)
    plt.xlabel('X Velocity')
    plt.ylabel('Z Angular Velocity')
    plt.title('Odom Angular Velocity vs Linear X Velocity')
    plt.grid()
    plt.savefig(f'{data_path}/odom_ang_vel_x_vel.png')
    plt.close()

    # Plot the y velocity as a function of the angular z velocity.
    plt.figure()
    plt.scatter(odom_df['zAngVel'], odom_df['yVel'], s=1)
    plt.xlabel('Z Angular Velocity')
    plt.ylabel('Y Velocity')
    plt.title('Odom Linear Y Velocity vs Angular Z Velocity')
    plt.grid()
    plt.savefig(f'{data_path}/odom_ang_vel_y_vel.png')
    plt.close()

    # Plot the command velocities as a function of time.
    plt.figure()
    plt.plot(cmdvel_df['Time']/60, cmdvel_df['linear.x'], linewidth=1)
    plt.plot(cmdvel_df['Time']/60, cmdvel_df['angular.z'], linewidth=1)
    plt.xlabel('Time [mins]')
    plt.ylabel('Velocity')
    plt.title('Commanded Velocities vs Time')
    plt.legend(['Linear X Velocity', 'Angular Z Velocity'])
    plt.grid()
    plt.savefig(f'{data_path}/cmd_vel_time.png')
    plt.close()

    # Plot the odom linear velocities as functions of time.
    plt.figure()
    plt.plot(odom_df['time']/60, odom_df['xVel'], linewidth=1)
    plt.plot(odom_df['time']/60, odom_df['yVel'], linewidth=1)
    plt.xlabel('Time [mins]')
    plt.ylabel('Velocity')
    plt.title('Odom Linear Velocities vs Time')
    plt.legend(['X Velocity', 'Y Velocity'])
    plt.grid()
    plt.savefig(f'{data_path}/odom_vel_time.png')
    plt.close()

    # # Plot the odom linear x velocity as a function of the commanded linear x velocity.
    # plt.figure()
    # plt.scatter(cmdvel_df['linear.x'], odom_df['xVel'], s=1)
    # plt.plot([cmdvel_df['linear.x'].min(), cmdvel_df['linear.x'].max()], 
    #         [cmdvel_df['linear.x'].min(), cmdvel_df['linear.x'].max()], 
    #         color='black', linewidth=1, linestyle='--', label='y=x')
    # plt.xlabel('Commanded Linear X Velocity')
    # plt.ylabel('Odom Linear X Velocity')
    # plt.title('Odom Linear X Velocity vs Commanded Linear X Velocity')
    # plt.grid()
    # plt.legend()
    # plt.axis('equal')
    # plt.savefig(f'{data_path}/odom_x_vel_cmd_x_vel.png')
    # plt.close()

    # # Plot the odom angular velocity as a function of the commanded angular velocity.
    # plt.figure()
    # plt.scatter(cmdvel_df['angular.z'], odom_df['zAngVel'], s=1)
    # plt.plot([cmdvel_df['angular.z'].min(), cmdvel_df['angular.z'].max()], 
    #         [cmdvel_df['angular.z'].min(), cmdvel_df['angular.z'].max()], 
    #         color='black', linewidth=1, linestyle='--', label='y=x')
    # plt.xlabel('Commanded Angular Z Velocity')
    # plt.ylabel('Odom Angular Z Velocity')
    # plt.title('Odom Angular Z Velocity vs Commanded Angular Z Velocity')
    # plt.grid()
    # plt.legend()
    # plt.axis('equal')
    # plt.savefig(f'{data_path}/odom_ang_vel_cmd_ang_vel.png')
    # plt.close()
