#!/usr/bin/env python3

import numpy as np
import os
import rospy
import sys
import torch
import time as timepkg
from mppi_rollouts.srv import MppiRollouts, MppiRolloutsResponse
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from functools import partial

# Add path to the FunctionEncoder package.
home = os.path.expanduser('~')
sys.path.append(f'{home}/FunctionEncoderMPPI')
from FunctionEncoder import FunctionEncoder, WarthogDataset, WarthogDatasetFull2D

# create a dataset
data_path = f"{home}/catkin_ws/src/mppi_rollouts/data/2025-03-22-10-35-47"
odom_path = f"{data_path}/warty-odom_processed_full2D-20Hz-CLEAN.csv"
cmdvel_path = f"{data_path}/warty-cmd_vel-CLEAN.csv"
dataset = WarthogDatasetFull2D(
    odom_csv = odom_path, 
    cmdvel_csv = cmdvel_path,
    n_examples = 1000
)

# Create a Function Encoder model. 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FunctionEncoder(
    input_size=dataset.input_size,  # del_time, states (yaw), controls (lin x, ang z vel)
    output_size=dataset.output_size, # next states (x, y, yaw)
    data_type=dataset.data_type,
    n_basis=11,
    model_type="FE_NeuralODE",
    method="least_squares",
    use_residuals_method=False
).to(device)

# Load pre-trained weights into the model. 
path = f'{home}/FunctionEncoderMPPI/logs/warthog_example/least_squares/shared_model'
model.load_state_dict(torch.load(f'{path}/2025-04-01_15-26-52_FE_NeuralODE/model.pth'))

# Sample data
example_xs, example_ys, _, _, _ = dataset.sample()
# Compute the coefficients for the function encoder using new data. 
with torch.no_grad():
    coeff, _ = model.compute_representation(example_xs, example_ys, method="least_squares")


# New kernel stuff
# train_data = dataset.train_data.cpu() # 90% of data
# test_data = dataset.test_data.cpu() # 10% of data

# X = train_data[:,:6]
# Y = train_data[:,6:]

# sigma = 0.2
# gamma = 1/ (2 * sigma ** 2)
# kernel = partial(rbf_kernel, gamma=gamma)
# Gx = kernel(X, X)
# G = Gx

# G[np.diag_indices_from(G)] += 1e-6

# G_inv = np.linalg.inv(G)

# def ker_model(x):
#     kx = kernel(X, x)

#     k = kx

#     pred = Y.T @ G_inv @ k
#     return pred


def handle_calc_rollouts(req):
    """ Calculates rollouts for MPPI. 
        INPUTS: 
            req.x0 = list encoding a 1-D array.
            req.U = list encoding a (req.K, req,T+1, req.M) array. 
            req.K = number of samples.
            req.T = number of integration time steps.
            req.M = number of control actions. 
            req.N = number of states. 
            req.dT = integration time step. 
        OUTPUTS:
            X = list encoder a (req.K, req.T+1. N) array. 
    """
    # print("[DEBUG]: Function Inputs") 
    # print("dT: ", req.dT)
    # print("K: ", req.K)
    # print("T: ", req.T)
    # print("M: ", req.M)
    # print("N: ", req.N)
    # print("len(x0): ", len(req.x0))
    # print("len(U): ", len(req.U))
    # print("-----")


    # Convert x0 and U into torch tensors. Make sure that the 
    # elements of U are loaded into the tensor correctly (so 
    # that the tensor matches the ArrayFire array in C++).
    x0 = torch.tensor(req.x0, dtype=torch.float32, device=device)
    U = torch.tensor(req.U, dtype=torch.float32, device=device).reshape(req.M, req.T+1, req.K).transpose(1, 2)

    # print("[DEBUG]: Lists Converted to Tensors")
    # print("Tensor x0: ", x0)
    # print("x0 shape: ", x0.shape)
    # print("Tensor U: ", U)
    # print("U shape: ", U.shape)

    # if (T == 0) {
    #     return tile(moddims(x0, 1, 1, N), K);
    # }

    # Define output tensor size to align with ArrayFire expectations. 
    X = torch.zeros((req.N, req.K, req.T + 1), dtype=torch.float32, device=device)

    # Define a tensor with integration steps for each rollout.
    time = torch.tensor([[req.dT]], device=device)
    time = time.repeat(1, req.K) 

    # Remove the previous control from the sequence.
    V = U[:, :, 1:]

    # print("[DEBUG]: Tensor Sizes for X, V, and time")
    # print("X: ", X.shape)
    # print("time: ", time.shape)
    # print("V: ", V.shape)
    # print("V: ", V)
    
    # Set initial state across all samples at t=0. Assign 
    # the initial state of each sample along X. 
    x0_reshaped = x0.reshape(req.N, 1, 1) 
    X[:, :, 0] = x0_reshaped.repeat(1, req.K, 1).squeeze(2) 

    # print("[DEBUG]: Setting the Initial States")
    # print("x0_reshaped: ", x0_reshaped)
    # print("x0_reshaped: ", x0_reshaped.shape)
    # print("x0_reshaped_repeated: ", x0_reshaped.repeat(1, req.K, 1))
    # print("x0_reshaped_repeated: ", x0_reshaped.repeat(1, req.K, 1).shape)
    # print("x0_reshaped_repeated_squeezed: ", x0_reshaped.repeat(1, req.K, 1).squeeze(2))
    # print("X with init cond: ", X)
    
    # Integrate over time. 
    for i in range(req.T):
        # Zero out the position and heading states in x0.
        zeros = torch.zeros(3, X.shape[1], device=device)
        # Concatenate all of the inputs.
        input = torch.cat((time, zeros, X[3:,:,i], V[:,:,i]), 0) # only use vels
        # if i == 0:
            # print("[DEBUG]: Concatenated Input Tensor")
            # print("input: ", input)
            # print("input: ", input.shape)

        # Transpose the inputs to be compatible w/ FEs
        input = input.transpose(0,1).unsqueeze(0)
        # if i == 0:
            # print("[DEBUG]: Transposed Input Tensor")
            # print("input: ", input)
            # print("input: ", input.shape)
        
        # Predict the change in pose (expressed in the body frame) and
        # the velocity of the next body frame. 
        with torch.no_grad():
            output = model.predict(input, coeff)
            # output = bicycle(input)
            # if i == 0:
            #     print("[DEBUG]: Output Tensor")
            #     print("output: ", output.shape)
            #     print("output: ", output.shape)
            #     print("output.squeeze(0).transpose(1,0): ", output.squeeze(0).transpose(1,0))

            # try using the kernel model
            # output = ker_model(torch.cat((time, X[3:,:,i], V[:,:,i])).transpose(1,0).cpu()).cuda().transpose(1,0).unsqueeze(0)
            # print("output: ", output.shape)

        # Transform the change in pose from the initial body frame
        # to the inertial frame.  
        del_states_I = body_to_inertial(
            X[:3,:,i].transpose(1,0), 
            output.squeeze(0)[:,:3]
        )
        
        # Translate the change in pose in the inertial frame
        # to get the pose of the next body frame (expressed in I).
        X[:3,:,i+1] =  del_states_I.transpose(1,0) + X[:3,:,i]

        # Add the initial velocity to the change in velocity to
        # get the velocity at the next frame expr. in the current frame. 
        next_vel_Bi = output.squeeze(0)[:,3:] + X[3:,:,i].transpose(0,1)

        # Rotate the next velocity from Bi to I and then to Bf.
        next_vel_I = body_to_inertial(
            X[:3,:,i].transpose(1,0),  # orientation at t of Bi relative to I
            next_vel_Bi # velocity of Bf relative to I expressed in Bi 
        )
        next_vel_Bf = inertial_to_body(
            X[:3,:,i+1].transpose(1,0),  # orientation at t+1 of Bf relative to I
            next_vel_I,    # (K, 3) matrix of vectors in the inertial frame
        )

        # Update the velocity of the next frame (expressed in next frame).
        X[3:,:,i+1] = next_vel_Bf.transpose(1,0)

    # print("[DEBUG]: Double Check the final shape of X")       
    # print("X after = ", X.shape)
    # Correct the angle.
    # X_np = output.squeeze(0).cpu().numpy()
    # plt.scatter(X_np[:,0], X_np[:,1], c='blue', alpha=0.6, edgecolors='k')
    # plt.xlabel("X-axis")
    # plt.ylabel("Y-axis")
    # plt.title("Scatter Plot of del_states_I final time")
    # plt.grid(True)
    # plt.xlim(-0.03, 0.1)
    # plt.ylim(-0.06, 0.03)
    # plt.savefig(f'text-{round(timepkg.time()*1e6)}.png')
    # plt.clf()

    # Wrap the angles to between -pi and pi.
    X[2,:,:] = wrap_to_pi(X[2,:,:])

    # Flatten the trajectories to a list in ROW major order so
    # that it is easy to unpack into an ArrayFire Array in C++. 
    X_flat = X.permute(0, 2, 1).contiguous().flatten().tolist()

    # print("[DEBUG]: Check that the tensor was flattened correctly")
    # print("X_flat: ", X_flat)
    # print("-----------------------")
    return MppiRolloutsResponse(X_flat)

def bicycle(input):
    """Same algorithm used in Phoenix to calculate skid_steer rollouts."""
    inputT = input[0].transpose(0,1)
    T, x, y, yaw, xvel, zvel = inputT
    dx1 = (xvel*torch.cos(yaw)*T).unsqueeze(0)
    dx2 = (xvel*torch.sin(yaw)*T).unsqueeze(0)
    dx3 = (zvel*T).unsqueeze(0)
    output = torch.cat((dx1,dx2,dx3), 0).transpose(1,0).unsqueeze(0)
    return output  

def dummy(input):
    """Used to debug the rollouts. Let's you set change in states to constant."""
    inputT = input[0].transpose(0,1)
    T, x, y, yaw, xvel, zvel = inputT
    dx1 = 0.0 + torch.zeros_like(x).unsqueeze(0)
    dx2 = 0.0 + torch.zeros_like(x).unsqueeze(0)
    dx3 = 2.0 + torch.zeros_like(x).unsqueeze(0)
    output = torch.cat((dx1,dx2,dx3), 0).transpose(1,0).unsqueeze(0)
    return output  

def wrap_to_pi(theta):
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi

def body_to_inertial(
        bIMat,    # (K, 3) matrix of body frame origin vectors in the inertial frame
        xBMat,    # (K, 3) matrix of vectors in the body frame
):
    """ Transforms body frame vectors into the inertial frame. """

    # Extract the rotation angles. Ensure separate memory by cloning.
    yaws = bIMat[:,2].clone()  

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0], device=device)
    ones = torch.ones(yaws.shape[0], device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, -sin_yaw, zeros], dim=1),
        torch.stack([sin_yaw, cos_yaw, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (K, 3, 3)

    # Perform batch matrix-vector multiplication
    xIMat = torch.bmm(R, xBMat.unsqueeze(-1)).squeeze(-1) #+ bIMat # Shape: (N, 3)
    return xIMat

def inertial_to_body(
        bIMat,    # (K, 3) matrix of body frame origin vectors in the inertial frame
        xIMat,    # (K, 3) matrix of vectors in the inertial frame
):
    """ Transforms inertial frame vectors into the body frame. """

    # Extract the rotation angles. Ensure separate memory by cloning.
    yaws = bIMat[:,2].clone()  

    cos_yaw = torch.cos(yaws)
    sin_yaw = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0], device=device)
    ones = torch.ones(yaws.shape[0], device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([cos_yaw, sin_yaw, zeros], dim=1),
        torch.stack([-sin_yaw, cos_yaw, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (K, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.bmm(R, xIMat.unsqueeze(-1)).squeeze(-1) #+ bIMat # Shape: (N, 3)
    return xBMat


if __name__ == "__main__":
    # Initialize the ROS server.
    rospy.init_node('rollouts_server')
    service = rospy.Service('warty/calc_rollouts', MppiRollouts, handle_calc_rollouts)
    rospy.loginfo("Service 'calc_rollouts' ready to calculate MPPI rollouts.")
    rospy.spin()
