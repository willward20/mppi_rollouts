#!/usr/bin/env python3

import numpy as np
import os
import rospy
import sys
import torch
from mppi_rollouts.srv import MppiRollouts, MppiRolloutsResponse
from terrain_adaptation.data.load_data import load_scenes, PhoenixDataset
from terrain_adaptation.models.neural_ode import load_model as load_model_ode
from terrain_adaptation.models.function_encoder import load_model as load_model_fe

home = os.path.expanduser('~')

# Load all scene data as a dictionary
indices = [1]
scene_data = load_scenes(indices)

# Get the scene the robot is deployed on
inputs = [scene_data[f"scene{i}"][0] for i in indices]
targets = [scene_data[f"scene{i}"][1] for i in indices]

dataset = PhoenixDataset(
    inputs, targets, n_example_points=100, n_points=1000
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
dataloader_iter = iter(dataloader)

# Load the model.
model_type = "function_encoder" # neural_ode or function_encoder
device = "cuda" if torch.cuda.is_available() else "cpu"
path = f'{home}/function-encoder-terrain-adaptation/{model_type}_model_interp_extrap.pth'

if model_type == "neural_ode":
    model = load_model_ode(
        device = device,
        path = path
    )
elif model_type == "function_encoder":
    model = load_model_fe(
        device = device,
        path = path
    )

if model_type == "function_encoder":
    # Get a batch of data.
    batch = next(dataloader_iter)

    # Compute the coefficients.
    xs, dt, ys, example_xs, example_dt, example_ys = batch
    xs = xs.to(device)
    dt = dt.to(device)
    ys = ys.to(device)
    example_xs = example_xs.to(device)
    example_dt = example_dt.to(device)
    example_ys = example_ys.to(device)
    coefficients, _ = model.compute_coefficients((example_xs, example_dt), example_ys)

# print("done")
# exit()




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
        input = torch.cat((zeros, X[3:,:,i], V[:,:,i]), 0) # only use vels
        # Transpose the inputs to be compatible w/ FEs
        input = input.transpose(0,1).unsqueeze(0)
        # if i == 0:
        #     print("[DEBUG]: Transposed Concatenated Input Tensor")
        #     print("input: ", input)
        #     print("input: ", input.shape)
        
        # Predict the change in pose (expressed in the body frame) and
        # the velocity of the next body frame. 
        with torch.no_grad():
            if model_type == "function_encoder":
                output = model((input, time), coefficients=coefficients) # xs = [bs, num_pts, 8], dt = [bs, num_pts]
            elif model_type == "neural_ode":
                output = model((input, time))

            # output = model.predict(input, coeff) # for function encoder models
            # output = model.model.forward(input).squeeze(-1) # for singal network models
            # print(output.shape)
            # if i == 0:
            #     print("[DEBUG]: Output Tensor")
            #     print("output: ", output.shape)
            #     print("output: ", output.shape)
            #     print("output.squeeze(0).transpose(1,0): ", output.squeeze(0).transpose(1,0))

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

    # Wrap the angles to between -pi and pi.
    X[2,:,:] = wrap_to_pi(X[2,:,:])

    # Flatten the trajectories to a list in ROW major order so
    # that it is easy to unpack into an ArrayFire Array in C++. 
    X_flat = X.permute(0, 2, 1).contiguous().flatten().tolist()

    # print("[DEBUG]: Check that the tensor was flattened correctly")
    # print("X_flat: ", X_flat)
    # print("-----------------------")
    return MppiRolloutsResponse(X_flat)  


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
