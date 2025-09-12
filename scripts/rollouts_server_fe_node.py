#!/usr/bin/env python3

import codecs
import csv
import os
import threading
import datetime
import time
import numpy as np
import rospy
import torch
from torch.utils.data import DataLoader

from mppi_rollouts.msg import OdomCmdVelProcessedFull2D
from mppi_rollouts.srv import MppiRollouts, MppiRolloutsResponse

from terrain_adaptation_rls.data.load_data import load_scenes, PhoenixDataset
from terrain_adaptation_rls.models.function_encoder import load_model as load_model_fe
from terrain_adaptation_rls.models.neural_ode import load_model as load_model_node


def rosmsg_error_handler(error):
    print("[ERROR]: ", error)
    return ('', error.end)

def make_dataloader(indices, n_example_points=100, n_points=1000):
    scene_data = load_scenes(indices, 'jackal_0770')
    inputs = [scene_data[f"scene{i}"][0] for i in indices]
    targets = [scene_data[f"scene{i}"][1] for i in indices]
    dataset = PhoenixDataset(inputs, targets, n_example_points, n_points)
    return iter(DataLoader(dataset, batch_size=1))

def get_coefficients(dataloader_iter, model, device):
    _, _, _, example_xs, example_dt, example_ys = next(dataloader_iter)
    example_xs = example_xs.to(device)
    example_dt = example_dt.to(device)
    example_ys = example_ys.to(device)
    return model.compute_coefficients((example_xs, example_dt), example_ys)

def wrap_to_pi(theta):
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi

def body_to_inertial(
        bIMat,    # (K, 3) matrix of body frame origin vectors in the inertial frame
        xBMat,    # (K, 3) matrix of vectors in the body frame
):
    """ Transforms body frame vectors into the inertial frame. """

    # Extract the rotation angles. Ensure separate memory by cloning.
    yaws = bIMat[:,2].clone()  

    c = torch.cos(yaws)
    s = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0], device=device)
    ones = torch.ones(yaws.shape[0], device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([c, -s, zeros], dim=1),
        torch.stack([s, c, zeros], dim=1),
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

    c = torch.cos(yaws)
    s = torch.sin(yaws)
    zeros = torch.zeros(yaws.shape[0], device=device)
    ones = torch.ones(yaws.shape[0], device=device)

    # Construct the batch of rotation matrices
    R = torch.stack([
        torch.stack([c, s, zeros], dim=1),
        torch.stack([-s, c, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)  # Shape: (K, 3, 3)

    # Perform batch matrix-vector multiplication
    xBMat = torch.bmm(R, xIMat.unsqueeze(-1)).squeeze(-1) #+ bIMat # Shape: (N, 3)
    return xBMat

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
    # print("[DEBUG]: Rollouts requested")
    # print("T: ", req.T)
    start_time = time.time()

    # Convert x0 and U into torch tensors. Make sure that the 
    # elements of U are loaded into the tensor correctly (so 
    # that the tensor matches the ArrayFire array in C++).
    x0 = torch.tensor(req.x0, dtype=torch.float32, device=device)
    U = torch.tensor(req.U, dtype=torch.float32, device=device).reshape(req.M, req.T+1, req.K).transpose(1, 2)

    # Define output tensor size to align with ArrayFire expectations. 
    X = torch.zeros((req.N, req.K, req.T + 1), dtype=torch.float32, device=device)

    # Define a tensor with integration steps for each rollout.
    times = torch.tensor([[req.dT]], device=device)
    times = times.repeat(1, req.K)

    # Remove the previous control from the sequence.
    V = U[:, :, 1:]
    
    # Set initial state across all samples at t=0. Assign 
    # the initial state of each sample along X. 
    x0_reshaped = x0.reshape(req.N, 1, 1) 
    X[:, :, 0] = x0_reshaped.repeat(1, req.K, 1).squeeze(2) 
    
    # Integrate over time. 
    for i in range(req.T):
        # Zero out the position and heading states in x0.
        zeros = torch.zeros(3, X.shape[1], device=device)
        # Concatenate all of the inputs.
        input = torch.cat((zeros, X[3:,:,i], V[:,:,i]), 0) # only use vels
        # Transpose the inputs to be compatible w/ FEs
        input = input.transpose(0,1).unsqueeze(0)
        
        # Predict the change in pose (expressed in the body frame) and
        # the velocity of the next body frame. 
        with torch.no_grad():
            if model_name == 'FE':
                output = fe_model((input, times), coefficients=coeffs) # xs = [bs, num_pts, 8], dt = [bs, num_pts]
            elif model_name == 'NODE':
                output = node_model((input, times))
            else:
                print("WRONG MODEL NAME")

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

    # Wrap the angles to between -pi and pi.
    X[2,:,:] = wrap_to_pi(X[2,:,:])

    # Flatten the trajectories to a list in ROW major order so
    # that it is easy to unpack into an ArrayFire Array in C++. 
    X_flat = X.permute(0, 2, 1).contiguous().flatten().tolist()
    # print(f"[DEBUG]: Rollouts sent {time.time() - start_time} seconds later.")
    print(time.time() - start_time)
    return MppiRolloutsResponse(X_flat)  


def make_hardware_dataloader(data_path, n_example_points=100, n_points=1000):

    # Load data as CSV
    test_inputs_np = load_csv(f"{data_path}/test_input.csv")
    test_targets_np = load_csv(f"{data_path}/test_target.csv")

    # Convert to torch tensor
    test_inputs = [torch.from_numpy(test_inputs_np).float()]
    test_targets = [torch.from_numpy(test_targets_np).float()]

    # Build the dataloader.
    dataset = PhoenixDataset(test_inputs, test_targets, n_example_points, n_points)
    return iter(DataLoader(dataset, batch_size=1))

def load_csv(full_path):
    data = []
    with open(full_path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            data.append([float(x) for x in row])
    return np.array(data)




# =========================== SETUP ======================================

# Choose (1) the FE model, (2) the NODE model, (3) the baseline coeff data
home = os.path.expanduser('~')
model_name = 'FE'
n_basis = 3
hidden_size = 16
fe_path = f'{home}/terrain-adaptation-rls/logs/grass_gym_ice_mulch_pavement_turf/function_encoder/seed=42/n_basis={n_basis}/hidden_size={hidden_size}/function_encoder_model.pth'
node_path = f'{home}/terrain-adaptation-rls/logs/grass_gym_ice_mulch_pavement_turf/neural_ode/seed=42/n_basis={n_basis}/hidden_size={hidden_size}/neural_ode_model.pth'
data_path = f"{home}/terrain-adaptation-rls/terrain_adaptation_rls/data_split/jackal_0770/seed_42/gym_floor" #ice/2025-08-08-15-57-39"

# Register the 'rosmsg' error handler
codecs.register_error("rosmsg", rosmsg_error_handler)

# Load the FE and NODE models.
device = "cpu" #if torch.cuda.is_available() else "cpu"
fe_model = load_model_fe(device = device, path = fe_path, n_basis=n_basis, hidden_size=hidden_size) 
node_model = load_model_node(device = device, path = node_path, n_basis=n_basis, hidden_size=hidden_size)

# Get coeffs from the historical trajectories.
dataloader_real = make_hardware_dataloader(data_path=data_path)
coeffs, _ = get_coefficients(dataloader_real, fe_model, device)


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node('rollouts_server')

    # Get robot name from a private parameter (default: "warty")
    name = rospy.get_param("~name", "jackal_0770")

    # Start the rollouts service.
    service = rospy.Service(f'{name}/calc_rollouts', MppiRollouts, handle_calc_rollouts)
    rospy.loginfo("Service 'calc_rollouts' ready to calculate MPPI rollouts.")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down rollouts server due to keyboard interrupt.")