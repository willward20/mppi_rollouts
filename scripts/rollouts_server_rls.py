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

from function_encoder.coefficients import recursive_least_squares_update
from terrain_adaptation_rls.models.function_encoder import load_model as load_model_fe


class GlobalState:
    def __init__(self, device, n_basis):
        self.lock = threading.Lock()
        self.coefficients = torch.zeros(1, n_basis, device=device)
        self.P = torch.eye(n_basis, device=device).unsqueeze(0)
        self.input_time = time.time()
        self.input_pose_I = torch.zeros((1,3), device=device)
        self.input_vel_B = torch.zeros(1,3, device=device)
        self.cmd = torch.zeros(2, device=device)
        self.first_cmd_received = False

        self.node_err = []
        self.fe_err = []
        self.rls_err = []
        self.time_array = []

def rosmsg_error_handler(error):
    print("[ERROR]: ", error)
    return ('', error.end)

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

    with gs.lock:
        # Use the coefficients from the global state.
        coeffs = gs.coefficients.clone()

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
            output = fe_model((input, times), coefficients=coeffs) # xs = [bs, num_pts, 8], dt = [bs, num_pts]

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


def rls_update(data):
    # print("[DEBUG]: Received a new command velocity for RLS update")

    # Unpack the data from the message (these are the targets). 
    target_time = data.time
    target_xPos = data.xPos
    target_yPos = data.yPos
    target_yaw = np.unwrap([gs.input_pose_I.cpu()[:,2].item(), data.yaw])[1]
    target_xVel = data.xVel
    target_yVel = data.yVel
    target_zAngVel = data.zAngVel

    # Maintain a certain framerate.
    del_t = target_time - gs.input_time

    # Make sure you have two points to process. 
    if gs.first_cmd_received:

        # Build x_step tensor from the previous states.
        x_step = torch.cat(
            (torch.zeros_like(gs.input_vel_B), gs.input_vel_B),  # Previous velocity in body frame
            dim=-1
        ).unsqueeze(0)

        # Build the u_step vector from the previous controls.
        u_step = torch.tensor(
            [gs.input_cmd_xVel, gs.input_cmd_zAngVel],dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0)

        # Build the dt_step tensor from the time difference.
        dt_step = torch.tensor(
            [del_t], dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Set the target pose and velocity tensors.
        target_pose_I = torch.tensor(
            [target_xPos, target_yPos, target_yaw], dtype=torch.float32, device=device
        ).unsqueeze(0)

        # Transform the target positions into the body frame of the previous state.
        target_del_pose_B = inertial_to_body(
            gs.input_pose_I,  target_pose_I - gs.input_pose_I
        )

        # Transform the target velocities into the body frame of the previous state.
        target_vel_B = body_to_inertial(
            target_pose_I, 
            torch.tensor(
                [target_xVel, target_yVel, target_zAngVel], 
                dtype=torch.float32, device=device
            ).unsqueeze(0)   # xBMat
        )
        target_vel_B = inertial_to_body(gs.input_pose_I, target_vel_B)

        # Build the y_step tensor (change in pose and vel) from the target.
        y_step = torch.cat(
            (target_del_pose_B, target_vel_B - gs.input_vel_B), dim=-1
        ).unsqueeze(0)

        with torch.no_grad():

            # Compute the basis functions 
            # [batch_size, n_points, n_features, n_basis]
            g = fe_model.basis_functions((torch.cat((x_step,u_step), dim=-1), dt_step))

            L = torch.linalg.cholesky(gs.P)

            with gs.lock:
                coeffs = gs.coefficients.clone()

            coeffs, gs.P = recursive_least_squares_update(
                method='qr',
                g=g,
                y=y_step,
                P=L,
                coefficients=coeffs,
                forgetting_factor=0.95,
            )

            with gs.lock:
                gs.coefficients = coeffs

            # Compute the recursive least squares prediction error
            pred = fe_model((torch.cat((x_step, u_step), dim=-1), dt_step), coefficients=coeffs)
            
            # # Compute the RLS error. 
            loss_rls = torch.nn.functional.mse_loss(pred, y_step)
            gs.rls_err.append(loss_rls.item())

            # # Record the target time (time of prediction)
            gs.time_array.append(target_time)

    # Save the new state and control for the next iteration.
    gs.input_time = target_time
    gs.input_pose_I = torch.tensor(
        [
            data.xPos, 
            data.yPos, 
            np.unwrap([data.yaw])[0]
        ], 
        dtype=torch.float32, device=device
    ).unsqueeze(0)
    
    gs.input_vel_B = torch.tensor(
        [data.xVel, data.yVel, data.zAngVel], 
        dtype=torch.float32, device=device
    ).unsqueeze(0)
    
    gs.input_cmd_xVel = data.cmd_xVel
    gs.input_cmd_zAngVel = data.cmd_zAngVel

    # Set the flag to true
    gs.first_cmd_received = True




# =========================== SETUP ======================================

# Choose (1) the FE model, (2) the NODE model, (3) the baseline coeff data
home = os.path.expanduser('~')
n_basis = 3
hidden_size = 16
fe_path = f'{home}/terrain-adaptation-rls/logs/grass_gym_ice_mulch_pavement_turf/function_encoder/seed=42/n_basis={n_basis}/hidden_size={hidden_size}/function_encoder_model.pth'

# Register the 'rosmsg' error handler
codecs.register_error("rosmsg", rosmsg_error_handler)

# Load the FE and NODE models.
device = "cpu" #if torch.cuda.is_available() else "cpu"
fe_model = load_model_fe(device = device, path = fe_path, n_basis=n_basis, hidden_size=hidden_size) 

# Initialize the global object to track information. 
gs = GlobalState(device, n_basis) 


if __name__ == "__main__":
    # Initialize the ROS node.
    rospy.init_node('rollouts_server')

    # Get robot name from a private parameter (default: "warty")
    name = rospy.get_param("~name", "jackal_0770")

    # Start the rollouts service.
    service = rospy.Service(f'{name}/calc_rollouts', MppiRollouts, handle_calc_rollouts)
    rospy.loginfo("Service 'calc_rollouts' ready to calculate MPPI rollouts.")

    # Initialize a ROS subscriber.
    rospy.Subscriber(f'{name}/odom_cmd_vel_processed_full2D', OdomCmdVelProcessedFull2D, rls_update)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down rollouts server due to keyboard interrupt.")
    finally:

        # Define CSV filename
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        csv_path = f"{home}/ws/src/mppi_rollouts/ice_autonomy_data/fe_rls_errors/n_basis={n_basis}_hidden_size={hidden_size}"

        # Create the directory path if it doesn't exist
        os.makedirs(csv_path, exist_ok=True)

        # Build full filename
        file_path = os.path.join(csv_path, f"{timestamp}.csv")

        # Pad shorter lists with NaNs to align lengths
        max_len = max(len(gs.time_array), len(gs.rls_err))
        def pad(lst): return lst + [float('nan')] * (max_len - len(lst))

        rows = zip(
            pad(gs.time_array),
            pad(gs.rls_err),
        )

        with open(file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time_array", "rls_err"])
            writer.writerows(rows)