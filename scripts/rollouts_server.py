#!/usr/bin/env python3

import numpy as np
import os
import rospy
import sys
import torch
from mppi_rollouts.srv import MppiRollouts, MppiRolloutsResponse

# Add path to the FunctionEncoder package.
home = os.path.expanduser('~')
sys.path.append(f'{home}/FunctionEncoderMPPI')
from FunctionEncoder import FunctionEncoder

# Create a Function Encoder model. 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FunctionEncoder(
    input_size=(6,), # 1 del_time + 3 states + 2 actions
    output_size=(3,), # 3 states
    data_type="deterministic",
    n_basis=11,
    model_type="MLP",
    method="least_squares",
    use_residuals_method=False
).to(device)

# Load pre-trained weights into the model. 
path = f'{home}/FunctionEncoderMPPI/logs/warthog_example/least_squares/shared_model'
model.load_state_dict(torch.load(f'{path}/2025-02-12_15-04-51/model.pth'))

# Load pre-collected CSV data to compute representations.
csv_file = '/home/arl/catkin_ws/src/mppi_rollouts/data/warty-warthog_velocity_controller-odom-TRIMMED.csv'
# Load CSV into a numpy array, then convert to torch tensor. 
array = np.loadtxt(csv_file, delimiter=',')
tensor = torch.tensor(array, device=device).to(torch.float32)
# Convert the time stamps column to changes in time. 
tensor[:-1, 0] = tensor[1:, 0] - tensor[:-1, 0]
# Get the "next" states from the data. 
next_states = tensor[1:, 1:4]
# Remove the bottom row from the data. 
new_tensor = tensor[:-1,:]
# Append the "next" states to the tensor. 
data = torch.cat((new_tensor, next_states), dim=1)

# Get random indices from the data tensor.
ex_indices = torch.randperm(data.size(0))[:1000]
# Sample random rows from the data tensor. 
ex_subset = data[ex_indices]
# Parse out the input and output data.
example_xs = ex_subset[:,:6].unsqueeze(dim=0)
example_ys = ex_subset[:,6:].unsqueeze(dim=0)
# Compute the coefficients for the function encoder using new data. 
with torch.no_grad():
    coeff, _ = model.compute_representation(example_xs, example_ys, method="least_squares")



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
    # Print the input constants.
    print("K: ", req.K)
    print("T: ", req.T)
    print("M: ", req.M)
    print("N: ", req.N)
    print("-----")


    # Convert the input lists into torch tensors. 
    # print("x0: ", len(req.x0))
    # print("U: ", len(req.U))
    x0 = torch.tensor(req.x0, dtype=torch.float32, device=device)
    U = torch.tensor(req.U, dtype=torch.float32, device=device).reshape(req.K, req.T+1, req.M) 
    print("x0: ", x0.shape)
    print("U: ", U.shape)

    # if (T == 0) {
    #     return tile(moddims(x0, 1, 1, N), K);
    # }

    # Define output tensor. 
    X = torch.zeros((req.K, req.T + 1, req.N), dtype=torch.float32, device=device)
    print("X: ", X.shape)

    # Define a tensor for the integration time step.
    time = torch.tensor([[req.dT]], device=device)
    time = time.repeat(req.K, 1) # repeat for each rollout. 
    print("time: ", time.shape)

    # Remove the previous control from the sequence.
    V = U[:, 1:, :]
    print("V: ", V.shape)
    
    # Set initial state across all samples at t=0
    x0_reshaped = x0.reshape(1, 1, req.N) 
    X[:, 0, :] = x0_reshaped.repeat(req.K, 1, 1).squeeze(1)  # Repeat K times along the first dimension
    
    # Integrate over time. 
    for i in range(req.T):
        # Concatenate all of the inputs.
        input = torch.cat((time, X[:,i,:], V[:,i,:]), 1).unsqueeze(0)
        # print("input: ", input.shape)
        
        # Predict change in states using the model. 
        with torch.no_grad():
            output = model.predict(input, coeff)
            # print("output: ", output.shape)

        # Add the change in states to the current state. 
        X[:,i+1,:] = (X[:,i,:] + output.squeeze(0).squeeze(0)).to("cpu")

    # Flatten the trajectories to a list and respond to the client. 
    X_flat = X.flatten().tolist()
    print("X_flat: ", len(X_flat))
    print("-----------------------")
    return MppiRolloutsResponse(X_flat)

if __name__ == "__main__":
    # Initialize the ROS server.
    rospy.init_node('rollouts_server')
    service = rospy.Service('warty/calc_rollouts', MppiRollouts, handle_calc_rollouts)
    rospy.loginfo("Service 'calc_rollouts' ready to calculate MPPI rollouts.")
    rospy.spin()
