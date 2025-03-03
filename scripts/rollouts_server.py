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
    input_size=(4,),  # del_time, states (yaw), controls (lin x, ang z vel)
    output_size=(3,), # next states (x, y, yaw)
    data_type="deterministic",
    n_basis=11,
    model_type="MLP",
    method="least_squares",
    use_residuals_method=False
).to(device)

# Load pre-trained weights into the model. 
path = f'{home}/FunctionEncoderMPPI/logs/warthog_example/least_squares/shared_model'
model.load_state_dict(torch.load(f'{path}/2025-03-03_12-31-48-WORKING-MODEL/model.pth'))

# Load pre-collected CSV data to compute representations.
csv_file = '/home/arl/catkin_ws/src/mppi_rollouts/data/2025-02-28-10-58-11/warty-odom_processed-10hz-20.csv'
# Load CSV into a numpy array. 
array = np.loadtxt(csv_file, delimiter=',')
# Unwrap the yaw meaurements.
array[:, 3] = np.unwrap(array[:,3])
# Convert numpy array to torch tensor. 
tensor = torch.tensor(array, device=device).to(torch.float32)
# Convert the time stamps column to changes in time. 
tensor[:-1, 0] = tensor[1:, 0] - tensor[:-1, 0]
# Get the change in states from the data. 
del_states = tensor[1:, 1:4] - tensor[:-1, 1:4]
# Remove the bottom row from the data. 
new_tensor = tensor[:-1,:]
# Remove the xPos and yPos from the data.
final_tensor = new_tensor[:,[0,3,4,5]]
# Append the change in states to the tensor
data = torch.cat((final_tensor, del_states), dim=1)

# Get random indices from the data tensor.
ex_indices = torch.randperm(data.size(0))[:1000]
# Sample random rows from the data tensor. 
ex_subset = data[ex_indices]
# Parse out the input and output data.
example_xs = ex_subset[:,:4].unsqueeze(dim=0)
example_ys = ex_subset[:,4:].unsqueeze(dim=0)
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
        # Concatenate all of the inputs.
        input = torch.cat((time, X[2,:,i].unsqueeze(0), V[:,:,i]), 0)
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
        
        # Predict the next state using the model. 
        with torch.no_grad():
            output = model.predict(input, coeff)
            # output = bicycle(input)
            # if i == 0:
            #     print("[DEBUG]: Output Tensor")
            #     print("output: ", output)
            #     print("output: ", output.shape)
            #     print("output.squeeze(0).transpose(1,0): ", output.squeeze(0).transpose(1,0))

        # Assign the output as the next state. 
        # X[:,:,i+1] = (output.squeeze(0).transpose(1,0)).to("cpu")
        
        # Assign the output as the change in state. 
        X[:,:,i+1] = X[:,:,i] + (output.squeeze(0).transpose(1,0))#.to("cpu")
        # if i == 0:
            # print("[DEBUG]: Assigning the output tensor to X")
            # print("X: ", X)

    # print("[DEBUG]: Double Check the final shape of X")       
    # print("X after = ", X.shape)

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


if __name__ == "__main__":
    # Initialize the ROS server.
    rospy.init_node('rollouts_server')
    service = rospy.Service('warty/calc_rollouts', MppiRollouts, handle_calc_rollouts)
    rospy.loginfo("Service 'calc_rollouts' ready to calculate MPPI rollouts.")
    rospy.spin()
