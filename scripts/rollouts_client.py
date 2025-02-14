#!/usr/bin/env python3

import numpy as np
import random
import rospy
import torch
from mppi_rollouts.srv import MppiRollouts, MppiRolloutsRequest

def call_calc_rollouts(x0, U, K, T, M, N, dT):
    rospy.wait_for_service('warty/calc_rollouts')
    try:
        rollouts_service = rospy.ServiceProxy('warty/calc_rollouts', MppiRollouts)
        request = MppiRolloutsRequest(x0, U, K, T, M, N, dT)
        response = rollouts_service(request)
        return response.X
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    # Initialize the client node. 
    rospy.init_node('rollouts_client')

    # Define constants. 
    dT = 0.1 # integration time step
    K = 2 # number of rollouts.........depth
    T = 2 # number of time steps........row
    M = 2 # number of control actions....col
    N = 3 # number of input states

    # Sample random control actions and flatten to a list. 
    U = np.random.rand(K, T+1, M)*10
    U_flat = U.flatten().tolist()
    # print("U: ", U)

    # Sample random initial states in a list. 
    x0 = random.sample(range(1, 101), N)

    # Call the service. 
    result = call_calc_rollouts(x0, U_flat, K, T, M, N, dT)

    # Convert the trajectory response back to torch tensor. 
    X = torch.tensor(result, dtype=torch.float32).reshape(K, T+1, N)
    rospy.loginfo(f"Sent back X: {X.shape}")
