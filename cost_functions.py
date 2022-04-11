import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import torch

'''
Cost function used in SnAr Experiment in the paper, see section 4.2 to see how we arrived to it.
'''

# need a function that takes as input two numpy arrays, and returns the cost matrix between the functions

def max_time_cost(vec_1, vec_2, constants = [5, 2, 3], thresholds = [2.5 / 80, 0.01 / 0.4, 0.05 / 1.5]):
    # check set of control variables matches dimension of given vectors
    assert vec_1.shape[1] == len(constants) == vec_2.shape[1], 'Number of constants does not match number of variables'
    # obtain number of rows in cost matrix, number of columns in cost matrix and number of variables (which we take max over)
    num_rows = vec_1.shape[0]
    num_columns = vec_2.shape[0]
    num_variables = len(constants)
    # initialise cost matrix
    C = np.zeros(shape = (num_rows, num_columns, num_variables))
    for var_num in range(num_variables):
        # calculate the distance between the points
        dT = distance_matrix(vec_1[:, var_num].reshape(-1, 1), vec_2[:, var_num].reshape(-1, 1))
        # use linear term and add log term
        C[:, :, var_num] = np.minimum(dT, thresholds[var_num]) + np.maximum(0, constants[var_num] * np.log(dT / thresholds[var_num]))
    # return maximum across all variables
    return np.max(C, axis = 2)

def max_time_cost_torch(vec_1, vec_2, constants = torch.tensor([5, 2, 3]), thresholds = torch.tensor([2.5 / 80, 0.01 / 0.4, 0.05 / 1.5])):
    # check set of control variables matches dimension of given vectors
    assert vec_1.shape[1] == len(constants) == vec_2.shape[1], 'Number of constants does not match number of variables'
    # obtain number of rows in cost matrix, number of columns in cost matrix and number of variables (which we take max over)
    num_rows = vec_1.shape[0]
    num_columns = vec_2.shape[0]
    num_variables = len(constants)
    # initialise cost matrix
    C = torch.zeros(size = (num_rows, num_columns, num_variables))
    for var_num in range(num_variables):
        # calculate the distance between the points
        dT = torch.cdist(vec_1[:, var_num].reshape(-1, 1), vec_2[:, var_num].reshape(-1, 1))
        # use linear term and add log term
        C_var_num = torch.minimum(dT, thresholds[var_num]) + torch.maximum(torch.tensor(0), constants[var_num] * torch.log(dT / thresholds[var_num]))
        C[:, :, var_num] = C_var_num.clone()
    # return maximum across all variables
    return torch.max(C, dim = 2)[0]

if __name__ == '__main__':
    A = torch.tensor([[0.1, 0.2], [0.1, 0.5], [0.5, 0.7], [0.6, 0.4], [0.3, 0.6]])
    B = torch.tensor([[0.1, 0.01]])

    a = max_time_cost_torch(B, A)
    print(a)