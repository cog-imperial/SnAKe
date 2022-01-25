import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

'''
Cost function used in SnAr Experiment in the paper, see section 4.2 to see how we arrived to it.
'''

# need a function that takes as input two numpy arrays, and returns the cost matrix between the functions

def max_time_cost(vec_1, vec_2, constants = [5, 2], thresholds = [2.5 / 80, 0.01 / 0.4]):
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