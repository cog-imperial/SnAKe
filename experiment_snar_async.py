import torch
from gp_utils import BoTorchGP
from functions import SnAr
from adaptive_thompson_scheduling import AdaptiveThompsonScheduling, RandomTSP
from bayes_op import UCBwLP, ThompsonSampling
from temperature_env import NormalDropletFunctionEnv
from scipy.spatial import distance_matrix
from cost_functions import max_time_cost
import numpy as np
import sys
import os

#method = 'UCBwLP'
#run_num = 9
#budget = 100
#epsilon = 0.1
#delay = 1

method = str(sys.argv[1])
run_num = int(sys.argv[2])
budget = 100
epsilon = float(sys.argv[3])
delay = int(sys.argv[4])

function_number = 0

print(method, run_num, budget, epsilon)

# Make sure problem is well defined
assert method in ['EaS', 'UCBwLP', 'TS', 'Random'], 'Method must be string in [EaS, UCBwLP, TS, Random]'
assert delay in [0, 1, 2, 3], \
    'Delay must be integer in [0, 1, 2, 3]'
assert epsilon in [0, 0.1, 0.25, 1], \
    'Epsilon must be in [0, 0.1, 0.25, 1]'

if delay == 0:
    delay = 5
elif delay == 1:
    delay = 10
elif delay == 2:
    delay = 25
else:
    delay = 50

# Define function name
functions = [SnAr()]
func = functions[function_number]

# We start counting from zero, so set budget minus one
budget = budget - 1

# Define cost function

cost_function = lambda x, y: max_time_cost(x, y)

# Define seed, sample initalisation points
seed = run_num + (function_number + 1) * 91
torch.manual_seed(seed)
np.random.seed(seed)

dim = func.t_dim
if func.x_dim is not None:
    dim = dim + func.x_dim

initial_temp = np.random.uniform(size = (1, dim)).reshape(1, -1)

x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

y_train = np.array(y_train)

# initalise hyper-parameters too
# Train and set educated guess of hyper-parameters
gp_model = BoTorchGP(lengthscale_dim = dim)

gp_model.fit_model(x_train, y_train)
#gp_model.set_hyperparams(hyperparams=(2, 1, 1e-4, 0))
gp_model.optim_hyperparams()

hypers = gp_model.current_hyperparams()

print('Initial hyper-parameters:', hypers)
# Define Normal BayesOp Environment without delay
env = NormalDropletFunctionEnv(func, budget, max_batch_size = delay)

# Choose the correct method
# Choose the correct method
if method == 'EaS':
    mod = AdaptiveThompsonScheduling(env, merge_method = 'e-Point Deletion', merge_constant = epsilon, cost_function = cost_function, initial_temp = initial_temp, \
        hp_update_frequency = 25)
elif method == 'UCBwLP':
    mod = UCBwLP(env, initial_temp = initial_temp, hp_update_frequency = 25)
elif method == 'TS':
    mod = ThompsonSampling(env, initial_temp = initial_temp, hp_update_frequency = 25)
elif method == 'Random':
    mod = RandomTSP(env, initial_temp = initial_temp)

mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = True)

X, Y = mod.run_optim(verbose = True)

print(X)
print(np.array(Y))

if method == 'EaS':
    folder_inputs = 'experiment_results_snar_async/' + f'{epsilon}-EaS/' + f'/delay{delay}/' + '/inputs/'
    folder_outputs = 'experiment_results_snar_async/' + f'{epsilon}-EaS/' + f'/delay{delay}/' + '/outputs/'
    file_name = f'run_{run_num}'
elif method == 'Random':
    folder_inputs = 'experiment_results_snar_async/' + f'Random/' + f'/budget{delay}/' + '/inputs/'
    folder_outputs = 'experiment_results_snar_async/' + f'Random/' + f'/budget{delay}/' + '/outputs/'
    file_name = f'run_{run_num}'
else:
    folder_inputs =  'experiment_results_snar_async/' + method + '/' + f'/budget{delay}/inputs/'
    folder_outputs =  'experiment_results_snar_async/' + method + '/' + f'/budget{delay}/outputs/'
    file_name = f'run_{run_num}'

# create directories if they exist
os.makedirs(folder_inputs, exist_ok = True)
os.makedirs(folder_outputs, exist_ok = True)

print(X.shape[0])
print(np.array(Y).shape[0])

np.save(folder_inputs + file_name, X)
np.save(folder_outputs + file_name, np.array(Y))