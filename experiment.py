import torch
from gp_utils import BoTorchGP
from functions import BraninFunction, Hartmann6D, Hartmann4D, Hartmann3D, Ackley4D, Perm8D, Perm10D
from adaptive_thompson_scheduling import AdaptiveThompsonScheduling, RandomTSP
from bayes_op import UCBwLP, oneExpectedImprovement, oneProbabilityOfImprovement
from temperature_env import NormalDropletFunctionEnv
from scipy.spatial import distance_matrix
import numpy as np
import sys

method = str(sys.argv[1])
function_number = int(sys.argv[2])
run_num = int(sys.argv[3])
budget = int(sys.argv[4])
epsilon = float(sys.argv[5])
cost_func = int(sys.argv[6])

# Make sure problem is well defined
assert method in ['EaS', 'EI', 'UCB', 'PI', 'Random'], 'Method must be string in [EaS, EI, UCB, PI, Random]'
assert function_number in range(7), \
    'Function must be integer between 0 and 6'
assert budget in [50, 100, 250], \
    'Budget must be integer in [50, 100, 250]'
assert epsilon in [0, 0.1, 0.25, 1, 2], \
    'Epsilon must be in [0, 0.1, 0.25, 1, 2]'
assert cost_func in [1, 2, 3], \
    'Cost function must be integer in [1, 2, 3] (where 3 corresponds to infinity norm)'

# Define function name
functions = [BraninFunction(), Hartmann3D(), Hartmann4D(), Hartmann6D(), Ackley4D(), Perm8D(), Perm10D()]
func = functions[function_number]

# Define cost function
if cost_func == 1:
    cost_function = lambda x, y: distance_matrix(x, y, p = 1)
    cost_name = '1norm'
elif cost_func == 2:
    cost_function = lambda x, y: distance_matrix(x, y, p = 2)
    cost_name = '2norm'
elif cost_func == 3:
    cost_function = lambda x, y: distance_matrix(x, y, p = float('inf'))
    cost_name = 'inftynorm'

# Define seed, sample initalisation points
seed = run_num + function_number * 335
torch.manual_seed(seed)
np.random.seed(seed)

initial_temp = np.random.uniform(size = (1, func.t_dim)).reshape(1, -1)

dim = func.t_dim
if func.x_dim is not None:
    dim = dim + func.x_dim

x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

y_train = np.array(y_train)

# initalise hyper-parameters too
# Train and set educated guess of hyper-parameters
gp_model = BoTorchGP()

gp_model.fit_model(x_train, y_train)
gp_model.set_hyperparams(hyperparams=(2, 1, 1e-4, 0))
gp_model.optim_hyperparams()

hypers = gp_model.current_hyperparams()

# Define Normal BayesOp Environment without delay
env = NormalDropletFunctionEnv(func, budget, max_batch_size = 1)

# Choose the correct method
if method == 'EaS':
    mod = AdaptiveThompsonScheduling(env, merge_method = 'e-Point Deletion', merge_constant = epsilon, cost_function = cost_function, initial_temp = initial_temp, \
        hp_update_frequency = 25)
elif method == 'EI':
    mod = oneExpectedImprovement(env, initial_temp = initial_temp, hp_update_frequency = 25)
elif method == 'UCB':
    mod = UCBwLP(env, initial_temp = initial_temp, hp_update_frequency = 25)
elif method == 'PI':
    mod = oneProbabilityOfImprovement(env, initial_temp = initial_temp, hp_update_frequency = 25)
elif method == 'Random':
    mod = RandomTSP(env, initial_temp = initial_temp)

mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = True)

X, Y = mod.run_optim(verbose = True)

print(X)
print(np.array(Y))

if method == 'EaS':
    folder_inputs = f'/{epsilon}-EaS/' + func.name + f'/budget{budget}/' + cost_name + '/inputs/'
    folder_outputs = f'/{epsilon}-EaS/' + func.name + f'/budget{budget}/' + cost_name + '/outputs/'
    file_name = f'run_{run_num}'
elif method == 'Random':
    folder_inputs = f'/Random/' + func.name + f'/budget{budget}/' + cost_name + '/inputs/'
    folder_outputs = f'/Random/' + func.name + f'/budget{budget}/' + cost_name + '/outputs/'
    file_name = f'run_{run_num}'
else:
    folder = '/' + method + func.name + '/' + f'/budget{budget}/inputs/'
    folder = '/' + method + func.name + '/' + f'/budget{budget}/outputs/'
    file_name = f'run_{run_num}'

#np.save(folder_inputs + file_name, X)
#np.save(folder_inputs + file_name, np.array(Y))