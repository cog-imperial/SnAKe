import numpy as np
from temperature_env import NormalDropletFunctionEnv
from functions import ConvergenceTest
import matplotlib.pyplot as plt
from adaptive_thompson_scheduling import SnAKe
import torch

'''
This script was used to create Figures 2 and 9.
'''

budget = 250
max_batch_size = 1
epsilon_list = [0.1]
methods = ['e-Point Deletion']
colors = ['r']
max_change = None

fig, ax = plt.subplots(nrows = 2, ncols = 1)
fig.set_figheight(6)
fig.set_figwidth(8)

pt = 0.74
escape_prediction = budget * pt

optimum_global = 0.78125
optimum_local = 0.15625
optimums = [optimum_local, optimum_global]

np.random.seed(2023)
torch.manual_seed(2023)

for i in range(0, len(epsilon_list)):
    initial_temp = np.array([0]).reshape(1, 1)
    epsilon = epsilon_list[i]
    #np.random.seed(47)
    func = ConvergenceTest()
    env = NormalDropletFunctionEnv(func, budget = budget, max_batch_size = max_batch_size)
    model = SnAKe(env, max_change = max_change, merge_method = methods[i], \
        merge_constant = epsilon, initial_temp = initial_temp, num_of_multistarts = 50)
    model.set_hyperparams(constant = 0.6, lengthscale = torch.tensor(0.1).reshape(-1, 1), noise = 1e-5, mean_constant=0)
    X, Y = model.run_optim(verbose = True)
    target_func = []
    grid = np.sort(model.global_grid0, axis = 0)

    for t in grid:
        target_func.append(func.query_function(t))

    times = np.array(range(0, env.budget+1)).reshape(-1, 1)
    # show posterior too
    posterior_mean, posterior_sd = model.model.posterior(grid)
    if methods[0] == 'Resampling':
        title = f'No Point Deletion'
    else:
        title = f'$\epsilon$ = {epsilon}.'

    ax[0].scatter(X, Y, s = 50, marker = 'x', c = 'k')
    if i == 0:
        ax[0].set_ylabel('Observations')
    ax[0].plot(grid, target_func, '--k', label = 'True function')
    ax[0].plot(grid, posterior_mean.detach().numpy(), colors[i], label = 'GP mean')
    ax[0].fill_between(grid.reshape(-1), posterior_mean.detach() - 1.96 * posterior_sd.detach(), \
            posterior_mean.detach() + 1.96 * posterior_sd.detach(), alpha = 0.2, color = colors[i])
    ax[0].set_xlim(0, 1)
    ax[0].grid()
    ax[0].legend(loc = 'lower right')
    ax[1].plot(X, times, colors[i])
    ax[1].set_xlabel('x')
    if i == 0:
        ax[1].set_ylabel('Iteration')
    ax[1].set_xlim(0, 1)
    ax[1].grid()
    if methods[0] == 'e-Point Deletion':
        ax[1].hlines(escape_prediction, 0, 1, colors = colors[i], linestyles = '--', label = 'Escape Prediction')
        ax[1].legend(loc = 'lower right')

filename = 'PDvsRSe-PointDeletion250' + '.pdf'
plt.savefig(filename, bbox_inches = 'tight')
plt.show()