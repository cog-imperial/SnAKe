#from gpytorch import constraints
from numpy.lib.function_base import average
from gp_utils import BoTorchGP
from functions import BraninFunction, TwoDSinCosine, Hartmann6D
from adaptive_thompson_scheduling import AdaptiveThompsonScheduling
from bayes_op import UCBwLP, oneExpectedImprovement
from temperature_env import NormalDropletFunctionEnv
from scipy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np

def map_into_cost_grid(cost_grid, costs, regret):
    regret_out = []
    for c_g in cost_grid:
        smaller_than_cg = [c <= c_g for c in costs]
        if False in smaller_than_cg:
            idx = smaller_than_cg.index(False) - 1
        else:
            idx = len(costs) - 1
        regret_out.append(regret[idx])

    return regret_out

budget = 1000
obs_delay = 1

n_runs = 1

max_change = None

models = [AdaptiveThompsonScheduling] * 3 + [oneExpectedImprovement]
labels = ['ep = 1', 'ep = .1', 'ep = 0.00001', 'EI']
merge_constants = [1, .1, 0.00001, None]
merge_methods = ['e-Point Deletion', 'e-Point Deletion', 'e-Point Deletion', 'e-Point Deletion']
cols = ['b', 'r', 'k', 'g']

#models = [AdaptiveThompsonScheduling, oneExpectedImprovement, UCBwLP]
#labels = ['ATS', 'EI', 'UCB']
#cols = ['b', 'r', 'g']

cost_grids = []
regrets_list = [[] for _ in range(len(models))]
costs = [[] for _ in range(len(models))]
total_costs = [0] * len(models)

gp_model = BoTorchGP()
func = Hartmann6D()

dim = func.t_dim
if func.x_dim is not None:
    dim = dim + func.x_dim

for run in range(n_runs):
    x_train = np.random.uniform(0, 1, size = (max(int(budget / 5), 10 * dim), dim))
    y_train = []
    for i in range(0, x_train.shape[0]):
        y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

    gp_model.fit_model(x_train, y_train)
    gp_model.set_hyperparams(hyperparams=(2, 1, 1e-4, 0))
    gp_model.optim_hyperparams()

    hypers = gp_model.current_hyperparams()
    print(f'Initial hyperparameters = {hypers}')
    for i, model in enumerate(models):
        print(f'\rCurrently on run = {run + 1}, evaluating model: {labels[i]} ({i+1})', end = '', flush = True)
        env = NormalDropletFunctionEnv(func, budget = budget, max_batch_size = obs_delay)
        if labels[i] in ['EI', 'UCB']:
            mod = model(env)
        else:
            mod = model(env, max_change = max_change, merge_constant = merge_constants[i], merge_method = merge_methods[i], \
                hp_update_frequency = max(int(budget * 0.2), 25))
        
        mod.set_hyperparams(constant = hypers[0], lengthscale = hypers[1], noise = hypers[2], mean_constant = hypers[3], \
            constraints = True)

        X, Y = mod.run_optim()

        cumulative_cost = 0
        cost_record = [0]

        best_obs = 0

        cumulative_regret = [func.optimum - best_obs]

        for j in range(len(Y) - 1):

            cost = norm(X[j, :] - X[j+1, :])
            cumulative_cost += cost
            cost_record.append(cumulative_cost)

            best_obs = max(float(best_obs), float(Y[j]))
            
            regret = func.optimum - best_obs

            if regret <= 0:
                print(f'Zero / Negative Regret!')
                print('Pausa')
                regret = 1e-50

            cumulative_regret.append(regret)
        
        total_costs[i] = total_costs[i] + cumulative_cost
        
#        cost_grid = cost_grids[i]
        regrets_list[i].append(cumulative_regret)
        costs[i].append(cost_record)
#        regrets_list[i][run, :] = map_into_cost_grid(cost_grid, cost_record, cumulative_regret)

max_costs = [0] * len(models)
for m in range(len(models)):
    # for every run, check the cost
    for run in range(n_runs):
        run_cost = costs[m][run][-1]
        max_costs[m] = float(max(max_costs[m], run_cost))

regrets = []
cost_grid_lens = []
#bar_idxs = []
#x_bar_locs = []

for lens in max_costs:
    cost_grid = np.linspace(0, lens, 250)
    cost_grids.append(cost_grid)
    regrets.append(np.zeros((n_runs, len(cost_grid))))

for m in range(len(models)):
    cost_grid = cost_grids[m]
    len_grid = len(cost_grid)
    step = int(len_grid / 10)
    bar_idx = [step*s for s in range(2, 10)]
    #x_bar_locs.append(cost_grid[bar_idx])
    #bar_idxs.append(bar_idx)
    for run in range(n_runs):
        regrets[m][run, :] = map_into_cost_grid(cost_grid, costs[m][run], regrets_list[m][run])

avg_log_best = []
std_log_best = []
mins = []
maxs = []
y_max = 0
y_min = 0

#y_bar_locs = []
#bar_lens = []

for i in range(len(models)):
    model_regrets = regrets[i]
    avg_log = np.average(np.log(model_regrets), axis = 0)
    avg_log_best.append(avg_log)
    std_log = np.std(np.log(model_regrets), axis = 0)
    std_log_best.append(std_log)
    #y_bar_locs.append(avg_log[bar_idxs[i]])
    #bar_lens.append(np.std(np.log(model_regrets), axis = 0)[bar_idxs[i]])
    #min_logs = np.min(np.log(model_regrets), axis = 0)
    min_logs = avg_log - 0.5 * std_log
    mins.append(min_logs)
    y_min = min(np.min(min_logs), y_min)
    #max_logs = np.max(np.log(model_regrets), axis = 0)
    max_logs = avg_log + 0.5 * std_log
    y_max = max(np.max(max_logs), y_max)
    maxs.append(max_logs)

fig, ax = plt.subplots(nrows = 1, ncols = 1)
fig.set_figheight(6)
fig.set_figwidth(8)
title = 'Testing On ' + func.name + f' Function: Linear Input Cost. Budget = {budget}, Delay = {obs_delay}, $\Delta x$ = {max_change}.'
fig.suptitle(title)


for i in range(len(models)):
    ax.plot(cost_grids[i], avg_log_best[i], cols[i], label = labels[i])
    ax.fill_between(cost_grids[i], mins[i], maxs[i], color = cols[i], alpha = 0.3)
    final_reg = avg_log_best[i][-1]
    #ax.errorbar(x_bar_locs[i], y_bar_locs[i], bar_lens[i], c = cols[i], fmt = 'none', capsize = 0.5, ls = ':')
    ax.hlines(final_reg, 0, max(max_costs), colors = cols[i], linestyles = 'dashed')
    ax.vlines(total_costs[i] / n_runs, ymin = y_min - 0.5, ymax = y_max + 0.5, colors = cols[i], linestyles = 'dashed')

ax.set_xlabel('cost')
ax.set_ylabel('AVG(log-regret)')
ax.legend()
ax.grid()

plt.show()

'hola'

# 9 / 16 PL 20-21 (11 conceded)
# 3 / 6 PL 21-22 (3 conceded)
# 0 / 1 SuperCup 21-22 (1 conceded)
# 5 / 7 CL 20-21 (2 conceded)
# 1 / 2 CL 21-22 (1 conceded)

# Total - 18 / 32
# Conceded - 17 conceded