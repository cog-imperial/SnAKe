import numpy as np
import matplotlib.pyplot as plt
from gp_utils import BoTorchGP
from sampling import EfficientThompsonSampler

'''
This script was used to create Figures 7 and 8 in the appendix.
'''

func = lambda x : np.sin(10 * x) + np.exp(-(x - 0.775) ** 2 / 0.1) / 3
run_num = 10
experiment_name = 'WithOptimum'
color = 'g'


'''
This first part was first used to create the data, which was saved.
'''
# initialize values of p_t
pt_optimum = []

max_num_obs = 15


# select area A lower bound and upper bound
interval_ub = 0.2
interval_lb = 0.1

# initialize observation list
x_list = []

# set seed
np.random.seed(run_num)
for q in range(15, max_num_obs + 1):
    '''
    Uncomment below for stochastic sampling of points, for final plot we used linspace
    '''
    # x_list.append(np.random.uniform() * (interval_ub - interval_lb) + interval_lb)
    # x_train = np.array(x_list).reshape(-1, 1)
    x_train = np.linspace(interval_lb, interval_ub, q).reshape(-1, 1)
    y_train = func(x_train)
    # train GP model
    model = BoTorchGP(lengthscale_dim = 1)
    model.fit_model(x_train, y_train)
    # set hyper-parameters manually
    model.set_hyperparams((0.6, 0.1, 1e-5, 0))
    # create samples
    num_of_samples = 5000
    sampler = EfficientThompsonSampler(model, num_of_samples = num_of_samples, num_of_multistarts = 10)
    sampler.create_sample()
    samples = sampler.generate_candidates()
    samples = samples.detach().numpy()
    # count number of points in inside interval
    in_area_optimum = (samples < interval_ub) & (samples > interval_lb)
    pt_optimum.append(sum(in_area_optimum) / num_of_samples)
    print('done with q:', q)

# grid for plot of mean and standard deviation
full_grid = np.linspace(0, 1, 101).reshape(-1, 1)
mean, std = model.posterior(full_grid)
# true function values
Ys = func(full_grid)

# create plot
fig, ax = plt.subplots(nrows = 1, ncols = 2)
fig.set_figheight(4)
fig.set_figwidth(15)
# GP mean and std
ub = mean.detach() + 1.96 * std.detach()
lb = mean.detach() - 1.96 * std.detach()
ub = ub.numpy()
lb = lb.numpy()
# print final estimate of p_t
print(pt_optimum[-1])

'''
Uncomment below to save the data
'''
# filename = 'results_pt/' + experiment_name + '/run' + str(run_num)
# np.save(filename, np.array(pt_optimum))

'''
We now used the saved data to create the plots quickly
'''

# obtain saved values of p_t for all runs
pt_opts = []
for run_num in range(1, 10 + 1):
    filename = 'results_pt/' + experiment_name + '/run' + str(run_num) + '.npy'
    pt_opt = np.load(filename)
    pt_opts.append(pt_opt.reshape(-1))

pt_optimum = np.array(pt_opts)

# plot mean value of p_t
ax[1].plot(range(1, max_num_obs + 1), pt_optimum.T.mean(axis = 1), alpha = 1, c = color)
# plot confidence bounds, bound by zero since  probability cannot be negative
ax[1].fill_between(range(1, max_num_obs + 1), np.maximum(pt_optimum.T.mean(axis = 1) - pt_optimum.T.std(axis = 1), 0), \
    pt_optimum.T.mean(axis = 1) + pt_optimum.T.std(axis = 1), alpha = 0.2, color = color)
# set labels
ax[1].set_xlabel(f'Number of queries in [{interval_lb}, {interval_ub}]', fontsize = 12)
ax[1].set_ylabel('Estimate of $p_t$', fontsize = 12)

# plot GP
ax[0].plot(full_grid.reshape(-1), mean.detach(), color, label = 'GP mean')
ax[0].fill_between(full_grid.reshape(-1), ub, lb, color = color, alpha = 0.2)
ax[0].plot(full_grid, Ys, 'k--', label = 'True function')
ax[0].set_xlabel('x', fontsize = 12)
ax[0].set_ylabel('y', fontsize = 12)
ax[0].legend(loc = 'lower left', prop={'size': 12})

# save figure
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)
plt.savefig('ProbOfEscape' + experiment_name + '.pdf', bbox_inches = 'tight')
plt.show()