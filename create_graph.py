import numpy as np
import matplotlib.pyplot as plt
from snake import SnAKe
from functions import BraninFunction
from temperature_env import NormalDropletFunctionEnv
import math
from gp_utils import BoTorchGP
import torch
from matplotlib.patches import Circle

'''
This script was used to create Figure 3 in the paper.
'''

# initialise a figure
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# select function, and corresponding grids
func = BraninFunction()
x_grid = np.linspace(0, 1, 101)
y_grid = np.linspace(0, 1, 101)
xmesh, ymesh = np.meshgrid(x_grid, y_grid)
# define lambda function to plot underlying contour
lambda_func = lambda x, y: -((15*y - 5.1 * (15*x-5)**2 / (4 * math.pi**2) + 5 * (15*x-5) / math.pi - 6)**2 \
    + (10 - 10 / (8 * math.pi)) * np.cos((15*x-5)) - 44.81)/51.95
z_grid = lambda_func(xmesh, ymesh)
# set variables of optimization
budget = 100
max_batch_size = 1
max_change = None
epsilon = 0.05
initial_temp = np.array([0.5, 0.5]).reshape(1, -1)

# initialize GP model for hyper-parameters
gp_model = BoTorchGP()
dim = func.t_dim
if func.x_dim is not None:
    dim = dim + func.x_dim

x_train = np.random.uniform(0, 1, size = (50, dim))
y_train = []
for i in range(0, x_train.shape[0]):
    y_train.append(func.query_function(x_train[i, :].reshape(1, -1)))

seed = 3
# set seeds for reproducibility
torch.manual_seed(seed)
np.random.seed(seed)

gp_model.fit_model(x_train, y_train)
# set hyper-parameters manually for simplicity
gp_model.set_hyperparams(hyperparams=(0.66, 1, 1e-4, 0))
hypers = gp_model.current_hyperparams()

# define environment
env = NormalDropletFunctionEnv(func, budget = budget, max_batch_size = max_batch_size)
# define SnAKe model
model = SnAKe(env, initial_temp = initial_temp, gen_animation = True, \
    merge_method = 'e-Point Deletion', merge_constant = epsilon, num_of_multistarts = 20)

# manually run optimization loop until time required to visialize the plot
while model.current_time <= model.budget:
    model.optim_loop()

    model.out0[2] = model.env.temperature_list
    model.out1[2] = model.env.temperature_list
    model.out2[2] = model.env.temperature_list

    a = model.out2
    # out yields which are normally used to generate an animation
    if model.current_time == 11:
        yield1 =  model.out0 # 'raw samples', samples, self.env.temperature_list, self.X
        yield2 =  model.out1 # 'deleted samples', self.working_grid[unique_sample_idx, :], self.env.temperature_list, self.X
        yield3 =  model.out2 # 'new_plan', self.query_plan, self.env.temperature_list, self.X
        yield4 =  ['move', a[1], a[2], a[3], model.env.t]
        # obtain and reformat deleted points
        deleted_points = np.array(model.deleted_points)
        break

# do we want to plot paths or Point Deletion Step?
plot = 'second'
# do we want to plot the deleted points?
with_deleted = True

# plot contour
ax.contourf(x_grid, y_grid, z_grid, levels = 100)

if plot == 'first':
    # plot evaluated points
    evaled_points = np.array(yield1[3])
    n_eval_points = evaled_points.shape[0]
    ax.plot(evaled_points[:, 0], evaled_points[:, 1], c = 'r', marker = 'x', markersize = 15, zorder = 1)
    # add epsilon circles
    for i in range(0, n_eval_points, 1):
        xx = evaled_points[i, 0]
        yy = evaled_points[i, 1]
        color = 'r'
        alpha = 0.4
        circ = Circle((xx,yy), epsilon, alpha = alpha, fill = False, color = color)
        ax.add_patch(circ)

    # plot samples
    samples = yield1[1]
    ax.scatter(samples[:, 0], samples[:, 1], c = 'k', marker = '.', s = 300, label = 'Thompson samples', zorder = 3)
    ax.scatter(deleted_points[:, 0], deleted_points[:, 1], c = 'g', marker = '.', s = 300, label = 'points to delete', zorder = 4)

if plot == 'second':
    # plot evaluated points
    evaled_points = np.array(yield1[3])
    n_eval_points = evaled_points.shape[0]
    ax.plot(evaled_points[:, 0], evaled_points[:, 1], c = 'r', marker = 'x', markersize = 15)
    # plot future paths
    future_points = yield3[1]
    ax.plot(future_points[:, 0], future_points[:, 1], c = 'b', marker = '.', markersize = 15, label = 'new optimization path')
    # connect
    connect = np.concatenate((evaled_points[-1, :].reshape(1, -1), future_points[0, :].reshape(1, -1)), axis = 0)
    ax.plot(connect[:, 0], connect[:, 1], 'b--', markersize = 1)
    # with deleted
    if with_deleted:
        ax.scatter(deleted_points[:, 0], deleted_points[:, 1], c = 'g', marker = '.', label = 'deleted points', s = 300)
    
    
# set limits for readibility and save figure
ax.set_xlim(0.375, 0.625)
ax.set_ylim(0.175, 0.425)
x_ticks = [0.4, 0.5, 0.6]
y_ticks = [0.2, 0.3, 0.4]
plt.xticks(x_ticks, fontsize = 20)
plt.yticks(y_ticks, fontsize = 20)
ax.set_xlabel('$x_1$', fontsize = 20)
ax.set_ylabel('$x_2$', fontsize = 20)
ax.set_aspect('equal')
plt.legend(loc = 'upper right', framealpha = 0.8, prop={'size': 25})
plt.savefig('EaSdemo3nd.pdf', bbox_inches = 'tight')
plt.show()