import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from gp_utils import BoTorchGP
from temperature_env import MultiObjectiveNormalDropletFunctionEnv
from snake import MultiObjectiveSnAKe
from bayes_op import TruncatedExpectedImprovement, EIperUnitCost, MultiObjectiveEIpu, MultiObjectiveTrEI
from functions import MultiSchekel2D, YpacaraiLake, YpacaraiLakeSingleObjective
import torch
import os


'''
This script creates:
(1) Ypacarai Lake Figure is purpose is set to 'plot'
(2) Runs the experiments if purpose is set to 'experiment'
(3) Analyses the results if purpose is set to 'process_results'
'''

purpose = 'plot'
cost_change = 3.3
save_plot = True

if purpose == 'plot':

    func = YpacaraiLake()
    grid_to_search = func.grid_to_search
    initial_idx = np.random.random_integers(len(grid_to_search))
    initial_point = grid_to_search[initial_idx, :].numpy().reshape(1, -1)

    env = MultiObjectiveNormalDropletFunctionEnv(func, max_batch_size = 1, budget = 100)
    model = MultiObjectiveSnAKe(env, initial_temp = initial_point, max_change = 0.2, exploration_constant = 0, merge_constant = 'lengthscale', objective_weights = [1, 1, 1])
    # model = TruncatedExpectedImprovement(env, initial_temp = initial_point)
    model.gp_hyperparams[0] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.5)
    model.gp_hyperparams[1] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.8)
    model.gp_hyperparams[2] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.5)

    X, Y = model.run_optim(verbose = True)

    grid_len_x = 2006
    grid_len_y = 2825

    func = MultiSchekel2D(n_optims = [2, 3, 2])

    x1 = np.linspace(0, 1, grid_len_x)
    x2 = np.linspace(0, 1, grid_len_y)

    xv, yv = np.meshgrid(x1, x2)

    zv1 = np.zeros_like(xv)
    zv2 = np.zeros_like(xv)
    zv3 = np.zeros_like(xv)

    for i in range(grid_len_x):
        x = x1[i] * np.ones_like(x2)
        y = x2

        x = x.reshape(-1, 1)
        y = np.flip(y.reshape(-1, 1))

        xy = np.concatenate((x, y), axis = 1)

        y1, y2, y3 = func.query_function(xy)

        zv1[:, i] = y1
        zv2[:, i] = y2
        zv3[:, i] = y3

    array_lake = np.load('ypacarai_array.npy')

    zv1 = zv1 * (array_lake - 1) * (-1)
    zv2 = zv2 * (array_lake - 1) * (-1)
    zv3 = zv3 * (array_lake - 1) * (-1)

    fig, ax = plt.subplots(nrows = 1, ncols = 3)
    fig.set_figheight(6)
    fig.set_figwidth(18)

    ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[0].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[1].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax[2].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    ax[0].tick_params(axis='both', labelsize=15)
    ax[1].tick_params(axis='x', labelsize=15)
    ax[1].tick_params(axis='y', labelsize=0)
    ax[2].tick_params(axis='x', labelsize=15)
    ax[2].tick_params(axis='y', labelsize=0)

    contour_obj1 = ax[0].contourf(x1, np.flip(x2), zv1, levels = 50)
    cbar1 = fig.colorbar(contour_obj1, ax = ax[0], format = FormatStrFormatter('%.1f'))
    cbar1.ax.tick_params(labelsize = 15)
    ax[0].set_title('Objective 1', fontsize = 20)
    ax[0].set_xlabel('x1', fontsize = 15)
    ax[0].set_ylabel('x2', fontsize = 15)
    ax[0].scatter(X[:, 0], X[:, 1], color = 'k')
    ax[0].plot(X[:, 0], X[:, 1], color = 'k', linewidth = 0.5, markersize = 0.001, linestyle = '--')
    contour_obj2 = ax[1].contourf(x1, np.flip(x2), zv2, levels = 50, cmap = 'RdBu_r')
    cbar2 = fig.colorbar(contour_obj2, ax = ax[1])
    cbar2.ax.tick_params(labelsize = 15)
    ax[1].set_title('Objective 2', fontsize = 20)
    ax[1].set_xlabel('x1', fontsize = 15)
    ax[1].scatter(X[:, 0], X[:, 1], color = 'k')
    ax[1].plot(X[:, 0], X[:, 1], color = 'k', linewidth = 0.5, markersize = 0.001, linestyle = '--')
    contour_obj3 = ax[2].contourf(x1, np.flip(x2), zv3, levels = 50, cmap = 'ocean')
    cbar3 = fig.colorbar(contour_obj3, ax = ax[2])
    cbar3.ax.tick_params(labelsize = 15)
    ax[2].set_title('Objective 3', fontsize = 20)
    ax[2].set_xlabel('x1', fontsize = 15)
    ax[2].scatter(X[:, 0], X[:, 1], color = 'k')
    ax[2].plot(X[:, 0], X[:, 1], color = 'k', linewidth = 0.5, markersize = 0.001, linestyle = '--')

    if save_plot == True:
        fig_name = 'YpacaraiExampleRun'
        save_name = fig_name + '.pdf'
        fig.savefig(save_name, bbox_inches = 'tight')
    
    plt.show()

if purpose == 'experiment':
    for method in ['TrEI', 'EIpu', 'SnAKe']:
    # for method in ['TrEI', 'EIpu']:
        for obj_number in [0, 1, 2, 3]:
            if obj_number == 0:
                func = YpacaraiLake()
            else:
                func = YpacaraiLakeSingleObjective(obj_to_query = obj_number - 1)
            
            for run_num in range(1, 11):
                # set random seed
                seed = run_num * 505
                np.random.seed(seed)
                torch.manual_seed(seed)
                grid_to_search = func.grid_to_search
                initial_idx = np.random.random_integers(len(grid_to_search))
                initial_point = grid_to_search[initial_idx, :].numpy().reshape(1, -1)

                if method == 'SnAKe':
                    budget = 200
                else:
                    budget = 200

                env = MultiObjectiveNormalDropletFunctionEnv(func, max_batch_size = 1, budget = budget)
                if method == 'SnAKe':
                    model = MultiObjectiveSnAKe(env, initial_temp = initial_point, exploration_constant = 0, merge_constant = 'lengthscale', objective_weights = [1, 1, 1])
                elif method == 'TrEI':
                    model = MultiObjectiveTrEI(env, initial_temp = initial_point, cost_switch = cost_change)
                elif method == 'EIpu':
                    model = MultiObjectiveEIpu(env, initial_temp = initial_point, cost_switch = cost_change)
                
                if obj_number == 0:
                    model.gp_hyperparams[0] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.5)
                    model.gp_hyperparams[1] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.8)
                    model.gp_hyperparams[2] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.5)
                else:
                    model.gp_hyperparams[0] = (0.12, torch.tensor([0.1, 0.1]), 1e-5, 0.5)

                X, Y = model.run_optim(verbose = True)

                if method in ['TrEI', 'EIpu']:
                    folder_inputs =  'ypacarai_results/' + f'objective{obj_number}' + str(cost_change) + '-' + method + '/inputs/'
                    folder_outputs = 'ypacarai_results/' + f'objective{obj_number}' + str(cost_change) + '-' + method + '/outputs/'
                    file_name = f'run_{run_num}'
                else:
                    folder_inputs =  'ypacarai_results/' + f'objective{obj_number}' + method + '/inputs/'
                    folder_outputs = 'ypacarai_results/' + f'objective{obj_number}' + method + '/outputs/'
                    file_name = f'run_{run_num}'
                # create directories if they exist
                os.makedirs(folder_inputs, exist_ok = True)
                os.makedirs(folder_outputs, exist_ok = True)
                # save the following ones
                np.save(folder_inputs + file_name, X)
                np.save(folder_outputs + file_name, np.array(Y))

if purpose == 'process_results':

    final_cost = 10
    mid_cost = 3.3
    mse_grid = np.load('mse_ypacarai_grid.npy')
    func = YpacaraiLake()
    for obj_number in [0, 1, 2, 3]:
        for method in ['SnAKe', 'EIpu', 'TrEI']:
            Regret1 = []
            Regret2 = []
            Regret3 = []
            Regret4 = []
            Costs = []
            num_of_samples = []
            for run_num in range(1, 11):
                if method in ['TrEI', 'EIpu']:
                    folder_inputs =  'ypacarai_results/' + f'objective{obj_number}' + str(cost_change) + '-' + method + '/inputs/'
                    folder_outputs = 'ypacarai_results/' + f'objective{obj_number}' + str(cost_change) + '-' + method + '/outputs/'
                    file_name = f'run_{run_num}.npy'
                else:
                    folder_inputs =  'ypacarai_results/' + f'objective{obj_number}' + method + '/inputs/'
                    folder_outputs = 'ypacarai_results/' + f'objective{obj_number}' + method + '/outputs/'
                    file_name = f'run_{run_num}.npy'

                X = np.load(folder_inputs + file_name)
                Y = np.load(folder_outputs + file_name)

                C = 0
                for i in range(len(X) - 1):
                    C = C + np.linalg.norm(X[i+1] - X[i])
                    if C > final_cost:
                        break
                

                x_optim1 = np.array([0.20916560292243958, 0.6638457775115967])
                x_optim2 = np.array([0.8908343315124512, 0.20615419745445251])
                x_optim3 = np.array([0.4012976884841919, 00.9626452326774597])
                x_optim4 = np.array([0.4275386929512024, 0.30612045526504517])
                optim_list = [x_optim1, x_optim2, x_optim3, x_optim4]

                regrets = []
                for j, optim in enumerate(optim_list):
                    distances = np.linalg.norm(X[:i+1, :] - optim, axis = 1)
                    closest_point = np.argmin(distances)
                    if j < 2:
                        y_star, _, _ = func.query_function(optim.reshape(1, -1))
                        y_best, _, _ = func.query_function(X[closest_point, :].reshape(1, -1))
                    elif j < 3:
                        _, y_star, _ = func.query_function(optim.reshape(1, -1))
                        _, y_best, _ = func.query_function(X[closest_point, :].reshape(1, -1))
                    else:
                        _, _, y_star = func.query_function(optim.reshape(1, -1))
                        _, _, y_best = func.query_function(X[closest_point, :].reshape(1, -1))
                    regrets.append(y_star - y_best)

                Regret1.append(regrets[0])
                Regret2.append(regrets[1])
                Regret3.append(regrets[2])
                Regret4.append(regrets[3])
                Costs.append(C)
                num_of_samples.append(i+1)


            Regret1 = np.array(Regret1)
            Regret2 = np.array(Regret2)
            Regret3 = np.array(Regret3)
            Regret4 = np.array(Regret4)

            print(method,f'Objective: {obj_number}',' : Reg1 : ', np.mean(Regret1) * 1000)
            print(method,f'Objective: {obj_number}',' : Std1 : ', np.std(Regret1) * 1000)
            print(method,f'Objective: {obj_number}',' : Reg2 : ', np.mean(Regret2) * 1000)
            print(method,f'Objective: {obj_number}',' : Std2 : ', np.std(Regret2) * 1000)
            print(method,f'Objective: {obj_number}',' : Reg3 : ', np.mean(Regret3) * 1000)
            print(method,f'Objective: {obj_number}',' : Std3 : ', np.std(Regret3) * 1000)
            print(method,f'Objective: {obj_number}',' : Reg4 : ', np.mean(Regret4) * 1000)
            print(method,f'Objective: {obj_number}',' : Std4 : ', np.std(Regret4) * 1000)
            print(method, ' : Cost : ', np.mean(Costs) )
            print(method, ' : Samples : ', np.mean(num_of_samples) )