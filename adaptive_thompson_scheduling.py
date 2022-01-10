from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem
import numpy as np
import math
from numpy.lib.arraysetops import unique
from gp_utils import SklearnGP, BoTorchGP
from temperature_env import NormalDropletFunctionEnv
from functions import GaussianMixture, TwoDSinCosine, BraninFunction, ConvergenceTest
import matplotlib.pyplot as plt
# newtorkx idea
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.linalg import norm
from scipy.stats.qmc import Sobol
import sobol_seq
from itertools import chain
from bayes_op import UCBwLP, ThompsonSampling
from sampling import EfficientThompsonSampler
import torch
import time

class AdaptiveThompsonScheduling():
    def __init__(self, env, initial_temp = None, \
        max_change = None, \
            gen_animation = False, \
                merge_method = 'Point Deletion', \
                    merge_constant = None, \
                        hp_update_frequency = None, \
                            num_of_multistarts = 10, \
                                cost_function = None):
        ''' 
        Method for Etch a Sketch Bayesian Optimization (previously called Adaptive Thompson Scheduling). 

        Required:
        env: An environment that runs the optimisation.
        
        Optional:
        initial_temp: The initial temperature of the optimisation.

        max change: Maximum change allowed in controlled variables. A float or None (default).

        merge method: Method for updating ordering. Default is Point Deletion.

        merge constant: Constant used in merging method, usually epsilon in Point Deletion. Default is None.

        hp_update_frequency: Frequency for re-training hyper-parameters. Default is None.

        num_of_multistarts: Number of multistarts when performing gradient optimisation on Thompson Samples. Will be multipled by
        problem dimension. Default is 10.

        cost_function: Function that takes two vectors and calculates the Cost matrix of the problem. Default uses simple Euclidean 
        distance.
        '''
        self.env = env
        self.max_change = max_change
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim

        if self.x_dim is not None:
            self.dim = self.t_dim + self.x_dim
        else:
            self.dim = self.t_dim

        if initial_temp is not None:
            self.initial_temp = initial_temp
        else:
            self.initial_temp = np.random.uniform(size = (1, self.dim))
        # define domain
        self.domain = np.zeros((self.dim,))
        self.domain = np.stack([self.domain, np.ones(self.dim, )], axis=1)
        # animation
        self.gen_animation = gen_animation
        # merge methods
        merge_methods = ['Point Deletion', 'Resampling', 'Moving Sample Size', 'e-Point Deletion']
        assert merge_method in merge_methods, 'Method Invalid, please use:' + str(merge_methods)
        # hyper-parameter update frequency
        self.hp_update_frequency = hp_update_frequency
        # number of multistarts for the gradient Thompson Samples
        self.num_of_multistarts = num_of_multistarts * self.dim

        # deletion method
        self.merge_method = merge_method
        self.merge_constant = merge_constant
        # cost function
        if cost_function is None:
            self.cost_function = distance_matrix
        else:
            self.cost_function = cost_function
        
        if self.merge_method == 'Moving Sample Size':
            assert merge_constant is not None, 'Moving Sample Size requires a sampling size value!'
            assert type(merge_constant) == int, 'Sampling size has to be an integer!'
            self.moving_sample_size = merge_constant

        self.initialise_stuff()
    
    def initialise_stuff(self):
        # list of future plans
        self.mega_batch = []
        # initialise count matrix
        self.count_list = None
        # list of queries
        self.queried_batch = []
        # list of queries and observations
        self.X = []
        self.Y = []
        # initial gp hyperparams
        self.set_hyperparams()
        # define model
        self.model = BoTorchGP(lengthscale_dim = self.dim)
        # try sobol sequences
        self.n_global = 100
        self.n_local = 25
        self.global_grid0 = sobol_seq.i4_sobol_generate(self.dim, self.n_global)
        self.local_grid0 = sobol_seq.i4_sobol_generate(self.dim, self.n_local) - 0.5
        self.local_grid = None
        # initialise working grid
        self.working_grid = None
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
        # initialise mega_batch 
        self.update_grid()
        self.create_mega_batch()
        self.order_mega_batch()
    
    def set_hyperparams(self, constant = None, lengthscale = None, noise = None, mean_constant = None, constraints = False):
        if constant == None:
            self.constant = 0.6
            self.length_scale = torch.tensor([0.15] * self.dim)
            self.noise = 1e-4
            self.mean_constant = 0
        
        else:
            self.length_scale = lengthscale
            self.noise = noise
            self.constant = constant
            self.mean_constant = mean_constant
        
        self.gp_hyperparams = (self.constant, self.length_scale, self.noise, self.mean_constant)
        # check if we want our constraints based on these hyperparams
        if constraints is True:
            self.model.define_constraints(self.length_scale, self.mean_constant, self.constant)
    
    def update_grid(self, diam = 0.1):
        '''
        Updates the grid: centering around the current temperature, optionally the diameter can be modified
        '''
        # set local grid around the current temperature
        if self.local_grid is None:
            self.local_grid = self.current_temp + diam * self.local_grid0
            # make sure it stays within the domain bounds
            for i in range(0, len(self.domain[:, 0])):
                self.local_grid[:, i] = np.maximum(self.local_grid[:, i], self.domain[i, 0])
                self.local_grid[:, i] = np.minimum(self.local_grid[:, i], self.domain[i, 1])
            # define new working grid
        self.working_grid = np.concatenate((self.global_grid0, self.local_grid), axis = 0)
    
    def run_optim(self, verbose = False):
        '''
        Runs the whole optimisation procedure, returns all queries and evaluations
        '''
        self.env.initialise_optim()
        while self.current_time <= self.budget:
            self.optim_loop()
            if verbose:
                print(f'Current time-step: {self.current_time}')
        X, Y = self.env.finished_with_optim()

        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)

        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        if (self.new_obs is not None) & (self.merge_method != 'Moving Sample Size'):
            # STEP 0: UPDATE SEARCH GRID
            self.update_grid()
            # STEP 1: CREATE MEGA BATCH
            self.create_mega_batch()
            # STEP 2: ARRANGE AND PLAN ORDER
            self.order_mega_batch()
        
        elif (self.merge_method == 'Moving Sample Size'):
            if (self.env.t % self.moving_sample_size == 0):
                # STEP 0: UPDATE SEARCH GRID
                self.update_grid()
                # STEP 1: CREATE MEGA BATCH
                self.create_mega_batch()
                # STEP 2: ARRANGE AND PLAN ORDER
                self.order_mega_batch()
        
        new_T = self.query_plan[0, :self.t_dim].reshape(1, -1)
        if self.max_change is not None:
            vec_norm = norm(new_T - self.current_temp[:, :self.t_dim])
            if vec_norm > self.max_change:
                new_T = self.current_temp[:, :self.t_dim] + self.max_change * (new_T - self.current_temp[:, :self.t_dim]) / vec_norm

        if self.env.x_dim == None:
            new_X = None
            query = new_T
        else:
            new_X = self.query_plan[0, self.t_dim:].reshape(-1, self.x_dim)
            query = np.concatenate((new_T, new_X), axis = 1)
        
        query = list(query.reshape(-1))
        
        self.query_plan = self.query_plan[1:, :]
        # queries used to be single numbers, which was fine, but now that they are multi-dimensional
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        self.queried_batch.append(query)
        # STEP 3: UPDATE MODEL
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
            self.update_model()
        
        if (self.hp_update_frequency is not None) & (len(self.X) > 0):
            if len(self.X) % self.hp_update_frequency == 0:
                self.model.optim_hyperparams()
                self.gp_hyperparams = self.model.current_hyperparams()
                print(f'New hyperparams: {self.model.current_hyperparams()}')
        
        self.current_temp = np.array(query).reshape(1, -1)
        self.current_time = self.current_time + 1
    
    def find_nearest_point(self, x0, new_sample):
        ''' 
        Takes as inputs:
        x0 - a (1 x d) array, where d is the dimension of the problem
        new_schedule - the new schedule, it is (budget x d) dimensional
        '''
        dist = np.linalg.norm(x0 - new_sample, axis = 1).reshape(-1, 1)
        min_dist = np.min(dist)
        idx = np.argmin(dist)
        if (self.merge_method == 'e-Point Deletion'):
            if min_dist > self.merge_constant:
                if new_sample.shape[0] == 1:
                    idx = 0
                else:
                    idx = np.random.randint(0, new_sample.shape[0] - 1)
        return idx
    
    def delete_points_from(self, new_schedule):
        # remove for each point in the batch
        self.deleted_points = []
        queried_batch_copy = self.queried_batch
        if self.merge_constant == 'Reverse':
            queried_batch_copy.reverse()
        for x0 in queried_batch_copy:
            # find the nearest point in the new schedule
            #new_schedule = self.working_grid[new_schedule_idx.reshape(-1), :]
            idx = self.find_nearest_point(x0, new_schedule)
            idx_range = np.arange(len(new_schedule))
            # save deleted point for animation
            if self.gen_animation == True:
                self.deleted_points.append(new_schedule[idx, :])

            new_schedule = new_schedule[idx_range != idx, :]
            # either remove, or subtract one from count list
            #if self.count_list[idx] == 1:
            #    new_schedule = new_schedule[idx_range!= idx, :]
            #    new_schedule_idx = new_schedule_idx[idx_range!= idx, :]
            #    self.count_list = self.count_list[idx_range!= idx, :]
            #else:
            #    self.count_list[idx] = self.count_list[idx] - 1
        return new_schedule

    def create_mega_batch(self):
        # get number of samples depending on the method
        if self.merge_method in ['Point Deletion', 'e-Point Deletion']:
            self.num_of_samples = self.budget + 1
        elif self.merge_method == 'Resampling':
            self.num_of_samples = self.budget + 1 - self.env.t
        elif self.merge_method == 'Moving Sample Size':
            self.num_of_samples = self.moving_sample_size
        
        # if there is no data yet, create batch uniformly
        if len(self.X) == 0:
            
            samples = np.random.uniform(size = (self.num_of_samples, self.dim))

            # sort samples by distance to current temperature
            dist_to_current = norm(self.current_temp - samples, axis = 1)
            sorted_samples = np.array([list(s) for _, s in sorted(zip(dist_to_current, samples), \
                key = lambda z: z[0])]).reshape(-1, self.dim)
            # define local and global samples
            self.local_grid = sorted_samples[:self.n_local, :]
            self.update_grid()
            # find distances to each grid point
            distances_to_grid = distance_matrix(self.working_grid, samples)
            max_idx = np.argmin(distances_to_grid, axis = 0).squeeze()
            # obtain unique samples and count list
            unique_sample_idx, self.count_list = np.unique(max_idx, axis = 0, return_counts = True)
            unique_sample_idx = unique_sample_idx.reshape(-1, 1)
            self.count_list = self.count_list.reshape(-1, 1)
            self.unique_samples = self.working_grid[unique_sample_idx.reshape(-1), :]
            # finally define the temperature samples
            self.temperature_samples = self.unique_samples[:, :self.t_dim]

            if self.gen_animation:
                self.out0 = ['raw samples', self.unique_samples, self.env.temperature_list, self.X]
                self.out1 = ['deleted samples', self.unique_samples, self.env.temperature_list, self.X]

            # add current temperature if needed
            if self.current_temp[:, :self.t_dim] in self.temperature_samples:
                self.current_temp_in_sample = True
                self.current_temp_idx = np.where(self.current_temp == self.temperature_samples)[0][0]
            else:
                self.current_temp_in_sample = False
                self.temperature_samples = np.concatenate((self.current_temp[:, :self.t_dim], self.temperature_samples))
                self.unique_samples = np.concatenate((self.current_temp, self.unique_samples))
                self.current_temp_idx = 0
            return

        if self.merge_method == 'Exploration Bias':
            # obtain standard deviation from posterior
            _, standard_dev = self.model.posterior(self.working_grid)
            exploration_bias = standard_dev.reshape(-1, 1) * self.exploration_bias_constant
            samples = self.model.sample(self.working_grid, n_samples = self.num_of_samples)
            samples = samples.reshape(len(self.working_grid), self.num_of_samples)
            samples = samples + exploration_bias
        else:
            # define the samples
            sampler = EfficientThompsonSampler(self.model, num_of_multistarts = self.num_of_multistarts, \
                num_of_bases = 1024, \
                    num_of_samples = self.num_of_samples)
            # create samples
            sampler.create_sample()
            # optimise samples
            samples = sampler.generate_candidates()
            samples = samples.numpy()
            
            if self.gen_animation:
                self.out0 = ['raw samples', samples, self.env.temperature_list.copy(), self.X.copy()]
            # delete points if necessary
            if self.merge_method in ['Point Deletion', 'e-Point Deletion']:
                samples = self.delete_points_from(samples)

            # sort samples by distance to current temperature
            dist_to_current = norm(self.current_temp - samples, axis = 1)
            sorted_samples = np.array([list(s) for _, s in sorted(zip(dist_to_current, samples), \
                key = lambda z: z[0])]).reshape(-1, self.dim)
            # define local and global samples
            self.local_grid = sorted_samples[:self.n_local, :]
            self.update_grid()

            distances_to_grid = distance_matrix(self.working_grid, samples)
            max_idx = np.argmin(distances_to_grid, axis = 0).squeeze()
            #samples = self.model.sample(self.working_grid, n_samples = self.num_of_samples)
            #samples = samples.reshape(len(self.working_grid), self.num_of_samples)


        #max_idx = np.argmax(samples, axis = 1).squeeze()
        if (self.merge_method == 'Exploration Bias') & (self.env.t == self.budget):
            max_idx = max_idx.reshape(-1)

        if self.current_time == self.budget:
            max_idx = [max_idx]

        # make them unique, obtain counts
        unique_sample_idx, self.count_list = np.unique(max_idx, axis = 0, return_counts = True)

        # reshape
        unique_sample_idx = unique_sample_idx.reshape(-1, 1)
        self.count_list = self.count_list.reshape(-1, 1)

        if self.gen_animation:
            self.out1 = ['deleted samples', self.working_grid[unique_sample_idx, :].copy(), self.env.temperature_list.copy(), self.X.copy()]
        self.unique_samples = self.working_grid[unique_sample_idx.reshape(-1), :]
        # ordered_queries = ordered_queries[ordered_queries[:, 0].argsort()][self.env.t:]
        # self.temperature_ordered = self.ordered_queries[:, 0]
        # self.ordered_queries = ordered_queries
        # pick out temperature from samples for sorting
        self.temperature_samples = self.unique_samples[:, :self.t_dim]
        # add current temperature
        if self.current_temp[:, :self.t_dim] in self.temperature_samples:
            self.current_temp_in_sample = True
            self.current_temp_idx = np.where(self.current_temp[:, :self.t_dim] == self.temperature_samples)[0][0]
        else:
            self.current_temp_in_sample = False
            self.temperature_samples = np.concatenate((self.current_temp[:, :self.t_dim], self.temperature_samples))
            self.unique_samples = np.concatenate((self.current_temp, self.unique_samples))
            self.current_temp_idx = 0
    
    def order_mega_batch(self):
        # create distance matrix
        dist = self.cost_function(self.temperature_samples, self.temperature_samples)
        # check if we only have one point
        if (dist == 0).all():
            # what if we only have one point, stay there
            new_path_idx = [0]
            expanded_path_idx = [[idx] * self.count_list[0][0] for idx in new_path_idx]
            new_path_idx = list(chain.from_iterable(expanded_path_idx))
            #self.query_plan = self.temperature_samples[new_path_idx, :]
            self.query_plan = self.unique_samples[new_path_idx, :]
            return None

        # define graph
        G = nx.convert_matrix.from_numpy_array(dist)
        # define tsp and method
        tsp = nx.algorithms.approximation.traveling_salesman_problem
        # TA_tsp = nx.algorithms.approximation.threshold_accepting_tsp
        SA_tsp = nx.algorithms.approximation.simulated_annealing_tsp
        method = lambda G, wt: SA_tsp(G, 'greedy', weight = wt, source = self.current_temp_idx)
        # obtain new path
        new_path_idx = tsp(G, cycle = True, method = method)

        # remove current_temp
        if self.current_temp_in_sample:
            new_path_idx = new_path_idx[:-1]
            # expand path list
            expanded_path_idx = [[idx] * self.count_list[idx][0] for idx in new_path_idx]
        else:
            new_path_idx = new_path_idx[1:-1]
            # expand path list
            expanded_path_idx = [[idx] * self.count_list[idx - 1][0] for idx in new_path_idx]
        new_path_idx = list(chain.from_iterable(expanded_path_idx))
        # finally, update query plan
        #self.query_plan = self.temperature_samples[new_path_idx, :]
        self.query_plan = self.unique_samples[new_path_idx, :]

        if self.gen_animation:
            self.out2 = ['new_plan', self.query_plan.copy(), self.env.temperature_list.copy(), self.X.copy()]
    
    def update_model(self):
        if self.x_dim == None:
            dim = self.t_dim
        else:
            dim = self.t_dim + self.x_dim

        X_numpy = np.array(self.X).reshape(-1, dim)
        #Y = np.array([y for y in self.Y])

        if self.new_obs is not None:
            self.model.fit_model(X_numpy, self.Y, previous_hyperparams = self.gp_hyperparams)

class RandomTSP():
    def __init__(self, env, initial_temp = None, \
            gen_animation = False, \
                cost_function = None):
        ''' 
        Method for creating a random sample and ordering using TSP.

        Required:
        env: An environment that runs the optimisation.
        
        Optional:
        initial_temp: The initial temperature of the optimisation. 

        cost_function: Function that takes two vectors and calculates the Cost matrix of the problem. Default uses simple Euclidean 
        distance.
        '''
        self.env = env
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim

        if self.x_dim is not None:
            self.dim = self.t_dim + self.x_dim
        else:
            self.dim = self.t_dim

        if initial_temp is not None:
            self.initial_temp = initial_temp
        else:
            self.initial_temp = np.random.uniform(size = (1, self.dim))
        # define domain
        self.domain = np.zeros((self.dim,))
        self.domain = np.stack([self.domain, np.ones(self.dim, )], axis=1)
        # animation
        self.gen_animation = gen_animation

        # cost function
        if cost_function is None:
            self.cost_function = distance_matrix
        else:
            self.cost_function = cost_function

        self.initialise_stuff()
    
    def initialise_stuff(self):
        # list of future plans
        self.mega_batch = []
        # initialise count matrix
        self.count_list = None
        # list of queries
        self.queried_batch = []
        # list of queries and observations
        self.X = []
        self.Y = []
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
        # initialise mega_batch 
        self.create_mega_batch()
        self.order_mega_batch()
    
    def set_hyperparams(self, constant, lengthscale, noise, mean_constant, constraints):
        pass

    def run_optim(self, verbose = False):
        '''
        Runs the whole optimisation procedure, returns all queries and evaluations
        '''
        self.env.initialise_optim()
        while self.current_time <= self.budget:
            self.optim_loop()
            if verbose:
                print(f'Current time-step: {self.current_time}')
        X, Y = self.env.finished_with_optim()

        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)

        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        new_T = self.query_plan[0, :self.t_dim].reshape(1, -1)

        if self.env.x_dim == None:
            new_X = None
            query = new_T
        else:
            new_X = self.query_plan[0, self.t_dim:].reshape(-1, self.x_dim)
            query = np.concatenate((new_T, new_X), axis = 1)
        
        query = list(query.reshape(-1))
        
        self.query_plan = self.query_plan[1:, :]
        # queries used to be single numbers, which was fine, but now that they are multi-dimensional
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        self.queried_batch.append(query)

        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
        
        self.current_temp = np.array(query).reshape(1, -1)
        self.current_time = self.current_time + 1

    def create_mega_batch(self):
        self.num_of_samples = self.budget + 1
        # create batch uniformly        
        # samples = np.random.uniform(size = (self.num_of_samples, self.dim))
        # create samples using sobol grid
        sobol = Sobol(self.dim)
        samples = sobol.random(self.num_of_samples)
        self.unique_samples = samples
        self.count_list = list(np.ones(self.num_of_samples))
        # finally define the temperature samples
        self.temperature_samples = self.unique_samples[:, :self.t_dim]

        if self.gen_animation:
            self.out0 = ['raw samples', self.unique_samples, self.env.temperature_list, self.X]
            self.out1 = ['deleted samples', self.unique_samples, self.env.temperature_list, self.X]

        # add current temperature if needed
        if self.current_temp[:, :self.t_dim] in self.temperature_samples:
            self.current_temp_in_sample = True
            self.current_temp_idx = np.where(self.current_temp == self.temperature_samples)[0][0]
        else:
            self.current_temp_in_sample = False
            self.temperature_samples = np.concatenate((self.current_temp[:, :self.t_dim], self.temperature_samples))
            self.unique_samples = np.concatenate((self.current_temp, self.unique_samples))
            self.current_temp_idx = 0
        return

    
    def order_mega_batch(self):
        # create distance matrix
        dist = self.cost_function(self.temperature_samples, self.temperature_samples)
        # check if we only have one point
        if (dist == 0).all():
            # what if we only have one point, stay there
            new_path_idx = [0]
            expanded_path_idx = [[idx] * self.count_list[0][0] for idx in new_path_idx]
            new_path_idx = list(chain.from_iterable(expanded_path_idx))
            #self.query_plan = self.temperature_samples[new_path_idx, :]
            self.query_plan = self.unique_samples[new_path_idx, :]
            return None

        # define graph
        G = nx.convert_matrix.from_numpy_array(dist)
        # define tsp and method
        tsp = nx.algorithms.approximation.traveling_salesman_problem
        # TA_tsp = nx.algorithms.approximation.threshold_accepting_tsp
        SA_tsp = nx.algorithms.approximation.simulated_annealing_tsp
        method = lambda G, wt: SA_tsp(G, 'greedy', weight = wt, source = self.current_temp_idx)
        # obtain new path
        new_path_idx = tsp(G, cycle = True, method = method)

        if self.gen_animation:
            self.out2 = ['new_plan', self.unique_samples[new_path_idx[:-1], :], self.env.temperature_list, self.X]

        # remove current_temp
        if self.current_temp_in_sample:
            new_path_idx = new_path_idx[:-1]
            # expand path list
            expanded_path_idx = [[idx] * self.count_list[idx][0] for idx in new_path_idx]
        else:
            new_path_idx = new_path_idx[1:-1]
            # expand path list
            expanded_path_idx = [[idx] * int(self.count_list[idx - 1]) for idx in new_path_idx]
        new_path_idx = list(chain.from_iterable(expanded_path_idx))
        # finally, update query plan
        #self.query_plan = self.temperature_samples[new_path_idx, :]
        self.query_plan = self.unique_samples[new_path_idx, :]

if __name__ == '__main__':

    exp = '1D'

    if exp == '1D':

        budget = 250
        max_batch_size = 1
        epsilon_list = [0.1]
        methods = ['e-Point Deletion']
        colors = ['b', 'tab:red']
        max_change = None

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        fig.set_figheight(3.5)
        fig.set_figwidth(7)
        #title = f'Budget = {budget} : Time delay = {max_batch_size - 1}'
        #fig.suptitle(title)
        pt = 0.7362
        escape_prediction = budget * pt

        np.random.seed(2022)

        for i in range(0, len(epsilon_list)):
            initial_temp = np.array([0]).reshape(1, 1)
            epsilon = epsilon_list[i]
            #np.random.seed(47)
            func = ConvergenceTest()
            env = NormalDropletFunctionEnv(func, budget = budget, max_batch_size = max_batch_size)
            model = AdaptiveThompsonScheduling(env, max_change = max_change, merge_method = methods[i], \
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
            '''
            ax[0].set_title(title) #do this for more in the list
            ax[0].set_title(title)
            ax[0].scatter(X, Y, s = 50, marker = 'x', c = 'r')
            if i == 0:
                ax[0].set_ylabel('Observations')
            ax[0].plot(grid, target_func, '--k', label = 'True function')
            ax[0].plot(grid, posterior_mean.detach().numpy(), 'b', label = 'GP mean')
            ax[0].fill_between(grid.reshape(-1), posterior_mean.detach() - 1.96 * posterior_sd.detach(), \
                 posterior_mean.detach() + 1.96 * posterior_sd.detach(), alpha = 0.2)
            ax[0].set_xlim(0, 1)
            ax[0].grid()
            ax[0].legend(loc = 'lower right')
            ax[1].plot(X, times)
            ax[1].set_xlabel('x')
            if i == 0:
                ax[1].set_ylabel('Iteration')
            ax[1].set_xlim(0, 1)
            ax[1].grid()
            ax[1].hlines(escape_prediction, 0, 1, colors = 'g', linestyles = '--', label = 'Escape Prediction')
            ax[1].legend(loc = 'lower right')
            if i == 1:
                label = '0.1-Point Deletion'
            else:
                label = 'Resampling'
            ax.plot(X, times, c = colors[i], label = label)
            ax.set_xlabel('x')
            if i == 0:
                ax.set_ylabel('Iteration')
            ax.set_xlim(0, 1)
            ax.grid()
            if i == 1:
                ax.hlines(escape_prediction, 0, 1, colors = colors[i], linestyles = '--', label = 'Escape Prediction')
            ax.legend(loc = 'lower right')

            '''
            ax.scatter(X, Y, s = 50, marker = 'x', c = 'k')
            if i == 0:
                ax.set_ylabel('f(x)')
            ax.plot(grid, target_func, '--k', label = 'True function')
            ax.plot(grid, posterior_mean.detach().numpy(), 'r', label = 'GP mean')
            ax.fill_between(grid.reshape(-1), posterior_mean.detach() - 1.96 * posterior_sd.detach(), \
                 posterior_mean.detach() + 1.96 * posterior_sd.detach(), alpha = 0.2, color = 'r')
            ax.set_xlim(0, 1)
            ax.grid()
            ax.legend(loc = 'lower right')
        
        filename = 'PDvsRS_PointDeletion_model' + '.pdf'
        plt.savefig(filename, bbox_inches = 'tight')
        plt.show()
    
    if exp == '2D':

        budget = 250
        max_batch_size = 10
        max_change_list = [None]

        fig, ax = plt.subplots(nrows = 2, ncols = 3, gridspec_kw={'height_ratios': [6, 3], 'width_ratios': [3, 6, 3]})
        fig.set_figheight(6)
        fig.set_figwidth(8 * len(max_change_list))
        title = f'Budget = {budget} : Time delay = {max_batch_size}'
        fig.suptitle(title)

        for i in range(0, len(max_change_list)):
            np.random.seed(47)
            max_change = max_change_list[i]
            initial_temp = np.array([0.5, 0.5]).reshape(1, -1)
            func = TwoDSinCosine(random=True)
            env = NormalDropletFunctionEnv(func, budget = budget, max_batch_size = max_batch_size)
            model = ThompsonSampling(env)
            #model = AdaptiveThompsonScheduling(env, max_change = max_change, initial_temp = initial_temp)
            X, Y = model.run_optim(verbose = True)

            #model.reset_model(initial_constant = 0.01)
            #Xo, Yo = model.run_optim()

            x_grid = np.linspace(0, 1, 101)
            y_grid = np.linspace(0, 1, 101)
            xmesh, ymesh = np.meshgrid(x_grid, y_grid)
            lambda_func = lambda x, y: np.sin(5*(x - func.mu1)) * np.cos(5*(y - func.mu2)) * np.exp((x-0.5)**2 / 2) * np.exp((y-0.5)**2 / 2)
            z_grid = lambda_func(xmesh, ymesh)

            times = np.array(range(0, env.budget+1)).reshape(-1, 1)

            #title = f'$\Delta x$ = {max_change}.'
            ax[0, 0].plot(times, X[:, 1])
            ax[0, 0].set_xlabel('time')
            ax[0, 0].set_ylabel('$x_2$')
            ax[0, 0].set_ylim(0, 1)
            #ax[0, i].scatter(X[:, 0], X[:, 1], s = 50, marker = 'x', c = 'r')
            ax[1, 1].plot(X[:, 0], times)
            ax[1, 1].set_xlabel('$x_1$')
            ax[1, 1].set_ylabel('times')
            ax[1, 1].set_xlim(0, 1)

            ax[1, 0].axis('off')
            ax[1, 2].axis('off')
            ax[0, 2].axis('off')

            ax[0, 1].plot(X[:, 0], X[:, 1], marker = 'x', c = 'r', alpha = 0.4)
            #ax[i].plot(Xo[:, 0], Xo[:, 1], marker = 'x', c = 'b', alpha = 0.2)
            b = ax[0, 1].contourf(x_grid, y_grid, z_grid, levels = 50)
            fig.colorbar(b, ax = ax[0, 2])
            ax[0, 1].set_xlim(0, 1)
            ax[0, 1].set_ylim(0, 1)
            ax[0, 1].grid()
            #ax[0, 1].set_xlabel('$x_1$')
            #ax[0, 1].set_ylabel('$x_2$')
            

        plt.show()
    
    if exp == 'UCBwLP':

        budget = 100
        max_batch_size = 10

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        #func = TwoDSinCosine(random = False)
        func = BraninFunction()
        x_grid = np.linspace(0, 1, 101)
        y_grid = np.linspace(0, 1, 101)
        xmesh, ymesh = np.meshgrid(x_grid, y_grid)
        #lambda_func = lambda x, y: np.sin(5*(x - func.mu1)) * np.cos(5*(y - func.mu2)) * np.exp((x-0.5)**2 / 2) * np.exp((y-0.5)**2 / 2)
        lambda_func = lambda x, y: -((15*y - 5.1 * (15*x-5)**2 / (4 * math.pi**2) + 5 * (15*x-5) / math.pi - 6)**2 \
            + (10 - 10 / (8 * math.pi)) * np.cos((15*x-5)) - 44.81)/51.95
        z_grid = lambda_func(xmesh, ymesh)

        real_path, = ax.plot([], [], "r", label = "Real path")

        func = BraninFunction()
        env = NormalDropletFunctionEnv(func, budget, max_batch_size)
        model = UCBwLP(env)
        X, Y = model.run_optim(verbose = True)

        contour_plot = ax.contourf(x_grid, y_grid, z_grid, levels = 100)
        fig.colorbar(contour_plot, ax = ax)

        real_path.set_data(X[:, 0], X[:, 1])
        ax.scatter(X[:, 0], X[:, 1], c = 'k', marker = 'x', s = 25)

        plt.show()
        'hola'