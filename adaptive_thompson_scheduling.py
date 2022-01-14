from networkx.algorithms.approximation.traveling_salesman import traveling_salesman_problem
import numpy as np
import math
from numpy.lib.arraysetops import unique
from gp_utils import SklearnGP, BoTorchGP
from temperature_env import NormalDropletFunctionEnv
from functions import BraninFunction, ConvergenceTest
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
        Method for Etch a Sketch Bayesian Optimization (previously called Adaptive Thompson Scheduling). For the code, we call
        'Temperature' all the variables that incur input cost, they should always be the first variables in the system, that is:
        X[:self.t_dim] - all variables that incur input cost
        X[self.t_dim:] - all variables that incur no input cost (there are self.x_dim of these ones)

        Required:
        env: An environment that runs the optimisation.
        
        Optional:
        initial_temp: The initial temperature of the optimisation.

        max change: Maximum change allowed in controlled variables. A float or None (default). This concept while interesting, was dropped
        for the paper to avoid cluttering it.

        merge method: Method for updating ordering. Default is Point Deletion, equivalent to e-Point Deletion with e = infinity.

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
        # define domain, we will use unit hypercube. Make sure the function class transforms to other bounds.
        self.domain = np.zeros((self.dim,))
        self.domain = np.stack([self.domain, np.ones(self.dim, )], axis=1)
        # if you want to create an animation
        self.gen_animation = gen_animation
        # merge methods
        merge_methods = ['Point Deletion', 'Resampling', 'Moving Sample Size', 'e-Point Deletion']
        assert merge_method in merge_methods, 'Method Invalid, please use:' + str(merge_methods)
        # hyper-parameter update frequency
        self.hp_update_frequency = hp_update_frequency
        # number of multistarts for the gradient Thompson Samples
        self.num_of_multistarts = num_of_multistarts * self.dim

        # parameters of deletion method
        self.merge_method = merge_method
        self.merge_constant = merge_constant
        # cost function
        if cost_function is None:
            self.cost_function = distance_matrix
        else:
            self.cost_function = cost_function
        # method of merging or updating schedule called 'moving sample size', not in paper to avoid cluttering
        if self.merge_method == 'Moving Sample Size':
            assert merge_constant is not None, 'Moving Sample Size requires a sampling size value!'
            assert type(merge_constant) == int, 'Sampling size has to be an integer!'
            self.moving_sample_size = merge_constant
        # initialise the optimisation
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
        # define Sobol sequences
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
        # initialise mega_batch (batch that approximates future observations)
        self.update_grid()
        self.create_mega_batch()
        self.order_mega_batch()
    
    def set_hyperparams(self, constant = None, lengthscale = None, noise = None, mean_constant = None, constraints = False):
        '''
        This function is used to set the hyper-parameters of the GP.
        INPUTS:
        constant: positive float, multiplies the RBF kernel and defines the initital variance
        lengthscale: tensor of positive floats of length (dim), defines the kernel of the rbf kernel
        noise: positive float, noise assumption
        mean_constant: float, value of prior mean
        constraints: boolean, if True, we will apply constraints from paper based on the given hyperparameters
        '''
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
        Initially we use a local Sobol grid, during optimization we replace the grid with the closest samples
        to the current temperature.
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
        # obtain all queries
        X, Y = self.env.finished_with_optim()
        # reformat all queries before returning
        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)

        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # check if we have received a new observation
        if (self.new_obs is not None):
            # STEP 0: UPDATE SEARCH GRID
            self.update_grid()
            # STEP 1: CREATE MEGA BATCH
            self.create_mega_batch()
            # STEP 2: ARRANGE AND PLAN ORDER
            self.order_mega_batch()
        
        # obtain new query from plans
        new_T = self.query_plan[0, :self.t_dim].reshape(1, -1)
        # check if planned change is larger than biggest allowed jump (not in paper)
        if self.max_change is not None:
            vec_norm = norm(new_T - self.current_temp[:, :self.t_dim])
            # if it is larger, restrict the jump size
            if vec_norm > self.max_change:
                new_T = self.current_temp[:, :self.t_dim] + self.max_change * (new_T - self.current_temp[:, :self.t_dim]) / vec_norm
        # check if all variables incur input cost
        if self.env.x_dim == None:
            new_X = None
            query = new_T
        else:
            new_X = self.query_plan[0, self.t_dim:].reshape(-1, self.x_dim)
            query = np.concatenate((new_T, new_X), axis = 1)
        # reshape query
        query = list(query.reshape(-1))
        # re-define the query plan without current query
        self.query_plan = self.query_plan[1:, :]
        # step in the environment
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        # append the query to our query batch
        self.queried_batch.append(query)
        # STEP 3: UPDATE MODEL
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
            self.update_model()
        # check the update frequency and make sure we actually have data to train
        if (self.hp_update_frequency is not None) & (len(self.X) > 0):
            if len(self.X) % self.hp_update_frequency == 0:
                self.model.optim_hyperparams()
                self.gp_hyperparams = self.model.current_hyperparams()
                print(f'New hyperparams: {self.model.current_hyperparams()}')
        # update current temperature and current time
        self.current_temp = np.array(query).reshape(1, -1)
        self.current_time = self.current_time + 1
    
    def find_nearest_point(self, x0, new_sample):
        ''' 
        Takes as inputs:
        x0 - a (1 x d) array, where d is the dimension of the problem
        new_schedule - the new schedule, it is (budget x d) dimensional
        Returns:
        the nearest point in the new schedule to x0

        if we are using epsilon point deletion, it may return a random point if the minimum distance is greater than epsilon
        '''
        # find all distances
        dist = np.linalg.norm(x0 - new_sample, axis = 1).reshape(-1, 1)
        # find minimum distance
        min_dist = np.min(dist)
        # idx of minimum distance
        idx = np.argmin(dist)
        # if doing e-Point deletion, check minimum distance against epsilon
        if (self.merge_method == 'e-Point Deletion'):
            # check if minimum distance is larger than epsilon
            if min_dist > self.merge_constant:
                if new_sample.shape[0] == 1:
                # if new schedule / sample is one-dimensional return the only possible value
                    idx = 0
                else:
                # else return a random number
                    idx = np.random.randint(0, new_sample.shape[0] - 1)
        return idx
    
    def delete_points_from(self, new_schedule):
        '''
        This function carries out the Point Deletion on a batch of Thompson Samples
        '''
        # save deleted point for creating graphics and animations
        self.deleted_points = []
        # copy the query batch to avoid python variable issues
        queried_batch_copy = self.queried_batch
        # loop through all previously queried points
        for x0 in queried_batch_copy:
            # find the nearest point in the new schedule
            idx = self.find_nearest_point(x0, new_schedule)
            # define the idx range for comprehension to delete the point
            idx_range = np.arange(len(new_schedule))
            # save deleted point for animation
            if self.gen_animation == True:
                self.deleted_points.append(new_schedule[idx, :])
            # define new schedule without deleted point
            new_schedule = new_schedule[idx_range != idx, :]
        return new_schedule

    def create_mega_batch(self):
        '''
        This function creates the batch of Thompson Samples that approximate the future
        '''
        # get number of samples depending on the method
        if self.merge_method in ['Point Deletion', 'e-Point Deletion']:
            self.num_of_samples = self.budget + 1
        elif self.merge_method == 'Resampling':
            self.num_of_samples = self.budget + 1 - self.env.t
        
        # if there is no data yet, create batch uniformly
        if len(self.X) == 0:
            # define samples randomly, uniformly
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
            # obtain unique samples and count list, remember there might be repetition since we assigned to adaptive grid
            unique_sample_idx, self.count_list = np.unique(max_idx, axis = 0, return_counts = True)
            unique_sample_idx = unique_sample_idx.reshape(-1, 1)
            self.count_list = self.count_list.reshape(-1, 1)
            self.unique_samples = self.working_grid[unique_sample_idx.reshape(-1), :]
            # finally define the temperature samples
            self.temperature_samples = self.unique_samples[:, :self.t_dim]
            # output information to generate animation
            if self.gen_animation:
                self.out0 = ['raw samples', self.unique_samples, self.env.temperature_list, self.X]
                self.out1 = ['deleted samples', self.unique_samples, self.env.temperature_list, self.X]
            # add current temperature if not in the batch, we need it to use as source of Travelling Salesman
            if self.current_temp[:, :self.t_dim] in self.temperature_samples:
                self.current_temp_in_sample = True
                self.current_temp_idx = np.where(self.current_temp == self.temperature_samples)[0][0]
            else:
                self.current_temp_in_sample = False
                self.temperature_samples = np.concatenate((self.current_temp[:, :self.t_dim], self.temperature_samples))
                self.unique_samples = np.concatenate((self.current_temp, self.unique_samples))
                self.current_temp_idx = 0
            return

        # define the sampler
        sampler = EfficientThompsonSampler(self.model, num_of_multistarts = self.num_of_multistarts, \
            num_of_bases = 1024, \
                num_of_samples = self.num_of_samples)
        # create samples
        sampler.create_sample()
        # optimise samples
        samples = sampler.generate_candidates()
        samples = samples.numpy()
        # outputs information for graphics and animation
        if self.gen_animation:
            self.out0 = ['raw samples', samples, self.env.temperature_list.copy(), self.X.copy()]
        # delete points if necessary
        if self.merge_method in ['Point Deletion', 'e-Point Deletion']:
            samples = self.delete_points_from(samples)

        # sort samples by distance to current temperature, to define the local grid
        dist_to_current = norm(self.current_temp - samples, axis = 1)
        sorted_samples = np.array([list(s) for _, s in sorted(zip(dist_to_current, samples), \
            key = lambda z: z[0])]).reshape(-1, self.dim)
        # define local and global samples
        self.local_grid = sorted_samples[:self.n_local, :]
        self.update_grid()
        # assign remaining points to grid
        # assign remaining points to grid: step 1, find the the point in the grid closest to each sample
        distances_to_grid = distance_matrix(self.working_grid, samples)
        max_idx = np.argmin(distances_to_grid, axis = 0).squeeze()
        # if we are at the last time-step, we run into trouble without the following step
        if self.current_time == self.budget:
            max_idx = [max_idx]

        # assign remaining points to grid: step 2, make them unique, obtain counts (how many samples were assigned to each grid-point)
        unique_sample_idx, self.count_list = np.unique(max_idx, axis = 0, return_counts = True)
        # reshape
        unique_sample_idx = unique_sample_idx.reshape(-1, 1)
        self.count_list = self.count_list.reshape(-1, 1)
        # if we are doing animation, output information
        if self.gen_animation:
            self.out1 = ['deleted samples', self.working_grid[unique_sample_idx, :].copy(), self.env.temperature_list.copy(), self.X.copy()]
        # assign remaining points to grid: step 3, finally re-define the samples to the corresponding grid-point
        self.unique_samples = self.working_grid[unique_sample_idx.reshape(-1), :]
        # pick out temperature variables from samples
        self.temperature_samples = self.unique_samples[:, :self.t_dim]
        # add current temperature if needed, to use as source of travelling salesman
        if self.current_temp[:, :self.t_dim] in self.temperature_samples:
            self.current_temp_in_sample = True
            self.current_temp_idx = np.where(self.current_temp[:, :self.t_dim] == self.temperature_samples)[0][0]
        else:
            self.current_temp_in_sample = False
            self.temperature_samples = np.concatenate((self.current_temp[:, :self.t_dim], self.temperature_samples))
            self.unique_samples = np.concatenate((self.current_temp, self.unique_samples))
            self.current_temp_idx = 0
    
    def order_mega_batch(self):
        '''
        This function orders the mega-batch to reduce input cost
        '''
        # create distance / cost matrix
        dist = self.cost_function(self.temperature_samples, self.temperature_samples)
        # check if we only have one point
        if (dist == 0).all():
            # if we only have one point, stay there
            new_path_idx = [0]
            expanded_path_idx = [[idx] * self.count_list[0][0] for idx in new_path_idx]
            new_path_idx = list(chain.from_iterable(expanded_path_idx))
            self.query_plan = self.unique_samples[new_path_idx, :]
            return None

        # define graph
        G = nx.convert_matrix.from_numpy_array(dist)
        # define tsp and method
        tsp = nx.algorithms.approximation.traveling_salesman_problem
        SA_tsp = nx.algorithms.approximation.simulated_annealing_tsp
        method = lambda G, wt: SA_tsp(G, 'greedy', weight = wt, source = self.current_temp_idx)
        # obtain new path
        new_path_idx = tsp(G, cycle = True, method = method)

        # remove current_temperature
        if self.current_temp_in_sample:
            new_path_idx = new_path_idx[:-1]
            # expand path list using counts (i.e. if a point was assigned two samples, we should 'visit it twice')
            expanded_path_idx = [[idx] * self.count_list[idx][0] for idx in new_path_idx]
        else:
            new_path_idx = new_path_idx[1:-1]
            # expand path list using counts (i.e. if a point was assigned two samples, we should 'visit it twice')
            expanded_path_idx = [[idx] * self.count_list[idx - 1][0] for idx in new_path_idx]
        # define new path
        new_path_idx = list(chain.from_iterable(expanded_path_idx))
        # finally, update query plan
        self.query_plan = self.unique_samples[new_path_idx, :]
        # save information for variable names
        if self.gen_animation:
            self.out2 = ['new_plan', self.query_plan.copy(), self.env.temperature_list.copy(), self.X.copy()]
    
    def update_model(self):
        '''
        This function updates the GP model
        '''
        # get input dimension correct
        if self.x_dim == None:
            dim = self.t_dim
        else:
            dim = self.t_dim + self.x_dim
        # reshape the data correspondingly
        X_numpy = np.array(self.X).reshape(-1, dim)
        # update model
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

        gen_animation: Used to save extra data to be able to create graphics and animations
        '''
        # define problem parameters
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
        # define domain, which will be unit cube, any others should be re-defined in function class
        self.domain = np.zeros((self.dim,))
        self.domain = np.stack([self.domain, np.ones(self.dim, )], axis=1)
        # save data for animation boolean
        self.gen_animation = gen_animation

        # cost function
        if cost_function is None:
            self.cost_function = distance_matrix
        else:
            self.cost_function = cost_function

        # initialise problem      
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
        # will not use GP so no need to update hyper-parameters
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
        # obtain all queries and observations
        X, Y = self.env.finished_with_optim()
        # reformat queries before returning
        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)

        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # obtain new query from query plan
        new_T = self.query_plan[0, :self.t_dim].reshape(1, -1)
        # check if we need to obtain non-temperature part of the query
        if self.env.x_dim == None:
            new_X = None
            query = new_T
        else:
            new_X = self.query_plan[0, self.t_dim:].reshape(-1, self.x_dim)
            query = np.concatenate((new_T, new_X), axis = 1)
        # redefine query, and reshape
        query = list(query.reshape(-1))
        # update query plan
        self.query_plan = self.query_plan[1:, :]
        # step in the environment with new query
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        # update queried batch
        self.queried_batch.append(query)
        # update data
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
        # update current temperature and current time
        self.current_temp = np.array(query).reshape(1, -1)
        self.current_time = self.current_time + 1

    def create_mega_batch(self):
        self.num_of_samples = self.budget + 1
        # create samples using sobol grid
        sobol = Sobol(self.dim)
        samples = sobol.random(self.num_of_samples)
        self.unique_samples = samples
        self.count_list = list(np.ones(self.num_of_samples))
        # finally define the temperature samples
        self.temperature_samples = self.unique_samples[:, :self.t_dim]
        # if creating graphics or animation
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
            self.query_plan = self.unique_samples[new_path_idx, :]
            return None

        # define graph
        G = nx.convert_matrix.from_numpy_array(dist)
        # define tsp and method
        tsp = nx.algorithms.approximation.traveling_salesman_problem
        SA_tsp = nx.algorithms.approximation.simulated_annealing_tsp
        method = lambda G, wt: SA_tsp(G, 'greedy', weight = wt, source = self.current_temp_idx)
        # obtain new path
        new_path_idx = tsp(G, cycle = True, method = method)
        # are we using new animation
        if self.gen_animation:
            self.out2 = ['new_plan', self.unique_samples[new_path_idx[:-1], :], self.env.temperature_list, self.X]

        # remove current_temp if in batch
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
        self.query_plan = self.unique_samples[new_path_idx, :]

if __name__ == '__main__':

    exp = '1D'

    if exp == '1D':

        budget = 100
        max_batch_size = 1
        epsilon_list = [0, 0.1]
        methods = ['Resampling', 'e-Point Deletion']
        colors = ['b', 'r']
        max_change = None

        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        fig.set_figheight(4)
        fig.set_figwidth(10)
        #title = f'Budget = {budget} : Time delay = {max_batch_size - 1}'
        #fig.suptitle(title)
        pt = 0.74
        escape_prediction = budget * pt

        optimum_global = 0.78125
        optimum_local = 0.15625
        optimums = [optimum_local, optimum_global]

        ax.vlines(optimum_local, ymin = 0, ymax = budget, colors = 'k', linestyles = '--', linewidth = 2, label = 'Optimums')
        ax.vlines(optimum_global, ymin = 0, ymax = budget, colors = 'k', linestyles = '--', linewidth = 2)

        np.random.seed(2023)
        torch.manual_seed(2023)

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
            '''
            if i == 1:
                label = '0.1-Point Deletion'
            else:
                label = 'Resampling'

            ax.plot(X, times, c = colors[i], label = label, linewidth = 2)
            ax.set_xlabel('x', fontsize = 20)
            if i == 0:
                ax.set_ylabel('Iteration', fontsize = 20)
            ax.set_xlim(0, 1)
            ax.grid()
            if i == 1:
                ax.hlines(escape_prediction, 0, 1, colors = colors[i], linestyles = '--', label = 'Escape Prediction', linewidth = 2)

            '''
            ax.scatter(X, Y, s = 100, marker = 'x', c = 'k')
            if i == 0:
                ax.set_ylabel('f(x)', fontsize = 20)
            ax.set_xlabel('x', fontsize = 20)
            ax.plot(grid, target_func, '--k', label = 'True function', linewidth = 2)
            ax.plot(grid, posterior_mean.detach().numpy(), 'r', label = 'GP mean', linewidth = 2)
            ax.fill_between(grid.reshape(-1), posterior_mean.detach() - 1.96 * posterior_sd.detach(), \
                 posterior_mean.detach() + 1.96 * posterior_sd.detach(), alpha = 0.2, color = 'r')
            ax.set_xlim(0, 1)
            ax.grid()
            ax.legend(loc = 'lower right', framealpha = 0.8, prop={'size': 20})

            plt.xticks(fontsize = 20)
            plt.yticks(fontsize = 20)
            '''
            
        ax.legend(loc = 'lower right', framealpha = 0.8, prop={'size': 20})
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)

        filename = 'PDvsRS_both_paths_w_optims' + '.pdf'
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
            func = BraninFunction(random=True)
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