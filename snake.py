import numpy as np
from gp_utils import BoTorchGP
import networkx as nx
from scipy.spatial import distance_matrix
from scipy.linalg import norm
from scipy.stats.qmc import Sobol
import sobol_seq
from itertools import chain
from sampling import EfficientThompsonSampler
import torch

'''
This python file implements the main method, SnAKe. It also has a class that implements a baseline consisting of a Random sample
which is then ordered by approximately solving the Travelling Salesman Problem.
'''

class SnAKe():
    def __init__(self, env, initial_temp = None, \
        max_change = None, \
            gen_animation = False, \
                merge_method = 'Point Deletion', \
                    merge_constant = None, \
                        hp_update_frequency = None, \
                            num_of_multistarts = 10, \
                                cost_function = None):
        ''' 
        Class for implementing SnAKe. For the code, we call
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
        # check if we are using non-parametric ell-SnAKe
        if self.merge_constant == 'lengthscale':
            self.parameter_free = True
        else:
            self.parameter_free = False
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
        if self.parameter_free == True:
            self.merge_constant = torch.min(self.length_scale).item()

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
                # update value of epsilon
                if self.parameter_free == True:
                    length_scale = self.gp_hyperparams[1]
                    self.merge_constant = torch.min(length_scale).item()
                    print(f'New $\epsilon = ${self.merge_constant}')
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

class MultiObjectiveSnAKe(SnAKe):
    '''
    Variant of SnAKe that allows for simultaneous optimization of various black-box functions.
    New inputs include:
    
    objective_weights: how to weight the importance of each objective? This should be list of length 'num_of_objectives'

    max_search_grid_size: in cases where we are optimizing over a grid, choose the maximum size of the grid over which to optimize.

    exploration_constant: a number between 0 and 1. It represents the percentage of exploratory samples are created in each batch. They are created using samples of the GP minus the mean.
    '''
    def __init__(self, env, initial_temp=None, max_change=None, gen_animation=False, merge_method='Point Deletion', merge_constant=None, hp_update_frequency=None, num_of_multistarts=10, cost_function=None, objective_weights = None, max_search_grid_size = 1000, exploration_constant = 1/3):
        # new params
        # number of objectives to maximize
        self.num_of_objectives = env.num_of_objectives
        # maximum search size for the grid, if grid is too large we randomly sample max_search_grid_size points from it
        self.max_grid_search_size = max_search_grid_size
        # determines the number of exploratory points
        self.exploration_constant = exploration_constant
        if objective_weights is None:
            self.objective_weights = [1 / self.num_of_objectives for _ in range(self.num_of_objectives)]
        else:
            self.objective_weights = objective_weights
        # initialize super
        super().__init__(env, initial_temp, max_change, gen_animation, merge_method, merge_constant, hp_update_frequency, num_of_multistarts, cost_function)
        self.X = []
        self.Y = [[] for _ in range(self.num_of_objectives)]
        # define model
        self.model = [BoTorchGP(lengthscale_dim = self.dim) for _ in range(self.num_of_objectives)]
        self.set_hyperparams()
        # check if grid needs initialization
        if self.env.function.grid_search is True:
            # initialize grid to search
            grid_to_search = self.env.function.grid_to_search
            idx_rand = torch.randperm(len(grid_to_search))[:self.max_grid_search_size]
            self.grid_to_search_sample = grid_to_search[idx_rand, :]

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
        
        self.gp_hyperparams = [(self.constant, self.length_scale, self.noise, self.mean_constant) for _ in range(self.num_of_objectives)]
        if self.parameter_free == True:
            self.merge_constant = torch.min(self.length_scale).item()

        # check if we want our constraints based on these hyperparams
        if constraints is True:
            for i in range(self.num_of_objectives):
                self.model[i].define_constraints(self.length_scale, self.mean_constant, self.constant)

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
                if self.env.function.grid_search is True:
                    distances_to_grid = np.sum((self.grid_to_search_sample - new_T).numpy()**2, axis = 1)
                    idx_min = np.argmin(distances_to_grid)
                    new_T = self.grid_to_search_sample[idx_min, :].numpy().reshape(1, -1)
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
            for obj in range(self.num_of_objectives):
                self.Y[obj].append(self.new_obs[obj])
                self.update_model(obj)
        # check the update frequency and make sure we actually have data to train
        for obj in range(self.num_of_objectives):
            if (self.hp_update_frequency is not None) & (len(self.X) > 0):
                if len(self.X) % self.hp_update_frequency == 0:
                    self.model[obj].optim_hyperparams()
                    self.gp_hyperparams[obj] = self.model[obj].current_hyperparams()
                    print(f'New hyperparams: {self.model[obj].current_hyperparams()}')
                    # update value of epsilon
                    if self.parameter_free == True:
                        length_scale = self.gp_hyperparams[obj][1]
                        self.merge_constant = torch.min(length_scale).item()
                        print(f'New $\epsilon = ${self.merge_constant}')
        # update current temperature and current time
        self.current_temp = np.array(query).reshape(1, -1)
        self.current_time = self.current_time + 1
    
    def create_mega_batch(self):
        '''
        This function creates the batch of Thompson Samples that approximate the future
        '''
        # get number of samples depending on the method
        if self.merge_method in ['Point Deletion', 'e-Point Deletion']:
            self.num_of_samples = self.budget + 1
        elif self.merge_method == 'Resampling':
            self.num_of_samples = self.budget + 1 - self.env.t

        samples_so_far = 0
        samples = np.empty(shape = (0, self.dim))
        for obj in range(self.num_of_objectives):
            if obj == self.num_of_objectives - 1:
                num_of_samples_obj = self.num_of_samples - samples_so_far
            else:
                num_of_samples_obj = int(self.num_of_samples * self.objective_weights[obj] / (np.sum(self.objective_weights)))
                samples_so_far = samples_so_far + num_of_samples_obj

            if len(self.X) == 0:
                samples_obj = np.random.uniform(size = (num_of_samples_obj, self.dim))

            elif self.env.function.grid_search is True:
                grid_to_search = self.env.function.grid_to_search
                idx_rand = torch.randperm(len(grid_to_search))[:self.max_grid_search_size]
                self.grid_to_search_sample = grid_to_search[idx_rand, :]
                gp_samples = self.model[obj].sample(self.grid_to_search_sample, n_samples = num_of_samples_obj)
                # we now check for exploratory samples
                exploratory_penalty = np.zeros_like(gp_samples)
                exploratory_sample_size = int(self.exploration_constant * num_of_samples_obj)
                with torch.no_grad():
                    mean, _ = self.model[obj].posterior(self.grid_to_search_sample)
                    mean = mean.numpy().reshape(1, -1).repeat(exploratory_sample_size, axis = 0)
                    exploratory_penalty[:exploratory_sample_size, :] = mean
                    gp_samples = gp_samples - exploratory_penalty
                # find the maximum of each grid search
                max_idx = np.argmax(gp_samples, axis = 1)
                samples_obj = self.grid_to_search_sample[max_idx, :]

            else:
                # define the sampler
                sampler = EfficientThompsonSampler(self.model[obj], num_of_multistarts = self.num_of_multistarts, \
                    num_of_bases = 1024, \
                    num_of_samples = num_of_samples_obj)
                # create samples
                sampler.create_sample()
                # optimise samples
                samples_obj = sampler.generate_candidates()
                samples_obj = samples_obj.numpy()
            samples = np.concatenate((samples, samples_obj), axis = 0)
        
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

    def update_model(self, obj):
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
            self.model[obj].fit_model(X_numpy, self.Y[obj], previous_hyperparams = self.gp_hyperparams[obj])