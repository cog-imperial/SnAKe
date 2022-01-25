import numpy as np
import torch
from gp_utils import BoTorchGP
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.optim.initializers import initialize_q_batch_nonneg
from sampling import EfficientThompsonSampler
import sobol_seq

'''
This script implements all Bayesian Optimization methods we compared in the paper.
'''

class UCBwLP():
    def __init__(self, env, initial_temp = None, beta = None, lipschitz_constant = 1, num_of_starts = 75, num_of_optim_epochs = 150, \
        hp_update_frequency = None):
        '''
        Upper Confidence Bound with Local Penalisation, see:

        Gonzalez, J., Dai, Z., Hennig, P., and Lawrence, N. 
        Batch Bayesian Optimization via Local Penalization. 
        In Proceedings of the 19th International Conference on Artificial Intelligence and Statistics, 
        pp. 648-657, 09-11 May 2016.

        Takes as inputs:
        env - optimization environment
        beta - parameter of UCB bayesian optimization, default uses 0.2 * self.dim * np.log(2 * (self.env.t + 1))
        lipschitz_consant - initial lipschitz_consant, will be re-estimated at every step
        num_of_starts - number of multi-starts for optimizing the acquisition function, default is 75
        num_of_optim_epochs - number of epochs for optimizing the acquisition function, default is 150
        hp_update_frequency - how ofter should GP hyper-parameters be re-evaluated, default is None

        '''
        # initialise the environment
        self.env = env
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim
        if self.x_dim is None:
            self.dim = self.t_dim
        else:
            self.dim = self.t_dim + self.x_dim

        # gp hyperparams
        self.set_hyperparams()

        # values of LP
        if beta == None:
            self.fixed_beta = False
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.t + 1)))
        else:  
            self.fixed_beta = True
            self.beta = beta

        # parameters of the method
        self.lipschitz_constant = lipschitz_constant
        self.max_value = 0
        # initalise grid to select lipschitz constant
        self.num_of_grad_points = 50 * self.dim
        self.lipschitz_grid = sobol_seq.i4_sobol_generate(self.dim, self.num_of_grad_points)
        # do we require transform?
        if (self.env.function.name in ['Perm10D', 'Ackley4D', 'SnarBenchmark']) & (self.env.max_batch_size > 1):
            self.soft_plus_transform = True
        else:
            self.soft_plus_transform = False

        # optimisation parameters
        self.num_of_starts = num_of_starts
        self.num_of_optim_epochs = num_of_optim_epochs
        # hp hyperparameters update frequency
        self.hp_update_frequency = hp_update_frequency

        # initial temperature, not needed I think
        if initial_temp is not None:
            self.initial_temp = initial_temp
        else:
            self.initial_temp = np.zeros((1, self.t_dim))
        
        # define domain
        self.domain = np.zeros((self.t_dim,))
        self.domain = np.stack([self.domain, np.ones(self.t_dim, )], axis=1)
        
        self.initialise_stuff()

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
            self.constant = constant
            self.length_scale = lengthscale
            self.noise = noise
            self.mean_constant = mean_constant
        
        self.gp_hyperparams = (self.constant, self.length_scale, self.noise, self.mean_constant)
        # check if we want our constraints based on these hyperparams
        if constraints is True:
            self.model.define_constraints(self.length_scale, self.mean_constant, self.constant)
    
    def initialise_stuff(self):
        # list of queries
        self.queried_batch = []
        # list of queries and observations
        self.X = []
        self.Y = []
        # define model
        self.model = BoTorchGP(lengthscale_dim = self.dim)
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
    
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
        # check if we need to update beta
        if self.fixed_beta == False:
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.t + 1)))
        # optimise acquisition function to obtain new query
        new_T, new_X = self.optimise_af()
        # check if all variables incur input cost
        if self.x_dim == None:
            query = new_T
        else:
            query = np.concatenate((new_T, new_X), axis = 1)
        # reformat query
        query = list(query.reshape(-1))
        # step forward in environment
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        # append query to queried batch
        self.queried_batch.append(query)
        # update model if there are new observations
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
            # redefine new maximum value
            self.max_value = float(max(self.max_value, float(self.new_obs)))
            self.update_model()
        
        # update hyperparams if needed
        if (self.hp_update_frequency is not None) & (len(self.X) > 0):
            if len(self.X) % self.hp_update_frequency == 0:
                self.model.optim_hyperparams()
                self.gp_hyperparams = self.model.current_hyperparams()
                print(f'New hyperparams: {self.model.current_hyperparams()}')
        # update current temperature and time
        self.current_temp = new_T
        self.current_time = self.current_time + 1
    
    def update_model(self):
        '''
        This function updates the GP model
        '''
        if self.new_obs is not None:
            # fit new model
            self.model.fit_model(self.X, self.Y, previous_hyperparams=self.gp_hyperparams)
            # we also update our estimate of the lipschitz constant, since we have a new model
            # define the grid over which we will calculate gradients
            grid = torch.tensor(self.lipschitz_grid, requires_grad = True).double()
            # we only do this if we are in asynchronous setting, otherwise this should behave as normal UCB algorithm
            if self.env.max_batch_size > 1:
                # calculate mean of the GP
                mean, _ = self.model.posterior(grid)
                # calculate the gradient of the mean
                external_grad = torch.ones(self.num_of_grad_points)
                mean.backward(gradient = external_grad)
                mu_grads = grid.grad
                # find the norm of all the mean gradients
                mu_norm = torch.norm(mu_grads, dim = 1)
                # choose the largest one as our estimate
                self.lipschitz_constant = max(mu_norm).item()

    def build_af(self, X):
        '''
        This takes input locations, X, and returns the value of the acquisition function
        '''
        # check the batch of points being evaluated
        batch = self.env.temperature_list
        # if there are no new observations return the prior
        if self.new_obs is not None:
            mean, std = self.model.posterior(X)
        else:
            mean, std = torch.tensor(self.mean_constant), torch.tensor(self.constant)
        # calculate upper confidence bound
        ucb = mean + self.beta * std
        # apply softmax transform if necessary
        if self.soft_plus_transform: 
            ucb = torch.log(1 + torch.exp(ucb))
        # penalize acquisition function, loop through batch of evaluations
        for i, penalty_point in enumerate(batch):
            # add x-variables if needed
            if self.env.x_dim is not None:
                query_x = self.env.batch[i]
                penalty_point = np.concatenate((penalty_point, query_x.reshape(1, -1)), axis = 1).reshape(1, -1)
            # re-define penalty point as tensor
            penalty_point = torch.tensor(penalty_point)
            # define the value that goes inside the erfc
            norm = torch.norm(penalty_point - X, dim = 1)
            # calculate z-value
            z = self.lipschitz_constant * norm - self.max_value + mean
            z = z / (std * np.sqrt(2))
            # define penaliser
            penaliser = 0.5 * torch.erfc(-1*z)
            # penalise ucb
            ucb = ucb * penaliser
        # return acquisition function
        return ucb
    
    def optimise_af(self):
        '''
        This function optimizes the acquisition function, and returns the next query point
        '''
        # if time is zero, pick point at random
        if self.current_time == 0:
            new_T = np.random.uniform(size = self.t_dim).reshape(1, -1)
            if self.x_dim is not None:
                new_X = np.random.uniform(size = self.x_dim).reshape(1, -1)
            else:
                new_X = None
            return new_T, new_X
        
        # optimisation bounds
        bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
        # random initialization, multiply by 100
        X = torch.rand(100 * self.num_of_starts, self.dim).double()
        X.requires_grad = True
        # define optimiser
        optimiser = torch.optim.Adam([X], lr = 0.0001)
        af = self.build_af(X)
        
        # do the optimisation
        for _ in range(self.num_of_optim_epochs):
            # set zero grad
            optimiser.zero_grad()
            # losses for optimiser
            losses = -self.build_af(X)
            loss = losses.sum()
            loss.backward()
            # optim step
            optimiser.step()

            # make sure we are still within the bounds
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub)
        # find the best start
        best_start = torch.argmax(-losses)
        # corresponding best input
        best_input = X[best_start, :].detach()
        # return the next query point
        if self.x_dim is not None:
            best = best_input.detach().numpy().reshape(1, -1)
            new_T = best[0, :self.t_dim].reshape(1, -1)
            new_X = best[0, self.t_dim:].reshape(1, -1)
        else:
            new_T = best_input.detach().numpy().reshape(1, -1)
            new_X = None

        return new_T, new_X

class ThompsonSampling():
    '''
    Method of Thompson Sampling for Bayesian Optimization, see the paper:

    Kandasamy, K., Krishnamurthy, A., Schneider, J., and Poczos, B. 
    Parallelised BayesianOptimisation via Thompson Sampling. 
    In Proceedings of the Twenty-First International Conference on Artificial Intelligence and Statistics, 
    pp. 133-142, 2018.
    '''
    def __init__(self, env, initial_temp = None, num_of_starts = 75, num_of_optim_epochs = 150, \
        hp_update_frequency = None):
        '''
        Takes as inputs:
        env - optimization environment
        beta - parameter of UCB bayesian optimization, default uses 0.2 * self.dim * np.log(2 * (self.env.t + 1))
        lipschitz_consant - initial lipschitz_consant, will be re-estimated at every step
        num_of_starts - number of multi-starts for optimizing the acquisition function, default is 75
        num_of_optim_epochs - number of epochs for optimizing the acquisition function, default is 150
        hp_update_frequency - how ofter should GP hyper-parameters be re-evaluated, default is None

        '''

        self.env = env
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim
        if self.x_dim is None:
            self.dim = self.t_dim
        else:
            self.dim = self.t_dim + self.x_dim

        # gp hyperparams
        self.set_hyperparams()


        # optimisation parameters
        self.num_of_starts = num_of_starts
        self.num_of_optim_epochs = num_of_optim_epochs
        # hp hyperparameters update frequency
        self.hp_update_frequency = hp_update_frequency

        # initial temperature, not needed I think
        if initial_temp is not None:
            self.initial_temp = initial_temp
        else:
            self.initial_temp = np.zeros((1, self.t_dim))
        
        # define domain
        self.domain = np.zeros((self.t_dim,))
        self.domain = np.stack([self.domain, np.ones(self.t_dim, )], axis=1)
        
        self.initialise_stuff()

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
            self.constant = constant
            self.length_scale = lengthscale
            self.noise = noise
            self.mean_constant = mean_constant
        
        self.gp_hyperparams = (self.constant, self.length_scale, self.noise, self.mean_constant)
        # check if we want our constraints based on these hyperparams
        if constraints is True:
            self.model.define_constraints(self.length_scale, self.mean_constant, self.constant)
    
    def initialise_stuff(self):
        # list of queries
        self.queried_batch = []
        # list of queries and observations
        self.X = []
        self.Y = []
        # define model
        self.model = BoTorchGP(lengthscale_dim = self.dim)
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
    
    def run_optim(self, verbose = False):
        '''
        Runs the whole optimisation procedure, returns all queries and evaluations
        '''
        self.env.initialise_optim()
        while self.current_time <= self.budget:
            self.optim_loop()
            if verbose:
                print(f'Current time-step: {self.current_time}')
        # obtain all inputs and outputs
        X, Y = self.env.finished_with_optim()
        # reformat the output
        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)
        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # if there are no observations, sample uniformly
        if len(self.X) == 0:
            query = np.random.uniform(size = (1, self.dim))

        else:
            # define the samples
            sampler = EfficientThompsonSampler(self.model, num_of_multistarts = self.num_of_starts, \
                num_of_bases = 1024, \
                    num_of_samples = 1)
            # create samples
            sampler.create_sample()
            # optimise samples
            samples = sampler.generate_candidates()
            query = samples.numpy()

        # check if there are x-variables to get formatting right
        if self.x_dim == None:
            new_T = query
            new_X = None
        else:
            new_T = query[0, :self.t_dim]
            new_X = query[0, self.t_dim:]
        # reformat query
        query = list(query.reshape(-1))
        # carry out step in environment
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        # add query to queried batch
        self.queried_batch.append(query)
        # update model
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
            self.update_model()
        
        # update hyperparams if needed
        if (self.hp_update_frequency is not None) & (len(self.X) > 0):
            if len(self.X) % self.hp_update_frequency == 0:
                self.model.optim_hyperparams()
                self.gp_hyperparams = self.model.current_hyperparams()
                print(f'New hyperparams: {self.model.current_hyperparams()}')
        # update current temperature and time
        self.current_temp = new_T
        self.current_time = self.current_time + 1
    
    def update_model(self):
        # updates model
        if self.new_obs is not None:
            self.model.fit_model(self.X, self.Y, previous_hyperparams=self.gp_hyperparams)

class oneExpectedImprovement():
    def __init__(self, env, initial_temp = None, num_of_starts = 75, num_of_optim_epochs = 150, \
        hp_update_frequency = None):
        '''
        Expected Improvement in a sequential setting i.e. one query per iteration. See paper:

        Mockus, J., Tiesis, V., and Zilinskas, A.
        The application of Bayesian methods for seeking the extremum. 
        Towards Global Optimization, 2:117-129, 09 2014.

        Inputs:
        num_of_starts - number of multi-starts to optimize acquisition function
        num_of_optim_epochs - number of epochs to optimize acquisition function
        hp_update_frequency - how frequently to optimize the hyper-parameters
        '''
        self.env = env
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim
        if self.x_dim is None:
            self.dim = self.t_dim
        else:
            self.dim = self.t_dim + self.x_dim
        
        assert self.env.max_batch_size == 1, 'Expected Improvement Requires Sequential Data!'

        self.set_hyperparams()

        # optimisation parameters
        self.num_of_starts = num_of_starts
        self.num_of_optim_epochs = num_of_optim_epochs

        # hp hyperparameters update frequency
        self.hp_update_frequency = hp_update_frequency
        # initial temperature, not needed I think
        if initial_temp is not None:
            self.initial_temp = initial_temp
        else:
            self.initial_temp = np.zeros((1, self.t_dim))
        
        # define domain
        self.domain = np.zeros((self.t_dim,))
        self.domain = np.stack([self.domain, np.ones(self.t_dim, )], axis=1)
        
        self.initialise_stuff()
    
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
            self.constant = constant
            self.length_scale = lengthscale
            self.noise = noise
            self.mean_constant = mean_constant
        
        self.gp_hyperparams = (self.constant, self.length_scale, self.noise, self.mean_constant)
        # check if we want our constraints based on these hyperparams
        if constraints is True:
            self.model.define_constraints(self.length_scale, self.mean_constant, self.constant)
    
    def initialise_stuff(self):
        # list of queries
        self.queried_batch = []
        # list of queries and observations
        self.X = []
        self.Y = []
        # define model
        self.model = BoTorchGP(lengthscale_dim = self.dim)
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
    
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
        # reformat queries
        X_out = X[0]
        for x in X[1:]:
            X_out = np.concatenate((X_out, x), axis = 0)
        return X_out, Y
    
    def optim_loop(self):
        '''
        Performs a single loop of the optimisation
        '''
        # optimise acquisition function to obtain new queries
        new_T, new_X = self.optimise_af()
        # check if we have x-variables (i.e. variables with no input cost)
        if self.x_dim == None:
            query = new_T
        else:
            query = np.concatenate((new_T, new_X), axis = 1)
        # reformat query
        query = list(query.reshape(-1))
        # step in environment
        obtain_query, self.new_obs = self.env.step(new_T, new_X)
        # add query to queried batch
        self.queried_batch.append(query)
        # update model
        if self.new_obs is not None:
            self.X.append(list(obtain_query.reshape(-1)))
            self.Y.append(self.new_obs)
            self.max_value = float(max(self.max_value, float(self.new_obs)))
            self.update_model()
        # update hyperparams if needed
        if (self.hp_update_frequency is not None) & (len(self.X) > 0):
            if len(self.X) % self.hp_update_frequency == 0:
                self.model.optim_hyperparams()
                self.gp_hyperparams = self.model.current_hyperparams()
                print(f'New hyperparams: {self.model.current_hyperparams()}')
        # update temperature and time
        self.current_temp = new_T
        self.current_time = self.current_time + 1
    
    def update_model(self):
        # update gp model
        if self.new_obs is not None:
            self.model.fit_model(self.X, self.Y, previous_hyperparams=self.gp_hyperparams)

    def build_af(self, X):
        # build acquisition function using BoTorch Expected Improvement
        EI = ExpectedImprovement(self.model.model, best_f = self.max_value)
        return EI(X.unsqueeze(1))
    
    def optimise_af(self):
        # if time is zero, pick point at random
        if self.current_time == 0:
            new_T = np.random.uniform(size = self.t_dim).reshape(1, -1)
            if self.x_dim is not None:
                new_X = np.random.uniform(size = self.x_dim).reshape(1, -1)
            else:
                new_X = None
            
            return new_T, new_X
        
        # optimisation bounds
        bounds = torch.stack([torch.zeros(self.dim), torch.ones(self.dim)])
        # random initialization
        Xraw = torch.rand(100 * self.num_of_starts, self.dim)
        Yraw = self.build_af(Xraw)
        # use BoTorch initializer
        X = initialize_q_batch_nonneg(Xraw, Yraw, self.num_of_starts)
        X.requires_grad = True
        # define optimizer
        optimiser = torch.optim.Adam([X], lr = 0.0001)
        
        # do the optimization
        for _ in range(self.num_of_optim_epochs):
            # set zero grad
            optimiser.zero_grad()
            # losses for optimizer
            losses = -self.build_af(X)
            loss = losses.sum()
            loss.backward()
            # optim step
            optimiser.step()

            # make sure we are still within the bounds
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        # obtain the best start
        best_start = torch.argmax(-losses)
        best_input = X[best_start, :].detach()

        if self.x_dim is not None:
            new_T = best_input[:self.t_dim].detach().numpy().reshape(1, -1)
            new_X = best_input[self.t_dim:].detach().numpy().reshape(1, -1)
        else:
            new_T = best_input.detach().numpy().reshape(1, -1)
            new_X = None
        # return next query
        return new_T, new_X

class oneProbabilityOfImprovement(oneExpectedImprovement):
    '''
    One Probability of Improvement, see paper:

    Kushner, H. J. 
    A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve in the Presence of Noise. 
    Journal of Basic Engineering, 86:97-106, 1964.
    '''
    def __init__(self, env, initial_temp=None, beta=1.96, lipschitz_constant=20, num_of_starts=75, num_of_optim_epochs=150, hp_update_frequency=None):
        '''
        Inputs:
        num_of_starts - number of multi-starts to optimize acquisition function
        num_of_optim_epochs - number of epochs to optimize acquisition function
        hp_update_frequency - how frequently to optimize the hyper-parameters
        '''
        # use Expected Improvement class, only change the acquisition function
        super().__init__(env, initial_temp=initial_temp, beta=beta, lipschitz_constant=lipschitz_constant, num_of_starts=num_of_starts, num_of_optim_epochs=num_of_optim_epochs, hp_update_frequency=hp_update_frequency)

    def build_af(self, X):
        # Probability of Improvement using BoTorch
        PI = ProbabilityOfImprovement(self.model.model, best_f = self.max_value)
        return PI(X.unsqueeze(1))