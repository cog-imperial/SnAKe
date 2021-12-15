from botorch import sampling
import numpy as np
import torch
from gp_utils import BoTorchGP
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.sampling import IIDNormalSampler
from botorch.optim.initializers import initialize_q_batch_nonneg

class UCBwLP():
    def __init__(self, env, initial_temp = None, beta = None, lipschitz_constant = 20, num_of_starts = 75, num_of_optim_epochs = 150, \
        hp_update_frequency = None):
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

        self.lipschitz_constant = lipschitz_constant
        self.max_value = 0

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
        if constant == None:
            self.constant = 0.6
            self.length_scale = torch.tensor([0.15] * self.x_dim)
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

        if self.fixed_beta == False:
            self.beta = float(0.2 * self.dim * np.log(2 * (self.env.t + 1)))
        
        new_T, new_X = self.optimise_af()

        if self.x_dim == None:
            query = new_T
        else:
            query = np.concatenate((new_T, new_X), axis = 1)
        
        query = list(query.reshape(-1))
        
        obtain_query, self.new_obs = self.env.step(new_T, new_X)

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
        
        self.current_temp = new_T
        self.current_time = self.current_time + 1
    
    def update_model(self):
        if self.new_obs is not None:
            self.model.fit_model(self.X, self.Y, previous_hyperparams=self.gp_hyperparams)

    def build_af(self, X):
        batch = self.env.temperature_list

        if self.new_obs is not None:
            mean, std = self.model.posterior(X)
        else:
            mean, std = 0, self.constant

        ucb = mean + self.beta * std

        for penalty_point in batch:
            penalty_point = torch.tensor(penalty_point)
            # define the value that goes inside the erfc
            norm = torch.norm(penalty_point - X, dim = 1)

            z = self.lipschitz_constant * norm - self.max_value + mean
            z = z / (std * np.sqrt(2))

            # define penaliser
            penaliser = 0.5 * torch.erfc(-1*z)

            # penalise ucb at specific fidelity
            ucb = ucb * penaliser
        
        return ucb
    
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
        # random initialisation
        X = torch.rand(self.num_of_starts, self.dim)
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
                X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself

        best_start = torch.argmax(-losses)
        best_input = X[best_start, :].detach()

        if self.x_dim is not None:
            new_T = best_input[0, :self.t_dim].detach.numpy().reshape(1, -1)
            new_X = best_input[0, self.t_dim:].detach.numpy().reshape(1, -1)
        else:
            new_T = best_input.detach().numpy().reshape(1, -1)
            new_X = None

        return new_T, new_X

class oneExpectedImprovement():
    def __init__(self, env, initial_temp = None, beta = 1.96, lipschitz_constant = 20, num_of_starts = 75, num_of_optim_epochs = 150, \
        hp_update_frequency = None):
        self.env = env
        self.t_dim = self.env.t_dim
        self.x_dim = self.env.x_dim
        if self.x_dim is None:
            self.dim = self.t_dim
        else:
            self.dim = self.t_dim + self.x_dim
        
        assert self.env.max_batch_size == 1, 'Expected Improvement Requires Sequential Data!'

        self.set_hyperparams()

        # values of LP
        self.beta = beta
        self.lipschitz_constant = lipschitz_constant
        self.max_value = 0

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
        if constant == None:
            self.constant = 0.5
            self.length_scale = 0.1
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
        self.model = BoTorchGP()
        # current temperature
        self.current_temp = self.initial_temp
        # budget
        self.budget = self.env.budget
        # time
        self.current_time = 0
        # initialise new_obs
        self.new_obs = None
    
    def run_optim(self, verbose = False):
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
        
        new_T, new_X = self.optimise_af()

        if self.x_dim == None:
            query = new_T
        else:
            query = np.concatenate((new_T, new_X), axis = 1)
        
        query = list(query.reshape(-1))
        
        obtain_query, self.new_obs = self.env.step(new_T, new_X)

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
        
        self.current_temp = new_T
        self.current_time = self.current_time + 1
    

    def update_model(self):
        if self.new_obs is not None:
            self.model.fit_model(self.X, self.Y, previous_hyperparams=self.gp_hyperparams)

    def build_af(self, X):

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
        # random initialisation
        Xraw = torch.rand(100 * self.num_of_starts, self.dim)
        Yraw = self.build_af(Xraw)
        X = initialize_q_batch_nonneg(Xraw, Yraw, self.num_of_starts)
        X.requires_grad = True
        # define optimiser
        optimiser = torch.optim.Adam([X], lr = 0.01)
        
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
                X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself

        best_start = torch.argmax(-losses)
        best_input = X[best_start, :].detach()

        if self.x_dim is not None:
            new_T = best_input[:self.t_dim].detach().numpy().reshape(1, -1)
            new_X = best_input[self.t_dim:].detach().numpy().reshape(1, -1)
        else:
            new_T = best_input.detach().numpy().reshape(1, -1)
            new_X = None

        return new_T, new_X

class oneProbabilityOfImprovement(oneExpectedImprovement):
    def __init__(self, env, initial_temp=None, beta=1.96, lipschitz_constant=20, num_of_starts=75, num_of_optim_epochs=150, hp_update_frequency=None):
        super().__init__(env, initial_temp=initial_temp, beta=beta, lipschitz_constant=lipschitz_constant, num_of_starts=num_of_starts, num_of_optim_epochs=num_of_optim_epochs, hp_update_frequency=hp_update_frequency)

    def build_af(self, X):
        PI = ProbabilityOfImprovement(self.model.model, best_f = self.max_value)
        return PI(X.unsqueeze(1))