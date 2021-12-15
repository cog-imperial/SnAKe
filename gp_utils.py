import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
import torch
from gpytorch.priors import SmoothedBoxPrior
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan, Interval, LessThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

class SklearnGP():

    def __init__(self, kernel = None, optimise_hyper_params = True, initial_lengthscale = 1, \
         initial_constant = 1, noise_assumption = 1e-10):

        self.initial_lengthscale = initial_lengthscale
        self.initial_constant = initial_constant

        if optimise_hyper_params == True:
            constant_bounds = (0.0001, 100)
            length_scale_bounds = (0.00001, 100)
        else:
            constant_bounds = 'fixed'
            length_scale_bounds = 'fixed'

        if kernel == None:
            self.kernel = ConstantKernel(constant_value = self.initial_constant, constant_value_bounds = constant_bounds) \
                * RBF(length_scale = self.initial_lengthscale, length_scale_bounds = length_scale_bounds)
        else:
            self.kernel = kernel
        
        self.model = GaussianProcessRegressor(kernel = self.kernel, alpha = noise_assumption)

        self.sample_int = np.random.randint(0, 10000000)
    
    def fit_data(self, X, Y):
        self.model.fit(X, Y)
    
    def posterior(self, X):
        mean, std = self.model.predict(X, return_std=True)
        return mean, std
    
    def sample(self, X, new_sample = False, n_samples = 1):
        # if we have a new sample draw a new seed
        if new_sample == True:
            self.sample_int = np.random.randint(0, 10000000)
        return self.model.sample_y(X, n_samples = n_samples, random_state = self.sample_int)

class BoTorchGP():
    def __init__(self, kernel = None, lengthscale_dim = None):
        if kernel == None:
            self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = lengthscale_dim))
        self.constraints_set = False
        self.lengthscale_dim = lengthscale_dim
        
    def fit_model(self, train_x, train_y, train_hyperparams = False, previous_hyperparams = None):
        # transform data to tensors
        self.train_x = torch.tensor(train_x)
        train_y = np.array(train_y)
        self.train_y = torch.tensor(train_y).reshape(-1, 1)
        # define model
        self.model = SingleTaskGP(train_X = self.train_x, train_Y = self.train_y, \
            covar_module = self.kernel)
        self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))
        
        # marginal likelihood
        self.mll = ExactMarginalLogLikelihood(likelihood = self.model.likelihood, model = self.model)

        if previous_hyperparams is not None:
            self.outputscale = float(previous_hyperparams[0])
            self.lengthscale = previous_hyperparams[1].detach()
            self.noise = float(previous_hyperparams[2])
            self.mean_constant = float(previous_hyperparams[3])
            self.set_hyperparams()
        
        if train_hyperparams == True:
            self.optim_hyperparams()
    
    def define_constraints(self, init_lengthscale, init_mean_constant, init_outputscale):
        # define lengthscale bounds
        self.lengthscale_ub = 2 * init_lengthscale
        self.lengthscale_lb = init_lengthscale / 2
        # define mean_constant bounds
        self.mean_constant_ub = init_mean_constant + 0.25 * init_outputscale
        self.mean_constant_lb = init_mean_constant - init_outputscale
        # define outputscale bounds
        self.outputscale_ub = 3 * init_outputscale
        self.outputscale_lb = init_outputscale / 3

        self.constraints_set = True


    def optim_hyperparams(self, num_of_epochs = 500, verbose = False):
        # set constraints if there are any
        if self.constraints_set is True:
            if verbose:
                print('Setting Constraints...')
                print(f'lengthscale lb {self.lengthscale_lb} : lengthscale ub {self.lengthscale_ub}')
                print(f'outputscale lb {self.outputscale_lb} : outputscale ub {self.outputscale_ub}')
                print(f'mean constant lb {self.mean_constant_lb} : mean constant ub {self.mean_constant_ub}')
            # for lengthscale
            prior_lengthscale = SmoothedBoxPrior(self.lengthscale_lb, self.lengthscale_ub, 0.001)
            self.model.covar_module.base_kernel.register_prior('Smoothed Box Prior', prior_lengthscale, "lengthscale")
            # for outputscale
            prior_outputscale = SmoothedBoxPrior(self.outputscale_lb, self.outputscale_ub, 0.001)
            self.model.covar_module.register_prior('Smoothed Box Prior', prior_outputscale, "outputscale")
            # for mean constant
            prior_constant = SmoothedBoxPrior(self.mean_constant_lb, self.mean_constant_ub, 0.001)
            self.model.mean_module.register_prior('Smoothed Box Prior', prior_constant, "constant")
        
        # define optimiser
        optimiser = Adam([{'params': self.model.parameters()}], lr=0.01)

        self.model.train()

        for epoch in range(num_of_epochs):
            # obtain output
            output = self.model(self.train_x)
            # calculate loss
            loss = - self.mll(output, self.train_y.view(-1))
            # optim step
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if ((epoch + 1) % 10 == 0) & (verbose):
                print(
                    f"Epoch {epoch+1:>3}/{num_of_epochs} - Loss: {loss.item():>4.3f} "
                    f"outputscale: {self.model.covar_module.outputscale.item():4.3f} "
                    f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.detach():>4.3f} " 
                    f"noise: {self.model.likelihood.noise.item():>4.3f} " 
                    f"mean constant: {self.model.mean_module.constant.item():>4.3f}"
         )
    
    def current_hyperparams(self):
        noise = self.model.likelihood.noise.item()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.model.covar_module.outputscale.item()
        mean_constant = self.model.mean_module.constant.item()
        return (outputscale, lengthscale, noise, mean_constant)

    def set_hyperparams(self, hyperparams = None):
        if hyperparams == None:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(self.noise),
                'covar_module.base_kernel.lengthscale': torch.tensor(self.lengthscale),
                'covar_module.outputscale': torch.tensor(self.outputscale),
                'mean_module.constant': torch.tensor(self.mean_constant)
            }
        else:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(hyperparams[2]).float(),
                'covar_module.base_kernel.lengthscale': hyperparams[1],
                'covar_module.outputscale': torch.tensor(hyperparams[0]).float(),
                'mean_module.constant': torch.tensor(hyperparams[3]).float()
            }
        self.model.initialize(**hypers)
    
    def posterior(self, test_x):
        test_x = torch.tensor(test_x)
        self.model.eval()
        test_x = test_x.double()
        model_posterior = self.model(test_x)
        mean = model_posterior.mean
        std = model_posterior.stddev
        return mean, std
    
    def sample(self, test_x, n_samples = 1):
        self.model.eval()
        test_x = torch.tensor(test_x)
        with torch.no_grad():
            model_posterior = self.model(test_x)
            samples = model_posterior.sample(sample_shape = torch.Size([n_samples]))
        return samples.numpy()




if __name__ == '__main__':
    # use regular spaced points on the interval [0, 1]
    train_X = torch.linspace(0, 0.5, 15)
    test_X = torch.linspace(0, 1, 100)
    # training data needs to be explicitly multi-dimensional
    train_X = train_X.reshape(-1, 1)
    test_X = test_X.reshape(-1, 1)

    # sample observed values and add some synthetic noise
    train_Y = torch.sin(train_X * (2 * 3.1416)) + 0.1 * torch.randn_like(train_X)

    model = BoTorchGP()
    model.fit_model(train_X, train_Y, train_hyperparams=True, previous_hyperparams = (1, .1, .1))
    model.optim_hyperparams(verbose=True, num_of_epochs = 150)

    mean, std = model.posterior(test_X)
    mean = mean.detach()
    std = std.detach()
    
    import matplotlib.pyplot as plt

    samples = model.sample(test_X, n_samples = 10).transpose(0, 1)

    plt.fill_between(test_X.reshape(-1), mean - 1.96*std, mean + 1.96*std, alpha = 0.5)
    plt.plot(test_X.reshape(-1), mean)
    plt.scatter(train_X, train_Y, c = 'r', marker = 'x')
    plt.plot(test_X, samples)
    plt.show()
    'hola'