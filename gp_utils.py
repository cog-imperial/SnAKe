import numpy as np
import torch
from gpytorch.priors import SmoothedBoxPrior
import gpytorch
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

'''
This python file defines the Gaussian Process class which is used in all optimization methods.
'''

class BoTorchGP():
    '''
    Our GP implementation using GPyTorch, to use with BoTorch models and SnAKe.
    '''
    def __init__(self, kernel = None, lengthscale_dim = None):
        # initialize kernel
        if kernel == None:
            self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = lengthscale_dim))
        # initialize if we should set contrainst and if we have a multi-dimensional lengthscale
        self.constraints_set = False
        self.lengthscale_dim = lengthscale_dim
        
    def fit_model(self, train_x, train_y, train_hyperparams = False, previous_hyperparams = None):
        '''
        This function fits the GP model with the given data.
        '''
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

        # check if we should set hyper-parameters or if we should optimize them
        if previous_hyperparams is not None:
            self.outputscale = float(previous_hyperparams[0])
            self.lengthscale = previous_hyperparams[1].detach()
            self.noise = float(previous_hyperparams[2])
            self.mean_constant = float(previous_hyperparams[3])
            self.set_hyperparams()
        
        if train_hyperparams == True:
            self.optim_hyperparams()
    
    def define_constraints(self, init_lengthscale, init_mean_constant, init_outputscale, init_noise = None):
        '''
        This model defines constraints on hyper-parameters as defined in the Appendix of the paper.
        '''
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

        if init_noise is not None:
            self.noise_ub = 3 * init_noise
            self.noise_lb = init_noise / 3
            self.noise_constraint = True
        else:
            self.noise_constraint = False

    def optim_hyperparams(self, num_of_epochs = 500, verbose = False):
        '''
        We can optimize the hype-parameters by maximizing the marginal log-likelihood.
        '''
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

            if self.noise_constraint:
                prior_noise = SmoothedBoxPrior(self.noise_lb, self.noise_ub, 0.001)
                self.model.likelihood.register_prior('Smoothed Box Prior', prior_noise, "noise")
        
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
        '''
        Returns the current values of the hyper-parameters.
        '''
        noise = self.model.likelihood.noise.item()
        lengthscale = self.model.covar_module.base_kernel.lengthscale.detach()
        outputscale = self.model.covar_module.outputscale.item()
        mean_constant = self.model.mean_module.constant.item()
        return (outputscale, lengthscale, noise, mean_constant)

    def set_hyperparams(self, hyperparams = None):
        '''
        This function allows us to set the hyper-parameters.
        '''
        if hyperparams == None:
            hypers = {
                'likelihood.noise_covar.noise': torch.tensor(self.noise),
                'covar_module.base_kernel.lengthscale': self.lengthscale,
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
        '''
        Calculates the posterior of the GP, returning the mean and standard deviation at a corresponding set of points.
        '''
        if type(test_x) is not torch.Tensor:
            test_x = torch.tensor(test_x).double()
        self.model.eval()
        model_posterior = self.model(test_x)
        mean = model_posterior.mean
        std = model_posterior.stddev
        return mean, std
    
    def sample(self, test_x, n_samples = 1):
        '''
        Allows us to samples the GP. For SnAKe and Thompson Sampling, we instead used EfficientThompsonSampler method.
        '''
        self.model.eval()
        test_x = torch.tensor(test_x)
        with torch.no_grad():
            model_posterior = self.model(test_x)
            samples = model_posterior.sample(sample_shape = torch.Size([n_samples]))
        return samples.numpy()