import numpy as np
import torch
from math import pi

class EfficientThompsonSampler():
    def __init__(self, model, num_of_multistarts = 5, num_of_bases = 1024, num_of_samples = 1):
        '''
        Implementation of 'Efficiently Sampling From Gaussian Process Posteriors' by Wilson et al. (2020). It allows
        us to create approximate samples of the GP posterior, which we can optimise using gradient methods. We do this
        to generate candidates using Thompson Sampling. Link to the paper: https://arxiv.org/pdf/2002.09309.pdf .
        '''
        # GP model
        self.model = model
        # inputs
        if type(self.model.train_x) == torch.Tensor:
            self.train_x = self.model.train_x
        else:
            self.train_x = torch.tensor(self.model.train_x)
        self.x_dim = torch.tensor(self.train_x.shape[1])
        self.train_y = self.model.train_y
        self.num_of_train_inputs = self.model.train_x.shape[0]
        # thompson sampling parameters
        self.num_of_multistarts = num_of_multistarts
        self.num_of_bases = num_of_bases
        self.num_of_samples = num_of_samples
        # optimisation parameters
        self.learning_rate = 0.01
        self.num_of_epochs = 10 * self.x_dim
        # obtain the kernel parameters
        self.sigma = self.model.model.likelihood.noise.item()
        self.lengthscale = self.model.model.covar_module.base_kernel.lengthscale.detach().float()
        self.outputscale = self.model.model.covar_module.outputscale.item()
        # obtain the kernel 
        self.kernel = self.model.model.covar_module
        # define the Knn matrix
        with torch.no_grad():
            self.Knn = self.kernel(self.train_x)
            self.Knn = self.Knn.evaluate()
            # precalculate matrix inverse
            self.inv_mat = torch.inverse(self.Knn + self.sigma * torch.eye(self.num_of_train_inputs))
        
        self.create_fourier_bases()
        self.calculate_phi()
    
    def create_fourier_bases(self):
        # sample thetas
        self.thetas = torch.randn(size = (self.num_of_bases, self.x_dim)) / self.lengthscale
        # sample biases
        self.biases = torch.rand(self.num_of_bases) * 2 * pi

    def create_sample(self):
        # sample weights
        self.weights = torch.randn(size = (self.num_of_samples, self.num_of_bases)).float()
    
    def calculate_phi(self):
        '''
        From the paper, we are required to calculate a matrix which includes the evaluation of the training set, X_train,
        at the fourier frequencies. This function calculates that matrix, Phi.
        '''
        # we take the dot product by element-wise multiplication followed by summation
        thetas = self.thetas.repeat(self.num_of_train_inputs, 1, 1)
        prod = thetas * self.train_x.unsqueeze(1)
        dot = torch.sum(prod, axis = -1)
        # add biases and take cosine to obtain fourier representations
        ft = torch.cos(dot + self.biases.unsqueeze(0))
        # finally, multiply by corresponding constants (see paper)
        self.Phi = (self.outputscale * np.sqrt(2 / self.num_of_bases) * ft).float()
    
    def calculate_V(self):
        '''
        From the paper, to give posterior updates we need to calculate the vector V. Since we are doing multiple samples
        at the same time, V will be a matrix. We can pre-calculate it, since its value does not depend on the query locations.
        '''
        # multiply phi matrix by weights
        # PhiW: num_of_train x num_of_samples
        PhiW = torch.matmul(self.Phi, self.weights.T)
        # add noise (see paper)
        PhiW = PhiW + torch.randn(size = PhiW.shape) * self.sigma
        # subtract from training outputs
        mat1 = self.train_y - PhiW
        # calculate V matrix by premultiplication by inv_mat = (K_nn + I_n*sigma)^{-1}
        # V: num_of_train x num_of_samples
        self.V = torch.matmul(self.inv_mat, mat1)

    def calculate_fourier_features(self, x):
        '''
        Calculate the Fourier Features evaluated at some input x
        '''
        # evaluation using fourier features
        self.posterior_update(x)
        # calculate the dot product between the frequencies, theta, and the new query points
        dot = x.matmul(self.thetas.T)
        # calculate the fourier frequency by adding bias and cosine
        ft = torch.cos(dot + self.biases.unsqueeze(0))
        # apply the normalising constants and return the output
        return self.outputscale * np.sqrt(2 / self.num_of_bases) * ft

    def sample_prior(self, x):
        '''
        Create a sample form the prior, evaluate it at x
        '''
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        # calculate the fourier features evaluated at the query points
        out1 = self.calculate_fourier_features(x)
        # extend the weights so that we can use element wise multiplication
        weights = self.weights.repeat(self.num_of_multistarts, 1, 1)
        # return the prior
        return torch.sum(weights * out1, axis = -1)

    def posterior_update(self, x):
        '''
        Calculate the posterior update at a location x
        '''
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
        # x: num_of_multistarts x num_of_samples x dim
        self.calculate_V()
        # train x: num_of_multistarts x num_of_train x dim
        train_x = self.train_x.repeat(self.num_of_multistarts, 1, 1)
        # z: num_of_multistarts x num_of_train x num_of_samples
        # z: kernel evaluation between new query points and training set
        z = self.kernel(train_x, x)
        z = z.evaluate()
        # we now repeat V the number of times necessary so that we can use element-wise multiplication 
        V = self.V.repeat(self.num_of_multistarts, 1, 1)
        out = z * V
        return out.sum(axis = 1) # we return the sum across the number of training point, as per the paper

    def query_sample(self, x):
        '''
        Query the sample at a location 
        '''
        prior = self.sample_prior(x)
        update = self.posterior_update(x)
        return prior + update
    
    def generate_candidates(self):
        '''
        Generate the Thompson Samples, this function optimizes the samples.
        '''
        # we are always working on [0, 1]^d
        bounds = torch.stack([torch.zeros(self.x_dim), torch.ones(self.x_dim)])
        # initialise randomly - there is definitely much better ways of doing this
        X = torch.rand(self.num_of_multistarts, self.num_of_samples, self.x_dim)
        X.requires_grad = True
        # define optimiser
        optimiser = torch.optim.Adam([X], lr = self.learning_rate)

        for _ in range(self.num_of_epochs):
            # set zero grad
            optimiser.zero_grad()
            # evaluate loss and backpropagate
            losses = - self.query_sample(X)
            loss = losses.sum()
            loss.backward()
            # take step
            optimiser.step()

            # make sure we are still within the bounds
            for j, (lb, ub) in enumerate(zip(*bounds)):
                X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
        # check the final evaluations
        final_evals = self.query_sample(X)
        # choose the best one for each sample
        best_idx = torch.argmax(final_evals, axis = 0)
        # return the best one for each sample, without gradients
        X_out = X[best_idx, range(0, self.num_of_samples), :]
        return X_out.detach()