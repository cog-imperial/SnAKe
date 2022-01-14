import numpy as np
import math
import torch
from summit.benchmarks import SnarBenchmark
from summit.utils.dataset import DataSet

'''
In this script we include all the Benchmark function which we will evaluate.
'''

class ConvergenceTest():
    def __init__(self):
        self.x_dim = None
        self.t_dim = 1
    
    def draw_new_function(self):
        pass

    def query_function(self, temp):
        return np.sin(10 * temp) + np.exp(-(temp - 0.775) ** 2 / 0.1) / 3

class TwoDSinCosine():
    def __init__(self, random = False):
        self.t_dim = 2
        self.x_dim = None
        self.random = random
        self.optimum = 1.15371572971344
        self.draw_new_function()
        self.name = 'SineCosine 2D'

    def draw_new_function(self):
        if self.random:
            self.mu1 = np.random.uniform()
            self.mu2 = np.random.uniform()
        else:
            self.mu1 = 0
            self.mu2 = 0
        pass

    def query_function(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return np.sin(5 * (x1 - self.mu1)) * np.cos(5 * (x2 - self.mu2)) * np.exp((x1-0.5)**2 / 2) * np.exp((x2-0.5)**2 / 2)

    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        return torch.sin(5 * (x1 - self.mu1)) * torch.cos(5 * (x2 - self.mu2)) * torch.exp((x1-0.5)**2 / 2) * torch.exp((x2-0.5)**2 / 2)

class BraninFunction():
    def __init__(self, t_dim = 2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # optimum calculated using gradient methods (see code at the bottom)
        self.optimum = 1.0473939180374146

        self.name = 'Branin2D'
    
    def draw_new_function(self):
        pass

    def query_function(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
        s2 = (10 - 10 / (8 * math.pi)) * np.cos(x1bar) - 44.81

        return -(s1 + s2) / 51.95
    
    def query_function_torch(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]

        x1bar = 15 * x1 - 5
        x2bar = 15 * x2

        s1 = (x2bar - 5.1 * x1bar**2 / (4 * math.pi**2) + 5 * x1bar / math.pi - 6)**2
        s2 = (10 - 10 / (8 * math.pi)) * torch.cos(x1bar) - 44.81

        return -(s1 + s2) / 51.95

class Hartmann3D():
    def __init__(self, t_dim = 3):
        self.t_dim = t_dim
        if self.t_dim == 3:
            self.x_dim = None
        else:
            self.x_dim = 3 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart3.html
        self.optimum = 3.8627797869493365

        self.name = 'Hartmann3D'

        self.A = np.array( \
            [[3, 10, 30], \
                [0.1, 10, 35], \
                    [3, 10, 30], \
                        [0.1, 10, 35]])
        
        self.P = 1e-4 * np.array( \
            [[3689, 1170, 2673], \
                [4699, 4387, 7470], \
                    [1091, 8732, 5547], \
                        [381, 5743, 8828]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 3):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Hartmann4D():
    def __init__(self, t_dim = 4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart3.html
        self.optimum = 3.7298407554626465

        self.name = 'Hartmann4D'

        self.A = np.array( \
            [[10, 3, 17, 3.5, 1.7, 8], \
                [0.05, 10, 17, 0.1, 8, 14], \
                    [3, 3.5, 1.7, 10, 17, 8], \
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        self.P = 1e-4 * np.array( \
            [[1312, 1696, 5569, 124, 8283, 5886], \
                [2329, 4135, 8307, 3736, 1004, 9991], \
                    [2348, 1451, 3522, 2883, 3047, 6650], \
                        [4047, 8828, 8732, 5743, 1091, 381]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 4):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

    def query_function_torch(self, x):
        A = torch.tensor(self.A)
        P = torch.tensor(self.P)
        alpha = torch.tensor(self.alpha)

        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 4):
                S2 += A[i, j] * (x[:, j] - P[i, j])**2
            S1 += alpha[i] * torch.exp(-S2)
        return S1

class Hartmann6D():
    def __init__(self, t_dim = 6):
        self.t_dim = t_dim
        if self.t_dim == 6:
            self.x_dim = None
        else:
            self.x_dim = 6 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/hart6.html
        self.optimum = 3.322368011391339

        self.name = 'Hartmann6D'

        self.A = np.array( \
            [[10, 3, 17, 3.5, 1.7, 8], \
                [0.05, 10, 17, 0.1, 8, 14], \
                    [3, 3.5, 1.7, 10, 17, 8], \
                        [17, 8, 0.05, 10, 0.1, 14]])
        
        self.P = 1e-4 * np.array( \
            [[1312, 1696, 5569, 124, 8283, 5886], \
                [2329, 4135, 8307, 3736, 1004, 9991], \
                    [2348, 1451, 3522, 2883, 3047, 6650], \
                        [4047, 8828, 8732, 5743, 1091, 381]])

        self.alpha = np.array([1, 1.2, 3, 3.2])

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        S1 = 0
        for i in range(0, 4):
            S2 = 0
            for j in range(0, 6):
                S2 += self.A[i, j] * (x[:, j] - self.P[i, j])**2
            S1 += self.alpha[i] * np.exp(-S2)
        return S1

class Michalewicz2D():
    # taken from website: https://www.sfu.ca/~ssurjano/michal.html
    def __init__(self, t_dim = 2):
        self.t_dim = t_dim
        if self.t_dim == 2:
            self.x_dim = None
        else:
            self.x_dim = 2 - self.t_dim
        # calculated below
        self.optimum = 0.6754469275474548

        self.name = 'Michaelwicz2D'

        self.m = 10

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        x = x * np.pi
        S1 = np.sin(x[:, 0]) * (np.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = np.sin(x[:, 1]) * (np.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        return S1 + S2
    
    def query_function_torch(self, x):
        x = x * 4
        S1 = torch.sin(x[:, 0]) * (torch.sin(x[:, 0] / np.pi))**(2*self.m)
        S2 = torch.sin(x[:, 1]) * (torch.sin(2 * x[:, 1] / np.pi))**(2*self.m)
        return S1 + S2

class Perm8D():
    def __init__(self, t_dim = 8):
        self.t_dim = t_dim
        if self.t_dim == 8:
            self.x_dim = None
        else:
            self.x_dim = 8 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/permdb.html
        self.optimum = 0

        self.name = 'Perm8D'

        self.beta = 0.5

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        x = (x - 0.5) * 16
        S1 = 0
        for i in range(1, 1 + 8):
            S2 = 0
            for j in range(1, 1 + 8):
                S2 += (j**i + self.beta) * ((x[:, j-1] / j)**i - 1)
            S1 += S2**2
        return - S1 / 10**13

class Perm10D():
    def __init__(self, t_dim = 10):
        self.t_dim = t_dim
        if self.t_dim == 10:
            self.x_dim = None
        else:
            self.x_dim = 10 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/permdb.html
        self.optimum = 0

        self.name = 'Perm10D'

        self.beta = 10

    def draw_new_function(self):
        pass
    
    def query_function(self, x):

        x = (x - 0.5) * 20

        S1 = 0
        for i in range(1, 1 + 10):
            S2 = 0
            for j in range(1, 1 + 10):
                S2 += (j**i + self.beta) * ((x[:, j-1] / j)**i - 1)
            S1 += S2**2
        
        out = - S1 / (10 ** 21)
        return out.astype(float)

class Ackley4D():
    def __init__(self, t_dim = 4):
        self.t_dim = t_dim
        if self.t_dim == 4:
            self.x_dim = None
        else:
            self.x_dim = 4 - self.t_dim
        # taken from website: https://www.sfu.ca/~ssurjano/ackley.html
        self.optimum = 0

        self.name = 'Ackley4D'

        self.a = 20
        self.b = 0.2
        self.c = 2 * np.pi

    def draw_new_function(self):
        pass
    
    def query_function(self, x):
        # new optimum
        #shift = np.array([0.4, 0.5, 0.45, 0.55])
        # first reparametrise x
        x = (x - 0.45) * (2 * 2)
        s1 = np.sum(x**2, axis = 1) / 4
        s2 = np.sum(np.cos(self.c * x), axis = 1) / 4
        return self.a * np.exp(-self.b * np.sqrt(s1)) + np.exp(s2) - self.a - np.exp(1)
    
    def query_function_torch(self, x):
        # first reparametrise x
        x = (x - 0.45) * (2 * 2)
        s1 = torch.sum(x**2, axis = 1) / 4
        s2 = torch.sum(torch.cos(self.c * x), axis = 1) / 4
        return self.a * torch.exp(-self.b * torch.sqrt(s1)) + torch.exp(s2) - self.a - np.exp(1)

class SnAr():
    def __init__(self, residence_time = 1):
        self.t_dim = 2
        self.x_dim = 1

        self.name = 'SnarBenchmark'
        self.residence_time = residence_time

        self.snar_bench = SnarBenchmark()
    
    def draw_new_function(self):
        pass

    def query_function(self, x):
        x = x.reshape(1, -1)
        temp = x[:, 0] * 80 + 40
        conc_dfnb = x[:, 1] * 0.4 + 0.1
        equiv_pldn = x[:, 2] * 4 + 1

        values = {
            ("tau", "DATA"): [self.residence_time],
            ("equiv_pldn", "DATA"): [equiv_pldn],
            ("conc_dfnb", "DATA"): [conc_dfnb],
            ("temperature", "DATA"): [temp],
        }

        conditions = DataSet(values)
        experiments = self.snar_bench.run_experiments(conditions, computation_time = False)
        return experiments['sty'][0] / 10000 - experiments['e_factor'][0] / 5

def find_optimum(func, n_starts = 25, n_epochs = 100):
    # find dimension
    if func.x_dim is not None:
        dim = func.x_dim + func.t_dim
    else:
        dim = func.t_dim
    # define bounds
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])
    # random multistart
    X = torch.rand(n_starts, dim)
    X.requires_grad = True
    optimiser = torch.optim.Adam([X], lr = 0.01)

    for i in range(n_epochs):
        # set zero grad
        optimiser.zero_grad()
        # losses for optimiser
        losses = - func.query_function_torch(X)
        loss = losses.sum()
        loss.backward()
        # optim step
        optimiser.step()

        # make sure we are still within the bounds
        for j, (lb, ub) in enumerate(zip(*bounds)):
            X.data[..., j].clamp_(lb, ub) # need to do this on the data not X itself
    
    final_evals = func.query_function_torch(X)
    best_eval = torch.max(final_evals)
    best_start = torch.argmax(final_evals)
    best_input = X[best_start, :].detach()

    return best_input, best_eval

# this last part is used to find the optimum of functions using gradient methods, if optimum is not available online
if __name__ == '__main__':
    func = Michalewicz2D()
    best_input, best_eval = find_optimum(func, n_starts = 100000, n_epochs = 1000)
    print(float(best_eval.detach()))