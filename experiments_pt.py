import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from gp_utils import BoTorchGP
from sampling import EfficientThompsonSampler

func = lambda x : np.sin(10 * x) + np.exp(-(x - 0.775) ** 2 / 0.1) / 3

# pt_no_optimum = []
pt_optimum = []

max_num_obs = 30

interval_ub = 0.1
interval_lb = 0

for q in range(1, max_num_obs + 1):
    x_train = np.linspace(interval_lb, interval_ub, q).reshape(-1, 1)
    y_train = func(x_train)

    model = BoTorchGP(lengthscale_dim = 1)
    model.fit_model(x_train, y_train)
    model.set_hyperparams((0.6, 0.1, 1e-5, 0))

    num_of_samples = 5000
    sampler = EfficientThompsonSampler(model, num_of_samples = num_of_samples, num_of_multistarts = 10)
    sampler.create_sample()
    samples = sampler.generate_candidates()
    samples = samples.detach().numpy()
    in_area_optimum = (samples < interval_ub) & (samples > interval_lb)
    pt_optimum.append(sum(in_area_optimum) / num_of_samples)
    # pt_no_optimum.append(sum(in_area_no_optimum) / num_of_samples)
    print('done with q:', q)

full_grid = np.linspace(0, 1, 101).reshape(-1, 1)
mean, std = model.posterior(full_grid)

Ys = func(full_grid)

fig, ax = plt.subplots(nrows = 2, ncols = 1)

ub = mean.detach() + 1.96 * std.detach()
lb = mean.detach() - 1.96 * std.detach()

ub = ub.numpy()
lb = lb.numpy()

ax[1].plot(range(1, max_num_obs + 1), pt_optimum)
ax[1].set_xlabel(f'number of queries in [{interval_lb}, {interval_ub}]')
ax[1].set_ylabel('estimate of $p_t$')
# ax[0].plot(range(1, max_num_obs + 1), pt_no_optimum, label = 'pt no optim')

ax[0].plot(full_grid.reshape(-1), mean.detach(), 'b', label = 'gp mean')
ax[0].fill_between(full_grid.reshape(-1), ub, lb, color = 'b', alpha = 0.2)
ax[0].plot(full_grid, Ys, 'k--', label = 'true function')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[0].set_title('Bimodal optimisation and evolution of escape probability')
plt.legend()
plt.show()