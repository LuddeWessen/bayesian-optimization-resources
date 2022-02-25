#!/usr/bin/env python
# coding: utf-8

import GPy, numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)

# ## Non-Gaussian likelihoods
"""
Considered the olympic marathon data
In 1904 we noted there was an outlier example.
We deal with that outlier by considering a non-Gaussian likelihood.
Noise sampled from a Student-t density is heavier tailed than that sampled from a Gaussian.
However, it cannot be trivially assimilated into the Gaussian process. Below we use the Laplace approximation to incorporate this noise model.
"""

# Download the marathon data from yesterday and plot
GPy.util.datasets.authorize_download = lambda x: True # prevents requesting authorization for download.
data = GPy.util.datasets.olympic_marathon_men()
X = data['X']
Y = data['Y']

#plt.plot(X, Y, 'bx')
#plt.xlabel('year')
#plt.ylabel('marathon pace min/km')


# We might want to use a non-Gaussian likelihood here. This also means that our inference method can no longer be exact, and we have to use one of the approximate methods. Lets use a Student-T distribution that has heavy tails to allow for the outliers without drastically effecting the posterior mean of the GP.
# Laplace:

t_distribution = GPy.likelihoods.StudentT(deg_free=5.0, sigma2=2.0)
laplace = GPy.inference.latent_function_inference.Laplace()

kern = GPy.kern.RBF(1, lengthscale=5) + GPy.kern.Bias(1, variance=4.0)
m_stut = GPy.core.GP(X, Y, kernel=kern, inference_method=laplace, likelihood=t_distribution)

m_stut.optimize()

print(m_stut)

# Show
plot = m_stut.plot(plot_density=False)
plt.xlabel('year')
plt.ylabel('marathon pace min/km')

plt.show(plot)


plot = m_stut.plot_f(plot_density=False)
plt.xlabel('year')
plt.ylabel('marathon pace min/km')

plt.show(plot)
