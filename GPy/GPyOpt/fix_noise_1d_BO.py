#!/usr/bin/env python
# coding: utf-8

# Global Optimization using GPs
# Based on material from GPSS21

# We demo:
# (1) the choice of the model
# (2) the choice of the acquisition function.


# ## 1. Getting started

# In addition to GPy, this lab uses GPyOpt (http://sheffieldml.github.io/GPy/), a satellite module of GPy useful to solve global optimization problems. Please be sure that it is correctly installed before starting by following the Getting Started page.

# Some of the options of GPyOpt depend on other external packages: DIRECT, cma, pyDOE. Please be sure that this are installed if you want to use all the options. With everything installed, you are ready to start.
#
# Now, just as in the previous lab, specify to include plots in the notebook and to import relevant libraries.

import GPy
import GPyOpt
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import matplotlib

from GPyOpt.models import GPModel
seed(12345)


# ### Exercise 1

# We use BO to find the minimum of f(x)= x^2 + 10*sin(x), x in [-10, 10].

# We define the bounds of the problem
bnds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-10,10)}]
print(type(bnds))
print(type(bnds[0]))
print(bnds)

# We define the objective function
def my_f(x):
    print("x == ", x)
    return x**2 + 10*np.sin(x)

# We define input data, and generate output
x = np.arange(-10.0,10.0, 0.5)[:,None]

y = my_f(x)
print(np.shape(y))


"""
 Create a GPyOpt object for global optimization
 using a Mattern52 kernel and
 adding a gitter of $0.1$ to the expected improvement acquisition (hence the mysterious acquisition_par = 0.1).
"""
my_k = GPy.kern.Matern52(1)

my_BOpt = GPyOpt.methods.BayesianOptimization(f = my_f,        # function to optimize
                                             domain = bnds,        # box-constrains of the problem
                                             kernel = my_k,             # kernel of the GP
                                             acquisition_type='EI',
                                             acquisition_par=0.1,
                                             X=x,
                                             Y=y,
                                             initial_design_numdata=1
                                             )       # acquisition = Expected improvement

"""
 Constrain the noise of the model to be 10e-4.

 (Constraints on model noise as constraints is set on the GP-model, _not_ on the BO-model)
"""
#my_BOpt.model = GPyOpt.models.gpmodel.GPModel(kernel=my_k, noise_var=0.0001)

print("type(my_BOpt.model)")
print(type(my_BOpt.model))
print("type(my_BOpt.model.model)")
print(type(my_BOpt.model.model))


# If we want to initialize (or fix) parameters of the kernel
print("\nmy_BOpt.model.kernel (default): ")
print(my_BOpt.model.kernel)
print(type(my_BOpt.model.kernel.variance))
#my_BOpt.model.kernel.variance.fix(0.0001) #variance is a object of type 'GPy.core.parameterization.param.Param'


print("my_BOpt.model.noise_var: ")
print(my_BOpt.model.noise_var)
my_BOpt.model.noise_var = 0.1 # this is used for the first GP-regression of the GPy-model, but then is no longer used or coupled to the value of the GPy-model Gaussian noise
print(my_BOpt.model.noise_var)

#print(my_BOpt.model.get_model_parameters_names())

print("\nmy_BOpt.model.kernel (updated): ")

# Run the BO (fixed no iterations)
print(my_BOpt.model)
max_iter = 15                       # evaluation budget
my_BOpt.run_optimization(0)   # run optimization

print("my_BOpt.model.model: (pre init)")
print(my_BOpt.model.model)

# This triggers the my_BOpt (a BayesianOptimization instance) to update _its_ self.model (in our case a GPModel(BOModel))
# An update of the GPModel means to update (or create) the self.model (a GPy model)
try:
    print("Next x,y: ", my_BOpt.acquisition.optimize())
except:
    print("Next x,y could not be calculated ")

print("my_BOpt.model.model: (post init)")
print(my_BOpt.model.model)

print("my_BOpt.model.model.['"'Gaussian_noise'"']: (pre fix)")
print(my_BOpt.model.model.Gaussian_noise)
my_BOpt.model.model.Gaussian_noise.fix(0.0001)
print("my_BOpt.model.model['"'Gaussian_noise'"']: (post fix)")

print("my_BOpt.model.noise_var: ")
print(my_BOpt.model.noise_var)

#print(my_BOpt.model.model["Gaussian_noise"])

my_BOpt.run_optimization(max_iter)   # run optimization
print(my_BOpt.model.model["Gaussian_noise"])

acq_plot = my_BOpt.plot_acquisition()
plt.show(acq_plot)


my_BOpt.plot_convergence()
