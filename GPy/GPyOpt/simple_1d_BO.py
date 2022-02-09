#!/usr/bin/env python
# coding: utf-8

# Global Optimization using GPs
# Based on material from GPSS21

# We demo:
# (1) the choice of the model
# (2) the choice of the acquisition function.

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
seed(12345)


# Before starting with the lab, remember that (BO) is an heuristic for global optimization of black-box functions. Let $f: {\mathcal X} \to R$ be a 'well behaved' continuous function defined on a compact subset ${\mathcal X} \subseteq R^d$. Our goal is to solve the global optimization problem of finding
# $$ x_{M} = \arg \min_{x \in {\mathcal X}} f(x). $$

# 1. A **Gaussian process** that will capture the our beliefs on $f$.
# 2. An **acquisition function** that based on the model will be useful to determine where to collect new evaluations of f.

# Remember that every time a new data point is collected the model is updated and the acquisition function optimized again.

# ### Running example

# We start with a one-dimensional example. Consider here the Forrester function
# $$f(x) =(6x-2)^2 \sin(12x-4),$$ defined on the interval $[0, 1]$.
# The minimum of this function is located at $x_{min}=0.78$. We assume that the evaluations of $f$ to are perturbed by zero-mean Gaussian noise with standard deviation 0.25. The Forrester function is part of the benchmark of functions of GPyOpt. To create the true function, the perturbed version and the boundaries of the problem you need to run the following cell.


f_true = GPyOpt.fmodels.experiments1d.forrester()             # true function object
f_sim = GPyOpt.fmodels.experiments1d.forrester(sd=.25)        # noisy version
bounds = f_true.bounds                                        # problem constrains (implemented by default)
f_objective = f_sim.f                                         # objective function


# To plot the true $f$, simply write:
f_true.plot(bounds)

# Bounds:
print(type(bounds))
print(type(bounds[0]))
print(bounds)


# f_objective contains the function that we are going to optimize. You can define your own objective but it should be able to map any numpy array of dimension $n\times d$ (inputs) to a numpy array of dimension $n\times 1$ (outputs). For instance:

n = 8
x = np.random.rand(n).reshape(n,1)
print("x shape: ", np.shape(x))


f_objective(x)


# The bounds of the problem should be defined as a tuple containing the upper and lower limits of the box in which the optimization will be performed. In our example:
bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]


# To use BO to solve this problem, we need to create a GPyOpt object in which we need to specify the following elements:
# * The function to optimize.
# * The box constrains of the problem.
# * The model, that is fixed by default to be a GP with a SE kernel.
# * The acquisition function (and its parameters).

# We create an SE kernel as we do in GPy
k = GPy.kern.RBF(1)

# And now we have all the elements to start optimizing $f$. We create the optimization problem instance. Note that you don't need to specify the evaluation budget of. This is because at this stage we are not running the optimization, we are just initializing the different elements of the BO algorithm.

# Creation of the object that we will use to run BO.
seed(1234)
myBopt = GPyOpt.methods.BayesianOptimization(f = f_objective,        # function to optimize
                                             domain = bounds,        # box-constrains of the problem
                                             kernel = k,             # kernel of the GP
                                             acquisition_type='EI')       # acquisition = Expected improvement


# At this point you can access to a number of elements in myBopt, including the GP model and the current dataset (initialized at 3 random locations by default).
print(myBopt.X)
print(np.shape(myBopt.X))

print(myBopt.Y)

# Run the optimization (may take a few senconds)
max_iter = 15                       # evaluation budget
myBopt.run_optimization(max_iter)   # run optimization


# And that's it! You should have receive a message describing if the method converged (two equal x's are selected in consecutive steps of the optimization) or if the maximum number of iterations was reached. In one dimensional examples, you can visualize the model and the acquisition function (normalized between 0 and 1) as follows.
acq_plot = myBopt.plot_acquisition()
plt.show(acq_plot)

# You can only make the previous plot if the dimension of the problem is 1 or 2. However, you can always how the optimization evolved by running:
c_plot = myBopt.plot_convergence()
plt.show(c_plot)

# The first plot shows the distance between the last two collected observations at each iteration. This plot is useful to evaluate the convergence of the method. The second plot shows the best found value at each iteration. It is useful to compare different methods. The fastest the curve decreases the better the method.
#
# Noise variance of the GP is automatically bounded to avoid numerical problems. In case of having a problem where the evaluations of $f$ are exact you only need to include 'exact_feval=True' when creating the BO object as above. Now, to run the optimization for certain number of iterations you only need to write:

# In[15]:


print("myBopt.model.model")
print(myBopt.model.model)

print(type(myBopt.model.model)) #.matern52.variance))
print(myBopt.model.model)
print(myBopt.model.model['Gaussian_noise'])
#myBopt.model.model['Gaussian_noise'].Value = 0.0001
myBopt.model.model['Gaussian_noise'].fix(0.0001)
print(myBopt.model.model['Gaussian_noise'])
print(myBopt.model.model)
#print(myBopt.model.model['Gaussian_noise'].Value)
#?myBopt.model.model
