#!/usr/bin/env python
# coding: utf-8

# Global Optimization with BO, using GPs
# Based on material from GPSS21

# We demo:
# (1) the choice of the model
# (2) the choice of the acquisition function.

# In addition to GPy, we use GPyOpt, building upon GPy.
# Some of the options of GPyOpt depend on other external packages: DIRECT, cma, pyDOE.

import GPy
import GPyOpt
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import matplotlib
from GPyOpt import Design_space  ## GPyOpt design space
from GPyOpt.models import GPModel
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionLCB, AcquisitionMPI

seed(12345)

# Acquisition functions

"""
 We look to different acquisition functions.
 In GPyOpt we can use:
    - the expected improvement ('EI')
    - the maximum probability of improvement ('MPI') and
    - the lower confidence bound.

 In GPyOpt we can specify acquisition fn when creating the BO object.
 We can also load these acquisitions as separate objects.
"""

# To access these acquisitions 'externally' we create a GP model using the objective function in Section 1 evaluated in 10 locations.

f_true = GPyOpt.fmodels.experiments1d.forrester()             # true function object
f_sim = GPyOpt.fmodels.experiments1d.forrester(sd=.25)        # noisy version
bounds = f_true.bounds                                        # problem constrains (implemented by default)
f_objective = f_sim.f                                         # objective function


# To plot the true $f$, simply write:
f_true.plot(bounds)

seed(1234)
n = 10
X = np.random.rand(n).reshape(n,1)
Y = f_objective(X)
m = GPy.models.GPRegression(X,Y)
m.optimize()
m.plot([0,1])

## Now we pass this model into a GPyOpt Gaussian process model


model = GPModel(optimize_restarts=5,verbose=False)
model.model = m

# We define the bounds of the input space to be between zero and one.



space = Design_space([{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}] )

# Now, let's have a look to see what do we need to create an acquisition, for instance the Expected improvement and the Lower Confidence Bound.

# Now we create thee objects, one for each acquisition. The jitter parameter, to balance exploration and exploitation, need to be specified.

acq_EI = AcquisitionEI(model,space, jitter = 0)
acq_LCB = AcquisitionLCB(model,space,exploration_weight = 2)
acq_MPI = AcquisitionMPI(model,space,jitter = 0)


"""
 The objects acq_EI, acq_LCB, acq_MPI contain the acquisition functions and their gradients.
 By running the following piece of code you can visualize the three acquisitions.
 Note that we add a negative sign before each acquisition.
 This is because within GPyOpt these functions are minimized (instead of maximized) using gradient optimizers (like BFGS) to select new locations.
 In this plot, however, the larger is the value of the acquisition, the better is the point.
"""

# Plot the three acquisition functions (factor 0.1 added in in the LCB for visualization)
X_grid = np.linspace(0,1,200)[:, None]
plt.figure(figsize=(10,6))
plt.title('Acquisition functions',size=25)
plt.plot(X_grid, - 0.1*acq_LCB.acquisition_function(X_grid),label='LCB')
plt.plot(X_grid, -acq_EI.acquisition_function(X_grid),label='EI')
plt.plot(X_grid, -acq_MPI.acquisition_function(X_grid),label='MPI')
plt.xlabel('x',size=15)
plt.ylabel('a(x)',size=15)
plt.legend()

plt.show()

"""
 Comparing the LCB acquisition (of GP-UCB in the literature) w parameters: [0,0.1,0.25,0.5,1,2,5]
"""

# Plot the three acquisition functions (LCB is scaled)
X_grid = np.linspace(0,1,200)[:, None]
plt.figure(figsize=(10,6))
plt.title('LCB Acquisition functions',size=25)
acq_LCBs = []
exploration_vals = [0,0.1,0.25,0.5,1,2,5]
for i, exp_val in zip(range(len(exploration_vals)), exploration_vals):
    acq_LCBs.append(AcquisitionLCB(model,space,exploration_weight = exp_val))
    plt.plot(X_grid, - 0.1*acq_LCBs[i].acquisition_function(X_grid),label='LCB exp val='+str(exp_val))

plt.xlabel('x',size=15)
plt.ylabel('a(x)',size=15)
plt.legend()

plt.show()
