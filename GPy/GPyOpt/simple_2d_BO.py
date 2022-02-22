#!/usr/bin/env python
# coding: utf-8

# Global Optimization using GPs
# Based on material from GPSS21

# We demo:
# (1) the choice of the model
# (2) the choice of the acquisition function.

# Some of the options of GPyOpt depend on other external packages: DIRECT, cma, pyDOE.

import numpy as np
from numpy.random import seed

import GPy
import GPyOpt
from GPyOpt import Design_space  ## GPyOpt design space
from GPyOpt.models import GPModel
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionLCB, AcquisitionMPI

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, Formatter

seed(11)

# We use a example function, available in GPyOpt
GPyOpt.fmodels.experiments2d.sixhumpcamel().plot()


# Extract the objective:
f_shc = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.1).f  # simulated version with some noise

"""
(a) Create three objects to optimize this function using
the 'EI' (with parameter equal to zero),
the LCB (with parameter equal to 2) and
the MPI (with parameter equal to zero).

Use the same initial data in the three cases, e.g by using the options 'X' and 'Y' when creating the BO object.
"""

# Use matern kernel:
shc_k = GPy.kern.Matern52(2)
shc_k = GPy.kern.RBF(2)

# Define bounds+ constraints (here we dont have any contraints)
shc_domain =[{'name': 'var_1', 'type': 'continuous', 'domain':(-2,2), 'dimensionality':1},
                      {'name': 'var_2', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1}]
shc_space = Design_space(shc_domain)

n = 3
X = np.random.rand(2*n).reshape(n,2)
Y = f_shc(X)

print("-------------------------------------")
print("X shape: ", np.shape(X))
print("-------------------------------------")


m = GPy.models.GPRegression(X,Y)
m.optimize()

#Create a GP-object, and give the model to this GP model
model = GPModel(optimize_restarts=5,verbose=False)
model.model = m

shc_acqEI = AcquisitionEI(model,shc_space,jitter = 1.0)
shc_acqLCB = AcquisitionLCB(model,shc_space,exploration_weight = 2.0)
shc_acqMPI = AcquisitionMPI(model,shc_space,jitter=1.0)


shc_kwargs = {'acquisition' : shc_acqEI}
my_BOpt2_EI = GPyOpt.methods.BayesianOptimization(f = f_shc,        # function to optimize
                                             domain = shc_domain, # box-constrains of the problem
                                             kernel = shc_k,             # kernel of the GP
                                             acquisition_type='EI',
                                             X=X,
                                             Y=Y,
                                             #acquisition_par=0.0,
                                             kwargs=shc_kwargs,
                                             )       # acquisition = Lower Confidence Bound

shc_kwargs = {'acquisition' : shc_acqLCB}
my_BOpt2_LCB = GPyOpt.methods.BayesianOptimization(f = f_shc,        # function to optimize
                                             domain = shc_domain, # box-constrains of the problem
                                             kernel = shc_k,             # kernel of the GP
                                             X=X,
                                             Y=Y,
                                             kwargs=shc_kwargs,
                                             )       # acquisition = Lower Confidence Bound

shc_kwargs = {'acquisition' : shc_acqMPI}
my_BOpt2_MPI = GPyOpt.methods.BayesianOptimization(f = f_shc,        # function to optimize
                                             domain = shc_domain, # box-constrains of the problem
                                             kernel = shc_k,             # kernel of the GP
                                             X=X,
                                             Y=Y,
                                             kwargs = shc_kwargs,
                                             )       # acquisition = Lower Confidence Bound


print(np.shape(X))
print(np.shape(Y))

fig = plt.figure()

plt.scatter(X[:,0], X[:,1], label="initial x,y")
#plt.legend()

# Make legend, set axes limits and labels
fig.legend()

plt.xlim(-2, 2)
plt.ylim(-1, 1)

plt.show()

model.model.plot()


# Acquisition functions

# Run the optimization (may take a few senconds)
max_iter = 30                       # evaluation budget
my_BOpt2_EI.run_optimization(max_iter)   # run optimization
print(my_BOpt2_EI.model.model)
my_BOpt2_EI.model.model.plot()
my_BOpt2_EI.plot_acquisition()


# Low BO
my_BOpt2_LCB.run_optimization(max_iter)   # run optimization
print(my_BOpt2_LCB.model.model)
my_BOpt2_LCB.model.model.plot()
my_BOpt2_LCB.plot_acquisition()


# MPI
my_BOpt2_MPI.run_optimization(max_iter)   # run optimization
print(my_BOpt2_MPI.model.model)
my_BOpt2_MPI.model.model.plot()
my_BOpt2_MPI.plot_acquisition()


# Individually
#my_BOpt2_EI.plot_convergence()
#y_BOpt2_LCB.plot_convergence()
#my_BOpt2_MPI.plot_convergence()

# Compare
def bo_plot(bo_model, lbl):
    no_iter = np.shape(bo_model.Y_best)[0]
    #plt.plot(range(no_iter), bo_model.Y, label=lbl)
    plt.plot(range(no_iter), bo_model.Y_best, label=lbl+"_best")

bo_plot(my_BOpt2_EI, "EI")
bo_plot(my_BOpt2_LCB, "LCB")
bo_plot(my_BOpt2_MPI, "MPI")

plt.legend()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

def bo_scatter(bo_model, lbl):
    no_iter = np.shape(bo_model.Y_best)[0]
    ax.scatter(bo_model.X[:,0], bo_model.X[:,1], bo_model.Y, marker='o', label=lbl)

bo_scatter(my_BOpt2_EI, "EI")
bo_scatter(my_BOpt2_LCB, "LCB")
bo_scatter(my_BOpt2_MPI, "MPI")

fig.legend()
plt.xlim(-2, 2)
plt.ylim(-1, 1)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

"""
 Plot the function
"""
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


# Make data.
X = np.arange(-2, 2, 0.25)[:,None]
Y = np.arange(-1, 1, 0.125)[:,None]

print(np.shape(X))
print(np.shape(Y))
print(np.shape(np.hstack((X,Y))))

#print(np.shape(Z))
X, Y = np.meshgrid(X, Y)
#print(Z)

no_d = np.shape(X)[0]
Z_mean = np.empty(np.shape(X), 'float')
Z_stddev = np.empty(np.shape(X), 'float')
Z_stddev = np.empty(np.shape(X), 'float')
for dx_i in range(np.shape(X)[0]):
    for dy_i in range(np.shape(X)[1]):
        Z_mean[dx_i, dy_i], Z_stddev[dx_i, dy_i] = my_BOpt2_EI.model.predict(np.asarray([X[dx_i,dy_i], Y[dx_i,dy_i]]))

#Z = f_shc(np.hstack((X,Y)))
# Plot the surface.
surf = ax.plot_surface(X, Y, Z_mean,
                       cmap=cm.coolwarm,
                       linewidth=0,
                       antialiased=False
                      )
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:.02f}'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
