#!/usr/bin/env python
# coding: utf-8

import GPy, numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)

# sample inputs and outputs from 2d model
X2 = np.random.uniform(-3.,3.,(50,2))
Y2 = np.sin(X2[:,0:1]) + np.sin(X2[:,1:2]) + np.random.randn(50,1)*0.05

# define a sum kernel, where the first dimension is modelled with an RBF, and the second dimension is modelled with a linear kerel
ker = GPy.kern.RBF(input_dim=1, active_dims=[0]) + GPy.kern.RBF(input_dim=1, active_dims=[1]) # GPy.kern.StdPeriodic(input_dim=1, active_dims=[1]) # GPy.kern.Linear(input_dim=1, active_dims=[1])


# create simple GP model
model = GPy.models.GPRegression(X2,Y2,ker)
print("Parameters Pre Optimizations: ")
print(model)
model.optimize()
print("Parameters Post Optimizations: ")
print(model)

plot = model.plot()
plt.show(plot)



"""
If we want to plot the marginal of the model where we fix one input to a certain value,
we can use "fixed_inputs", the format for which is a list of tuples,
where we specify the dimension, and the fixed value.
"""
# Fix 1st dimension (the domain of the linear kernel) to 0.2
plt.show(model.plot(fixed_inputs=[(1, 0.2)]))

# Fix the 0th dimension (domain of rbf) to 1.0
plt.show(model.plot(fixed_inputs=[(0, 1.0)]))

"""
As expected, the data doesn't sit on the posterior distribution,
as this is a projection of all the data onto one dimension.
"""

# EP

# ### 2D example

# sample inputs and outputs
X = X2 #np.random.uniform(-3.,3.,(50,2))
Y = np.sin(X[:,0:1]) * np.sin(X[:,1:2])+np.random.randn(50,1)*0.05

# define kernel
ker = GPy.kern.Matern52(2,ARD=True)# + GPy.kern.White(2)

# create simple GP model
model2 = GPy.models.GPRegression(X,Y,ker)

print("Parameters Pre Optimizations: ")
print(model2)

# optimize and plot
model2.optimize(messages=True,max_f_eval = 1000)
print("Parameters Post Optimizations: ")
print(model)
fig = model2.plot()
plt.show(fig)
print(model2)


# Plotting slices
#
# To see the uncertaintly associated with the above predictions, we can plot slices through the surface.
# This is done by passing the optional fixed_inputs argument to the plot function.
# Fixed_inputs is a list of tuples containing which of the inputs to fix, and to which value.
#
# To get horixontal slices of the above GP, we'll fix second (index 1) input to -1, 0, and 1.5:


import matplotlib
#matplotlib.use("Agg") # if we want to save figure
GPy.plotting.change_plotting_library("matplotlib")

#figure, axes = plt.subplots(3, 1, constrained_layout=True)
slices = [-1, 0, 1.5]
figure = GPy.plotting.plotting_library().figure(3, 1)#, constrained_layout=True, tight_layout=True)

figure.suptitle('This is the figure title', fontsize=16)


for i, y in zip(range(3), slices):
    coll_dict = model2.plot(figure=figure, fixed_inputs=[(1,y)], row=(i+1), legend=True) #, plot_raw=True, plot_density=True) #)#, plot_data=False))
    figure.axes[i].set_title("subplot "+str(i))
    figure.axes[i].set_xlabel("xlabel "+str(i))
    figure.axes[i].set_ylabel("ylabel "+str(i))

gs = figure.gridspec
print(type(gs))

gs.tight_layout(figure, rect=[0, 0.03, 1, 0.95])

plt.show()
