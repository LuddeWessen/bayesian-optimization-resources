#!/usr/bin/env python
# coding: utf-8

import GPy, numpy as np
from matplotlib import pyplot as plt
np.random.seed(1)


figure, axes = plt.subplots(4,3, figsize=(10,10), tight_layout=True)
kerns = [GPy.kern.RBF(1), GPy.kern.Exponential(1), GPy.kern.Matern32(1),
         GPy.kern.Matern52(1), GPy.kern.Brownian(1),GPy.kern.Bias(1),
         GPy.kern.Linear(1), GPy.kern.PeriodicExponential(1), GPy.kern.White(1),
         GPy.kern.StdPeriodic(1), GPy.kern.Poly(1), GPy.kern.MLP(1)]

#GPy.kern.Matern52(2,ARD=True) + GPy.kern.White(2)

for k,a in zip(kerns, axes.flatten()):
    k.plot(ax=a, x=1)
    a.set_title(k.name.replace('_', ' '))

plt.show()
