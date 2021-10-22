#!/usr/bin/env python
"""
    This bug report is due shows that parameter values cannot be constrained if being choice parameters, despite being int
"""


import numpy as np
import time

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from ax.modelbridge.registry import Models

use_choice_value = False
use_parameter_constraint = True

def dist(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(2)])
    return {"l2norm": (np.sqrt(((x) ** 2).sum()), 0.0)}

if use_choice_value:
    parameters=[
        {
            "name": "x1",
            "type": "choice",
            "values": [0,1,2,3,4,5,6,7,8,9,10],
            "value_type": "int"  # Optional, defaults to inference from type of "bounds".
            #"is_ordered": True
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
    ]
else:
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "is_ordered": True
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
    ]

if use_parameter_constraint:
    parameter_constraints=["x2 - x1 >= 1"]
else:
    parameter_constraints=[]#["x2 - x1 >= 1"]#, "x3 - x2 >= 1", "x4 - x3 >= 1", "x5 - x4 >= 1", "x6 - x5 >= 1"]

outcome_constraints=[] #"l1norm <= 1.25"]
cfun = "l2norm"


best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name="test",
    objective_name="l2norm",
    evaluation_function=dist,
    minimize=True,
    parameter_constraints=parameter_constraints,
    total_trials=13,
)


"""Plot argmin"""
print("\nArgmin: ", best_parameters)


"""Plot argmin"""
print("\nModel: ", type(model))
print(model)


"""Plot evolution cost function value:"""

best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
print("args:\n")
pars = [key for key in model.model_space.parameters.keys()]
for par_name in pars:
    print(par_name, " values: ", np.array([[trial.generator_runs[0].arms[0].parameters[par_name] for trial in experiment.trials.values()]]))

#best_args = np.array([[trial.parameters for trial in experiment.trials.values()]])
print("best_objectives")
print(best_objectives)
#print(best_args)

best_objective_plot = optimization_trace_single_method(
    #y=np.minimum.accumulate(best_objectives, axis=1), #keeps returning minimum of all solutions thus far
    y=best_objectives, #returns the best cost of the actual optimizations
    optimum=0,
    title="Model performance vs. # of iterations",
    ylabel=cfun,
)

print("best_objective_plot")
render(best_objective_plot)
print(type(best_objective_plot))
