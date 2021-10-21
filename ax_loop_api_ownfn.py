#!/usr/bin/env python
"""
    This example was based on the example in
    https://ax.dev/tutorials/gpei_hartmann_loop.html
    However, here, I have changed the function to be sampled, and tested other metrics, for my own understanding.

    The loop API is the most lightweight way to do optimization in Ax.
    The user makes one call to `optimize`, which performs all of the optimization under the hood and returns the optimized parameters.
    For more customizability of the optimization procedure, use the Service or Developer API.
"""


import numpy as np
import time

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

use_float_values = False
optimize_sum_of_violations = True

# 1. Define evaluation function

"""
    First, we define an evaluation function that is able to compute all the metrics needed for this experiment.
    This function needs to accept a set of parameter values and can also accept a weight.
    It should produce a dictionary of metric names to tuples of mean and standard error for those metrics.
"""
def own_eval_function(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    print("no_elem: ", type(x[0]))


    #lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
    return {"l1norm": (np.abs(x-0.5).sum(), 0.0), "l2norm": (np.sqrt(((x - 0.5) ** 2).sum()), 0.0)} # standard error is 0.0, since we only have 1 function eval (which is deterministic)
    #return (np.sqrt((x ** 2).sum()), 0.0) # standard error is 0.0, since we only have 1 function eval (which is deterministic)

def own_maxsat_alldiff(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(6)])
    no_elem = np.shape(x)[0]
    sum_of_violations = 0
    maxval = max(x)
    for i in range(no_elem):
        for j in range(i+1, no_elem,1):
            if abs(x[i]-x[j]) < 0.9999999:
                sum_of_violations += 10
            #sum_of_violations =+ 1.0 / (0.001 + abs(x[i]-x[j]))



    #lambda p: (p["x1"] + 2*p["x2"] - 7)**2 + (2*p["x1"] + p["x2"] - 5)**2,
    return {"sumofviolations": (sum_of_violations + maxval, 0.0), "l1norm": (np.abs(x-0.5).sum(), 0.0), "l2norm": (np.sqrt(((x - 0.5) ** 2).sum()), 0.0)} # standard error is 0.0, since we only have 1 function eval (which is deterministic)
    #return (np.sqrt((x ** 2).sum()), 0.0) # standard error is 0.0, since we only have 1 function eval (which is deterministic)



"""
 If there is only one metric in the experiment – the objective – then evaluation function can return a single tuple of mean and SEM,
 in which case Ax will assume that evaluation corresponds to the objective.
 It can also return only the mean as a float, in which case Ax will treat SEM as unknown and use a model that can infer it. For more details on evaluation function, refer to the "Trial Evaluation" section in the docs.
"""
# 2. Run optimization
"""
    Parameters:
    From ax.service.utils.instantiation.make_experiment(..) @ https://ax.dev/api/service.html
     "
     parameters – List of dictionaries representing parameters in the experiment search space.
     Required elements in the dictionaries are:
     1. “name” (name of parameter, string),
     2. “type” (type of parameter: “range”, “fixed”, or “choice”, string),
     and
     3. one of the following:
        3a. “bounds” for range parameters (list of two values, lower bound first),
        3b. “values” for choice parameters (list of values), or
        3c. “value” for fixed parameters (single value).

    Optional elements are:
     1. “log_scale” (for float-valued range parameters, bool),
     2. “value_type” (to specify type that values of this parameter should take; expects “float”, “int”, “bool” or “str”),
     3. “is_fidelity” (bool) and “target_value” (float) for fidelity parameters,
     4. “is_ordered” (bool) for choice parameters,
     5. “is_task” (bool) for task parameters, and
     6. “digits” (int) for float-valued range parameters.

"""
if use_float_values:
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 10.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 10.0],
        },
    ]
else:

    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        },
    ]


"""
    parameter_constraints –
    List of string representation of parameter constraints,
    such as “x3 >= x4” or “-x3 + 2*x4 - 3.5*x5 >= 2”.
    For the latter constraints, any number of arguments is accepted,
    and acceptable operators are “<=” and “>=”.
"""
#parameter_constraints=["x1 + x2 >= 1.5"]
parameter_constraints=["x2 > x1", "x3 > x2", "x4 > x3", "x5 > x4", "x6 > x5"]
"""
    outcome_constraints –
    List of string representation of outcome constraints of form
    “metric_name >= bound”, like “m1 <= 3.”
"""
#outcome_constraints=["l2norm <= 1.25"]
outcome_constraints=[] #"l1norm <= 1.25"]
"""
    The setup for the loop is fully compatible with JSON.
    According to documentation and original tutorial  (@ https://ax.dev/tutorials/gpei_hartmann_loop.html):
    "The optimization algorithm is selected based on the properties of the problem search space."
    (
        For details: see ax.modelbridge.dispatch_utils.choose_generation_strategy(..)
        @ https://ax.dev/api/modelbridge.html?highlight=choose_generation_strategy#ax.modelbridge.dispatch_utils.choose_generation_strategy)
    )
    This creates a OptimizationLoop object, and returns the outcome as-is.
    (The source code is quite charming, as it is evident this project is still in its infancy - there are magic numbers, and even a "ToDo[Lena]" (she is in charge of making it run async))
"""

if optimize_sum_of_violations:
    cfun = 'sumofviolations'
else:
    cfun = "l2norm"



best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    experiment_name="test",
    #objective_name="hartmann6",
    #objective_name="l2norm",
    objective_name=cfun, #'sumofviolations', #"l2norm",
    evaluation_function=own_maxsat_alldiff, #own_eval_function,
    minimize=True,  # Optional, defaults to False.
    parameter_constraints=["x1 + x2 >= 1"],#parameter_constraints,  # Optional.
    outcome_constraints=outcome_constraints,  # Optional.
    total_trials=40, # Optional.
)

"""
    The output of "optimize(..)":
        Optimal parameters in Dict, stored as string key-value pairs
        Values is a (means,covariance) tuple-of-dictionaries, one dictionary per metric - and each such dictionary returning covariance of results (?)
        An experiment object (class: ax.core.experiment.Experiment), holding
        An model object (class: ax.modelbridge.torch.TorchModelBridge), holding


"""
print("values")
print(values)
print(type(values))

print("experiment")
print(type(experiment))

print("model")
print(type(model))



# And we can introspect optimization results:

# In[4]:

print("best_parameters, ", type(best_parameters))
print(best_parameters)


means, covariances = values
print("means, ", type(means))
print(means)

print("covariances, ", type(covariances))
print(covariances)

#print("For comparison, minimum of Hartmann6 is: ", hartmann6.fmin)

"""
    Plot results
    Arbitrarily select "x1" and "x2" to plot for both metrics.
"""
print("Plot w l2norm metric ")
render(plot_contour(model=model, param_x='x1', param_y='x2', metric_name='l2norm'))

print("Plot w l2norm metric ")
render(plot_contour(model=model, param_x='x1', param_y='x2', metric_name='sumofviolations'))


print("Plot w l1norm ")
render(plot_contour(model=model, param_x='x1', param_y='x2', metric_name='l1norm'))

"""
    Plot evolution cost function value:
    `plot_single_method` expects a 2d array of means, because it expects to average means from multiple
    optimization runs, so we wrap out best objectives array in another array.

"""

best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
print("best_objectives")
print(best_objectives)
print(type(best_objectives))
print(np.shape(best_objectives))

best_objective_plot = optimization_trace_single_method(
    #y=np.minimum.accumulate(best_objectives, axis=1), #keeps returning minimum of all solutions thus far
    y=best_objectives, #returns the best cost of the actual optimizations
    optimum=0,#hartmann6.fmin,
    title="Model performance vs. # of iterations",
    ylabel="Hartmann6",
)

print("best_objective_plot")
render(best_objective_plot)
print(type(best_objective_plot))
