#!/usr/bin/env python
"""
    This bug report is due shows that parameter values cannot be constrained if being choice parameters, despite being int
"""


import numpy as np
from ax.service.managed_loop import optimize

def dist(parameterization):
    x = np.array([parameterization.get(f"x{i+1}") for i in range(2)])
    return {"l2norm": (np.sqrt(((x) ** 2).sum()), 0.0)}

parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",  # Optional, defaults to inference from type of "bounds".
        }
    ]

"""
 True,True raises error, while all other combinations are ok
 Hence, giving parameter constraint on choice variables does not work
"""
use_choice_value = True
use_parameter_constraint = True


if use_choice_value:
    second_par = {
        "name": "x2",
        "type": "choice",
        "bounds": [0, 10],
        "values": [0,1,2,3,4,5,6,7,8,9,10],
        "value_type": "int"
    }
else:
    second_par = {
            "name": "x2",
            "type": "range",
            "bounds": [0, 10],
            "value_type": "int",
    }

parameters.append(second_par)


if use_parameter_constraint:
    parameter_constraints=["x2 - x1 >= 1"]
else:
    parameter_constraints=[]

best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    objective_name="l2norm",
    evaluation_function=dist,
    minimize=True,
    parameter_constraints=parameter_constraints,
    total_trials=13,
)

print("\nObjectives")
best_objectives = [trial.objective_mean for trial in experiment.trials.values()]
print(best_objectives)

print("\nArgs:")
pars = [key for key in model.model_space.parameters.keys()]
for par_name in pars:
    print(par_name, " values: ", [trial.generator_runs[0].arms[0].parameters[par_name] for trial in experiment.trials.values()])
