# CUATRO
As a quadratic model-based derivative-free optimization solver, [CUATRO](https://www.sciencedirect.com/science/article/pii/S0009250921007004) (short for Convex qUAdratic Trust-Region Optimizer) is similar to COBYQA and Py-BOBYQA, but with specialized routines for black-box optimization problems that frequently arise within chemical engineering applications: explicit constraint handling, noisy evaluations, high-dimensional decision spaces, safe constraint satisfaction, and sample-efficient trust region exploration.

## Installation

The latest release is available on PyPI via 
```pip install cuatro```

<!-- ## Tests
Run `python -m unittest` to check if the dependencies are correctly installed -->

## Examples of use
The folder `CUATRO/demos` contains examples of how to use CUATRO. `CUATRO_PLS.py` for instance compares various high-dimensional DFO routines for CUATRO.

Let's walk through another example where we want to use CUATRO to solve a constrained Rosenbrock implementation:

CUATRO only accepts simulations which take the vector of decision-variables as input, and output a tuple of the objective evaluation and a list of the constraint evaluations such that each constraint evaluation should be below 0: 
```
def f(x)
  return obj(x), [g1(x), g2(x)]
```

Let's start by defining our black-box:
```
def sim(x):
    g1 = lambda x: (x[0] - 1)**3 - x[1] + 1
    g2 = lambda x: x[0] + x[1] - 1.8
    f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return f(x), [g1(x), g2(x)]
```


the list should remain empty if the black-box evaluations are unconstrained. We then call the CUATRO optimizer as follows:

```
import numpy as np
from cuatro import CUATRO

x0 = np.array([-2., 2.])
bounds = np.array([(-5., 5.) for _ in range(len(x0))])
budget = 100

solver_instance = CUATRO(
                    init_radius = 0.1, # how much radius should the initial area cover 
                    beta_red = 0.001**(2/budget), # trust region radius reduction heuristic
                    rescale_radius=True, # scale radii to unit box
                    method = 'local',
                    N_min_samples = 6, # 
                    constr_handling = 'Discrimination', # or 'Regression'
                    sampling = 'base', # maximize closest distance in trust region exploration
                    explore = 'feasible_sampling', 
                    # reject exploration samples that are predicted to violate constraints
                )
 
res = solver_instance.run_optimiser(sim=sim, x0=x0, bounds=bounds, max_f_eval=budget, )
print(res['f_best_so_far'], res['x_best_so_far'])

```

We define the initial guess `x0` as a numpy array of size d, and the box bounds `bounds` as a list of d tuples containing the upper and lower bound on the decision variables.

The solver instance is then run and we print the best objective evaluation and decision variables found in the budget of 100 evaluations. Other interesting arguments include 'constr_violation', 'radius_list', 'f_eval_list', 'x_eval_list'.

The documentation of `CUATRO/optimizer/CUATRO_optimizer_use.py` contains more information on possible solver configurations.

### Citing

If this repository is used in published work, please cite as:

```
@article{VANDEBERG2022117135,
title = {Data-driven optimization for process systems engineering applications},
journal = {Chemical Engineering Science},
volume = {248},
pages = {117135},
year = {2022},
issn = {0009-2509},
doi = {https://doi.org/10.1016/j.ces.2021.117135},
url = {https://www.sciencedirect.com/science/article/pii/S0009250921007004},
author = {Damien {van de Berg} and Thomas Savage and Panagiotis Petsagkourakis and Dongda Zhang and Nilay Shah and Ehecatl Antonio {del Rio-Chanona}},
}
```



