# CUATRO
Convex qUAdratic Trust-Region Optimizer - As a quadratic model-based derivative-free optimization solver, CUATRO is similar to COBYQA and Py-BOBYQA, but with different routines to optimize black-box optimization problems that frequently arise within chemical engineering applications: explicit constraint handling, noisy evaluations, high-dimensional decision spaces, safe constraint satisfaction, and sample-efficient trust region exploration.

## Installation and dependencies

Using `conda env create --file environment.yml` will recreate an environment with the required dependencies.
Running `pip install -e .` from the project directory installs CUATRO as a package.

## Tests
Run `python -m unittest` to check if the dependencies are correctly installed

## Examples of use
The folder `CUATRO/demos` contains examples of how to use CUATRO. `CUATRO_PLS.py` for instance compares various high-dimensional DFO routines for CUATRO.

Let's walk through another example where we want to use CUATRO to solve a constrained Rosenbrock implementation:

Let's start by defining our black-box:

```
class RB:
    def __init__(self, ):
        self.bounds = np.array([(-2,2) for _ in range(2)])
        self.x0 = np.array([[-1., -1.], [-1., 1.]])
        
    def rosenbrock_(self, x):
        x = np.array(x).squeeze()
        assert x.ndim == 1
        return 100*(x[1]-x[0]**2)**2 + (x[0]-1)**2
    
    def fun_test(self, x):
        return self.rosenbrock_(x)
    
    def con_test(self, x):
        return [self.rosenbrock_g1(x), self.rosenbrock_g2(x)]
    
    def rosenbrock_g1(self, x):
        return (x[0] - 1)**3 - x[1] + 1

    def rosenbrock_g2(self, x):
        print(x)
        return x[0] + x[1] - 1.8
  ```

CUATRO only accepts simulations which take the vector of decision-variables as input, and output a tuple of the objective evaluation and a list of the constraint evaluations: 
```
def f(x)
  return obj(x), [g1(x), g2(x), ...]
```
the list should remain empty if the black-box contains evaluates no constraints.
