from turbo import Turbo1
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

from CUATRO.test_functions.high_dim.RB import RB

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Levy:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        
    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        w = 1 + (x - 1.0) / 4.0
        val = np.sin(np.pi * w[0]) ** 2 + \
            np.sum((w[1:self.dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:self.dim - 1] + 1) ** 2)) + \
            (w[self.dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[self.dim - 1])**2)
        return val
    
class RB_TURBO:
    def __init__(self, dim=10):
        self.dim = dim
        self.lb = -5 * np.ones(dim) # since bounds is (-5,5) for all dimensions
        self.ub = 5*np.ones(dim)
        self.counter = 0

    def __call__(self, x):
        RB_inst = RB(self.dim)
        self.counter += 1
        return RB_inst.rosenbrock_higher(x)

# f = Levy(10)
f = RB_TURBO(100)

turbo1 = Turbo1(
    f=f,  # Handle to objective function
    lb=f.lb,  # Numpy array specifying lower bounds
    ub=f.ub,  # Numpy array specifying upper bounds
    n_init=10,  # Number of initial bounds from an Latin hypercube design
    max_evals = 500,  # Maximum number of evaluations
    batch_size=10,  # How large batch size TuRBO uses
    verbose=True,  # Print information from each batch
    use_ard=True,  # Set to true if you want to use ARD for the GP kernel
    max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
    n_training_steps=50,  # Number of steps of ADAM to learn the hypers
    min_cuda=1024,  # Run on the CPU for small datasets
    device="cpu",  # "cpu" or "cuda"
    dtype="float64",  # float64 or float32
)

turbo1.optimize()

X = turbo1.X  # Evaluated points
fX = turbo1.fX  # Observed values
ind_best = np.argmin(fX)
f_best, x_best = fX[ind_best], X[ind_best, :]

print("Best value found:\n\tf(x) = %.3f\nObserved at:\n\tx = %s" % (f_best, np.around(x_best, 3)))

print(f.counter)

# fig = plt.figure(figsize=(7, 5))
# matplotlib.rcParams.update({'font.size': 16})
# plt.plot(fX, 'b.', ms=10)  # Plot all evaluated points as blue dots
# plt.plot(np.minimum.accumulate(fX), 'r', lw=3)  # Plot cumulative minimum as a red line
# plt.xlim([0, len(fX)])
# plt.ylim([0, 30])
# plt.title("10D Levy function")
# 
# plt.tight_layout()
# plt.show()

