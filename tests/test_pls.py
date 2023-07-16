import unittest
from functools import partial

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

from scipy.optimize import minimize
import numpy as np
from scipy.stats import ortho_group

from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g1
from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g2

def quadratic_f(x):
    '''
    test objective 
    '''
    return (x[0]-1)**2 + 10 * (x[1]-1)**2 + (x[0]-1) * (x[1]-1)

def quadratic_higher(Q, x):
    # Q is used to 'mix up' the fake and effective dimensions
    x = np.array(x).squeeze()
    assert x.ndim == 1
    x = x.reshape(-1,1)

    new_x = (Q @ x).squeeze()
    return quadratic_f(new_x)

def sim_quadr(high_dim, x, seed=0):
    np.random.seed(seed)
    Q = ortho_group.rvs(high_dim)
    f1 = partial(quadratic_higher, Q)

    return f1(x), []

def rosenbrock_(x, seed=0):
    x = np.array(x).squeeze()
    assert x.ndim == 1
    return 100*(x[1]-x[0]**2)**2 + (x[0]-1)**2

def rosenbrock_higher(Q, x):
    # Q is used to 'mix up' the fake and effective dimensions
    x = np.array(x).squeeze()
    assert x.ndim == 1
    x = x.reshape(-1,1)

    new_x = (Q @ x).squeeze()
    return rosenbrock_(new_x)

def sim(high_dim, x, seed=0):
    np.random.seed(seed)
    Q = ortho_group.rvs(high_dim)
    f1 = partial(rosenbrock_higher, Q)
    
    return f1(x), []

# N=1000
N = 500
N_high=100
N_pls=2

optimum = 0
conv_rad = 0.01

# bounds = np.array([(-5.0, 5.0) for i in range(N_high)])
bounds = np.array([(-5.0, 5.0) for i in range(N_high)])
x0 = np.array([(b[0] + b[1])/2 for b in bounds]) 
init_radius = np.max([(b[1] - b[0])/5 for b in bounds]) ## change back
beta = 1e-3**(1/N)
N_min_s = N/10
tol= 1e-8

CUATRO_inst = CUATRO(sampling='g', explore=None, method='local', N_min_samples=6, beta_red=beta, tolerance=tol, init_radius=init_radius, dim_red=True)
CUATRO_bench = CUATRO(sampling='g', explore=None, method='local', N_min_samples=N_min_s, beta_red=beta, tolerance=tol, init_radius=init_radius)


f_RB = partial(sim, N_high)
f_quadr = partial(sim_quadr, N_high)

class TestConvergence_PLS_CUATRO(unittest.TestCase):
    def runTest(self):
        """
        check whether CUATRO_PLS outperforms CUATRO_g in high dimensions
        """  
        res = CUATRO_inst.run_optimiser(sim=f_RB, x0=x0, bounds=bounds, max_f_eval=N, rnd=0, n_pls=N_pls)
        res_bench = CUATRO_bench.run_optimiser(sim=f_RB, x0=x0, bounds=bounds, max_f_eval=N, rnd=0)
        print(f"Best objective found: {res['f_best_so_far'][-1]} compared to baseline CUATRO {res_bench['f_best_so_far'][-1]}")
        self.assertTrue((res['f_best_so_far'][-1]  < res_bench['f_best_so_far'][-1]), f"CUATRO_PLS did not outperform CUATRO_g at high dimension: {res['f_best_so_far'][-1]} > {res_bench['f_best_so_far'][-1]}")
    

class TestConvergence_PLS_quadr(unittest.TestCase):   
    def runTest(self):
        """
        check whether convergence is achieved for the quadratic test function in high dimensions
        """  
        res = CUATRO_inst.run_optimiser(sim=f_quadr, x0=x0, bounds=bounds, max_f_eval=N, rnd=0, n_pls=N_pls)
        print(f"Objective: {res['f_best_so_far'][-1]}")
        self.assertTrue((abs(res['f_best_so_far'][-1] - optimum)**2 < conv_rad), f"CUATRO_g did not converge to {conv_rad} accuracy for a budget of {N}; best_f: {res['f_best_so_far'][-1]}, optimum: {optimum}")
        
        

