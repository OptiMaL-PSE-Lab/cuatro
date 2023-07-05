import unittest

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import CUATRO.functions as f

from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g1
from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g2


def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def sim(x):
    f1 = Rosenbrock
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    
    return f1(x), [g1(x)]


no_of_priors = 8

# will use these parameters for both the prior data generating and 'real' runs
bounds = np.array([[-5,5],[-5,5]])
x0 = np.array([-1.5,-0.5])
N_min_s = 20

prior_instance = CUATRO(x0=x0, N_min_samples=N_min_s)
results_prior = prior_instance.run_optimiser(sim, bounds=bounds)

X_prior = results_prior['x_store'][:no_of_priors]
f_prior = results_prior['f_store'][:no_of_priors]
g_prior = results_prior['g_store'][:no_of_priors]


class TestPriorDataInit(unittest.TestCase):
    def runTest(self):
        """
        check whether prior evaluation data was initialised correctly
        """
        solver_instance = CUATRO(x0=x0, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, bounds=bounds, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        # before the first iteration we should have the prior evals (no_of_priors) + center's eval in the f_eval_list
        self.assertEqual(results['samples_at_iteration'][0], no_of_priors+1, "some prior data was initialised incorrectly")
        self.assertEqual(results_prior['samples_at_iteration'][0], 1, "more than one sample (the center) was evaluated before the first iteration for the no prior data case")


class TestTolerancePriorData(unittest.TestCase):
    def runTest(self):
        """
        check whether tolerance is used successfully as a termination criterion when there's prior data available;
        set high value for tolerance, max_iter and max_f_eval
        """
        tol = 1e-03
        max_it = 1000
        max_eval = 1000

        solver_instance = CUATRO(x0=x0, tolerance=tol, max_iter=max_it, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, bounds=bounds, max_f_eval=max_eval, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        self.assertTrue((results['N_eval'] < max_eval), "number of function evaluations is over the max allowed")
        self.assertTrue((results['N_iter'] < max_it), "number of iterations is over the max allowed")
        self.assertTrue((results['TR'][-1] < tol), "radius was not smaller than tolerance")


class TestBudgetPriordata(unittest.TestCase):
    def runTest(self):
        """
        check whether evaluation budget is used successfully as a termination criterion when there's prior data available;
        set high value for max_iter, low values for tolerance and max_f_eval
        """
        tol = 1e-10
        max_it = 10000
        max_eval = 50

        solver_instance = CUATRO(x0=x0, tolerance=tol, max_iter=max_it, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, bounds=bounds, max_f_eval=max_eval, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        self.assertTrue(((results['N_eval'] == max_eval) or (results['N_eval'] == max_eval-1)), f"number of function evaluations was not equal to max allowed, it was {results['N_eval']}")
        self.assertTrue((results['N_iter'] < max_it), "number of iterations is over the max allowed")
        self.assertTrue((results['TR'][-1] > tol), "radius was not bigger than tolerance")


class TestIterationsPriordata(unittest.TestCase):
    def runTest(self):
        """
        check whether number of iterations is used successfully as a termination criterion when there's prior data available;
        set high value for max_f_eval, low values for tolerance and max_iter
        """
        tol = 1e-10
        max_it = 100
        max_eval = 10000

        solver_instance = CUATRO(x0=x0, tolerance=tol, max_iter=max_it, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, bounds=bounds, max_f_eval=max_eval, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        self.assertTrue((results['N_eval'] < max_eval), "number of function evaluations was not smaller than max allowed")
        self.assertTrue((results['N_iter'] == max_it+1), "number of iterations was not one greater than max allowed")
        self.assertTrue((results['TR'][-1] > tol), "radius was not bigger than tolerance")
        

if __name__ == '__main__':
    unittest.main()