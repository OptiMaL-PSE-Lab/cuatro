import unittest

from cuatro import CUATRO
import numpy as np

from cuatro.test_functions.constraints.rosenbrock_constrained import rosenbrock_g1


def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def sim(x):
    f1 = Rosenbrock
    g1 = rosenbrock_g1
    
    return f1(x), [g1(x)]


no_of_priors = 8

# will use these parameters for both the prior data generating and 'real' runs
bounds = np.array([[-5,5],[-5,5]])
x0 = np.array([-1.5,-0.5])
N_min_s = 20

prior_instance = CUATRO(N_min_samples=N_min_s)
results_prior = prior_instance.run_optimiser(sim, x0=x0, bounds=bounds)

X_prior = results_prior['x_store'][:no_of_priors]
f_prior = results_prior['f_store'][:no_of_priors]
g_prior = results_prior['g_store'][:no_of_priors]


class TestPriorDataInit(unittest.TestCase):
    def runTest(self):
        """
        check whether prior evaluation data was initialised correctly
        """
        solver_instance = CUATRO(N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, x0=x0, bounds=bounds, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        # before the first iteration we should have the prior evals (no_of_priors) + center's eval in the f_eval_list
        self.assertEqual(results['samples_at_iteration'][0], no_of_priors+1, "some prior data was initialised incorrectly")
        self.assertEqual(results_prior['samples_at_iteration'][0], 1, "more than one sample (the center) was evaluated before the first iteration for the no prior data case")


class TestTolerancePriorData(unittest.TestCase):
    def runTest(self):
        """
        check whether tolerance is used successfully as a termination criterion when there's prior data available;
        set high value for tolerance, and max_f_eval
        """
        tol = 1e-03
        max_eval = 1000

        solver_instance = CUATRO(tolerance=tol, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, x0=x0, bounds=bounds, max_f_eval=max_eval, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        self.assertTrue((results['N_eval'] < max_eval), "number of function evaluations is over the max allowed")
        self.assertTrue((results['TR'][-1] < tol), "radius was not smaller than tolerance")


class TestBudgetPriordata(unittest.TestCase):
    def runTest(self):
        """
        check whether evaluation budget is used successfully as a termination criterion when there's prior data available;
        low values for tolerance and max_f_eval
        """
        tol = 1e-10
        max_eval = 50

        solver_instance = CUATRO(tolerance=tol, N_min_samples=N_min_s)
        results = solver_instance.run_optimiser(sim, x0=x0, bounds=bounds, max_f_eval=max_eval, \
            prior_evals={'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
        'bounds': [], 'x0_method': 'best eval'})

        self.assertTrue(((results['N_eval'] == max_eval) or (results['N_eval'] == max_eval-1)), f"number of function evaluations was not equal to max allowed, it was {results['N_eval']}")
        self.assertTrue((results['TR'][-1] > tol), "radius was not bigger than tolerance")


if __name__ == '__main__':
    unittest.main()