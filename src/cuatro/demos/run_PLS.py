
import numpy as np
import math
import matplotlib.pyplot as plt

from time import perf_counter

import cuatro.functions as f
from cuatro.test_functions.high_dim.RB import RB
from cuatro.optimizer.CUATRO_optimizer_use import CUATRO
#import utils as ut

n_e = 2 # effective dimensionality
dims = [3, 5, 10, 25, 50, 100, 250, 500, 1000, 2500]
budget = 500

N_min_PLS, N_min_g = int((n_e+1)*(n_e+2)/2), 10
init_radius = 1
beta = 1e-3**(2/budget)
tol=1e-8

fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.set_size_inches(10,10)

perf_res = {
    'base': [],
    'PLS': [],
    'PLS_expl': [],
    'PLS_bandit': [],
    'embed': [],
}

time_res = {
    'base': [],
    'PLS': [],
    'PLS_expl': [],
    'PLS_bandit': [],
    'embed': [],
}

for dim in dims:
    
    RB_inst = RB(dim)
    def obj(x):
        return RB_inst.rosenbrock_higher(x), []

    bounds = np.array([(-5,5) for _ in range(dim)])
    x0 = np.array([0 for _ in range(dim)])

    CUATRO_PLS = CUATRO(sampling='g', explore=None, method='local', N_min_samples=N_min_PLS, beta_red=beta, tolerance=tol, init_radius=init_radius, dim_red='PLS')
    CUATRO_PLS_expl = CUATRO(sampling='base', explore=None, method='local', N_min_samples=N_min_PLS, beta_red=beta, tolerance=tol, init_radius=init_radius, dim_red='explore')
    CUATRO_embed = CUATRO(sampling='g', explore=None, method='local', N_min_samples=N_min_PLS, beta_red=beta, tolerance=tol, init_radius=init_radius, dim_red='embed')
    CUATRO_PLS_bandit = CUATRO(sampling='base', explore=None, method='local', N_min_samples=N_min_PLS, beta_red=beta, tolerance=tol, init_radius=init_radius, dim_red='bandit')
    CUATRO_bench = CUATRO(sampling='g', explore=None, method='local', N_min_samples=N_min_g, beta_red=beta, tolerance=tol, init_radius=init_radius)

    t0 = perf_counter()
    res_PLS = CUATRO_PLS.run_optimiser(sim=obj, x0=x0, bounds=bounds, max_f_eval=budget, rnd=0, n_pls=n_e)
    t1=perf_counter()

    perf_res['PLS'] += [res_PLS['f_best_so_far'][-1]]
    time_res['PLS'] += [t1-t0]

    t0 = perf_counter()
    res_PLS_expl = CUATRO_PLS_expl.run_optimiser(sim=obj, x0=x0, bounds=bounds, max_f_eval=budget, rnd=0, n_pls=n_e)
    t1 = perf_counter()

    perf_res['PLS_expl'] += [res_PLS_expl['f_best_so_far'][-1]]
    time_res['PLS_expl'] += [t1-t0]

    t0 = perf_counter()
    res_embed = CUATRO_embed.run_optimiser(sim=obj, x0=x0, bounds=bounds, max_f_eval=budget, rnd=0, n_pls=n_e)
    t1 = perf_counter()

    perf_res['embed'] += [res_embed['f_best_so_far'][-1]]
    time_res['embed'] += [t1-t0]

    if dim < 300:
        res_g = CUATRO_bench.run_optimiser(sim=obj, x0=x0, bounds=bounds, max_f_eval=budget, rnd=0)
        t2=perf_counter()

        perf_res['base'] += [res_g['f_best_so_far'][-1]]
        time_res['base'] += [t2-t1]

    if dim < 600:

        t0 = perf_counter()
        res_PLS_bandit = CUATRO_PLS_bandit.run_optimiser(sim=obj, x0=x0, bounds=bounds, max_f_eval=budget, rnd=0, n_pls=n_e)
        t1 = perf_counter()

        perf_res['PLS_bandit'] += [res_PLS_bandit['f_best_so_far'][-1]]
        time_res['PLS_bandit'] += [t1-t0]

    print(f'Done with dim {dim}')

ax1.plot(dims[:len(perf_res['base'])], perf_res['base'], label='base', c='blue')
ax1.plot(dims, perf_res['PLS'], c='orange', label='PLS')
ax1.plot(dims, perf_res['PLS_expl'], label='PLS_expl', c='green')
ax1.plot(dims, perf_res['embed'], label='embed', c='brown')
ax1.plot(dims[:len(perf_res['PLS_bandit'])], perf_res['PLS_bandit'], label='PLS_bandit', c='grey')
ax1.legend()
ax1.set_xscale('log')
ax1.set_xlabel('Original dimensionality')
ax1.set_ylabel('Best objective found (optimum: 0)')

ax2.plot(dims[:len(time_res['base'])], time_res['base'], c='blue')
ax2.plot(dims, time_res['PLS'], c='orange')
ax2.plot(dims, time_res['PLS_expl'], label='PLS_expl', c='green')
ax2.plot(dims, time_res['embed'], label='embed', c='brown')
ax2.plot(dims[:len(time_res['PLS_bandit'])], time_res['PLS_bandit'], label='PLS_bandit', c='grey')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Original dimensionality')
ax2.set_ylabel('Runtime in s')

plt.tight_layout()
plt.show()

path = 'CUATRO/demos/figures/PLS_plot.jpg'
fig.savefig(path)
