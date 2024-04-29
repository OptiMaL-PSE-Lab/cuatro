import numpy as np
from cuatro import CUATRO

def sim(x):
    g1 = lambda x: (x[0] - 1)**3 - x[1] + 1
    g2 = lambda x: x[0] + x[1] - 1.8
    f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2
    return f(x), [g1(x), g2(x)]

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
