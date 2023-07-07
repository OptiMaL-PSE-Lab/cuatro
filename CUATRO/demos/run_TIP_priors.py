# NEED TO CHANGE HOW CUATRO IS CALLED

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO
import numpy as np
import math
import matplotlib.pyplot as plt
import CUATRO.functions as f
from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g1
from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g2
from scipy.stats import qmc

from scipy.optimize import minimize

#%% Problem solution

def sobol(center, radius, d = 2, m = 5):
        
    sampler = qmc.Sobol(d, scramble = True)
    data_points = np.array(sampler.random_base2(m))

    l_bounds = [center[i] - radius for i in range(d)]
    u_bounds = [center[i] + radius for i in range(d)]
    
    data_points = qmc.scale(data_points, l_bounds, u_bounds)
    return data_points

a = 1
def generate_init_points(center, radius, bounds = None, N = 10):
    
    # Because we initialise new_samples with the center, we find new_samples
    # until the length is N+1 and then remove the centre
    
    samples = np.array([center])
    
    while len(samples) < N:
        
        def closest_distance_edge(x):  
            nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples], default = math.inf)
            distance_to_edge = radius - np.linalg.norm(x-center)
            nearest_distance = min(nearest_distance_to_sample, distance_to_edge/a)
            return -nearest_distance
        def closest_distance(x):
            nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples])
            return -nearest_distance_to_sample
        def bounds_const(x): 
            if (bounds == None).all(): 
                return 1
            else: 
                return min([min(x[i] - bounds[i,0] , bounds[i,1] - x[i]) \
                        for i in range(len(center))])
        def trust_const(x): 
            return radius - np.linalg.norm(x-center)
        
        initial_choice = sobol(center, radius, m = 6)
        # ititial_choice = lattice(center, radius, (8, 8))
        # closest_distances = [closest_distance(p) for p in ititial_choice]
        closest_distances = [closest_distance_edge(p) for p in initial_choice]
        ind_best_distance = np.argmin(closest_distances)
        s_0 = initial_choice[ind_best_distance]
        
        cons = [{'type': 'ineq', 'fun': bounds_const},
                {'type': 'ineq', 'fun': trust_const}]
                
        s_i = minimize(closest_distance_edge, s_0, constraints = cons).x
        samples = np.vstack((samples,s_i))
    
    return samples

def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

def Easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - math.pi)**2 + (x[1] - math.pi)**2))

def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def sim(x):
    f1 = Rosenbrock
    f2 = Easom
    f3 = Himmelblau
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    
    # return f1(x), [g1(x), g2(x)]
    # return f1(x), [x[0]-0.5]
    # return f1(x), [(x[0]+2)**2 + (x[1]+2)**2 - 9]
    return f3(x), []
    # return x[0]*x[1], [-(x[0]**2+(0.5*x[1])**2 - 4), -(x[0]-0.3), -(x[1]-0.3)]
    # return x[1], [np.sin(2*math.pi*x[0])*np.exp(-0.1*x[0])-x[1]]
    # return x[0]**2 + x[1]**2, []
    
bounds = np.array([[-5,5],[-5,5]])
# x0 = f.random_bounded_sample(bounds)
# x0 = np.array([[ -1.000, 2.500],
#                 [ 1.000, 3.000]])
# x0 = np.array([[ 3.0, 3.5],
#                 [ 1.0, 4.0],
#                 [ 4.0, 2.0]])
# x0 = np.array([[3,3]])
# x0 = np.array([[0.5, 4.5]])
x0 = sobol(np.array([0,0]), 5, m=3)
# x0 = generate_init_points(np.array([0,0]), 10, bounds, 10)
# known_opt = np.array([1,1])

max_f_eval = 200
max_it = 100

# Please note, N_min_s will be distributed across all trajectories for each
# step, so a higher value is recommended
N_min_s = 10
init_radius = 2
method = 'Discrimination'

sampl = 'base'


#####################

# CUATRO instance initialization and optimization #1 - to be used as prior evals in #2
custom_params={'max_iter': max_it, 'N_min_samples': N_min_s, 'init_radius': init_radius, 'explore': 'TIP', 'sampling': sampl}
CUATRO_inst_prior = CUATRO(x0=x0, **custom_params)


results_prior = CUATRO_inst_prior.run_optimiser(sim, bounds = bounds, \
                             max_f_eval = max_f_eval)

print("\nDone generating prior evals\n\nOptimising using prior evals...\n")

no_x0 = 5

X_prior = results_prior['x_store'][:no_x0]
f_prior = results_prior['f_store'][:no_x0]
g_prior = results_prior['g_store'][:no_x0]

# X_prior = results_prior['x_store']
# f_prior = results_prior['f_store']
# g_prior = results_prior['g_store']

#print(X_prior[0])
#print("\n")
#print(f_prior[0])
#print("\n")
#print(g_prior[0])

# CUATRO instance initialization and optimization #2 - use prior evals
custom_params={'max_iter': max_it, 'N_min_samples': N_min_s, 'init_radius': init_radius, 'explore': 'TIP', 'sampling': sampl}
CUATRO_inst = CUATRO(x0=x0, **custom_params) # TODO: x0 is not used, another reason why x0 should be inside run_optimiser()


results = CUATRO_inst.run_optimiser(sim, bounds = bounds, max_f_eval = max_f_eval, \
    prior_evals = {'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
    'bounds': [], 'x0_method': 'best eval'})

print("\nDone with optimisation\n")

######################


#%% Plot 1

colors = ['black','brown','darkorange','darkkhaki','yellowgreen','green',\
          'mediumturquoise','dodgerblue','blue','blueviolet','magenta','crimson'] 

log_contour = True
scale_point_size = True

x = np.linspace(*bounds[0],500)
y = np.linspace(*bounds[1],500)
X, Y = np.meshgrid(x,y)
Z, constr = sim([X,Y])
if log_contour:
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i,j] = math.log10(Z[i,j])

fig, ax = plt.subplots()
fig.set_size_inches(10,10)

ctr = ax.contour(X, Y, Z, cmap='summer', linewidths = [1])
ax.clabel(ctr, ctr.levels, inline=True, fontsize=8)
for Z_ in constr:
    ax.contour(X, Y, Z_, 0, linewidths = [0.1])

rejected_steps = results['rejected_steps']

for i in range(len(rejected_steps)):
    ax.plot(*rejected_steps[i], 'rx', color = "#909090", markersize = 1,  markeredgewidth = 0.5)

samples = results['x_store'][len(X_prior):]
for i in range(len(samples)):
    ax.plot(*samples[i], 'o', color = "#2020f0", markersize  = 0.5, \
            markerfacecolor = 'none', markeredgewidth = 0.1)

for t in range(no_x0):
    
    steps = results['steps'][t]
    radii = results['radii'][t]
         
    min_point_scale = 0.15
    init_point_scale = 1
    point_scale = init_point_scale
    
    ax.annotate(t, np.array(steps[0]) + 0.1 * np.array([-1,1]), fontsize = 5, color = "red")
    
    for i in range(len(steps)):
        
        if scale_point_size and i != 0:
            step_size = np.linalg.norm(np.array(steps[i])-np.array(steps[i-1]))
            point_scale = max(min_point_scale, init_point_scale * step_size / init_radius)
            
        if i != 0:
            ax.plot(*f.path([steps[i],steps[i-1]]), linestyle = '--', \
                        color = "#888888", linewidth = 1 * point_scale)
        
        if i == 0 or steps[i] != steps[i-1]:
            ax.annotate(i, np.array(steps[i]) + 0.1 * point_scale * np.ones(2), fontsize = 5 * point_scale)
            ax.plot(*steps[i], 'x', color = "#333333", markersize = 4 * point_scale, \
                    markeredgewidth = 1 * point_scale)
        if i == 0: 
            color = 'black' ; linewidth = 2; linestyle = 'solid'
        else:
            color = colors[t]; linewidth = 0.05; linestyle = 'dashed'
        ax.plot(*f.circle(steps[i], radii[i], 40), '--', color = color, \
                linewidth = linewidth, linestyle = linestyle)
           
        # coeff = results['ineq_steps'][t][i]
        # if coeff != None and type(coeff) != float:
        #     print("Plotting inequality contour for step {}".format(i))
        #     P_iq = coeff[0]
        #     q_iq = coeff[1]
        #     r_iq = coeff[2]
        #     Z = np.ones(X.shape)
        #     color = colors[i]
        #     # color = 'red'
        #     for i in range(X.shape[0]):
        #         for j in range(X.shape[1]):
        #             x = np.array([X[i][j],Y[i][j]]).reshape(-1,1)
        #             Z[i][j] = x.T @ P_iq @ x + q_iq.T @ x + r_iq
        #     ctr = ax.contour(X, Y, Z, 0, colors = [color] , linewidths = [0.05])
        
    ax.plot(*steps[-1], 'x', color = "#ff0000", markersize = 4 * point_scale, \
        markeredgewidth = 1 * point_scale)

ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])

plt.tight_layout()
plt.show()

path = 'CUATRO/demos/figures/TIP_priors_plot.jpg'
fig.savefig(path)


# #%% Plot 2
# # plotting best_f over iterations

# fig, ax = plt.subplots()
# fig.set_size_inches(6,6)

# best_f = np.log(results['f_best_so_far'])
# x = np.arange(0,len(best_f))
# ax.step(x,best_f)
# f_of_step = np.log(results['f_of_step'])
# ax.step(x,f_of_step, color = 'red')
# ax.set_xlabel('Iteration')
# ax.set_ylabel('Best evaluation of f')
# ax.legend(['best evaluation', 'evaluation of current step'])

# plt.tight_layout()
# plt.show()

