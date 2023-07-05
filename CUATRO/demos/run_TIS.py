from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import CUATRO.functions as f
from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g1
from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g2

#%% Problem solution

def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

def Easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - math.pi)**2 + (x[1] - math.pi)**2))

def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def Rastrigin(x):
    return 20 + (x[0]**2) + (x[1]**2) - (10*np.cos(2*math.pi*x[0])) - \
        (10*np.cos(2*math.pi*x[1]))

def Ackley(x):
    return -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2 + x[1]**2))) - \
        np.exp(0.5*(np.cos(2*math.pi*x[0]) + np.cos(2*math.pi*x[1]))) + \
            np.exp(1) + 20
            
def Eggholder(x):
    return (-x[1]+47)*np.sin(np.sqrt(abs((x[0]/2) + x[1] + 47))) - \
                             x[0]*np.sin(np.sqrt(abs(x[0]-x[1]-47)))

def Hölder(x):
    return

def sim(x):
    f1 = Rosenbrock
    f2 = Easom
    f3 = Himmelblau
    f4 = Rastrigin
    f5 = Ackley
    f6 = Eggholder
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    
    # return f1(x), [g1(x), g2(x)]
    # return f1(x), []
    # return f2(x), []
    # return f1(x), [x[0]-0.5]
    # return f1(x), [(x[0]+2)**2 + (x[1]+2)**2 - 9]
    # return f3(x), []
    # return f4(x), []
    # return f5(x), []
    return f6(x), []
    # return x[0]*x[1], [-(x[0]**2+(0.5*x[1])**2 - 4), -(x[0]-0.3), -(x[1]-0.3)]
    # return x[1], [np.sin(2*math.pi*x[0])*np.exp(-0.1*x[0])-x[1]]
    # return x[0]**2 + x[1]**2, []
    
bounds = np.array([[-500,500],[-500,500]])
x0 = np.array([0.75, 1])



max_f_eval = 200
max_it = 50

N_min_s = 15
init_radius = 0.1*sum(bounds[i][1]-bounds[i][0] \
                    for i in range(len(bounds)))/len(bounds)
min_radius = 0.05*init_radius
min_restart_radius = 2*init_radius
conv_radius = 0.25*init_radius
method = 'Discrimination'

sampl = 'base'


# # CUATRO instance initialization and optimization
# CUATRO_inst = CUATRO(x0=x0, custom_params={'max_iter': max_it, 'N_min_samples': N_min_s, 'init_radius': init_radius,
# 'explore': 'TIS', 'sampling': sampl, 'min_radius': min_radius, 'min_restart_radius': min_restart_radius,
#     'conv_radius': conv_radius})


# results = CUATRO_inst.optimise(sim, bounds = bounds, \
#                              max_f_eval = max_f_eval)

# CUATRO instance initialization and optimization
CUATRO_inst = CUATRO(x0=x0, max_iter=max_it, N_min_samples=N_min_s, \
     init_radius=init_radius, explore='TIS', sampling=sampl, solver_to_use='SCS', min_radius=min_radius, \
         min_restart_radius=min_restart_radius, conv_radius=conv_radius)

results = CUATRO_inst.run_optimiser(sim=sim, bounds=bounds, max_f_eval=max_f_eval)


print('\n')
#print(results['status'])
print('{} black box evaluations carried out'.format(len(results['f_store'])))

#%% Plot 1

log_contour = False
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

ctr = ax.contour(X, Y, Z, cmap='summer', linewidths = 1)
ax.clabel(ctr, ctr.levels, inline=True, fontsize=8)
for Z_ in constr:
    ax.contour(X, Y, Z_, 0)

steps = results['steps']
radii = results['TR']

# ax.plot(*f.path(*steps), linestyle = '--', \
#         color = "#888888", linewidth = 0.5)

samples = results['x_store']
for i in range(len(samples)):
    ax.plot(*samples[i], 'o', color = "#2020f0", markersize  = 2, \
            markerfacecolor = 'none', markeredgewidth = 0.1)

min_point_scale = 0.15
init_point_scale = 1
point_scale = init_point_scale
colour_no = 0
colour_store = ["k","#E50000","c","#00FFFF"]
colour = colour_store[colour_no]
for i in range(len(steps)):
    
    if i>0 and radii[i]/radii[i-1] > 1.2 and colour_no<len(colour_store)-1:
        colour_no += 1
        colour = colour_store[colour_no]
    
    if scale_point_size and i != 0:
        step_size = np.linalg.norm(np.array(steps[i])-np.array(steps[i-1]))
        point_scale = max(min_point_scale, init_point_scale * step_size / init_radius)
        
    if i != 0:
        ax.plot(*f.path([steps[i],steps[i-1]]), linestyle = '--', \
                    color = "#888888", linewidth = 1 * point_scale)
    
    if i == 0 or (steps[i] != steps[i-1]):
        ax.annotate(i, np.array(steps[i]) + 0.1 * point_scale * np.ones(2), fontsize = 5 * point_scale)
        ax.plot(*steps[i], 'x', color = "#333333", markersize = 4 * point_scale, \
                markeredgewidth = 1 * point_scale)
    ax.plot(*f.circle(steps[i], radii[i], 40), '--',color = colour, linewidth = 0.05)
    
        
ax.plot(*steps[-1], 'x', color = "#ff0000", markersize = 4 * point_scale, \
            markeredgewidth = 1 * point_scale)
# ax.plot(*known_opt, 'ro', markersize = 4, linewidth = 0.2, markerfacecolor='none')

ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])

plt.tight_layout()
plt.show()

# pdf_path = 'plot.pdf'
# fig.savefig(pdf_path)
# subprocess.Popen([pdf_path], shell=True)


#%% Plot 2
# plotting best_f over iterations

fig, ax = plt.subplots()
fig.set_size_inches(6,6)
if log_contour:
    best_f = np.log(results['f_best_so_far'])
else:
    best_f = np.array(results['f_best_so_far'])
x = np.arange(0,len(best_f))
ax.step(x,best_f)
if log_contour:
    f_of_step = np.log(results['f_of_step'])
else:
    f_of_step = np.array(results['f_of_step'])
ax.step(x,f_of_step, color = 'red')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best evaluation of f')
ax.legend(['best evaluation', 'evaluation of current step'])

plt.tight_layout()
plt.show()

