from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO
import numpy as np
import math
import matplotlib.pyplot as plt
import CUATRO.functions as f

from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g1
from CUATRO.test_functions.constraints.rosenbrock_constrained import rosenbrock_g2


def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


def sim(x):
    f1 = Rosenbrock
    g1 = rosenbrock_g1
    g2 = rosenbrock_g2
    
    #return f1(x), [g1(x), g2(x)]
    return f1(x), [g1(x)]


# need to specify x0 if working without prior evaluations
# input to optimise() method
x0 = np.array([-1.5,-1.5])

# custom parameter inputs at initialization of CUATRO instance
max_it = 200
N_min_s = 15
init_radius = 0.5
constr_handl = 'Discrimination'
sampl = 'g'
expl = None

# inputs to optimise() method
bounds = np.array([[-5,5],[-5,5]])
max_f_eval = 200

# CUATRO instance initialization and optimization
CUATRO_inst = CUATRO(x0=x0, max_iter=max_it, N_min_samples=N_min_s, \
     init_radius=init_radius, explore=expl, sampling=sampl, solver_to_use='SCS')

results = CUATRO_inst.run_optimiser(sim=sim, bounds=bounds, max_f_eval=max_f_eval)


# demonstrative plot
# TODO: add title etc.
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

ctr = ax.contour(X, Y, Z, cmap='summer', linewidths = 1)
ax.clabel(ctr, ctr.levels, inline=True, fontsize=8)
for Z_ in constr:
    ax.contour(X, Y, Z_, 0)

steps = results['steps']
radii = results['TR']

samples = results['x_store']
for i in range(len(samples)):
    ax.plot(*samples[i], 'o', color = "#2020f0", markersize  = 2, \
            markerfacecolor = 'none', markeredgewidth = 0.1)

min_point_scale = 0.15
init_point_scale = 1
point_scale = init_point_scale
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
    ax.plot(*f.circle(steps[i], radii[i], 40), 'k--', linewidth = 0.05)
    
        
ax.plot(*steps[-1], 'x', color = "#ff0000", markersize = 4 * point_scale, \
            markeredgewidth = 1 * point_scale)

ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])

plt.tight_layout()
plt.show()

# # pdf_path = 'plot.pdf'
# # fig.savefig(pdf_path)
# # subprocess.Popen([pdf_path], shell=True)