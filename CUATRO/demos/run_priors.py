from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO
import numpy as np
import math
import subprocess
import matplotlib.pyplot as plt
import CUATRO.functions as f
from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g1
from benchmarking.benchmark_problems.constraints.rosenbrock_constrained import rosenbrock_g2
#import utils as ut


def Himmelblau(x):
    return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

def Easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - math.pi)**2 + (x[1] - math.pi)**2))

def Rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2


# funcs for TIS and TIP
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
    
    return f1(x), [g1(x), g2(x)]

    #for TIP
    #return f3(x), []

    #for TIS
    #return f6(X), []


bounds = np.array([[-5,5],[-5,5]])
x0 = np.array([-1.5,-0.5])
#x0 = np.array([-2,-2])

# for TIP
#x0 = ut.sobol(np.array([0,0]), 5, m=3)

max_f_eval = 100
max_it = 50
N_min_s = 10
init_radius = 0.5

# for expl_expl:
#tolerance = 1e-3

constr_handl = 'Discrimination'
sampl = 'base'


# CUATRO instance initialization and optimization #1 - to be used as prior evals in #2
CUATRO_inst_prior = CUATRO(x0=x0, custom_params={'max_iter': max_it, 'N_min_samples': N_min_s, 'init_radius': init_radius, 'explore': 'sampling_region', 'sampling': sampl})


results_prior = CUATRO_inst_prior.optimise(sim, bounds = bounds, \
                             max_f_eval = max_f_eval)

print("\nDone generating prior evals\n\nOptimising using prior evals...\n")

X_prior = results_prior['x_store'][:2]
f_prior = results_prior['f_store'][:2]
g_prior = results_prior['g_store'][:2]

# X_prior = results_prior['x_store']
# f_prior = results_prior['f_store']
# g_prior = results_prior['g_store']

#print(X_prior[0])
#print("\n")
#print(f_prior[0])
#print("\n")
#print(g_prior[0])

# CUATRO instance initialization and optimization #2 - use prior evals
CUATRO_inst = CUATRO(x0=x0, custom_params={'max_iter': max_it, 'N_min_samples': N_min_s, 'init_radius': init_radius, 'explore': 'sampling_region', 'sampling': sampl})


results = CUATRO_inst.optimise(sim, bounds = bounds, max_f_eval = max_f_eval, \
    prior_evals = {'X_samples_list' : X_prior, 'f_eval_list': f_prior, 'g_eval_list': g_prior,
    'bounds': [], 'x0_method': 'best eval'})

print("\nDone with optimisation\n")

# Plotting routine from 4th Yr's 'CUATRO_run.py' script
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

# ax.plot(*f.path(*steps), linestyle = '--', \
#         color = "#888888", linewidth = 0.5)

samples = results['x_store'][len(X_prior):] # only plotting the samples that were evaluated now
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
# ax.plot(*known_opt, 'ro', markersize = 4, linewidth = 0.2, markerfacecolor='none')

ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])

plt.tight_layout()
plt.show()

pdf_path = 'plot.pdf'
fig.savefig(pdf_path)
subprocess.Popen([pdf_path], shell=True)


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
'''

# TIP plots, 4th Yr routines
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

samples = results['x_store']
for i in range(len(samples)):
    ax.plot(*samples[i], 'o', color = "#2020f0", markersize  = 0.5, \
            markerfacecolor = 'none', markeredgewidth = 0.1)

for t in range(len(x0)):
    
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

pdf_path = 'plot.pdf'
fig.savefig(pdf_path)
subprocess.Popen([pdf_path], shell=True)


#%% Plot 2
# plotting best_f over iterations

fig, ax = plt.subplots()
fig.set_size_inches(6,6)

best_f = np.log(results['f_best_so_far'])
x = np.arange(0,len(best_f))
ax.step(x,best_f)
f_of_step = np.log(results['f_of_step'])
ax.step(x,f_of_step, color = 'red')
ax.set_xlabel('Iteration')
ax.set_ylabel('Best evaluation of f')
ax.legend(['best evaluation', 'evaluation of current step'])

plt.tight_layout()
plt.show()
'''