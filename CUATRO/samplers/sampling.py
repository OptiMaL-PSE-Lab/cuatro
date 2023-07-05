import cvxpy as cp
import numpy as np
import scipy.linalg as LA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import random
from scipy.stats import qmc
from statistics import median
from CUATRO.utilities import sample_simulation

def LHS(bounds, N, rnd_seed = 1):
    np.random.seed(rnd_seed)
    matrix = np.zeros((len(bounds), N))
    for i in range(len(bounds)):
        l, u = bounds[i]
        rnd_ind = np.arange(N)
        np.random.shuffle(rnd_ind)
        # print(rnd_ind)
        rnd_array = l + (np.random.rand(N)+ rnd_ind)*(u-l)/N
        matrix[i] = rnd_array
    return matrix

def sample_LHS(sim, bounds, N, rnd_seed = 1):
    data_points = LHS(bounds, N, rnd_seed = rnd_seed).T
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), sim)
    
    return data_points, func_eval, g_eval, feas

a = 0.1
def sample_points_opt(center, radius, f, samples, bounds, N = 10):
    # used for:
    # base, sampling_region, exploit_explore
    new_samples = np.array([center])
    
    while len(new_samples) < N + 1:
            
        def closest_distance_edge(x):  
            nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples])
            distance_to_edge = radius - np.linalg.norm(x-center)
            nearest_distance = min(nearest_distance_to_sample, distance_to_edge/a)
            return -nearest_distance
        def bounds_const(x): 
            return min([min(x[i] - bounds[i,0] , bounds[i,1] - x[i]) \
                        for i in range(len(center))])
        def trust_const(x): 
            return radius - np.linalg.norm(x-center)
        
        initial_points = sobol(center, radius, d = len(center))
        closest_distances = [closest_distance_edge(p) for p in initial_points]
        ind_best_distance = np.argmin(closest_distances)
        s_0 = initial_points[ind_best_distance]
        
        cons = [{'type': 'ineq', 'fun': bounds_const},
                {'type': 'ineq', 'fun': trust_const}]
                
        s_i = minimize(closest_distance_edge, s_0, constraints = cons).x
        new_samples = np.vstack((new_samples,s_i))
        samples = np.vstack((samples,s_i))
    
    new_samples = new_samples[1:] # Removing the center from samples
    func_eval, g_eval, feas = sample_simulation(new_samples.tolist(), f)
    
    return new_samples, func_eval, g_eval, feas

def sample_points_feas_samp(center, radius, f, samples, bounds, ineq_coeff, N = 10):
    
    new_samples = np.array([center])
    
    while len(new_samples) < N + 1:
        
        def closest_distance_edge(x):  
            nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples])
            distance_to_edge = radius - np.linalg.norm(x-center)
            nearest_distance = min(nearest_distance_to_sample, distance_to_edge/a)
            return -nearest_distance
        def bounds_const(x): 
            return min([min(x[i] - bounds[i,0] , bounds[i,1] - x[i]) \
                        for i in range(len(center))])
        def trust_const(x): 
            return radius - np.linalg.norm(x-center)
        
        "Added constraint for new samples, such that they must also satisfy the "\
        "problem's quadratic approximation of all the inequality constraints"
        
        def ineq_const(x):
            for coeff in ineq_coeff:
                P_ineq, q_ineq, r_ineq = coeff
                if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                    constr = -((x.T @ P_ineq @ x) + q_ineq.T @ x + r_ineq)
                else:
                    constr = 1
                return constr
                
        initial_points = sobol(center, radius, d = len(center))
        closest_distances = [closest_distance_edge(p) for p in initial_points]
        ind_best_distance = np.argmin(closest_distances)
        s_0 = initial_points[ind_best_distance]
        
        cons = [{'type': 'ineq', 'fun': bounds_const},
                {'type': 'ineq', 'fun': trust_const},
                {'type': 'ineq', 'fun': ineq_const}]
                
        s_i = minimize(closest_distance_edge, s_0, constraints = cons).x
        new_samples = np.vstack((new_samples,s_i))
        samples = np.vstack((samples,s_i))
    
    new_samples = new_samples[1:] # Removing the center from samples
    func_eval, g_eval, feas = sample_simulation(new_samples.tolist(), f)
    
    return new_samples, func_eval, g_eval, feas

def sample_points_TIS(center, radius, f, samples, bounds, ineq_coeff, N = 10, \
                      ineq = False):
    
    new_samples = np.array([center])
    
    for i in range(N):
        # Because we initialise new_samples with the center, we find new_samples
        # until the length is N+1 and then remove the centre
        while len(new_samples) < N + 1:
            
            def closest_distance_edge(x):  
                nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples])
                distance_to_edge = radius - np.linalg.norm(x-center)
                nearest_distance = min(nearest_distance_to_sample, distance_to_edge/a)
                return -nearest_distance
            def bounds_const(x): 
                return min([min(x[i] - bounds[i,0] , bounds[i,1] - x[i]) \
                            for i in range(len(center))])
            def trust_const(x): 
                return radius - np.linalg.norm(x-center)
            
            def ineq_const(x):
                for coeff in ineq_coeff:
                    P_ineq, q_ineq, r_ineq = coeff
                    if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                        constr = -((x.T @ P_ineq @ x) + q_ineq.T @ x + r_ineq)
                    else:
                        constr = 1
                    return constr
            
            initial_points = sobol(center, radius, d = len(center))
            closest_distances = [closest_distance_edge(p) for p in initial_points]
            ind_best_distance = np.argmin(closest_distances)
            s_0 = initial_points[ind_best_distance]
            
            if ineq:
                cons = [{'type': 'ineq', 'fun': bounds_const},
                        {'type': 'ineq', 'fun': trust_const},
                        {'type': 'ineq', 'fun': ineq_const}]                
            else:
                cons = [{'type': 'ineq', 'fun': bounds_const},
                        {'type': 'ineq', 'fun': trust_const}]
                    
            s_i = minimize(closest_distance_edge, s_0, constraints = cons).x
            new_samples = np.vstack((new_samples,s_i))
            samples = np.vstack((samples,s_i))
    
    new_samples = new_samples[1:] # Removing the center from samples
    func_eval, g_eval, feas = sample_simulation(new_samples.tolist(), f)
    
    return new_samples, func_eval, g_eval, feas

def sample_points_TIP(center, radius, f, samples, ineq_list, bounds, N = 10, \
                      ineq_active = False):
    
    # Because we initialise new_samples with the center, we find new_samples
    # until the length is N+1 and then remove the centre
    new_samples = np.array([center])
    
    while len(new_samples) < N + 1:
        
        def closest_distance_edge(x):  
            nearest_distance_to_sample = min([np.linalg.norm(x-s) for s in samples])
            distance_to_edge = radius - np.linalg.norm(x-center)
            nearest_distance = min(nearest_distance_to_sample, distance_to_edge/a)
            return -nearest_distance
        def bounds_const(x): 
            return min([min(x[i] - bounds[i,0] , bounds[i,1] - x[i]) \
                        for i in range(len(center))])
        def ineq_const(x):
            if ineq_list == None or type(ineq_list) == float:
                return 1
            P_iq = ineq_list[0]
            q_iq = ineq_list[1]
            r_iq = ineq_list[2]
            X = np.array(x).reshape(-1,1)
            return -np.squeeze((X.T @ P_iq @ X + q_iq.T @ X + r_iq))
        def trust_const(x): 
            return radius - np.linalg.norm(x-center)
        
        initial_points = sobol(center, radius, d = len(center))
        closest_distances = [closest_distance_edge(p) for p in initial_points]
        ind_best_distance = np.argmin(closest_distances)
        s_0 = initial_points[ind_best_distance]
        
        if ineq_active:
            cons = [{'type': 'ineq', 'fun': bounds_const},
                    {'type': 'ineq', 'fun': trust_const},
                    {'type': 'ineq', 'fun': ineq_const}]
        else:
            cons = [{'type': 'ineq', 'fun': bounds_const},
                    {'type': 'ineq', 'fun': trust_const}]
                
        s_i = minimize(closest_distance_edge, s_0, constraints = cons).x
        new_samples = np.vstack((new_samples,s_i))
        samples = np.vstack((samples,s_i))
    
    new_samples = new_samples[1:] # Removing the center from samples
    func_eval, g_eval, feas = sample_simulation(new_samples.tolist(), f)
    
    return new_samples, func_eval, g_eval, feas

def sobol(center, radius, d = 2, m = 5):
        
    sampler = qmc.Sobol(d, scramble = True)
    data_points = np.array(sampler.random_base2(m))

    l_bounds = [center[i] - radius for i in range(d)]
    u_bounds = [center[i] + radius for i in range(d)]
    
    data_points = qmc.scale(data_points, l_bounds, u_bounds)
    return data_points

def sample_points(center, radius, f, bounds, N = 10):
    
    if bounds is None:
        data_points = np.array(center*N).reshape(N, len(center)) + \
                      np.random.uniform(-radius, radius, (N, len(center)))
    else:
        uniform_sampling = np.zeros((N, len(center)))
        for i in range(len(center)):
            lower_bound = - radius ; upper_bound = radius
            if center[i] - radius < bounds[i,0]:
                lower_bound = bounds[i,0] - center[i]
            if center[i] + radius > bounds[i,1]:
                upper_bound = bounds[i,1] - center[i]
            uniform_sampling[:,i] = np.random.uniform(lower_bound, upper_bound, N)
            
        data_points = np.array(center*N).reshape(N, len(center)) + \
                        uniform_sampling
    func_eval, g_eval, feas = sample_simulation(data_points.tolist(), f)
    
    return data_points, func_eval, g_eval, feas



