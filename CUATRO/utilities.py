# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:12:38 2021

@author: dv516
"""

import cvxpy as cp
import numpy as np
import scipy.linalg as LA
from statistics import median

def quadratic_LA(X, Y, P, q, r):
    N = len(X)
    Z = np.zeros(X.shape)
    for i in range(N):
        for j in range(N):
            X_ = np.array([X[i,j], Y[i,j]]).reshape(-1,1)
            Z[i,j] = float(X_.T @ P @ X_ + q.T @ X_ + r)
    return Z

def make_PSD(P):
    eig_val, eig_vec = LA.eigh(P)
    # print(eig_val)
    eig_val = np.array([max(val, 1e-8) for val in eig_val])
    # print(eig_val)
    P = np.dot(eig_vec, eig_val[:, np.newaxis]*eig_vec.T)
    return P


def update_best_lists(X_list, f_list, g_list, X_best, f_best, g_best):
    g_feas = constr_creation(X_list, g_list)
    f = np.array(f_list)
    ind = np.where(f == np.min(f[g_feas == 1]))
    X_best += np.array(X_list)[ind].tolist()[:1]
    f_best += f[ind].tolist()[:1]
    g_best += np.array(g_list)[ind].tolist()[:1]
    
    return X_best, f_best, g_best

def samples_in_trust(center, radius, \
                     X_samples_list, y_samples_list, g_list):
    X = np.array(X_samples_list) 
    y = np.array(y_samples_list) 
    g = np.array(g_list)
    ind = np.where(np.linalg.norm(X - np.array(center), axis = 1,\
                                  keepdims = True) < radius)[0]
    X_in_trust = X[ind] ; y_in_trust = y[ind] ; g_in_trust = g[ind]
    feas_in_trust = constr_creation(X_in_trust, g_in_trust.tolist())
    
    return X_in_trust, y_in_trust, g_in_trust, feas_in_trust

def quadratic_fitting(X_mat, y_mat, solver_to_use, discr = False):
    N, M = X_mat.shape[0], X_mat.shape[1]
    P = cp.Variable((M, M), PSD = True)
    q = cp.Variable((M, 1))
    r = cp.Variable()
    X = cp.Parameter(X_mat.shape)
    y = cp.Parameter(y_mat.shape)
    X.value = X_mat
    y.value = y_mat
    quadratic = cp.bmat([cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r - y.value[i] for i in range(N)])
    obj = cp.Minimize(cp.norm(quadratic))
    if not discr:
        prob = cp.Problem(obj)
    else:
        const_P = [P >> np.eye(M)*1e-9]
        prob = cp.Problem(obj, constraints = const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve(solver=solver_to_use)
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX objective fitting call at: ')
        print('X matrix', X_mat)
        print('y array', y_mat)
        raise ValueError

def quadratic_discrimination(x_inside, y_outside, solver_to_use):
    N, M, D = x_inside.shape[0], y_outside.shape[0], x_inside.shape[1]
    u = cp.Variable(N, pos = True)
    v = cp.Variable(M, pos = True)
    P = cp.Variable((D,D), PSD = True)
    q = cp.Variable((D, 1))
    r = cp.Variable()
    X = cp.Parameter(x_inside.shape, value = x_inside)
    Y = cp.Parameter(y_outside.shape)
    X.value = x_inside ; Y.value = y_outside
    const_u = [cp.quad_form(X.value[i].reshape(-1,1), P) + \
                        q.T @ X.value[i].reshape(-1,1) + r <= -(1 - u[i]) for i in range(N)]
    const_v = [cp.quad_form(Y.value[i].reshape(-1,1), P) + \
                        q.T @ Y.value[i].reshape(-1,1) + r >= (1 - v[i]) for i in range(M)]
    const_P = [P >> np.eye(D)*1e-9]
    prob = cp.Problem(cp.Minimize(cp.sum(u) + cp.sum(v)), \
                      constraints = const_u + const_v + const_P)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve(solver=solver_to_use, verbose=False)
    if prob.status not in ['unbounded', 'infeasible']:
        return P.value, q.value, r.value
    else:
        print(prob.status, ' CVX ineq. classification call at: ')
        print('x_inside', x_inside)
        print('x_outside', y_outside)
        raise ValueError

def quadratic_min(P_, q_, r_, center, radius, bounds, solver_to_use, ineq = None):
    X = cp.Variable((len(center), 1))
    # P = cp.Parameter(P_.shape, value = P_, PSD = True)
    try:
        P = cp.Parameter(P_.shape, value = P_, PSD = True)
    except:
        P_ = make_PSD(P_)
        if (P_ == 0).all():
            P_ = np.eye(len(P_))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        try:
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        except:
            P_ = np.eye(len(P))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
    q = cp.Parameter(q_.shape, value = q_)
    r = cp.Parameter(r_.shape, value = r_)
    objective = cp.Minimize(cp.quad_form(X, P) + q.T @ X + r)
    trust_center = np.array(center).reshape((P_.shape[0], 1))
    constraints = []
    if ineq != None:
        for coeff in ineq:
            P_ineq, q_ineq, r_ineq = coeff
            if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                P_iq = cp.Parameter(P_ineq.shape, value = P_ineq, PSD = True)
                q_iq = cp.Parameter(q_ineq.shape, value = q_ineq)
                r_iq = cp.Parameter(r_ineq.shape, value = r_ineq)
                constraints += [cp.norm(X - trust_center) <= radius,
                           cp.quad_form(X, P_iq) + q_iq.T @ X + r_iq <= 0]

    else:
        constraints = [cp.norm(X - trust_center) <= radius]
    if bounds is not None:
        constraints += [bounds[i,0] <=  X[i] for i in range(P_.shape[0])]
        constraints += [X[i] <= bounds[i,1] for i in range(P_.shape[0])]
    prob = cp.Problem(objective, constraints)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve(solver=solver_to_use, verbose=False)
    if prob.status not in ['unbounded', 'infeasible']:
        return X.value.reshape(P_.shape[0])
    else:
        print(prob.status, ' CVX min. call at: ')
        print('Center', center)
        print('Radius', radius)
        print('P_', P_)
        print('q_', q_)
        print('r_', r_)
        print('Ineq', ineq)
        raise ValueError
        
def quadratic_min_expl_expl(P_, q_, r_, center, radius, sample_input, bounds, f_center, solver_to_use, \
                  ineq = None):
    
    const = []
    center = np.array(center)
    
    try:
        P = cp.Parameter(P_.shape, value = P_, PSD = True)
    except:
        P_ = make_PSD(P_)
        if (P_ == 0).all():
            P_ = np.eye(len(P_))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        try:
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
        except:
            P_ = np.eye(len(P))*1e-8
            P = cp.Parameter(P_.shape, value = P_, PSD = True)
    q = cp.Parameter(q_.shape, value = q_)
    r = cp.Parameter(r_.shape, value = r_)
    
    
    # def min_underestimator_tightL1(sample_input, center, radius):
    N, D = sample_input.shape
    big_M = 1000 # 4*radius^2
    x = cp.Variable(D)
    # center = center.reshape((D,1))
    y = cp.Variable((N,D), boolean = True)
    xc = cp.Parameter(center.shape, value = center)
    # xL = cp.Parameter(center.shape, value= center-radius)
    # xU = cp.Parameter(center.shape, value= center+radius)
    alpha = cp.Variable(nonneg=True)
    r_plus = cp.Variable((N,D), nonneg=True)
    r_minus = cp.Variable((N,D), nonneg=True)
    xd = cp.Parameter(sample_input.shape, value = sample_input)
    const = [cp.norm(x - xc) <= radius]
    
    for i in range(N):
      const += [alpha <= cp.sum([r_plus[i,j] + r_minus[i,j] for j in range(D)])] 
      for j in range(D):
        const += [x[j] - xd[i,j] == r_plus[i,j] - r_minus[i,j]]
        const += [r_plus[i,j] <= big_M*(1-y[i,j]), r_minus[i,j] <= big_M*y[i,j]]
    """
    """
    objective = cp.Minimize((cp.quad_form(x, P) + q.T @ x + r)/max(abs(f_center), \
                                                    1e-4) + (-alpha)/radius**2)
    """
    """
    # trust_center = np.array(center).reshape((P_.shape[0], 1))
    
    if ineq != None:
        for coeff in ineq:
            P_ineq, q_ineq, r_ineq = coeff
            if not ((P_ineq is None) or (q_ineq is None) or (r_ineq is None)):
                P_iq = cp.Parameter(P_ineq.shape, value = P_ineq, PSD = True)
                q_iq = cp.Parameter(q_ineq.shape, value = q_ineq)
                r_iq = cp.Parameter(r_ineq.shape, value = r_ineq)
                const += [cp.quad_form(x, P_iq) + q_iq.T @ x + r_iq <= 0]

    # else:
    #     constraints = [cp.norm(x - trust_center) <= radius]
    if bounds is not None:
        const += [bounds[i,0] <=  x[i] for i in range(P_.shape[0])]
        const += [x[i] <= bounds[i,1] for i in range(P_.shape[0])]
    prob = cp.Problem(objective, constraints = const)
    if not prob.is_dcp():
        print("Problem is not disciplined convex. No global certificate")
    prob.solve(solver=solver_to_use)
    if prob.status not in ['unbounded', 'infeasible']:
        return x.value.reshape(P_.shape[0])
    else:
        print(prob.status, ' CVX min. call at: ')
        print('Center', center)
        print('Radius', radius)
        print('P_', P_)
        print('q_', q_)
        print('r_', r_)
        print('Ineq', ineq)
        raise ValueError



def constr_creation(x, g):
    if g is None:
        if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
            feas = 1
        else:
            feas = np.ones(len(np.array(x)))
    elif any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        feas = np.product((np.array(g) <= 0).astype(int))
    else:
        feas = np.product( (np.array(g) <= 0).astype(int), axis = 1)
    return feas

def sample_oracle(x, f, ineq = []):
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        y = [f(x)]
        if ineq == []:
            g_list = None
        else:
            g_list = [[g_(x) for g_ in ineq]]
    else:
        y = []
        g_list = []
        for x_ in x:
            y += [f(x_)]
            if ineq != []:
                g_list += [[g_(x_) for g_ in ineq]]
    if g_list == []:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return y, g_list, feas


def sample_simulation(x, sim):
    f_list = [] ; g_list = []
    if any(isinstance(item, float) for item in x) or any(isinstance(item, int) for item in x):
        obj, constr_vec = sim(x)
        f_list += [obj]
        if constr_vec is not None:
            g_list = [constr_vec]
        # print('Yes')
        
    else:
        for x_ in x:
            obj, constr_vec = sim(x_)
            f_list += [obj]
            if constr_vec is not None:
                g_list += [constr_vec]
    if constr_vec is None:
        g_list = None
    
    feas = constr_creation(x, g_list)
    
    return f_list, g_list, feas

def generate_init_radii(init_points, default_radius):
    
    if type(init_points) == np.ndarray: init_points = init_points.tolist()
    if len(init_points) == 1: return init_points, [default_radius]
    
    init_radii = [0]*len(init_points)
    largest_spans_raw = []
    
    # Generates list of each point's 'largest raw span'. If you have a circle
    # centered on that point, the LRS is the largest the radius of that circle
    # can be before the circle overlaps any other points
    for i in range(len(init_points)):
        point_spans_raw = [np.linalg.norm(np.array(init_points[i]) - np.array(init_points[j]))\
                           for j in range(len(init_points))]
        largest_span_raw = np.partition(point_spans_raw,1)[1]
        largest_spans_raw += [largest_span_raw]
    
    def get_key(point):
        return sorted(largest_spans_raw).index(largest_spans_raw[init_points.index(point)])
    
    # Sorts the list of initial points according to their LRS, smallest first
    # This is to ensure that the radii are built smallest first so as to not
    # exclude a point from having a radius
    init_points_sorted = sorted(init_points, key = get_key)
    
    for i in range(len(init_points_sorted)):
        sorted_spans = sorted(largest_spans_raw)
        if sorted_spans[i] in sorted_spans[i+1:]:
            init_radii[i] = sorted_spans[i]/2
        else:
            point_spans = [np.linalg.norm(np.array(init_points_sorted[i]) - np.array(init_points_sorted[j])) - init_radii[j] \
                           for j in range(len(init_points_sorted))]
            largest_span = np.partition(point_spans,1)[1]
            init_radii[i] = largest_span

    return init_points_sorted, init_radii

def generate_new_center(old_center, radius, bounds, X_samples, y_samples, g_eval, feas, constr_handling, solver_to_use):
    from CUATRO.minimise_methods.minimise import minimise_TIP

    P, q, r = quadratic_fitting(X_samples, np.array(y_samples), solver_to_use)
    feas_X = X_samples.copy()[feas == 1]
    infeas_X = X_samples.copy()[feas != 1]

    if not ((P is None) or (q is None) or (r is None)):
        # print("feas_X:\n{}".format(feas_X))
        # print("infeas_X:\n{}".format(infeas_X))
        center_, ineq_list = minimise_TIP(X_samples, feas_X, infeas_X, np.array(g_eval), P, q, \
                           r, bounds, old_center, radius, constr_handling, solver_to_use)
    else:
        return ValueError("P is None")
    
    center = [float(c) for c in center_]
    
    X = np.array(center).reshape(-1,1)
    new_pred_f = X.T @ P @ X + q.T @ X + r
    X_old = np.array(old_center).reshape(-1,1)
    old_pred_f = X_old.T @ P @ X_old + q.T @ X_old + r
    
    pred_dec = old_pred_f - new_pred_f
    
    return center, pred_dec, ineq_list

def rescale_radius(input_rad, bounds):
    bounds = np.mean(np.array(bounds), axis=0)

    # input_min = 0
    # input_max = 1

    # output_min = bounds[0]
    # output_max = bounds[1]

    input_min = bounds[0]
    input_max = bounds[1]

    output_min = 0
    output_max = 1

    init_radius = abs((((output_max - output_min) * (input_rad - input_min)) / (input_max - input_min)) + output_min)

    return init_radius

def assign_solver(solver_name):
    if solver_name == 'SCS':
        solver_to_use = cp.SCS
    elif solver_name == 'MOSEK':
        solver_to_use = cp.MOSEK
    else:
        raise ValueError('Incorrect solver specified')
    
    return solver_to_use

