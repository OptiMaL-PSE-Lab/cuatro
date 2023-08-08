
"""
Created on Sun Jan 17 21:12:38 2021
@author: dv516
"""

"""
NOTES:
-
"""

import numpy as np
from CUATRO.utilities import *

from CUATRO.samplers.sampling import sample_LHS
from CUATRO.samplers.sampling import sample_points_TIS
from CUATRO.minimise_methods.minimise import minimise_TIS

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

from typing import Optional

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)


class CUATRO_TIS(CUATRO):
    
    def __init__(self):
        super().__init__()
        
    def optimise(
        self,
        sim,
        x0: np.ndarray = None,
        constraints: Optional[list] = None, #might change to [] i.e. a empty list
        bounds: Optional[list] = None,
        max_f_eval: int = 100,
        rnd: int = 1,
        prior_evals: dict = {'X_samples_list' : [], 'f_eval_list': [], 'g_eval_list': [], 'bounds': [], 'x0_method': 'best eval'}
    ):
        
        if (len(prior_evals['X_samples_list']) == 0) and (not (isinstance(x0, np.ndarray))):
            raise ValueError("You've specified neither prior function evaluations nor a valid x0 array")
        
        if (len(prior_evals['X_samples_list']) != len(prior_evals['f_eval_list'])) or (len(prior_evals['X_samples_list']) != len(prior_evals['g_eval_list'])):
            raise ValueError('Elements of prior evaluation input lists should correspond to each other')

        if prior_evals['x0_method'] not in ['best eval', 'bound center']:
            raise ValueError('Please enter a valid method of obtaining the initial guess value')
    

        X_samples_list = prior_evals['X_samples_list'].copy()
        f_eval_list = prior_evals['f_eval_list'].copy()
        g_eval_list = prior_evals['g_eval_list'].copy()
        
        steps = [] ; f_step = []
        best_x = [] ; best_f = [] ; best_g = []
        radius_list = [] ; nbr_samples_list = []
        step_size_list = [] ; restart_reason = []
        f_change_list = [] ; 
        ineq_list = [(None, None, None)]
        
        np.random.seed(rnd)
        
        if len(X_samples_list) == 0:
            center_ = list(x0)
            no_of_prior_feas_x = 0
            no_of_prior_infeas_x = 0

        else:
            feas_prior = constr_creation(X_samples_list, g_eval_list)
            no_of_prior_feas_x = len(np.array(X_samples_list.copy())[feas_prior == 1])
            no_of_prior_infeas_x = len(np.array(X_samples_list.copy())[feas_prior != 1])

            best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
            

            if prior_evals['x0_method'] == 'best eval':
                center_ = best_x[0]
            else:
                center_ = prior_evals['bounds'].mean(axis=1) 


        radius = self.init_radius
        center = [float(c) for c in center_]
        
        f_eval, g_eval, feas = sample_simulation(center, sim)
        new_f = f_eval[0]
        
        if feas == 0:
            raise ValueError("Please enter feasible starting point")
        
        no_of_feas_X = 1 + no_of_prior_feas_x
        no_of_infeas_X = no_of_prior_infeas_x

        X_samples_list += [center]
        steps += [center]
        f_step += [new_f]
        f_eval_list += [new_f]
        g_eval_list += g_eval

        best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
        
        radius_list += [self.init_radius]
        nbr_samples_list += [len(f_eval_list)]

        X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, radius, \
                                                                    X_samples_list, f_eval_list, g_eval_list)


        if len(X_in_trust) < self.N_min_samples: 

            N_s = self.N_min_samples - len(X_in_trust)

            X_samples, y_samples, g_eval, feas = sample_LHS(sim, bounds, \
                                                            N_s, rnd_seed = rnd)

            feas_new_X = X_samples.copy()[feas == 1]
            infeas_new_X = X_samples.copy()[feas != 1]
            
            no_of_feas_X += len(feas_new_X)
            no_of_infeas_X += len(infeas_new_X)
        
            X_samples_list += X_samples.tolist()
            f_eval_list += y_samples
            g_eval_list += g_eval

            
            X_samples = np.array(X_samples.copy().tolist() + X_in_trust.copy().tolist())
            y_samples = y_samples.copy() + y_in_trust.copy().tolist()
            g_eval = g_eval.copy() + g_in_trust.copy().tolist()
            feas = np.array(feas.copy().tolist() + feas_in_trust.copy().tolist())


        else:
            X_samples = X_in_trust.copy()
            y_samples = y_in_trust.copy()
            g_eval = g_eval.copy()
            feas = feas_in_trust.copy()

        
        old_trust = center
        old_f = best_f[0]


        P, q, r = quadratic_fitting(X_samples, np.array(y_samples), self.solver_to_use)
        feas_X = X_samples.copy()[feas == 1]
        infeas_X = X_samples.copy()[feas != 1]
        

        if not ((P is None) or (q is None) or (r is None)):
            # print("feas_X:\n{}".format(feas_X))
            # print("infeas_X:\n{}".format(infeas_X))
            center_, ineq_list = minimise_TIS(X_samples, feas_X, infeas_X, np.array(g_eval), P, q, \
                            r, bounds, center, radius, self.constr_handling, self.solver_to_use)
        else:
            print('P is None in first iteration')
            # logger.warn("P is None in first iteration")
            center_ = list(x0)
        
        center = [float(c) for c in center_]
        
        f_eval, g_eval, new_feas = sample_simulation(center, sim)
        
        if new_feas == 1:
            no_of_feas_X += 1
        else:
            no_of_infeas_X += 1
            
        new_f = f_eval[0]
        # g_eval = oracle.sample_g(center)
        # print(center)
        # print(g_eval)
        # new_feas = oracle.sample_constr(center, g_list = g_eval) 
        # print(new_feas)
        X_samples_list += [center]
        f_eval_list += [new_f]
        f_step += [new_f]
        g_eval_list += g_eval
        
        best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
        
        X = np.array(center).reshape(-1,1)
        new_pred_f = X.T @ P @ X + q.T @ X + r
        X_old = np.array(old_trust).reshape(-1,1)
        old_pred_f = X_old.T @ P @ X_old + q.T @ X_old + r
        
        pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
        
        N = 1
        N_last_restart = 0
        
                            # 0.1*sum(bounds[i][1]-bounds[i][0] \
                            #    for i in range(len(bounds)))/len(bounds)) + init_radius
        restart = False

        
        while ((len(f_eval_list) - len(prior_evals['f_eval_list'])) < max_f_eval - 1) and (radius > self.tolerance):
            
            rnd += 1
            np.random.seed(rnd)
        
            if restart and new_feas == 1:
                old_trust = center
                f_change_list += [old_f - new_f]
                old_f = new_f
                restart = False
            
            else:
                
                if (new_feas == 0) or (new_f - old_f > 0):
                    radius *= self.beta_red
                    center = old_trust
                    f_change_list += [0]
                else:
                    if (dec >= self.eta2*pred_dec) and (abs(np.linalg.norm(np.array(old_trust) - np.array(center)) - radius) < 1e-8).any():
                        radius *= self.beta_inc
                        old_trust = center
                        f_change_list += [old_f - new_f]
                        old_f = new_f
                        print(f'TR update: iteration {N}, objective {new_f:.3f}, evaluation {len(f_eval_list)}')

        
                    elif dec <= self.eta1*pred_dec:
                        radius *= self.beta_red
                        center = old_trust
                        f_change_list += [0]

                    else:
                        old_trust = center
                        f_change_list += [old_f - new_f]
                        old_f = new_f
                        print(f'TR update: iteration {N}, objective {new_f:.3f}, evaluation {len(f_eval_list)}')

            
            
            steps += [center] 
            step_size_list += [np.linalg.norm(np.array(steps[len(steps)-1]) \
                                    -np.array(steps[len(steps)-2]))]
            nbr_samples_list += [len(f_eval_list)]
            
            "tuning parameters"
            min_zero_steps = 8
            min_small_steps = 3
            min_imprv = 1
            
            if len(step_size_list) >= min_zero_steps:
                recent_steps = []
                for j in range(min_zero_steps):
                    recent_steps += [step_size_list[len(step_size_list)-(1+j)]]
                if min(recent_steps) == 0 and max(recent_steps) == 0:
                    restart = True
                    restart_reason += ["For iteration {}, not moving at all".format(N)]
            
            newtrajlen = len(f_change_list) - (N_last_restart + 1)
            if newtrajlen >= min_small_steps:
                recent_f_changes = []
                for j in range(newtrajlen):
                    if f_change_list[len(f_change_list)-(1+j)] != 0:
                        recent_f_changes += [f_change_list[len(f_change_list)-(1+j)]]
                if len(recent_f_changes) >= min_small_steps:
                    obj_imprv = sum(recent_f_changes[i] for i in range(min_small_steps))
                    if obj_imprv <= min_imprv:
                        restart = True
                        restart_reason += \
                        ["For iteration {}, not improving enough".format(N)]
        
                    
            if N_last_restart > 0 and not restart:
                for j in range(N_last_restart):
                    dist_to_oldtraj = np.linalg.norm(np.array(center) - \
                                    np.array(steps[j]))
                    convdist = min(self.conv_radius, radius_list[j])
                    if dist_to_oldtraj < convdist and dist_to_oldtraj > 1e-8:
                        restart = True
                        restart_reason += \
                        ["For iteration {}, reconverged".format(N)]
                        break

            if radius < self.min_radius:
                restart = True
                restart_reason += ["For iteration {}, radius too small".format(N)]
            
            if restart:
                restart_radius = max(np.linalg.norm(np.array(center) - np.array(x0))+ \
                                    self.init_radius, self.min_restart_radius)
                radius = restart_radius
                self.beta_red = 0.6
                N_last_restart = N
        
            radius_list += [radius]

            
            if P is not None:
                X = np.array(old_trust).reshape(-1,1)
                old_pred_f = X.T @ P @ X + q.T @ X + r
            
            X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, radius, \
                                                                    X_samples_list, f_eval_list, g_eval_list)
            N_samples = len(X_in_trust)
            if N_samples >= self.N_min_samples:
                N_s = 1
            else:
                N_s = self.N_min_samples - N_samples
            if (len(f_eval_list) - len(prior_evals['f_eval_list']) + N_s) > max_f_eval - 1:
                N_s = max(max_f_eval - 1 - (len(f_eval_list) - len(prior_evals['f_eval_list'])), 1)
            
            if restart:
                X_samples, y_samples, g_eval, feas_samples =  \
                    sample_points_TIS(center, radius, sim, X_samples_list, bounds, \
                                    ineq_list, N = 1, ineq = True)
                center = X_samples[0].tolist()
                new_f = y_samples[0]
                f_step += [new_f]            
                new_feas = feas_samples
            else:
                X_samples, y_samples, g_eval, feas_samples =  \
                    sample_points_TIS(center, radius, sim, X_samples_list, bounds, \
                                    ineq_list, N = N_s, ineq = False)
                
            feas_new_X = X_samples.copy()[feas_samples == 1]
            infeas_new_X = X_samples.copy()[feas_samples != 1]
            
            no_of_feas_X += len(feas_new_X)
            no_of_infeas_X += len(infeas_new_X)
                
            X_samples_list += X_samples.tolist()
            f_eval_list += y_samples
            g_eval_list += g_eval
            
            X_samples = np.array(X_in_trust.tolist() + X_samples.tolist())
            y_samples = np.array(y_in_trust.tolist() + y_samples)
            g_samples = np.array(g_in_trust.tolist() + g_eval)
            feas_samples = np.array(feas_in_trust.tolist() + feas_samples.tolist())
            
            try:
                P, q, r = quadratic_fitting(X_samples, y_samples, self.solver_to_use)
            except:
                print(f'Solver failed to find convex quadratic fit. N_iter: {N}')
                # logger.warn("Solver failed to find convex quadratic fit")
                
            feas_X = X_samples.copy()[feas_samples == 1]
            infeas_X = X_samples.copy()[feas_samples != 1]
                
            
            if not ((P is None) or (q is None) or (r is None)) and not restart:
                # print("iteration {}".format(N))
                # print("feas_X:\n{}".format(feas_X))
                # print("infeas_X:\n{}".format(infeas_X))
                center_, ineq_list = minimise_TIS(X_samples, feas_X, infeas_X, g_samples, \
                                P, q, r, bounds, center, radius, self.constr_handling, self.solver_to_use)
                
                center = [float(c) for c in center_]
            
                f_eval, g_eval, new_feas = sample_simulation(center, sim)
                new_f = f_eval[0]
            
                if new_feas == 1:
                    no_of_feas_X += 1
                else:
                    no_of_infeas_X += 1   
                    
                X_samples_list += [center]
                # steps += [center]
                # step_size_list += [np.linalg.norm(np.array(steps[len(steps)-1]) \
                #                             -np.array(steps[len(steps)-2]))]            
                f_eval_list += [new_f]
                f_step += [new_f]
                g_eval_list += g_eval
                X = np.array(center).reshape(-1,1)
                new_pred_f = X.T @ P @ X + q.T @ X + r
        
                pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
            
            best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
                
            N += 1
        
        N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])
        nbr_samples_list += [len(f_eval_list)]
    
        status = ""
        if radius < self.tolerance:
            status = "Radius below threshold"
        else:
            status = "Max # of function evaluations"
        
        if self.print_status:
            print('Minimisation terminated: ', status)

        # logger.warn(f"Minimisation terminated: {status}")
            
        constr_violation = 1 - (no_of_feas_X/len(X_samples_list))
        # percentage_violation = "{:.0%}".format(constr_violation)
        # print(no_of_feas_X)
        # print(no_of_infeas_X)
        # print(len(X_samples_list))
        
        output = {'steps': steps, 'step_sizes': step_size_list, 'f_of_step': f_step, 'x_best_so_far': best_x, \
                'f_best_so_far': best_f, 'g_best_so_far': best_g, \
                'x_store': X_samples_list, 'f_store': f_eval_list, \
                'g_store': g_eval_list, 'N_eval': N_evals, 'N_iter': N, \
                'TR': radius_list, 'samples_at_iteration': nbr_samples_list, \
                'restart_reason': restart_reason, \
                'f_changes': f_change_list, 'last_restart_iteration': N_last_restart, \
                'recent_f_changes': recent_f_changes, \
                'constr_violation': constr_violation}
            
        return output
