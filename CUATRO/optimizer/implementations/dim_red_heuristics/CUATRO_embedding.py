# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:12:38 2021

@author: dv516
"""

import CUATRO.utilities as ut
from CUATRO.samplers.sampling import sample_LHS, sample_points
from CUATRO.minimise_methods.minimise import minimise
from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

import warnings
import numpy as np
from typing import Optional

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class CUATRO_embedding(CUATRO):
    def __init__(self):
        super().__init__()

    def _scaling_properties(self, X):
        return X.mean(axis=0), X.std(axis=0, ddof=1)

    def _embed(self, x_high, scaling, Q):
        '''x_low = x_high @ Q.T @ inv(Q @ Q.T), 
        where x_high (n_samples, n), x_low (n_samples, n_pls),
        Q (n_pls, n) random embedding matrix'''
        mean, std = scaling
        x_scaled = (x_high - mean) / std
        return x_scaled @ Q.T @ np.linalg.inv(Q @ Q.T)
    
    def _reconstruct(self, x_low, scaling, Q):
        '''x_high = x_low @ Q, 
        where x_high (n_samples, n), x_low (n_samples, n_pls),
        Q (n_pls, n) random embedding matrix'''
        mean, std = scaling
        x_scaled = x_low @ Q
        return x_scaled * std + mean
        
    def optimise(
        self,
        sim,
        x0: np.ndarray = None,
        constraints: Optional[list] = None, # 
        bounds: Optional[list] = None,
        max_f_eval: int = 100,
        rnd: int = 1,
        n_pls: int = None, # effective dimensionality of the embedding
        prior_evals: dict = {'X_samples_list' : [], 'f_eval_list': [], 'g_eval_list': [], 'bounds': [], 'x0_method': 'best eval'}
    ):

        if not (constraints is None or constraints in [[], [0]]):
            raise NotImplementedError("CUATRO_embedding is not yet implemented for explicit constraint handling \n Use soft penalizations to handle constraints and set constraints to None")

        if prior_evals['g_eval_list']!= []:
            raise NotImplementedError("CUATRO_embedding is not yet implemented for explicit constraint handling \n g_eval_list in prior_evals should be an empty list")
    

        if n_pls is None: # if no knowledge about n_e is given, use heuristics to determine effective dim.
            n_pls = int(max_f_eval / 10)

        if (len(prior_evals['X_samples_list']) == 0) and (not (isinstance(x0, np.ndarray))):
            raise ValueError("You've specified neither prior function evaluations nor a valid x0 array")

        if (len(prior_evals['X_samples_list']) != len(prior_evals['f_eval_list'])) or (len(prior_evals['X_samples_list']) != len(prior_evals['g_eval_list'])):
            raise ValueError('Elements of prior evaluation input lists should correspond to each other')

        if prior_evals['x0_method'] not in ['best eval', 'bound center']:
            raise ValueError('Please enter a valid method of obtaining the initial guess value')


        X_samples_list = prior_evals['X_samples_list'].copy()
        f_eval_list = prior_evals['f_eval_list'].copy()
        g_eval_list = prior_evals['g_eval_list'].copy()


        steps = []
        best_x = [] ; best_f = [] ; best_g = []
        radius_list = [] ; nbr_samples_list = []
        
        np.random.seed(rnd)
        if len(X_samples_list) == 0:
            center_ = list(x0)
            no_of_prior_feas_x = 0
            no_of_prior_infeas_x = 0

        else:
            feas_prior = ut.constr_creation(X_samples_list, g_eval_list)
            no_of_prior_feas_x = len(np.array(X_samples_list.copy())[feas_prior == 1])
            no_of_prior_infeas_x = len(np.array(X_samples_list.copy())[feas_prior != 1])

            best_x, best_f, best_g = ut.update_best_lists(X_samples_list,
                                                f_eval_list, g_eval_list,  
                                                best_x, best_f, best_g)

            if prior_evals['x0_method'] == 'best eval':
                center_ = best_x[0]
            else:
                center_ = prior_evals['bounds'].mean(axis=1) 

        radius = self.init_radius
        center = [float(c) for c in center_]
        
        
        f_eval, g_eval, feas = ut.sample_simulation(center, sim)
        new_f = f_eval[0]

        if feas == 0:
            raise ValueError("Please enter feasible starting point")

        X_samples_list += [center]
        steps += [center]
        f_eval_list += [new_f]
        g_eval_list += g_eval

        # TODO fix (?) would we have to add 1 if center was already an evaluated x (e.g. corresponding to best f eval, as in heuristic #2)
        no_of_feas_X = 1 + no_of_prior_feas_x
        no_of_infeas_X = no_of_prior_infeas_x

        best_x, best_f, best_g = ut.update_best_lists(X_samples_list, 
                                                f_eval_list, g_eval_list,  
                                                best_x, best_f, best_g)
        radius_list += [self.init_radius]
        nbr_samples_list += [len(f_eval_list)]
    
        # import time
        # t = time.process_time()

        X_in_trust, y_in_trust, g_in_trust, feas_in_trust = ut.samples_in_trust(center, radius, 
                                                                    X_samples_list, f_eval_list, g_eval_list)

        if len(X_in_trust) < self.N_min_samples: 

            N_s = self.N_min_samples - len(X_in_trust)

            if self.method == 'local':
                X_samples, y_samples, g_eval, feas =  sample_points(center, radius, sim, 
                                                                bounds, N = N_s)
            elif self.method == 'global':
                X_samples, y_samples, g_eval, feas = sample_LHS(sim, bounds, 
                                                            N_s, rnd_seed = rnd)
            else:
                raise ValueError('Invalid input for method')

            feas_new_X = X_samples.copy()[feas == 1]
            infeas_new_X = X_samples.copy()[feas != 1]
            
            no_of_feas_X += len(feas_new_X)
            no_of_infeas_X += len(infeas_new_X)

            X_samples_list += X_samples.tolist()
            f_eval_list += y_samples
            g_eval_list += g_eval

            # the final results here are the samples (just taken) and the prior samples from within th trust region
            X_samples = np.array(X_samples.copy().tolist() + X_in_trust.copy().tolist())
            y_samples = y_samples.copy() + y_in_trust.copy().tolist()
            g_eval = g_eval.copy() + g_in_trust.copy().tolist()
            feas = np.array(feas_in_trust.copy().tolist() + feas.copy().tolist())

        else:
            X_samples = X_in_trust.copy()
            y_samples = y_in_trust.copy()
            g_eval = g_eval.copy()
            feas = feas_in_trust.copy()
        # t_sampling = time.process_time() - t
        old_trust = center ## trust should be in high-dim
        old_f = best_f[0]

        # t = time.process_time()
        Q = np.random.normal(size=(n_pls, len(bounds)))
        scaling = self._scaling_properties(X_samples)
        X_samples_embed = self._embed(X_samples, scaling, Q)
        center_embed = self._embed(np.array(center).reshape(1, -1), scaling, Q).squeeze()
        # t_pls_fit = time.process_time() - t

        # t = time.process_time()
        P, q, r = ut.quadratic_fitting(X_samples_embed, np.array(y_samples), self.solver_to_use)
        feas_X = X_samples_embed.copy()[feas == 1]
        infeas_X = X_samples_embed.copy()[feas != 1]

        if not ((P is None) or (q is None) or (r is None)): ## no constraint handling for now
            heuristic_radius = np.max(np.linalg.norm(X_samples_embed - center_embed))
            heuristic_bounds = np.array([(-heuristic_radius, heuristic_radius) for _ in range(n_pls)])
            center_ = minimise(X_samples_embed, feas_X, infeas_X, np.array(g_eval), P, q, r, heuristic_bounds, 
                                   center_embed, heuristic_radius, self.constr_handling, 
                                   0, 0, self.solver_to_use)
            center_ = ut.bound_sample(center_, bounds)
            center_ = self._reconstruct(np.array(center_).reshape(1, -1), scaling, Q).squeeze()

        else:
            print('P is None in first iteration')
            # logger.warn("P is None in first iteration")
            center_ = old_trust  # need to fix _______
        # t_quad_fit = time.process_time() - t
   
    
        # t = time.process_time()
        center = [float(c) for c in center_]

        f_eval, g_eval, new_feas = ut.sample_simulation(center, sim) # in higher_dim domain

        if new_feas == 1:
            no_of_feas_X += 1
        else:
            no_of_infeas_X += 1

        new_f = f_eval[0]
    
        X_samples_list += [center]
        f_eval_list += [new_f]
        g_eval_list += g_eval
    
        best_x, best_f, best_g = ut.update_best_lists(X_samples_list, 
                                               f_eval_list, g_eval_list,  
                                               best_x, best_f, best_g)
    
        
        X = self._embed(np.array(center).reshape(1,-1), scaling, Q).squeeze().reshape(-1,1)
        new_pred_f = X.T @ P @ X + q.T @ X + r
        X_old = self._embed(np.array(old_trust).reshape(1,-1), scaling, Q).squeeze().reshape(-1,1)
        old_pred_f = X_old.T @ P @ X_old + q.T @ X_old + r
        pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f

        # t_sim_1pt = time.process_time() - t
    
        N = 1
    
        # t1 = time.process_time() 
        # t_it = []; t_center_select = []; t_sample = []; t_minimise = []
        while ((len(f_eval_list) - len(prior_evals['f_eval_list'])) < max_f_eval - 1)  and (radius > self.tolerance):
            # ti = time.process_time()
            N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])

            rnd += 1
            np.random.seed(N*10)

            if self.method == 'local':
                if (new_feas == 0) or (new_f - old_f > 0):
                    radius *= self.beta_red
                    center = old_trust
                else:
                    if (dec >= self.eta2*pred_dec):
                        radius *= self.beta_inc
                        old_trust = center
                        old_f = new_f

                    elif dec <= self.eta1*pred_dec:
                        radius *= self.beta_red
                        center = old_trust
                    else:
                        old_trust = center
                        old_f = new_f
            else:
                radius *= self.beta_red
                if (new_feas == 0) or (new_f - old_f > 0):
                    center = old_trust
                else:
                    old_trust = center
                    old_f = new_f

            steps += [center]         

            # t_center_select += [time.process_time()-ti]
            # tii = time.process_time()

            radius_list += [radius]
            nbr_samples_list += [len(f_eval_list)]
            

            if P is not None:
                X = self._embed(np.array(old_trust).reshape(1,-1), scaling, Q).squeeze().reshape(-1,1)
                old_pred_f = X.T @ P @ X + q.T @ X + r

            # low-dim
            X_in_trust, y_in_trust, g_in_trust, feas_in_trust = ut.samples_in_trust(center, radius, X_samples_list, f_eval_list, g_eval_list)
            N_samples, N_x = X_in_trust.shape
            if N_samples >= self.N_min_samples:
                N_s = 1
            else:
                N_s = self.N_min_samples - N_samples
            if (len(f_eval_list) - len(prior_evals['f_eval_list']) + N_s) > max_f_eval - 1:
                N_s = max(max_f_eval - 1 - (len(f_eval_list) - len(prior_evals['f_eval_list'])), 1)

            X_samples, y_samples, g_eval, feas_samples =  sample_points(center, radius, sim, bounds, N = N_s)
            
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

            Q = np.random.normal(size=(n_pls, len(bounds)))
            scaling = self._scaling_properties(X_samples)
            X_samples_embed = self._embed(X_samples, scaling, Q)
            center_embed = self._embed(np.array(center).reshape(1, -1), scaling, Q).squeeze()

            heuristic_radius = np.max(np.linalg.norm(X_samples_embed - center_embed))
            heuristic_bounds = np.array([(-heuristic_radius, heuristic_radius) for _ in range(n_pls)])

            try: 
                P, q, r = ut.quadratic_fitting(X_samples_embed, y_samples, self.solver_to_use)
            except:
                print(f"Solver failed to find convex quadratic fit in iteration {N}")
                # logger.warn("Solver failed to find convex quadratic fit")

            feas_X = X_samples_embed.copy()[feas_samples == 1]
            infeas_X = X_samples_embed.copy()[feas_samples != 1]

            # t_sample += [time.process_time()-tii]
            # tiii = time.process_time()


            if not ((P is None) or (q is None) or (r is None)):
                center_ = minimise(X_samples_embed, feas_X, infeas_X, g_samples, P, q, r, heuristic_bounds, 
                                   center_embed, heuristic_radius, self.constr_handling, 
                                   N, N_evals, self.solver_to_use)
                center_ = ut.bound_sample(center_, bounds)
                center_ = self._reconstruct(np.array(center_).reshape(1, -1), scaling, Q).squeeze()
                center = [float(c) for c in center_]

                f_eval, g_eval, new_feas = ut.sample_simulation(center, sim) # low
                new_f = f_eval[0]

                if new_feas == 1:
                    no_of_feas_X += 1
                else:
                    no_of_infeas_X += 1

                X_samples_list += [center]
                f_eval_list += [new_f]
                g_eval_list += g_eval

                X = self._embed(np.array(center).reshape(1,-1), scaling, Q).squeeze().reshape(-1,1)
                new_pred_f = X.T @ P @ X + q.T @ X + r
    

                pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f

            best_x, best_f, best_g = ut.update_best_lists(X_samples_list, f_eval_list, 
                                                          g_eval_list, best_x, best_f, best_g)

            # t_minimise += [time.process_time()-tiii]  
            N += 1
            # t_it += [time.process_time() - ti] # Processing time at each iteration

        N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])
        radius_list += [radius] 
        nbr_samples_list += [len(f_eval_list)]

        if radius < self.tolerance:
            status = "Radius below threshold"
        else:
            status = "Max # of function evaluations"

        if self.print_status:
            print('Minimisation terminated: ', status)

        # logger.warn(f"Minimisation terminated: {status}")
        constr_violation = 1 - (no_of_feas_X/len(X_samples_list))

        output = {'steps': steps, 'x_best_so_far': best_x, 'f_best_so_far': best_f, 
                  'g_best_so_far': best_g, 'x_store': X_samples_list, 
                  'f_store': f_eval_list, 'g_store': g_eval_list, 
                  'N_eval': N_evals, 'N_iter': N, 'TR': radius_list, 
                  'samples_at_iteration': nbr_samples_list,
                  'constr_violation': constr_violation}

        return output
