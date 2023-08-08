# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 21:12:38 2021

@author: dv516
"""

import CUATRO.utilities as ut
from CUATRO.samplers.sampling import sample_points_expl, sample_points_opt, sample_LHS
from CUATRO.minimise_methods.minimise import minimise, minimise_PLS
from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from typing import Optional

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class CUATRO_PLS_bandit(CUATRO):
    def __init__(self):
        super().__init__()
        
    def optimise(
        self,
        sim,
        x0: np.ndarray = None,
        constraints: Optional[list] = None, # 
        bounds: Optional[list] = None,
        max_f_eval: int = 100,
        rnd: int = 1,
        n_pls: int = None, # effective dimensionality inputted to pls
        n_t: int = 5,
        prior_evals: dict = {'X_samples_list' : [], 'f_eval_list': [], 'g_eval_list': [], 'bounds': [], 'x0_method': 'best eval'}
    ):

        if not (constraints is None or constraints in [[], [0]]):
            raise NotImplementedError("CUATRO_PLS is not yet implemented for explicit constraint handling \n Use soft penalizations to handle constraints and set constraints to None")

        if prior_evals['g_eval_list']!= []:
            raise NotImplementedError("CUATRO_PLS is not yet implemented for explicit constraint handling \n g_eval_list in prior_evals should be an empty list")

        if len(prior_evals['X_samples_list']) != 0: ## TODO
            raise NotImplementedError('prior_evals not yet implemented for CUATRO_PLS_bandit')

        if n_pls is None: # if no knowledge about n_e is given, use heuristics to determine effective dim.
            n_pls = int(max_f_eval / 10)

        if (len(prior_evals['X_samples_list']) != len(prior_evals['f_eval_list'])) or (len(prior_evals['X_samples_list']) != len(prior_evals['g_eval_list'])):
            raise ValueError('Elements of prior evaluation input lists should correspond to each other')

        if prior_evals['x0_method'] not in ['best eval', 'bound center']:
            raise ValueError('Please enter a valid method of obtaining the initial guess value')

        # if self.method != 'local':
        #     print("method overwritten as 'local' since the LHS sampling of global is not compatible with the multi trust region approach")

        if x0 is not None: # TODO: 
            print('CUATRO_PLS_bandit ignores x0 for now and implements LHS for the initialization')
        
        # should be input arguments
        exploration_param = [0.1, 0.8]
        population_threshold = [0.3, 0.7]
        #

        np.random.seed(rnd)

        X_samples_list = []
        f_eval_list = []
        g_eval_list = []
        steps = []

        center_dummy = [float(b[1] + b[0])/2 for b in bounds]

        f_eval, g_eval, feas = ut.sample_simulation(center_dummy, sim)

        X_samples_list += [center_dummy]
        steps += [center_dummy]
        f_eval_list = f_eval.copy()
        g_eval_list += g_eval.copy()
        feas_dummy = [feas.copy()]

        # Use MCD sampling to find the initial TR centers
        X_samples, y_samples, g_eval, feas_samples =  sample_points_expl(center_dummy,  sim, \
                                                                            X_samples_list, \
                                                                            bounds, a=1, N = n_t - 1,
                                                                            candidates = 2000)
        X_samples_list += X_samples.tolist()
        f_eval_list += y_samples.copy()
        g_eval_list += g_eval
        feas_dummy += feas_samples.tolist()

        trusts = {idx: {
            'center': X_samples_list[idx], 
            'old_f': f_eval_list[idx].copy(),
            'old_trust': X_samples_list[idx],
            'radius': self.init_radius,
            'new_pred_f': None,
            'old_pred_f': None,
            'pred_dec': None,
            'candidate': None,
            'dec': None,
            # 'X_samples': None,
            'y_samples': None,
            # 'g_eval': None,
            # 'feas': None,
            } for idx in range(n_t)
        }

        # X_samples_list = prior_evals['X_samples_list'].copy()
        
        # f_eval_list = prior_evals['f_eval_list'].copy()
        # g_eval_list = prior_evals['g_eval_list'].copy()
        
        best_x = [] ; best_f = [] ; best_g = []
        radius_list = [] ; nbr_samples_list = []

        # TODO fix (?) would we have to add 1 if center was already an evaluated x (e.g. corresponding to best f eval, as in heuristic #2)
        no_of_feas_X = np.sum(feas_dummy)
        no_of_infeas_X = len(feas_dummy) - no_of_feas_X

        best_x, best_f, best_g = ut.update_best_lists(X_samples_list, 
                                                f_eval_list, g_eval_list,  
                                                best_x, best_f, best_g)
        # radius_list += [self.init_radius]
        nbr_samples_list += [len(f_eval_list)]
    
        # import time
        # t = time.process_time()

        for idx, TR in trusts.items():
            center = TR['center']
            radius = TR['radius']
            X_in_trust, y_in_trust, g_in_trust, feas_in_trust = ut.samples_in_trust(center, radius, 
                                                                    X_samples_list, f_eval_list, g_eval_list)

            if len(X_in_trust) < self.N_min_samples: 

                N_s = self.N_min_samples - len(X_in_trust)

                X_samples, y_samples, g_eval, feas =  sample_points_opt(center, radius, sim, X_in_trust.tolist(),
                                                                    bounds, N = N_s)

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
                feas = np.array(feas.copy().tolist() + feas_in_trust.copy().tolist())

            else:
                X_samples = X_in_trust.copy()
                y_samples = y_in_trust.copy()
                g_eval = g_eval.copy()
                feas = feas_in_trust.copy()
            # t_sampling = time.process_time() - t
            
            trusts[idx]['y_samples'] = y_samples

            # t = time.process_time()
            pls2 = PLSRegression(n_components=n_pls)
            pls2.fit(X_samples, y_samples)
            X_samples_pls = pls2.transform(X_samples)
            center_pls = pls2.transform(np.array(center).reshape(1, -1)).squeeze()
            # t_pls_fit = time.process_time() - t

            # t = time.process_time()
            P, q, r = ut.quadratic_fitting(X_samples_pls, np.array(y_samples), self.solver_to_use)
            feas_X = X_samples_pls.copy()[feas == 1]
            infeas_X = X_samples_pls.copy()[feas != 1]

            if not ((P is None) or (q is None) or (r is None)): ## no constraint handling for now
                # center_ = minimise_PLS(X_samples_pls, feas_X, infeas_X, np.array(g_eval), P, q, 
                #                    r, bounds, center, center_pls, radius, self.constr_handling, 0, 0, self.solver_to_use, pls2)
                heuristic_radius = np.max(np.linalg.norm(X_samples_pls - center_pls))
                heuristic_bounds = np.array([(-heuristic_radius, heuristic_radius) for _ in range(n_pls)])
                center_ = minimise(X_samples_pls, feas_X, infeas_X, np.array(g_eval), P, q, r, heuristic_bounds, 
                                       center_pls, heuristic_radius, self.constr_handling, 
                                       1, len(f_eval_list), self.solver_to_use)
                center_ = ut.bound_sample(center_, bounds)
                center_ = pls2.inverse_transform(np.array(center_).reshape(1, -1)).squeeze() ## centers should only be handled 

            else:
                print('P is None in first iteration')
                # logger.warn("P is None in first iteration")
                center_ = TR['old_trust']  # need to fix _______

            center = [float(c) for c in center_]
            trusts[idx]['candidate'] = center

            X = pls2.transform(np.array(center).reshape(1,-1)).squeeze().reshape(-1,1)
            trusts[idx]['new_pred_f'] = float((X.T @ P @ X + q.T @ X + r).squeeze())
            X_old = pls2.transform(np.array(TR['old_trust']).reshape(1,-1)).squeeze().reshape(-1,1)
            trusts[idx]['old_pred_f'] = float((X_old.T @ P @ X_old + q.T @ X_old + r).squeeze())
            trusts[idx]['pred_dec'] = TR['old_pred_f'] - TR['new_pred_f']


        TR_pred = np.array([TR['new_pred_f'] for TR in trusts.values()])
        # if self.method == 'local':
        #     TR_pred = np.exp(np.min(TR_pred) - TR_pred)
        # else:
        TR_pred = np.max(TR_pred) - TR_pred
        prob_dummy = TR_pred/np.sum(TR_pred)
        best_TR = np.where(TR_pred == float(np.random.choice(TR_pred, 1, p=prob_dummy)))[0][0]


        
        f_eval, g_eval, new_feas = ut.sample_simulation(trusts[best_TR]['candidate'], sim) # in higher_dim domain

        if new_feas == 1:
            no_of_feas_X += 1
        else:
            no_of_infeas_X += 1

        new_f = f_eval[0]
        trusts[best_TR]['dec'] = trusts[best_TR]['old_f'] - new_f
    
        X_samples_list += [trusts[best_TR]['candidate']]
        f_eval_list += [new_f]
        g_eval_list += g_eval
    
        best_x, best_f, best_g = ut.update_best_lists(X_samples_list, 
                                               f_eval_list, g_eval_list,  
                                               best_x, best_f, best_g)
    
        N = 1
    
        # print('best: ', best_TR)

        # t1 = time.process_time() 
        # t_it = []; t_center_select = []; t_sample = []; t_minimise = []
        while ((len(f_eval_list) - len(prior_evals['f_eval_list'])) < max_f_eval - 1)  and (radius > self.tolerance):
            # ti = time.process_time()
            N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])

            rnd += 1
            np.random.seed(rnd)
            
            if (new_feas == 0) or (new_f - trusts[best_TR]['old_f'] > 0):
                trusts[best_TR]['radius'] *= self.beta_red
                trusts[best_TR]['center'] = trusts[best_TR]['old_trust']
            else:
                if (trusts[best_TR]['dec'] >= self.eta2*trusts[best_TR]['pred_dec']):
                    trusts[best_TR]['radius'] *= self.beta_inc
                    trusts[best_TR]['old_trust'] = trusts[best_TR]['candidate']
                    trusts[best_TR]['center'] = trusts[best_TR]['candidate']
                    trusts[best_TR]['old_f'] = new_f
                    print(f'TR {best_TR} update: iteration {N}, objective {new_f:.3f}, evaluation {len(f_eval_list)}')
                elif trusts[best_TR]['dec'] <= self.eta1*trusts[best_TR]['pred_dec']:
                    trusts[best_TR]['radius'] *= self.beta_red
                    trusts[best_TR]['center'] = trusts[best_TR]['old_trust']
                else:
                    trusts[best_TR]['old_trust'] = trusts[best_TR]['candidate']
                    trusts[best_TR]['center'] = trusts[best_TR]['candidate']
                    trusts[best_TR]['old_f'] = new_f
                    print(f'TR {best_TR} update: iteration {N}, objective {new_f:.3f}, evaluation {len(f_eval_list)}')


            steps += [center]         


            # radius_list += [radius]

            nbr_samples_list += [len(f_eval_list)]

            ## 1 MCD sample
            exploration_ = exploration_param[1]-len(f_eval_list)/max_f_eval*(exploration_param[1]-exploration_param[0])
            if np.random.random_sample() <= exploration_:
                X_samples, y_samples, g_eval, feas_samples =  sample_points_expl(center_dummy,  sim, \
                                                                                X_samples_list, \
                                                                                bounds, a=1, N = 1,
                                                                                candidates = 2000)
            else:
                idx = np.random.randint(n_t)
                X_in_trust, _, __, ___ = ut.samples_in_trust(trusts[idx]['center'], trusts[idx]['radius'], 
                                                                        X_samples_list, f_eval_list, g_eval_list)
                X_samples, y_samples, g_eval, feas_samples =  sample_points_opt(trusts[idx]['center'], trusts[idx]['radius'], sim, \
                                                                            X_in_trust, \
                                                                            bounds, N = 1)
            X_samples_list += X_samples.tolist()
            f_eval_list += y_samples.copy()
            g_eval_list += g_eval
            feas_dummy += feas_samples.tolist()

            # if sample is better than half of samples, replace worst trust region
            dummy = np.array([TR['old_f'] for TR in trusts.values()])
            # exploration_ = exploration_param[1]-len(f_eval_list)/max_f_eval*(exploration_param[1]-exploration_param[0])
            exploration_ = population_threshold[1]-len(f_eval_list)/max_f_eval*(population_threshold[1]-population_threshold[0])
            third_best = np.sort(dummy)[:int(len(dummy)*exploration_)][-1] # 3rd best sample
            if y_samples[0] <= third_best:
                worst = np.where(dummy == np.max(dummy))[0][0]
                # print(y_samples[0], 'to replace ', worst, ' in ', dummy)
                trusts[worst]['center'] = X_samples[0].tolist().copy()
                trusts[worst]['old_trust'] = X_samples[0].tolist().copy()
                trusts[worst]['old_f'] = float(y_samples[0])
                # trusts[worst]['y_samples'] = y_samples
                trusts[worst]['radius'] = self.init_radius

            
            for idx, TR in trusts.items():
                center = TR['center']
                radius = TR['radius']
                X_in_trust, y_in_trust, g_in_trust, feas_in_trust = ut.samples_in_trust(center, radius, 
                                                                        X_samples_list, f_eval_list, g_eval_list)

                if np.array(y_in_trust).shape != np.array(TR['y_samples']).shape: ## TODO: Is there any way that len would be the same, but different values?

                    if (len(X_in_trust) < self.N_min_samples): 
                        N_s = self.N_min_samples - len(X_in_trust)

                        X_samples, y_samples, g_eval, feas =  sample_points_opt(center, radius, sim, X_in_trust.tolist(),
                                                                            bounds, N = N_s)

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
                        feas = np.array(feas.copy().tolist() + feas_in_trust.copy().tolist())

                    else:
                        X_samples = np.array(X_in_trust.copy().tolist())
                        y_samples = y_in_trust.copy().tolist()
                        g_eval = g_in_trust.copy().tolist()
                        feas = np.array(feas_in_trust.copy().tolist())

                    # t_sampling = time.process_time() - t
                    trusts[idx]['y_samples'] = y_samples

                    # t = time.process_time()
                    pls2 = PLSRegression(n_components=n_pls)
                    pls2.fit(X_samples, y_samples)
                    X_samples_pls = pls2.transform(X_samples)
                    center_pls = pls2.transform(np.array(center).reshape(1, -1)).squeeze()
                    # t_pls_fit = time.process_time() - t

                    # t = time.process_time()
                    P, q, r = ut.quadratic_fitting(X_samples_pls, np.array(y_samples), self.solver_to_use)
                    feas_X = X_samples_pls.copy()[feas == 1]
                    infeas_X = X_samples_pls.copy()[feas != 1]

                    if not ((P is None) or (q is None) or (r is None)): ## no constraint handling for now
                        # center_ = minimise_PLS(X_samples_pls, feas_X, infeas_X, np.array(g_eval), P, q, 
                        #                    r, bounds, center, center_pls, radius, self.constr_handling, 0, 0, self.solver_to_use, pls2)
                        heuristic_radius = np.max(np.linalg.norm(X_samples_pls - center_pls))
                        heuristic_bounds = np.array([(-heuristic_radius, heuristic_radius) for _ in range(n_pls)])
                        center_ = minimise(X_samples_pls, feas_X, infeas_X, np.array(g_eval), P, q, r, heuristic_bounds, 
                                               center_pls, heuristic_radius, self.constr_handling, 
                                               N, len(f_eval_list), self.solver_to_use)
                        center_ = ut.bound_sample(center_, bounds)
                        center_ = pls2.inverse_transform(np.array(center_).reshape(1, -1)).squeeze() ## centers should only be handled 

                    else:
                        print('P is None in first iteration')
                        # logger.warn("P is None in first iteration")
                        center_ = TR['old_trust']  # need to fix _______

                    center = [float(c) for c in center_]
                    trusts[idx]['candidate'] = center

                    X = pls2.transform(np.array(center).reshape(1,-1)).squeeze().reshape(-1,1)
                    trusts[idx]['new_pred_f'] = float((X.T @ P @ X + q.T @ X + r).squeeze())
                    X_old = pls2.transform(np.array(TR['old_trust']).reshape(1,-1)).squeeze().reshape(-1,1)
                    trusts[idx]['old_pred_f'] = float((X_old.T @ P @ X_old + q.T @ X_old + r).squeeze())
                    trusts[idx]['pred_dec'] = TR['old_pred_f'] - TR['new_pred_f']


            TR_pred = np.array([TR['new_pred_f'] for TR in trusts.values()])
            # print(TR_pred)
            # if self.method == 'local':
            #     TR_pred = np.exp(np.min(TR_pred) - TR_pred)
            # else:
            TR_pred = np.max(TR_pred) - TR_pred
            prob_dummy = TR_pred/np.sum(TR_pred)
            # print(TR_pred, prob_dummy)
            np.random.seed(rnd*100+idx)
            best_TR = np.where(TR_pred == float(np.random.choice(TR_pred, 1, p=prob_dummy)))[0][0]

            # print('best: ', best_TR)

            f_eval, g_eval, new_feas = ut.sample_simulation(trusts[best_TR]['candidate'], sim) # in higher_dim domain

            if new_feas == 1:
                no_of_feas_X += 1
            else:
                no_of_infeas_X += 1

            new_f = f_eval[0]
            trusts[best_TR]['dec'] = trusts[best_TR]['old_f'] - new_f

            X_samples_list += [trusts[best_TR]['candidate']]
            f_eval_list += [new_f]
            g_eval_list += g_eval

            best_x, best_f, best_g = ut.update_best_lists(X_samples_list, 
                                                   f_eval_list, g_eval_list,  
                                                   best_x, best_f, best_g)

            # t_minimise += [time.process_time()-tiii]  
            # print('evals: ', len(f_eval_list) )
            N += 1
            # t_it += [time.process_time() - ti] # Processing time at each iteration

        N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])
        # print('fin evals: ', N_evals)
        # radius_list += [radius] 
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
