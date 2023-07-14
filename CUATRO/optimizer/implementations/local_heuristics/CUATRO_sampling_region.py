
"""
Created on Sun Jan 17 21:12:38 2021
@author: dv516
"""

import numpy as np
from statistics import median
from CUATRO.utilities import * # get back to this

from CUATRO.samplers.sampling import sample_LHS
from CUATRO.samplers.sampling import sample_points_opt
from CUATRO.minimise_methods.minimise import minimise

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

from typing import Optional

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)

class CUATRO_sampling_region(CUATRO):
    
    def __init__(self, x0):
        super().__init__(x0)
        
    def optimise(
        self,
        sim,
        constraints: Optional[list] = None, #might change to [] i.e. a empty list
        bounds: Optional[list] = None,
        max_f_eval: int = 100,
        rnd: int = 1,
        prior_evals: dict = {'X_samples_list' : [], 'f_eval_list': [], 'g_eval_list': [], 'bounds': [], 'x0_method': 'best eval'}
    ):
        
        if (len(prior_evals['X_samples_list']) == 0) and (not (isinstance(self.x0, np.ndarray))):
            raise ValueError("You've specified neither prior function evaluations nor a valid x0 array")
        
        if (len(prior_evals['X_samples_list']) != len(prior_evals['f_eval_list'])) or (len(prior_evals['X_samples_list']) != len(prior_evals['g_eval_list'])):
            raise ValueError('Elements of prior evaluation input lists should correspond to each other')

        if prior_evals['x0_method'] not in ['best eval', 'bound center']:
            raise ValueError('Please enter a valid method of obtaining the initial guess value')




        X_samples_list = prior_evals['X_samples_list'].copy()
        f_eval_list = prior_evals['f_eval_list'].copy()
        g_eval_list = prior_evals['g_eval_list'].copy()
        
        steps = [] ; f_step = [] ; feas_steps = []
        best_x = [] ; best_f = [] ; best_g = []
        tr_list = [] ; sr_list = [] ; nbr_samples_list = []
        
        np.random.seed(rnd)
        
        if len(X_samples_list) == 0:
            center_ = list(self.x0)
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


        trust_radius = self.init_radius
        sample_radius = self.init_radius * self.sampling_trust_ratio[1]
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
        feas_steps += [feas]
        f_eval_list += [new_f]
        g_eval_list += g_eval
        
        best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
        
        
        tr_list += [trust_radius]
        sr_list += [sample_radius]
        nbr_samples_list += [len(f_eval_list)]
        
        X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, sample_radius, \
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
            feas = np.array(feas_in_trust.copy().tolist() + feas.copy().tolist())

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
            center_ = minimise(X_samples, feas_X, infeas_X, np.array(g_eval), P, q, \
                            r, bounds, center, trust_radius, self.constr_handling, 0, 0, self.solver_to_use)
        else:
            print('P is None in first iteration')
            # logger.warn("P is None in first iteration")
            center_ = list(self.x0)
        
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
        feas_steps += [new_feas]
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
        
        while ((len(f_eval_list) - len(prior_evals['f_eval_list'])) < max_f_eval - 1) and (sample_radius > self.tolerance):
            
            N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])

            rnd += 1
            np.random.seed(rnd)
            step_size = np.linalg.norm(np.array(old_trust) - np.array(center))
            
            if (new_feas == 0) or (new_f - old_f > 0):
                trust_radius *= self.beta_red
                center = old_trust
            else:
                if (dec >= self.eta2*pred_dec) and abs(step_size - trust_radius) < 1e-8:
                    trust_radius *= self.beta_inc
                    old_trust = center
                    old_f = new_f

                elif dec <= self.eta1*pred_dec:
                    trust_radius *= self.beta_red
                    center = old_trust
                else:
                    old_trust = center
                    old_f = new_f
                    if (sample_radius - step_size) > 1e-8:
                        sample_radius *= self.beta_red
                        if trust_radius > sample_radius / self.sampling_trust_ratio[0]:
                            trust_radius = self.sampling_trust_ratio[0] * sample_radius
                    
            
        
            min_sample_radius = self.sampling_trust_ratio[0] * trust_radius
            max_sample_radius = self.sampling_trust_ratio[1] * trust_radius
            # Ensures sample_radius is between min and max
            sample_radius = median([min_sample_radius, sample_radius, max_sample_radius])
        
            steps += [center]
            tr_list += [trust_radius]
            sr_list += [sample_radius]
            nbr_samples_list += [len(f_eval_list)]
            
            if P is not None:
                X = np.array(old_trust).reshape(-1,1)
                old_pred_f = X.T @ P @ X + q.T @ X + r
            
            X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(center, sample_radius, \
                                                                    X_samples_list, f_eval_list, g_eval_list)
            N_samples = len(X_in_trust)
            if N_samples >= self.N_min_samples:
                N_s = 1
            else:
                N_s = self.N_min_samples - N_samples
            if (len(f_eval_list) - len(prior_evals['f_eval_list']) + N_s) > max_f_eval - 1:
                N_s = max(max_f_eval - 1 - (len(f_eval_list) - len(prior_evals['f_eval_list'])), 1)
                
            X_samples, y_samples, g_eval, feas_samples =  sample_points_opt(center, sample_radius, sim, \
                                                                            X_samples_list, \
                                                                            bounds, N = N_s)

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
        
            
            if not ((P is None) or (q is None) or (r is None)):
                # print("iteration {}".format(N))
                # print("feas_X:\n{}".format(feas_X))
                # print("infeas_X:\n{}".format(infeas_X))
                center_ = minimise(X_samples, feas_X, infeas_X, g_samples, P, q, r, bounds, \
                            center, trust_radius, self.constr_handling, N_iter=N, N_eval=N_evals, solver_to_use=self.solver_to_use)
                
                center = [float(c) for c in center_]
            
                f_eval, g_eval, new_feas = sample_simulation(center, sim)
                new_f = f_eval[0]
            
                if new_feas == 1:
                    no_of_feas_X += 1
                else:
                    no_of_infeas_X += 1   
                    
                X_samples_list += [center]
                f_eval_list += [new_f]
                f_step += [new_f]
                feas_steps += [new_feas]
                g_eval_list += g_eval
                X = np.array(center).reshape(-1,1)
                new_pred_f = X.T @ P @ X + q.T @ X + r
        
                pred_dec = old_pred_f - new_pred_f ; dec = old_f - new_f
            
            best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                f_eval_list, g_eval_list,  \
                                                best_x, best_f, best_g)
            N += 1
        
        N_evals = len(f_eval_list) - len(prior_evals['f_eval_list'])
        tr_list += [trust_radius] 
        sr_list += [sample_radius]
        nbr_samples_list += [len(f_eval_list)]
    
        status = ""
        if trust_radius < self.tolerance:
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
        

        # #TODO: change below back to be the output; changed it to 
        # # be the same as that of CUATRO_g
        
        # output = {'steps': steps, 'f_of_step': f_step, 'feas_of_step': feas_steps, \
        #           'x_best_so_far': best_x, \
        #           'f_best_so_far': best_f, 'g_best_so_far': best_g, \
        #           'x_store': X_samples_list, 'f_store': f_eval_list, \
        #           'g_store': g_eval_list, 'N_eval': N_evals, 'N_iter': N, \
        #           'TR': tr_list, 'SR': sr_list, 'samples_at_iteration': nbr_samples_list, \
        #           'constr_violation': constr_violation}

        output = {'steps': steps, 'x_best_so_far': best_x, 'f_best_so_far': best_f, \
                'g_best_so_far': best_g, 'x_store': X_samples_list, \
                'f_store': f_eval_list, 'g_store': g_eval_list, \
                'N_eval': N_evals, 'N_iter': N, 'TR': tr_list, \
                'samples_at_iteration': nbr_samples_list, \
                'constr_violation': constr_violation}
            
        return output



    # def trust_fig(oracle, bounds):
    #     N = 200
    #     lim = 2
    #     x = np.linspace(-lim, lim, N)
    #     y = np.linspace(-lim, lim, N)
    #     X,Y = np.meshgrid(x, y)
    #     Z = oracle.sample_obj(X,Y)
    #     constr = oracle.sample_constr(X,Y)

    #     level_list = np.logspace(-0.5, 4, 10)

    #     fig = plt.figure(figsize = (6,4))
    #     ax = fig.add_subplot()
        
    #     ax.contour(X,Y,Z*constr, levels = level_list)
    #     ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,0], bounds[1, 0]], c = 'k')
    #     ax.plot([bounds[0,0], bounds[0, 1]], [bounds[1,1], bounds[1, 1]], c = 'k')
    #     ax.plot([bounds[0,0], bounds[0, 0]], [bounds[1,0], bounds[1, 1]], c = 'k')
    #     ax.plot([bounds[0,1], bounds[0, 1]], [bounds[1,0], bounds[1, 1]], c = 'k')
        
    #     return ax


    # class RB:
    #     def __init__(self, objective, ineq = []):
    #         self.obj = objective ; self.ieq = ineq
    #     def sample_obj(self, x, y):
    #         return self.obj(x, y)
    #     def sample_constr(self, x, y):
    #         if self.ieq == []:
    #             if (type(x) == float) or (type(x) == int):
    #                 return 1
    #             else:
    #                 return np.ones(len(x))
    #         elif (type(x) == float) or (type(x) == int):
    #             temporary = [int(g(x, y)) for g in self.ieq]
    #             return np.product(np.array(temporary))
    #         else:
    #             temporary = [g(x, y).astype(int) for g in self.ieq]
    #             return np.product(np.array(temporary), axis = 0)



    # f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    # g1 = lambda x: (x[0]-1)**3 - x[1] + 1
    # # g2 = lambda x, y: x + y - 2 <= 0
    # g2 = lambda x: x[0] + x[1] - 1.8

    # quadratic_f = lambda x: x[0]**2 + 10*x[1]**2 + x[0]*x[1]
    # quadratic_g = lambda x: 1 - x[0] - x[1]

    # def sim_RB(x):
    #     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    #     g1 = lambda x: (x[0]-1)**3 - x[1] + 1
    # # g2 = lambda x, y: x + y - 2 <= 0
    #     g2 = lambda x: x[0] + x[1] - 1.8
        
    #     return f(x), [g1(x), g2(x)]

    # def sim_RB_test(x):
    #     f = lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

    #     # g1 = lambda x: (x[0]-1)**3 - x[1] + 1
    # # g2 = lambda x, y: x + y - 2 <= 0
    #     # g2 = lambda x: x[0] + x[1] - 1.8
        
    #     return f(x), []

    # x0 = [-0.5, 1.5]


    # bounds = np.array([[-1.5,1.5],[-1.5,1.5]])

    # # bounds = np.array([[-1000,1000],[-1000,1000]])

    # # bounds = np.array([[-0.6,-0.4],[1,2]])

    # bounds_quadratic = np.array([[-1.5,1.5],[-0.5,0.5]])

    # # solution_output = cvx_quad_surr_opt(f, x0, init_radius, bounds = bounds, \
    # #                                     beta_red = 0.5, constraints = [g1, g2])

    # method = 'Fitting'
    # # method = 'Discrimination'
    # rnd_seed = 10

    # N_min_s = 6
    # init_radius = .1
    # solution_output = CUATRO(sim_RB, x0, init_radius, bounds = bounds, \
    #                           N_min_samples = N_min_s, tolerance = 1e-10, \
    #                           beta_red = 0.5, rnd = 1, method = 'local', \
    #                           constr_handling = 'Fitting')


    # N_min_s = 20
    # init_radius = 2
    # solution_output = CUATRO(sim_RB, x0, init_radius, bounds = bounds, \
    #                           N_min_samples = N_min_s, tolerance = 1e-10,\
    #                           beta_red = 0.9, rnd = rnd_seed, method = 'global', \
    #                           constr_handling = method)



    # x_all = solution_output['x_store']
    # x_best = solution_output['x_best_so_far']

    # f_RB = lambda x, y: (1 - x)**2 + 100*(y - x**2)**2

    # # oracle = RB(f)

    # g1_RB = lambda x, y: (x-1)**3 - y + 1 <= 0
    # # g2 = lambda x, y: x + y - 2 <= 0
    # g2_RB = lambda x, y: x + y - 1.8 <= 0
    # g_RB = [g1_RB, g2_RB]


    # oracle = RB(f_RB, ineq = [g1_RB, g2_RB])

    # ax = trust_fig(oracle, bounds)

    # ax.scatter(np.array(x_all)[:N_min_s+1,0], np.array(x_all)[:N_min_s+1,1], s = 10, c = 'blue')
    # ax.scatter(np.array(x_all)[N_min_s+1:,0], np.array(x_all)[N_min_s+1:,1], s = 10, c = 'green')
    # ax.plot(np.array(x_best)[:,0], np.array(x_best)[:,1], '--r')
    # ax.set_title('CQSO: ' + str(method) + ' , rnd seed: ' + str(rnd_seed))


    # solution_list = []

    # N_fail = 0
    # N = 100
    # for i in range(N):
    #     print('Iteration ', i+1)
    #     sol = cvx_quad_surr_opt(f, x0, init_radius, bounds = bounds, \
    #                                     beta_red = 0.5, constraints = [g1, g2])
    #     solution_list += [sol]
        
    # fig = plt.figure(figsize = (6,8))
    # ax = fig.add_subplot(211)

    # nbr_eval = np.zeros(N)

    # for i in range(N):
    #     y = np.array(solution_list[i]['f_best_so_far'])
    #     nbr_eval[i] = len(solution_list[i]['f_store'])
    #     ax.plot(np.arange(len(y)),y, '--k')
        
    # ax.set_yscale('log')

    # ax.set_xlim([0, 40])

    # ax2 = fig.add_subplot(212)

    # ax2.hist(nbr_eval)
