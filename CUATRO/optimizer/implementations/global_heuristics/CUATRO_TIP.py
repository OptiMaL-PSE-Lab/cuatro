
import numpy as np
from CUATRO.utilities import *

from CUATRO.samplers.sampling import sample_points_TIP

from CUATRO.optimizer.CUATRO_optimizer_use import CUATRO

from typing import Optional

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)


class CUATRO_TIP(CUATRO):
    
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


        best_x = [] ; best_f = [] ; best_g = []
        nbr_samples_list = [] ; rejected_steps = []
        
        np.random.seed(rnd)

        # initialise x0 such that it contains the specified number of best prior candidates
        if len(X_samples_list) != 0:
            best_indices = np.argsort(np.array(f_eval_list.copy()))[:self.no_x0]
            self.x0 = np.array([X_samples_list.copy()[i] for i in best_indices])
        
        n_T = len(self.x0)
        trajectories_active = np.ones(n_T)
        self.x0, radii = generate_init_radii(self.x0, self.init_radius)
        radii = np.array(radii).reshape((n_T,1)).tolist()
        steps = np.array(self.x0).reshape((n_T,1,-1)).tolist()
        f_steps, g_steps, feas_steps = sample_simulation(self.x0, sim)  
        
        X_samples_list += self.x0
        f_eval_list += f_steps
        g_eval_list += g_steps
    
        f_steps = np.array(f_steps).reshape((n_T, 1)).tolist()
        g_steps = np.array(g_steps).reshape((n_T, -1)).tolist()
        feas_steps = np.array(feas_steps).reshape((n_T,1)).tolist()

        no_of_feas_X = n_T
        no_of_infeas_X = 0    
        
        ineq_const_steps = np.ones(n_T).reshape((n_T,-1)).tolist()

        if (np.array(feas_steps) == 0).any():
            raise ValueError("Please make sure all points are feasible\n\
                            The following points are not:\n\
                            {}".format(np.array(self.x0)[(np.array(feas_steps) == 0).reshape(n_T)]))
        
        N = n_T
        T = 1
        
        while ((len(f_eval_list) - len(prior_evals['f_eval_list'])) < max_f_eval - n_T):
            
            nbr_active = np.sum(trajectories_active)
            
            active_ind = np.where(trajectories_active == 1)[0]
            # print("Trajectories_active: {}".format(trajectories_active))
            # print("Active ind: {}".format(active_ind))
            print("\nTurn {} starting\n".format(T))
            
            for i in active_ind:
                            
                old_center = steps[i][-1]
                old_f = f_steps[i][-1]
                radius = radii[i][-1]
                
                X_in_trust, y_in_trust, g_in_trust, feas_in_trust = samples_in_trust(old_center, radius, \
                                                                        X_samples_list, f_eval_list, g_eval_list)
                
                N_min_samples_i = max(2,int(self.N_min_samples//nbr_active))
                N_samples = len(X_in_trust)
                if N_samples >= N_min_samples_i:
                    N_s = 1
                else:
                    N_s = N_min_samples_i
                if (len(f_eval_list) - len(prior_evals['f_eval_list']) + N_s) > max_f_eval - 1:
                    N_s = max(max_f_eval - 1 - (len(f_eval_list) - len(prior_evals['f_eval_list'])), 1)
                
                ineq_list = ineq_const_steps[i][-1]
                
                X_samples, y_samples, g_eval, feas_samples =  sample_points_TIP(old_center, radius, sim, \
                                                                                X_samples_list, ineq_list, \
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
                
                center, pred_dec, ineq_list = generate_new_center(old_center, radius, bounds, X_samples, y_samples, \
                                                    g_samples, feas_samples, self.constr_handling, solver_to_use=self.solver_to_use)
                
                ineq_const_steps[i] += ineq_list
                
                    
                f_eval, g_eval, new_feas = sample_simulation(center, sim)
                
                if new_feas == 1:
                    no_of_feas_X += 1
                else:
                    no_of_infeas_X += 1
                
                X_samples_list += [center]
                f_eval_list += f_eval
                g_eval_list += g_eval
                    
                new_f = f_eval[0]
                dec = old_f - new_f
                
                best_x, best_f, best_g = update_best_lists(X_samples_list, \
                                                    f_eval_list, g_eval_list,  \
                                                    best_x, best_f, best_g)
                
                step_size = np.linalg.norm(np.array(old_center) - np.array(center))
                    
                if (new_feas == 0) or (new_f > old_f):
                        radius *= self.beta_red
                        rejected_steps += [center]
                        center = old_center
                        new_f = old_f
                else:
                    if (dec >= self.eta2*pred_dec) and step_size - radius < 1e-8:
                        radius *= self.beta_inc
        
                    elif dec <= self.eta1*pred_dec:
                        radius *= self.beta_red
                        rejected_steps += [center]
                        center = old_center
                        new_f = old_f
                    else:
                        pass
            
                
                steps[i] += [center]
                f_steps[i] += [new_f]
                radii[i] += [radius]
                nbr_samples_list += [len(f_eval_list)]
                
                N += 1
        
            # if T == 2 : trajectories_active = np.array([1,0])    
            
            
            
            # Check overlap:
            for i in active_ind:
                last_step = np.array(steps[i][-1])
                for j in active_ind:
                    if i == j: continue
                    for k in range(len(steps[j])):
                        point_k = np.array(steps[j][k])
                        distance_between_points = np.linalg.norm(last_step - point_k)
                        radius = radii[j][k]
                        if distance_between_points < radius + 1e-8:
                            # print("Overlap of trajectory {}, point {} on trajectory {}, point {}"\
                            #        .format(i,len(steps[i])-1,j,k))
                            best_f_of_i = min(f_steps[i])
                            best_f_of_j = min(f_steps[j])
                            # print("Trajectory i has best f_eval: {}, trajectory j's best is: {}"\
                                # .format(best_f_of_i,best_f_of_j))
                            trajectories_active[i] = int(best_f_of_i < best_f_of_j)
                            trajectories_active[j] = int(best_f_of_i > best_f_of_j)
                            trajectory_eliminated = j if best_f_of_i < best_f_of_j else i
                            print("Trajectory {} eliminated on turn {}"\
                                .format(trajectory_eliminated,T))
                            break
            print("Trajectories active: \n{}".format(trajectories_active))
            print("Function evaluations: {}".format(len(f_eval_list)))
            print("Iterations: {}".format(N))
            
            T += 1
        
        N_evals = len(f_eval_list)
        # nbr_samples_list += [len(f_eval_list)]
    
        status = ""
        if N_evals > max_f_eval:
            status = "Max # of function evaluations"
        print("\nAlgorithm terminated: {}".format(status))
        
        if self.print_status:
            print('Minimisation terminated: ', status)
        
        # logger.warn(f"Minimisation terminated: {status}")
        
        constr_violation = 1 - (no_of_feas_X/len(X_samples_list))
        # percentage_violation = "{:.0%}".format(constr_violation)
        # print(no_of_feas_X)
        # print(no_of_infeas_X)
        # print(len(X_samples_list))
        
        output = {'steps': steps, 'f_steps': f_steps, 'x_best_so_far': best_x, \
                'f_best_so_far': best_f, 'g_best_so_far': best_g, \
                'x_store': X_samples_list, 'f_store': f_eval_list, \
                'g_store': g_eval_list, 'N_eval': N_evals, 'N_iter': N, \
                'radii': radii, 'samples_at_iteration': nbr_samples_list, \
                'status': status, 'rejected_steps' : rejected_steps, \
                'ineq_steps': ineq_const_steps, \
                'constr_violation': constr_violation}
            
        return output
