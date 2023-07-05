
from statistics import median
import CUATRO.utilities as ut

# import logging
# import logging.config

# logging.config.fileConfig(fname='logger_config.conf', disable_existing_loggers=False)
# logger = logging.getLogger(__name__)


def minimise(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius, method, N_iter, N_eval, solver_to_use):
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
        # logger.warn("P is of type None. Jump step..")
        raise RuntimeWarning("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        if method == 'Discrimination':
            try:
                P_ineq, q_ineq, r_ineq = ut.quadratic_discrimination(feas_X, infeas_X, solver_to_use)
                #logger.info("test that logging in minimise() works")
            except:
                P_ineq, q_ineq, r_ineq = None, None, None
                all_feas = False
                # logger.warn(f"Feasible quadratic coefficients can't be found by discrimination. N_evals: {N_eval}. N_iter: {N_iter}")
                raise RuntimeWarning(f"Feasible quadratic coefficients can't be found by discrimination. N_evals: {N_eval}. N_iter: {N_iter}")
            ineq_list = [(P_ineq, q_ineq, r_ineq)]
        
        else:
            ineq_list = []
            n_ineq = g_array.shape[1]
            for i in range(n_ineq):
                g_pred = g_array[:,i]
                try:
                    fitting_out = ut.quadratic_fitting(X_samples, g_pred, solver_to_use, discr = True)
                    ineq_list += [fitting_out]
                except:
                    # logger.warn("Feasible quadratic coefficients can't be found by regression. N_evals: {N_eval}. N_iter: {N_iter}")
                    raise RuntimeWarning(f"Feasible quadratic coefficients can't be found by regression. N_evals: {N_eval}. N_iter: {N_iter}")
                    ineq_list += [(None, None, None)]
        
        if all_feas:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use,\
                                             ineq = ineq_list))
            except:
                P = ut.make_PSD(P)
                # logger.warn("Failed to find center by minimising quadratic surrogate (found feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate (found feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                               ineq = ineq_list))
                except:
                    center_ = center
                    raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate, even after updating P (found feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                    # logger.warn("Failed to find center by minimising quadratic surrogate, even after updating P (found feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
        else:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                P = ut.make_PSD(P)
                raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate (failed to find feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                # logger.warn("Failed to find center by minimising quadratic surrogate (failed to find feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
                except:
                    center_ = center
                    raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate, even after updating P (failed to find feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
                    # logger.warn("Failed to find center by minimising quadratic surrogate, even after updating P (failed to find feasible coefficients by discrimination previously). N_evals: {N_eval}. N_iter: {N_iter}")
    else:
        try:
            center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
        except:
            P = ut.make_PSD(P)
            raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate (all samples were feasible)")
            # logger.warn("Failed to find center by minimising quadratic surrogate (all samples were feasible)")
            # print(P)
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                center_ = center
                raise RuntimeWarning(f"Failed to find center by minimising quadratic surrogate, even after updating P (all samples were feasible))")
                # logger.warn("Failed to find center by minimising quadratic surrogate, even after updating P (all samples were feasible))")
    return center_

def minimise_expl_expl(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius,\
             f_center, solver_to_use):

    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        
        try:
            P_ineq, q_ineq, r_ineq = ut.quadratic_discrimination(feas_X, infeas_X, solver_to_use)
        except:
            P_ineq, q_ineq, r_ineq = None, None, None
            all_feas = False
            
        ineq_list = [(P_ineq, q_ineq, r_ineq)]
       
        
        if all_feas:
            try:
                center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, bounds, \
                                             f_center, solver_to_use, ineq = ineq_list))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, bounds, \
                               f_center, solver_to_use, ineq = ineq_list))
                except:
                    center_ = center
        else:
            try:
                center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, bounds, \
                                             f_center, solver_to_use))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, \
                                                 bounds, f_center, solver_to_use))
                except:
                    center_ = center
    else:
        try:
            center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, bounds,\
                                         f_center, solver_to_use))
        except:
            P = ut.make_PSD(P)
            # print(P)
            try:
                center_ = list(ut.quadratic_min_expl_expl(P, q, r, center, radius, X_samples, bounds,\
                                             f_center, solver_to_use))
            except:
                center_ = center
    return center_

def minimise_feas_samp(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds,\
     center, radius, method, solver_to_use):
    
    ineq_list = [(None, None, None)]
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        if method == 'Discrimination':
            try:
                P_ineq, q_ineq, r_ineq = ut.quadratic_discrimination(feas_X, infeas_X, solver_to_use)
            except:
                P_ineq, q_ineq, r_ineq = None, None, None
                all_feas = False
            ineq_list = [(P_ineq, q_ineq, r_ineq)]
            # print('Discrimination constants: ', P_ineq, q_ineq, r_ineq)
        
        else:
            ineq_list = []
            n_ineq = g_array.shape[1]
            # print(g_array)
            for i in range(n_ineq):
                g_pred = g_array[:,i]
                try:
                    fitting_out = ut.quadratic_fitting(X_samples, g_pred, solver_to_use, discr = True)
                # print(i, fitting_out)
                # print(g_pred)
                    ineq_list += [fitting_out]
                    # print('Yes')
                except:
                    ineq_list += [(None, None, None)]
                    # print('No')
                #     print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
                #     print('Yes')
                # except:
                #     all_feas = False
                #     print('Nope')
            
        
        # g_predicted = np.max(g_list, axis = 1)
        # try:
        #     P_ineq, q_ineq, r_ineq = quadratic_fitting(X_samples, g_predicted)
        # except:
        #     P_ineq, q_ineq, r_ineq = None, None, None
        #     print('Inequality constraint fitting failed')
        # print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
        
        if all_feas:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                                             ineq = ineq_list))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use,  \
                               ineq = ineq_list))
                except:
                    center_ = center
        else:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
                except:
                    center_ = center
    else:
        try:
            center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
        except:
            P = ut.make_PSD(P)
            # print(P)
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                center_ = center
    "Now forces the function to return list of inequality quadratic approx. coeffs"
    "In the case of no inequality constraints, all coeffs set by default to 'None' "
    return center_, ineq_list

def minimise_TIS(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius, method, solver_to_use):
    
    ineq_list = [(None, None, None)]
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        if method == 'Discrimination':
            try:
                P_ineq, q_ineq, r_ineq = ut.quadratic_discrimination(feas_X, infeas_X, solver_to_use)
            except:
                P_ineq, q_ineq, r_ineq = None, None, None
                all_feas = False
            ineq_list = [(P_ineq, q_ineq, r_ineq)]
            # print('Discrimination constants: ', P_ineq, q_ineq, r_ineq)
        
        else:
            ineq_list = []
            n_ineq = g_array.shape[1]
            # print(g_array)
            for i in range(n_ineq):
                g_pred = g_array[:,i]
                try:
                    fitting_out = ut.quadratic_fitting(X_samples, g_pred, solver_to_use, discr = True)
                # print(i, fitting_out)
                # print(g_pred)
                    ineq_list += [fitting_out]
                    # print('Yes')
                except:
                    ineq_list += [(None, None, None)]
                    # print('No')
                #     print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
                #     print('Yes')
                # except:
                #     all_feas = False
                #     print('Nope')
            
        
        # g_predicted = np.max(g_list, axis = 1)
        # try:
        #     P_ineq, q_ineq, r_ineq = quadratic_fitting(X_samples, g_predicted)
        # except:
        #     P_ineq, q_ineq, r_ineq = None, None, None
        #     print('Inequality constraint fitting failed')
        # print('Fitting constants: ', P_ineq, q_ineq, r_ineq)
        
        if all_feas:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                                             ineq = ineq_list))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                               ineq = ineq_list))
                except:
                    center_ = center
        else:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
                except:
                    center_ = center
    else:
        try:
            center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
        except:
            P = ut.make_PSD(P)
            # print(P)
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                center_ = center
    return center_, ineq_list

def minimise_TIP(X_samples, feas_X, infeas_X, g_array, P, q, r, bounds, center, radius, method, solver_to_use):
    
    ineq_list = [(None)]
    if (P is None) or (q is None) or (r is None):
        print("P is of type None. Jump step..")
    elif len(feas_X) != len(X_samples) :
        
        all_feas = True
        
        try:
            P_ineq, q_ineq, r_ineq = ut.quadratic_discrimination(feas_X, infeas_X, solver_to_use)
        except:
            P_ineq, q_ineq, r_ineq = None, None, None
            all_feas = False
        ineq_list = [(P_ineq, q_ineq, r_ineq)]
        
        if all_feas:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                                             ineq = ineq_list))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use, \
                               ineq = ineq_list))
                except:
                    center_ = center
        else:
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                P = ut.make_PSD(P)
                # print(P)
                try:
                    center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
                except:
                    center_ = center
    else:
        try:
            center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
        except:
            P = ut.make_PSD(P)
            # print(P)
            try:
                center_ = list(ut.quadratic_min(P, q, r, center, radius, bounds, solver_to_use))
            except:
                center_ = center
                
    return center_ , ineq_list
