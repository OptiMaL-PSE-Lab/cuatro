
from multiprocessing.sharedctypes import Value
import numpy as np
from statistics import median
from CUATRO.utilities import rescale_radius
import CUATRO.utilities as ut
from typing import Optional


class CUATRO():
    
    '''
    INPUTS
    ------------------------------------
    f:                function to be optimised
        
    init_radius:      initial trust region radius
        
    constraints:      constraint functions in form [g1,g2,...,gm]
                      all have form g(x) <= 0 and return g(x)

    sampling:         choose whether to implement MCD (Maximum Closest Distance}
                      for sample collection or not
                      Can take values: 'g', 'base' ('base' implements MCD)
                      Default: 'g'

    explore:          specify whether to implement any additional
                      exploration heuristics
                      Can take values: None, 'feasible_sampling', 
                      'exploit_explore', 'sampling_region', 'TIS', 'TIP'
                      Default: None
                      If set to anything else than None, change 'sampling' to 'base'

    prior_evals:      dictionary taking prior sampled points, prior function and 
                      constraint evaluations, and bounds in default form:
                      {'f_eval_list' : [] 'X_samples_list' : []
                      'g_eval_list' : [], 'bounds': [], 'x0_method': 'enter method here'}
                      Possible values for 'x0_method': 'best eval',
                      'bound center' (default: 'best eval')
    
    rescale_radius:   True / False, default: False

    solver_to_use:    solver to be used in quadratic fitting, discrimination and minimisation
                      Possible values: 'SCS', 'MOSEK'. Default: 'SCS'

    automatic_params: dictionary specifying the data based on which hyperparameters should be set automatically TODO: clarify this



    OUTPUTS
    ------------------------------------
    output_dict: 
        - 'x'           : final input variable
            
        - 'f'           : final function value
            
        - 'f_evals'     : total number of function evaluations
            
        - 'f_store'     : best function value at each iteration
                                
        - 'x_store'     : list of all previous best variables (per iteration)
                                
        - 'g_store'     : list of all previous constraint values (per iteration)
            
        - 'g_viol'      : total constraint violation (sum over constraints)
        
    NOTES
    --------------------------------------

    '''

    def __init__(
        self,
        init_radius: int = 1,
        tolerance: int = 1e-08,
        beta_inc: int = 1.2,
        beta_red: int = 0.8,
        eta1: int = 0.2,
        eta2: int =  0.8,
        method: str = 'local',
        N_min_samples: int = 6,
        print_status: str = False,
        constr_handling: str = 'Discrimination',
        sampling: str = 'g',
        dim_red = None, # in [None,'PLS','explore','embed', 'bandit']
        explore: Optional[str] = None,
        sampling_trust_ratio: list = [0.1, 0.9],
        min_radius: int = 0.05,
        min_restart_radius: int  = 2.0,
        conv_radius: int = 0.2,
        no_x0: int = 5,
        rescale_radius: bool = False, # TODO: default should be true + change it to new implementation
        solver_to_use: str = 'SCS',
        automatic_params: Optional[dict] = None,
    ):
        
        self.init_radius = init_radius
        self.tolerance = tolerance
        self.beta_inc = beta_inc
        self.beta_red = beta_red
        self.eta1 = eta1
        self.eta2 = eta2

        self.dim_red = dim_red
        
        self.method = method
        if self.method not in ['local', 'global']:
            raise ValueError('Please provide valid method')
        
        self.N_min_samples = N_min_samples
        self.print_status = print_status
        
        self.constr_handling = constr_handling
        if self.constr_handling not in ['Discrimination', 'Regression']:
            raise ValueError('Please provide valid constraint handling method')
        
        self.sampling = sampling
        if self.sampling not in ['base', 'g']:
            raise ValueError('Please enter a valid sampling method')
        
        self.explore = explore
        if self.explore not in [None, 'feasible_sampling', 'exploit_explore',\
             'sampling_region', 'TIS', 'TIP']:
            raise ValueError("Please enter valid exploration heuristics or None") 
        
        self.sampling_trust_ratio = sampling_trust_ratio
        self.min_radius = min_radius
        self.min_restart_radius = min_restart_radius
        self.conv_radius = conv_radius
        self.no_x0 = no_x0
        
        self.rescale_radius = rescale_radius
        
        # to pass the solver instance itself, e.g. cp.MOSEK instead of string value, e.g. 'Mosek'
        if solver_to_use not in ['SCS', 'MOSEK']:
            raise ValueError('Please enter a valid solver (SCS or MOSEK)')
        assigned_solver = ut.assign_solver(solver_to_use)
        self.solver_to_use = assigned_solver

        # TODO: implement automatic_params
        self.automatic_params = automatic_params
        if self.automatic_params:
            if self.automatic_params['exp'] == 'expensive':
                pass
                # set params for expensive scenario, e.g. self.N_min_s = ..., self.beta_red = ...
            else:
                pass
                # set params for non-expensive scenario
            if self.automatic_params['safety'] == 'safety':
                pass
                # set params for safe scenario, e.g. self.N_min_s = ..., self.beta_red = ...
            else:
                pass
                # set params for more exploration
            if self.automatic_params['convex'] == 'convex':
                pass
                # set params for convex problem, e.g. self.N_min_s = ..., self.beta_red = ...
            else:
                pass
            

    # def change_params(self, new_params):
    #     for key, value in new_params.items():
    #         setattr(self, key, value)


    def run_optimiser(
        self,
        sim,
        x0: np.ndarray = None, # x0: initial guess in form [x1,x2,...,xn]
        constraints: Optional[list] = None, #might have to change to [] i.e. an empty list
        bounds: Optional[list] = None,
        max_f_eval: int = 100,
        rnd: int = 1,
        n_pls: int = None,
        n_t: int = 5,
        prior_evals: dict = {'X_samples_list' : [], 'f_eval_list': [], 'g_eval_list': [], 'bounds': [], 'x0_method': 'best eval'}
    ):
        
        if (self.rescale_radius and bounds):
            new_rad = rescale_radius(self.init_radius, bounds)
            self.init_radius = new_rad

        if self.dim_red is not None:
            if self.dim_red=='PLS':
                from CUATRO.optimizer.implementations.dim_red_heuristics.CUATRO_PLS import CUATRO_PLS
                output = CUATRO_PLS().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, n_pls, prior_evals) 
            elif self.dim_red=='explore':
                from CUATRO.optimizer.implementations.dim_red_heuristics.CUATRO_PLS_explore import CUATRO_PLS_expl
                output = CUATRO_PLS_expl().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, n_pls, n_t, prior_evals)
            elif self.dim_red=='embed':
                from CUATRO.optimizer.implementations.dim_red_heuristics.CUATRO_embedding import CUATRO_embedding
                output = CUATRO_embedding().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, n_pls, prior_evals)
            elif self.dim_red=='bandit':
                from CUATRO.optimizer.implementations.dim_red_heuristics.CUATRO_PLS_bandit import CUATRO_PLS_bandit
                output = CUATRO_PLS_bandit().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, n_pls, n_t, prior_evals)
            else:
                raise NotImplementedError("Not yet implemented. dim_red should be in ['PLS', 'explore', 'embed', 'bandit']")

        elif self.sampling == 'g':
            if self.explore != None:
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.CUATRO_g import CUATRO_g
            output = CUATRO_g().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)
            
        
        elif (self.sampling == 'base') and (self.explore == None):
            from CUATRO.optimizer.implementations.CUATRO_base import CUATRO_base
            output = CUATRO_base().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals) 
                   
        
        elif self.explore == 'feasible_sampling':
            if self.sampling != 'base':
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.local_heuristics.CUATRO_feasible_sampling import CUATRO_feas_samp
            output = CUATRO_feas_samp().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)
            
        
        elif self.explore == 'exploit_explore':
            if self.sampling != 'base':
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.local_heuristics.CUATRO_exploit_explore import CUATRO_expl_expl
            output = CUATRO_expl_expl().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)
        
        
        elif self.explore == 'sampling_region':
            if self.sampling != 'base':
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.local_heuristics.CUATRO_sampling_region import CUATRO_sampling_region
            output = CUATRO_sampling_region().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)
        
        
        elif self.explore == 'TIS':
            if self.sampling != 'base':
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.global_heuristics.CUATRO_TIS import CUATRO_TIS
            output = CUATRO_TIS().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)
        

        elif self.explore == 'TIP':
            if self.sampling != 'base':
                raise ValueError("Exploration heuristics were developed to be used with MCD implemented (set 'sampling': 'base' in custom_params dictionary to use heuristics or set 'explore': None in custom_params to sample without MCD)")
            
            from CUATRO.optimizer.implementations.global_heuristics.CUATRO_TIP import CUATRO_TIP
            output = CUATRO_TIP().optimise(sim, x0, constraints, bounds, max_f_eval, rnd, prior_evals)

        
        return output