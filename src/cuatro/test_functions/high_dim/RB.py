import numpy as np
from scipy.stats import ortho_group

class RB:
    def __init__(self, n_high, seed = 0):
        np.random.seed(seed)
        self.dim = n_high
        self.Q = ortho_group.rvs(n_high)
        
    def rosenbrock_(self, x):
        x = np.array(x).squeeze()
        assert x.ndim == 1
        return 100*(x[1]-x[0]**2)**2 + (x[0]-1)**2
    
    def rosenbrock_higher(self, x):
        # Q is used to 'mix up' the fake and effective dimensions
        x = np.array(x).squeeze()
        assert x.ndim == 1
        x = x.reshape(-1,1)
    
        new_x = (self.Q @ x).squeeze()
        return self.rosenbrock_(new_x)
    


     
    
