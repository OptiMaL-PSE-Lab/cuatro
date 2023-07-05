from functools import partial
import numpy as np

def RB_decoupled(dim, x):
    it = int(dim/2)
    return np.sum([100*(x[2*i-2]**2 - x[2*i-1])**2+(x[2*i-2]-1)**2 for i in range(1,it)])

dim_decoupled = [2, 4, 6, 8, 10, 20, 50, 100]

RB_list = [partial(RB_decoupled, dim) for dim in dim_decoupled]
