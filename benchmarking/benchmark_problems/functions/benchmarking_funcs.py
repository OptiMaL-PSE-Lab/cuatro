from functools import partial
import numpy as np
from numpy import sin
from numpy import pi

dims = [2, 4, 6, 8, 10, 20, 50, 100]

# many local minima, https://www.sfu.ca/~ssurjano/levy.html
# usually evaluated [-10, 10] for all dims
def Levy(dim, x):
    d = dim
    w = [(1 + (x[i] - 1)/4) for i in range(d)]

    f = (sin(pi * w[0]))**2 + np.sum([((w[i] - 1)**2 * (1 + 10*(sin(pi*w[i] + 1))**2) + (w[d-1] - 1)**2 * (1 + (sin(2*pi*w[d-1]))**2)) for i in range(d-1)])

    return f

Levy_list = [partial(Levy, dim) for dim in dims]


# plate shaped, https://www.sfu.ca/~ssurjano/zakharov.html
# usually evaluated [-5, 10] for all dims
def Zakharov(dim, x):
    d = dim

    f = np.sum([x[i]**2 for i in range(d)]) + np.sum([(0.5*i*x[i])**2 for i in range(d)]) + np.sum([(0.5*i*x[i])**4 for i in range(d)])

    return f

Zakharov_list = [partial(Zakharov, dim) for dim in dims]


# has steep ridges / drops, https://www.sfu.ca/~ssurjano/michal.html
# usually evaluated [0, pi] for all dims
def Michalewicz(dim, x):
    d = dim
    # m is the parameter defining the steepnes of the valleys and ridges, recommended m = 10; increase for more difficult search
    m = 10

    f = (-1) * np.sum([sin(x[i]) * (sin((i*x[i]**2)/pi))**(2*m) for i in range(d)])

    return f

Michalewicz_list = [partial(Michalewicz, dim) for dim in dims]

# bowl-like in 2D, https://www.sfu.ca/~ssurjano/stybtang.html
# # usually evaluated [-5, 5] for all dims
def Styblinski_Tang(dim, x):
    d = dim

    f = 1/2 * np.sum([(x[i]**4 - 16*x[i]**2 + 5*x[i]) for i in range(d)])

    return f

Styblinski_Tang_list = [partial(Styblinski_Tang, dim) for dim in dims]


def RB_decoupled(dim, x):
    it = int(dim/2)
    return np.sum([100*(x[2*i-2]**2 - x[2*i-1])**2+(x[2*i-2]-1)**2 for i in range(1,it)])

dim_decoupled = [2, 4, 6, 8, 10, 20, 50, 100]

RB_list = [partial(RB_decoupled, dim) for dim in dim_decoupled]




