def rosenbrock_g1(x):
    '''
    Rosenbrock cubic constraint
    g(x) <= 0 
    '''
    return (x[0] - 1)**3 - x[1] + 1


def rosenbrock_g2(x):
    '''
    Rosenbrock linear constraint
    g(x) <= 0 
    '''
    return x[0] + x[1] - 1.8