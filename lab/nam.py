# library for numerical approximation methods.

import numpy as np
# solving function.
def solve(constructor, initial_condition, tt):
    # extract data from tt.
    n = len(tt)
    t_min, t_max = tt[0], tt[n-1]
    h = (t_max - t_min) / (n - 1)
    
    # method function.
    method_function = constructor(h)
    
    # create vector solution set.
    solution = [None for _ in range(n)]
    
    # compute solutions.
    for i, t in enumerate(tt):
        if i == 0:
            solution[0] = initial_condition
            pass
        else:
            solution[i] = method_function(t, solution[i-1])
            pass
        pass
    return solution

# supplementary vectorization function.
def vectorize(ff):
    def vectorized_function(t, x):
        # the problem with this function is that every element in x must have equal dimensions.
        
        return np.array([f(t, x) for f in ff])
    return vectorized_function

# solve by euler method.
def euler_method_constructor(ff):
    # ff: a list of functions for each entry in a vector differential equation.
    # The order of the differential equation is the length of ff and denoted n.
    # The number of arguments in each function of ff must be equal to (n+1).
    
    vectorized_function = vectorize(ff)
    
    def euler_method_stepsize(h):
        def euler_method_function(t, x):
            
            return x + vectorized_function(t, x)*h
        return euler_method_function
    return euler_method_stepsize

# solve by RK4
def runge_kutta_constructor(ff):
    # vectorize function.
    vectorized_function = vectorize(ff)
    
    def runge_kutta_stepsize(h):
        def runge_kutta_method(t, x):
            
            k1 = h*vectorized_function(t, x)
            k2 = h*vectorized_function(t + h/2, x + k1/2)
            k3 = h*vectorized_function(t + h/2, x + k2/2)
            k4 = h*vectorized_function(t + h, x + k3)
            
            return x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
        return runge_kutta_method
    return runge_kutta_stepsize