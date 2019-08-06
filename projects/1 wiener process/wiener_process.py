import numpy as np

def wiener_process(initial_state, tt):
    n = len(tt)
    t_min, t_max = tt[0], tt[n-1]
    dt = (t_max - t_min) / (n - 1)
    dimension = len(initial_state)
    
    solution = [None for _ in range(n)]
    solution[0] = initial_state
    
    for i in range(1, n):
        solution[i] = solution[i-1] + np.sqrt(dt)*np.random.normal(0, 1, dimension)
        pass
    return solution