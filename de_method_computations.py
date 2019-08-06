import numpy as np
from numpy import linalg

"""
Computing diagonalizable differential equations:
Given an invertible matrix P, P_inv, and an eigenvalue array D, we can solve for the system of differential equations
    X' = P.diag[D].P_inv.X
Using the formula,
    X[t] = P.diag[Exp[D.t]].P_inv.X[0]
As well as find an iterative method of computing X[t],
    X[t + dt] = P.diag[Exp[D.dt]].P_inv.X[t]
"""

def construct_solver(P, D, C = None):
    """
    Arguments:
        - P: an n×n invertible matrix.
        - D: an n-size array of eigenvalues.
        - C: an n-size array.
    Returns two functions:
        - construct_solution: constructs an exact solution.
        - construct_iterator: constructs the discrete iterator.
    """
    
    # compute inverse of P and diagonal matrix.
    P_inverse = linalg.inv(P)
    diagonal = lambda t: np.diag(np.exp(t*D))
    
    def construct_solution(initial_condition):
        
        if isinstance(C, type(None)):
            def solution(tt):
                return np.array([P.dot(diagonal(t).dot(P_inverse.dot(initial_condition))) for t in tt])
            return solution
        else:
            def solution(tt):
                return np.array([P.dot(diagonal(t).dot(P_inverse.dot(initial_condition))) + (t*C).flatten() for t in tt])
            return solution
        pass

    def construct_iterator(tt):
        # compute timestep.
        n = len(tt)
        t_min, t_max = tt[0], tt[n - 1]
        dt = (t_max - t_min) / (n - 1)

        # compute relevant diagonal matrix.
        discrete_diagonal = diagonal(dt)
        
        # cases for the main algorithm.
        if isinstance(C, type(None)):
            def iterate(x, t):
                return P.dot(discrete_diagonal.dot(P_inverse.dot(x)))
            pass
        else:
            def iterate(x, t):
                return P.dot(discrete_diagonal.dot(P_inverse.dot(x - (t*C).flatten()))) + ((t + dt)*C).flatten()
            pass
        
        def solve(initial_condition):
            solution = [None for _ in range(n)]
            solution[0] = initial_condition
            for i in range(1, n):
                solution[i] = iterate(solution[i-1], tt[i])
                pass
            return np.array(solution)
        return solve
    return construct_solution, construct_iterator