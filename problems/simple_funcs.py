import numpy as np

"""Define simple functions for testing the optimization performance
e.g. quadratic functions, convex quartic, Rosenbrock function, etc.
"""


def quadratic(x: np.ndarray) -> float:
    """Quadratic function

    Args:
        x (np.ndarray): input

    Returns:
        float: simple quadratic function value
    """
    return np.sum(x**2)


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock function

    Args:
        x (np.ndarray): input

    Returns:
        float: rosebrock function value
    """
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


def convex_quartic(x: np.ndarray, A: np.ndarray = None) -> float:
    """Convex quartic function

    Args:
        x (np.ndarray): input
        A (np.ndarray): matrix, positive-definite. if not given, randomly generated positive-definite matrix

    Returns:
        float: convex quartic function value: 0.1*\sum_{i=1}^n (x_i-0.1)^4 + 0.5*x^T A x + 0.01*\sum_{i=1}^n x_i^2
    """

    if A is None:
        if not hasattr(convex_quartic, 'A'):
            A = np.random.randn(len(x), len(x))
            A = np.matmul(A, A.T)
            A /= np.trace(A)*len(x) # scale A so that the trace = dim    
            convex_quartic.A = A
        else:
            A = convex_quartic.A
    x_shft = x - 0.1
    # x_shft = x
    return 0.1 *np.sum((x_shft) ** 4) + 0.5*np.dot(x_shft, np.dot(A, x_shft)) + 0.01*np.sum((x_shft)**2)
