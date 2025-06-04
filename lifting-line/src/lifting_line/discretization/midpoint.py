import numpy as np
from scipy.linalg import toeplitz

def isvector(x: np.array) -> bool:
    """Check if an array is a vector"""
    non_one_dimensions = 0
    for dimension in x.shape:
        if dimension != 1:
            non_one_dimensions += 1
    return non_one_dimensions == 1

def enforce_column_vector(x: np.array) -> np.array:
    """Convert a vector to a column vector"""
    if x.ndim == 1:
        return np.atleast_2d(x).T
    if x.ndim == 2 and x.shape[1] != 1:
        return x.T

def difference(x: np.array) -> np.array:
    """Calculate the midpoint difference between consecutive values in an array.
    
    Delta x[i] = (x[i+1] - x[i-1]) / 2
    
    Using the midpoint difference returns an array of the same size as the input array."""
    assert isvector(x)
    
    dx = np.full(x.shape, np.nan)
    dx[0] = (x[1] - x[0]) / 2
    dx[-1] = (x[-1] - x[-2]) / 2
    dx[1:-1] = (x[2:] - x[:-2]) / 2
    return dx

def difference_operator(x: np.array) -> np.array:
    """Return the difference operator using a midpoint discretization.
    
    The difference operator D returns the difference if multiplied against the column vector:
    
    Delta f = D * f
    """
    assert isvector(x)
    nx = x.size
    D = toeplitz(
        np.concatenate(([0, -1/2], np.zeros([nx-2]))),
        np.concatenate(([0, +1/2], np.zeros([nx-2])))
    )
    D[0, 0] = -1/2
    D[-1, -1] = 1/2
    return D
    
def derivative(f: np.array, x: np.array) -> np.array:
    """Calculate the derivative of a function using the midpoint rule."""
    assert isvector(f)
    assert isvector(x)
    assert f.shape == x.shape
    
    df = difference(f)
    dx = difference(x)
    return df / dx

def derivative_operator(x: np.array) -> np.array:
    """Return the derivative operator given a midpoint discretization.
    
    The derivative operator D returns the derivative if multiplied against the  column vector:
    
    f' = D * f
    """
    x = enforce_column_vector(x)
    assert isvector(x)
    D = difference_operator(x)
    return D/(D@x)
    
def integrate(f: np.array, x: np.array) -> np.array:
    """Integrate a function using the midpoint rule."""
    assert isvector(f)
    assert isvector(x)
    assert f.shape == x.shape
    
    dx = difference(x)
    return np.cumsum(f * dx)

def integrate_cauchy(f: np.array, x: np.array, x_singular: np.array) -> np.array:
    """Integrate a function using the midpoint rule allowing singular points using the Cauchy principal value.
    
    Only the intermediate singular points are allowed now with no support for endpoints."""
    assert isvector(f)
    assert isvector(x)
    assert f.shape == x.shape
    
    if x_singular.dtype != bool:
        x_singular = np.isin(x, x_singular)
    f[x_singular] = 0
    
    return integrate(f, x)


    