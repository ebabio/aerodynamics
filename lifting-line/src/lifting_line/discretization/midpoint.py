import numpy as np

def isvector(x: np.array) -> bool:
    """Check if an array is a vector"""
    non_one_dimensions = 0
    for dimension in x.shape:
        if dimension != 1:
            non_one_dimensions += 1
    return non_one_dimensions == 1


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

def derivative(f: np.array, x: np.array) -> np.array:
    """Calculate the derivative of a function using the midpoint rule."""
    assert isvector(f)
    assert isvector(x)
    assert f.shape == x.shape
    
    df = difference(f)
    dx = difference(x)
    return df / dx
    