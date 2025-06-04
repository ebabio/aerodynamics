import math
import numpy as np
import pytest

import lifting_line.discretization.midpoint as discretization

def test_derivative_x():
    x_range = np.array([-1, 2])
    x = np.linspace(x_range[0], x_range[1], 1000)
    f = x
    
    df = discretization.derivative(f, x)
    df_expected = 1
    tolerance = 1e-6
    assert(abs(df - df_expected).max() < tolerance)
    
def test_derivative_polynomial():
    x_range = np.array([-1, 2])
    x = np.linspace(x_range[0], x_range[1], 1000)
    f = x ** 2
    
    df = discretization.derivative(f, x)
    df_expected = 2 * x
    tolerance = 1e-2
    assert(abs(df - df_expected).max() < tolerance)

def test_cauchy_integration():
    x_range = np.array([-1, 2])
    x = np.linspace(x_range[0], x_range[1], 1000)
    f = np.divide(1, x, out=np.zeros_like(x), where=x!=0) # nice way of handling the singular values
    x_singular = np.array([0])
    
    F_indefinite = discretization.integrate_cauchy(f, x, x_singular)
    F_definite = F_indefinite[-1] - 0
    F_expected = math.log(x_range[1])-math.log(-x_range[0])
    tolerance = 1e-6
    assert(abs(F_definite - F_expected) < tolerance)

if __name__ == "__main__":
    pytest.main()
