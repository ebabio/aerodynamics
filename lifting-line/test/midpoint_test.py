import math
import numpy as np
import pytest

import lifting_line.discretization.midpoint as discretization

def x_range():
    x_range = np.array([-1, 2])
    n = 1000
    x = np.linspace(x_range[0], x_range[1], n)
    return x

@pytest.fixture
def derivative_params(request):
    x = x_range()
    params = {}
    if request.param == 'x':
        params["x"]             = x
        params["f"]             = x
        params["df_expected"]   = 1
        params["tolerance"]     = 1e-6
    elif request.param == 'x_squared':
        params["x"]             = x
        params["f"]             = x**2
        params["df_expected"]   = 2*x
        params["tolerance"]     = 1e-2
    return params

@pytest.mark.parametrize(
    "derivative_params", ["x", "x_squared"], indirect=True
)
def test_derivative(derivative_params):
    df = discretization.derivative(derivative_params["f"], derivative_params["x"])
    assert(abs(df - derivative_params["df_expected"]).max() < derivative_params["tolerance"])


@pytest.mark.parametrize(
    "derivative_params", ["x", "x_squared"], indirect=True
)
def test_difference_operator(derivative_params):
    D = discretization.derivative_operator(derivative_params["x"])
    df = D @ derivative_params["f"]
    assert(abs(df - derivative_params["df_expected"]).max() < derivative_params["tolerance"])


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
