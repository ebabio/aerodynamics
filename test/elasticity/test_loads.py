import matplotlib.pyplot as plt
import numpy as np
import pytest

import wing_modelling.elasticity.loads as loads

def test_static_loads(plot = False):
    b = 10
    y = np.linspace(-b/2, b/2, 1000)
    
    q_y = np.full(y.shape, 1.0)
    moment_y, shear_y, force_y = loads.compute_static_loads(y, q_y)
    
    if plot:
        loads.plot(y, q_y = q_y, shear_y = shear_y, moment_y = moment_y)
        plt.show()
    
    assert np.isclose(shear_y[-1],  0.0, atol = 1e-3)
    assert np.isclose(moment_y[-1], 0.0, atol = 1e-3)
    
if __name__ == "__main__":
    pytest.main([__file__])