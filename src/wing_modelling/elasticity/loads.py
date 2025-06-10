import matplotlib.pyplot as plt
import numpy as np

import wing_modelling.discretization.midpoint as discretization

def compute_static_loads(y, q_y, w_y = None):
    """Compute shear and moment loads on the wing."""
    
    if w_y is None:
        # add a load at the center to get force balance
        w_y = np.zeros(y.shape)
        w = discretization.integrate(q_y, y)[-1]
        
        idx = np.argmax(y >= 0)
        if y[idx] == 0:
            w_y[idx] = -w
        else:
            w_y[idx-1] = -w/2
            w_y[idx] = -w/2
    
    dy = discretization.difference(y)
    force_y = q_y + w_y/dy
    shear_y = discretization.integrate(force_y, y)
    moment_y = discretization.integrate(shear_y, y)
    
    return moment_y, shear_y, force_y


def plot(y, q_y = None, shear_y = None, moment_y = None):
        if q_y is not None:
            plt.plot(y, q_y, label='q')
        if shear_y is not None:
            plt.plot(y, shear_y, label='shear')
        if moment_y is not None:
            plt.plot(y, moment_y, label='moment')
        plt.xlabel('y [m]')
        plt.ylabel('q [N/m], shear [N], moment [Nm]')
        plt.grid(True)
        plt.legend()
        plt.title('Loads Diagram')
        return plt
