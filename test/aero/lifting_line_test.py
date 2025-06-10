import math
import matplotlib.pyplot as plt
import numpy as np
import pytest

import wing_modelling.aero.lifting_line as lifting_line
import wing_modelling.discretization.midpoint as discretization



def test_ellitic_wing(plot = False):
    """Check the result of the lifting line solver using a elliptic wing.
    
    We'll use a spitfire in a cruise condition to check the result."""
    
    # Flight condition
    alpha_0 = math.radians(5)
    velocity = 130
    density = 1.225 # At about 20,000ft

    # Aircraft geometry
    aspect_ratio = 10
    b = 11
    surface = 22
    
    y = np.linspace(-b/2, b/2, 1000)
    unit_chord_y = np.vectorize(math.sqrt)(1 - (2*y/b)**2)
    mean_chord = surface/discretization.integrate(unit_chord_y, y)[-1]
    chord_y = mean_chord * unit_chord_y
    alpha_y = np.full(chord_y.shape, alpha_0)

    # Compute results
    lift_y, drag_y, induced_alpha_y = \
    wing_modelling.compute_lift_y(alpha_y, chord_y, y, velocity)
    
    q_bar = 0.5 * density * velocity**2
    cl = discretization.integrate(lift_y, y)[-1] / (q_bar * surface)
    cd_i = discretization.integrate(drag_y, y)[-1] / (q_bar * surface)
    
    # Theoretical results
    cl_alpha_ideal = 2 * math.pi
    aspect_ratio = b**2 / surface
    efficiency = 1 # Oswald efficiency
    cl_alpha_theory = cl_alpha_ideal/(1 + cl_alpha_ideal / (math.pi * efficiency * aspect_ratio))
    cl_theory = cl_alpha_theory * alpha_0
    cd_i_theory = cl_theory**2 / (math.pi * efficiency * aspect_ratio)
    
    # Output results
    if plot:
        # Lift and drag distribution
        plt.plot(y, lift_y,         label='Finite Span Lift Distribution')
        plt.plot(y, -drag_y,        label='Finite Span Drag Distribution')
        plt.xlabel('y [m]')
        plt.ylabel('Lift [N/m]')
        plt.title('Spanwise Lift Distribution')
        plt.suptitle(f'velocity = {velocity:.1f} m/s, alpha = {math.degrees(alpha_0):.1f} deg')
        plt.grid(True)
        plt.legend()
        plt.show()
        plt.clf()

        # Induced angle of attack
        plt.plot(y, np.vectorize(math.degrees)(induced_alpha_y))
        plt.xlabel('y [m]')
        plt.ylabel('Induced AoA [deg]')
        plt.title('Spanwise Induced AoA Distribution')
        plt.grid(True)
        plt.show()
        plt.clf()
        
    # Test results    
    assert np.isclose(cl,   cl_theory,      rtol=1e-2)
    assert np.isclose(cd_i, cd_i_theory,    rtol=1e-2)
    
if __name__ == "__main__":
    pytest.main([__file__])
    
