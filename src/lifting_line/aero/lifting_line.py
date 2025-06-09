import math
import numpy as np

import lifting_line.discretization.midpoint as discretization

def compute_lift_ideal_y(alpha_y, chord_y, y, velocity, cl_alpha_y = None, density = 1.225):
    """Compute the ideal circulation along the wingspan. Ideal in the sense of disregarding vortex effects.
    
    The function allows injecting the lift coefficient as a function of the angle of attack and span position."""
    if cl_alpha_y is None:
        cl_alpha_y = 2*math.pi
            
    circulation_y = 1/2 * chord_y * velocity * cl_alpha_y * alpha_y
    return density * velocity * circulation_y

def compute_lift_y(alpha_y, chord_y, y, velocity, cl_alpha_y = None, density = 1.225, b_scale = 1.0):
    
    if cl_alpha_y is None:
        cl_alpha_y = 2*math.pi
    chord_y[[0,-1]] = 0
    column = discretization.enforce_column_vector
    y = column(y)
    y_diff = y-y.T
    np.fill_diagonal(y_diff, np.inf)
    y_diff = column(y_diff*b_scale)
    induced_velocity_matrix = 1/(4*math.pi*velocity) * ((1/y_diff) @ discretization.difference_operator(y))
    dcirculation_dalpha_ideal = velocity/2 * chord_y * cl_alpha_y
    circulation_y =  \
        np.linalg.inv(np.eye(y.size) - column(dcirculation_dalpha_ideal) * induced_velocity_matrix) @ \
        dcirculation_dalpha_ideal * alpha_y
    lift_y = density * velocity * circulation_y
    induced_alpha_y = induced_velocity_matrix @ circulation_y
    drag_y = -density * velocity * circulation_y * induced_alpha_y
    return lift_y, drag_y, induced_alpha_y
        
    