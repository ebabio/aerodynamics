"""This script computes the ideal lift distribution along the wingspan resulting from ignoring vortex effects."""

import math
import numpy as np
import matplotlib.pyplot as plt

import lifting_line.aero.lifting_line as lifting_line
import lifting_line.discretization.midpoint as discretization


# Flight condition
alpha_0 = math.radians(10) # degrees
velocity = 12 # m/s
density = 1.225 # kg/m^3
q_bar = 0.5 * density * velocity**2

# Geometry
b = 72 # m
y = np.linspace(-b/2, b/2, 1001)

def chord_skd(y):
    y_q = [b/2-11.35, b/2]
    c_q = [8.935-5.000, 8.147-5.451]
    
    chord_y = np.interp(abs(y), y_q, c_q, left=c_q[0], right=0)
    return chord_y
def alpha_skd(y):    
    alpha_y = np.zeros(y.shape)
    return alpha_y

alpha_y = alpha_skd(y) + alpha_0
chord_y = chord_skd(y)
surface = discretization.integrate(chord_y, y)[-1]

# Compute lift
circulation_y = lifting_line.compute_circulation_ideal_y(alpha_y, chord_y, velocity)
lift_y = lifting_line.compute_lift_y(density, velocity, circulation_y)
lift = discretization.integrate(lift_y, y)[-1]

# Alternate computation
lift_alt = q_bar * surface * (2 * math.pi * alpha_0)

# Display results
plt.plot(y, lift_y)
plt.xlabel('y')
plt.ylabel('Lift (N/m)')
plt.title('Ideal Lift Distribution')
plt.suptitle(f'velocity = {velocity:.1f} m/s, alpha = {math.degrees(alpha_0):.1f} deg, lift = {lift:.0f} N')
plt.show()
    