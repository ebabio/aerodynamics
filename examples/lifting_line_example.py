"""This script computes the ideal lift distribution along the wingspan resulting from ignoring vortex effects."""

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import wing_modelling.aero.lifting_line as lifting_line
import wing_modelling.discretization.midpoint as discretization
import aircraft_example

#Input
# Flight condition
alpha_0 = math.radians(10) # degrees
velocity = 10 # m/s
density = 1.225 # kg/m^3
q_bar = 0.5 * density * velocity**2

# Geometry
b = aircraft_example.geometry_example()['b']
y = np.linspace(-b/2, b/2, 1000)

chord_y = aircraft_example.tapered_chord(y) 
alpha_y= aircraft_example.untwisted_alpha(y) + alpha_0
surface = discretization.integrate(chord_y, y)[-1]

#Compute lift
lift_y, drag_y, induced_alpha_y = \
    wing_modelling.compute_lift_y(alpha_y, chord_y, y, velocity, b_scale=1.0)
lift_y_ideal, drag_y_ideal, _  = \
    wing_modelling.compute_lift_y(alpha_y, chord_y, y, velocity, b_scale=np.inf)
lift = discretization.integrate(lift_y, y)[-1]
drag = discretization.integrate(drag_y, y)[-1]
lift_ideal = discretization.integrate(lift_y_ideal, y)[-1]
cl_alpha = lift / (q_bar * surface) / alpha_0

# Theoretical results
cl_alpha_ideal = 2 * math.pi
aspect_ratio = b**2 / surface
efficiency = .9 # Oswald efficiency
cl_alpha_theory = cl_alpha_ideal/(1 + cl_alpha_ideal / (math.pi * efficiency * aspect_ratio))
cl_theory = cl_alpha_theory * alpha_0
cd_i_theory = cl_theory**2 / (math.pi * efficiency * aspect_ratio)

#Output results
output_path = Path(__file__).parent / '../doc/input'

# Lift and drag distribution
plt.plot(y, lift_y_ideal,   label='Infinite Span Lift Distribution')
plt.plot(y, -drag_y_ideal,  label='Infinite Span Drag Distribution')
plt.plot(y, lift_y,         label='Finite Span Lift Distribution')
plt.plot(y, -drag_y,        label='Finite Span Drag Distribution')
plt.xlabel('y [m]')
plt.ylabel('Lift [N/m]')
plt.title('Spanwise Lift Distribution')
plt.suptitle(f'velocity = {velocity:.1f} m/s, alpha = {math.degrees(alpha_0):.1f} deg')
plt.grid(True)
plt.legend()
plt.savefig(output_path / 'ex_lift_drag_distribution.png')
plt.clf()

# Induced angle of attack
plt.plot(y, np.vectorize(math.degrees)(induced_alpha_y))
plt.xlabel('y [m]')
plt.ylabel('Induced AoA [deg]')
plt.title('Spanwise Induced AoA Distribution')
plt.grid(True)
plt.savefig(output_path / 'ex_induced_alpha.png')
plt.clf()

# Performance values
performance_values = {
    'lift': lift,
    'drag': drag,
    'cl_alpha': cl_alpha,
    'lift_to_drag': lift/drag,
    'efficiency': efficiency,
    'cl_alpha_theory': cl_alpha_theory,
    'lift_to_drag_theory': cl_theory / cd_i_theory
}
with open(output_path / 'ex_performance_values.tex', 'w') as f:
    for key, value in performance_values.items():
        f.write(f"\\def\\{key.replace("_", "")}{{{value:.2f}}}\n")
    