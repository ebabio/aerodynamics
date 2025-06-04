import math

def compute_circulation_ideal_y(alpha_y, chord_y, velocity):
    """Compute the ideal circulation along the wingspan. Ideal in the sense of disregarding vortex effects."""
    circulation_y = 2 * math.pi * chord_y * velocity * alpha_y
    return circulation_y

def compute_lift_y(density, velocity, circulation_y):
    """Compute the lift distrbution along the wingspan."""
    lift_y = density * velocity * circulation_y
    return lift_y