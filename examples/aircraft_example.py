import numpy as np


def geometry_example(params = None):
    if params is None:
        params = {}
    default_params = {
        'b': 70,
        'c': 4,
        'b_tapering_ratio': 1/3,
        'c_tip_ratio': 0.6,
    }
    params = default_params | params
    return params

def tapered_chord(y, params = None):
    if params is None:
        params = geometry_example()
    
    y_q = params["b"] * np.array([1/2*(1-params["b_tapering_ratio"]), 1/2])
    c_q = params["c"] * np.array([1, params["c_tip_ratio"]])
    
    chord_y = np.interp(abs(y), y_q, c_q, left=c_q[0], right=0)
    return chord_y

def untwisted_alpha(y):
    alpha_y = np.zeros(y.shape)
    return alpha_y
