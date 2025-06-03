import numpy as np



def _kepler_potential(r, mvir, G):
    return  -G * mvir / (r + 1E-5)


def _nfw_potential(r, mvir, rs, c, G):
    x = r / rs
    A_nfw = np.log(1 + c) - c / (1 + c)
    return -G * mvir * rs / A_nfw * np.log(1 + x) / x 

def potential(r, mode="kepler", **kwargs):
    if mode == "kepler":
        mvir, G = kwargs["mass"], kwargs["G"]
        return _kepler_potential(r, mvir, G)
    if mode == "nfw":
        mvir, rs, c, G = kwargs["mass"], kwargs["rs"], kwargs["c"], kwargs["G"]
        return _nfw_potential(r, mvir, rs, c, G)