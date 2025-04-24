#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:17:24 2024

@author: asier
"""

import numpy as np
from scipy.optimize import root_scalar

def half_mass_radius(pos, 
                     mass, 
                     center=None,
                     mfrac=0.5,
                     project=False
                    ):
    """By default, it computes half mass radius of a given particle ensemble. If the center of the particles 
    is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
    only the particles inside r < 0.5*rmax.

    There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
    is computed via rootfinding using scipy's implementation of brentq method.

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    center : array, optional
        Center of mass position. If not provided it is estimated as explained above.
    mfrac : float, optional
        Mass fraction of desired radius. Default: 0.5 (half, mass radius).

    Returns
    -------
    MFRAC_mass_radius : float
        Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
    """
    if project and pos.shape[1] < 3:
        raise Exception(f"To get a projected estimate of {mfrac:.2f}-mass-radius the data must be three dimensional!")
    if mfrac > 1:
        raise Exception(f"Mass fraction MFRAC must be between 0 and 1! Your input was {mfrac:.2f} > 1")
    if mfrac < 0:
        raise Exception(f"Mass fraction MFRAC must be between 0 and 1! Your input was {mfrac:.2f} < 0")
        
    if center is not None:
        pass
    else:
        center = np.median(pos, axis=0)
        radii = np.linalg.norm(pos - center, axis=1)
        center = np.average(pos[radii < 0.5 * radii.max()], axis=0, weights=mass[radii < 0.5 * radii.max()])

    
    coords = pos - center
    if project:
        coords = coords[:, 0:2]
        
    radii = np.sqrt(np.sum(coords**2, axis=1))
    
    halfmass_zero = root_scalar(lambda r: mass[radii < r].sum()/mass.sum() - mfrac, method="brentq", bracket=[0, radii.max()])

    if halfmass_zero.flag != "converged":
        raise Exception(f"Could not converge on {mfrac:.2f} mass radius!")
    else:
        try:
            return halfmass_zero.root * coords.units #, center * coords.units
        except:
            return halfmass_zero.root # , center
            

