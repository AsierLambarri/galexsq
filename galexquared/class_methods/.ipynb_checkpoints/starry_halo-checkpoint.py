#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:47:22 2024

@author: asier
"""

import numpy as np
from unyt import unyt_array, unyt_quantity
from scipy.optimize import root_scalar
from scipy.stats import binned_statistic
from copy import copy

from .center_of_mass import center_of_mass_pos

import pprint
pp = pprint.PrettyPrinter(depth=4)



def zero_disc(x, y):
    """Find the root of enclosed mass distributions that are discontinuous, arising from a discrete nature of the set 
    of particles (forming a galaxy) in a halo. Only works for one independent variable. scipy's root_scalar works well
    but is sometimes unreliable.

    Because of the discontinuous nature of y, the root may not be exact. This function always returns the closest value to
    the root.

    Parameters
    ----------
    x : array
        Array of independent variable. Usually radius. Can have units.
    y : array
        Array of dependent variable. Usually enclosed mass. Can have units.


    Returns
    -------
    root : float
        Root of the function. Nan means the root hasn't been found.
    """
    root = np.nan
    mask = y <= 0

    if True in mask:
        if np.abs(y[mask][-1]) < np.abs(y[~mask][0]):
            root = x[mask][-1]
        else:
            root = x[~mask][0] 
    else:
        pass
    return root
    
def encmass(pos, mass, r0, cm = None):
    """Given the massess and positions of a set of particles it computes the enclosed mass
    at each particles location M(<r_p). By default, it does so w.r.t. the cm of the massess.
    Alternativelly, another cm may be provided (halo, cm of a set of particles, ...)

    Parameters
    ----------
    pos : array
        Array of positions. Shape (N,3).
    mass : array
        Array of massess. Shape (N,).
    cm : array, optional
        Alterrnative cm position. Shape (3,). Default: cm of (pos, mass).

    Returns
    -------
    rp : array
        Spherical radial distance from the cm. Shape (N,).
    encmass : array
        Enclosed maass at each value of rp. Shape (N,).
    """
    if cm:
        pass
    else:
        cm = center_of_mass_pos(pos, mass)

    relpos = pos - cm
    r = np.sqrt(relpos[:,0]**2 + relpos[:,1]**2 + relpos[:,2]**2)
    if 0 in r:
        pass
    else:
        rbins = np.insert(np.sort(r), 0, unyt_quantity(0, r.units))

    print(rbins, r, mass)
    binmass, rp, _ = binned_statistic(r, mass, bins = np.unique(rbins), statistic="sum")

    return rp[1:], np.cumsum(binmass)




def compute_stars_in_halo(pos,
                          masses,
                          vels,
                          pindices,
                          halo_params,
                          max_radius=(30, 'kpc'), 
                          nmin=6,
                          imax=200,
                          verbose=False,
                          **kwargs
                          ):
    """Computes the stars that form a galaxy inside a given halo using the recipe of Jenna Samuel et al. (2020). 
    For this one needs a catalogue of halos (e.g. Rockstar). The steps are the following:

        1. All the stars inside the min(0.8*Rvir, 30) kpccm of the host halo are considered as 
           candidates to form the given galaxy.
        2. The stars with relative speed bigger than 2*V_{circ, max} (usually a quantity computed in the
           catalogues) are removed.
        3. An iterative process is started where:
              (i) All the particles outside of 1.5*R90 (radius where m(R)/M_T = 0.9) are removed. We take into 
                  account both the stars cm and the halo center.
             (ii) We remove all the particles that have velocities 2*sigma above the mean.
            (iii) A convergence criterion of deltaM*/M*<0.01 is stablished.

        4. We only keep the galaxy if it has more than six stellar particles.

    Parameters
    ----------
    pos, masses, vels : array-like[float] with units
        Absolute position, mass and velocity of particles.
    pindices : array[int]
        Indices of particles.
    halo_params : dict[str : unyt_quantity or unyt_array]
        Parameters of the halo in which we are searching: center, center_vel, Rvir, vmax and vrms.
    max_radius : float, optional
        Maximum radius to consider for particle unbinding. Default: 30 kpc.
    nmin : int
        Minimum number of stellar particles to consider an ensemble a galaxy. Default: 6
    imax : int, optional
        Maximum number of iterations. Default 200.
    verbose : bool
        Wether to verbose or not. Default: False.
        
    Returns
    -------
    indices : array
        Array of star particle indices belonging to the halo.
    mask : boolean-array
        Boolean array for masking quantities
    delta_rel : float
        Obtained convergence for selected total mass after imax iterations. >1E-2.
    """
    if len(masses) == 0:
             return np.array([]), np.zeros_like(masses, dtype=bool), np.nan
         
    halo_center = halo_params['center'].in_units(pos.units)
    halo_center_vel = halo_params['center_vel'].in_units(vels.units)
    halo_Rvir = halo_params['rvir'].in_units(pos.units)
    halo_vmax = halo_params['vmax'].in_units(vels.units)
    halo_vrms = halo_params['vrms'].in_units(vels.units)
    max_radius = unyt_quantity(*max_radius).in_units(pos.units)

    vmax_lim = kwargs.get("fmax", 2)
    std_lim = kwargs.get("fstd", 2)
    fmass = kwargs.get("fmass", 0.9)
    frad = kwargs.get("frad", 2.5)

    halorel_positions = (pos - halo_center)
    halorel_velocities = (vels - halo_center_vel)
    
    halorel_absvel = np.sqrt(halorel_velocities[:,0]**2 + halorel_velocities[:,1]**2 + halorel_velocities[:,2]**2)
    halorel_R = np.sqrt(halorel_positions[:,0]**2 + halorel_positions[:,1]**2 + halorel_positions[:,2]**2)

    mask_vmax = (halorel_absvel < vmax_lim*halo_vmax)
    mask_rad = (halorel_R <= np.minimum(0.8 * halo_Rvir, max_radius))                                        
    mask_loop = copy(mask_vmax) & mask_rad                                                                   

    
    if verbose:
        print("\nHalo center:")
        pp.pprint(halo_center)
        print("\nHalo velocity:")
        pp.pprint(halo_center_vel)
        print(f"\nHalo virial radius: {halo_Rvir:.4f}")
        print(f"Halo maximum Vcirc: {halo_vmax:.4f}")
        print(f"Halo Vrms: {halo_vrms:.4f}")
        print(f"Stellar mass inside {np.minimum(0.8 * halo_Rvir, max_radius):3f}: {masses[mask_rad].sum()}. Total of {len(masses[mask_rad])} particles.")
        print(f"Accepted by Vmax criterion: {masses[mask_loop].sum()}. Total of {len(masses[mask_loop])} particles.")

    if np.count_nonzero(mask_loop)==0:
        if verbose:
            print("Interations terminated on 0. No coherent star structures where found.")
                
        return np.array([]), np.zeros_like(masses, dtype=bool), np.nan

    delta_mm = []
    for i in range(imax):
        old_mmass = masses[mask_loop].sum()
        cm = np.average(pos[mask_loop], axis=0, weights=masses[mask_loop])
        vcm = np.average(vels[mask_loop], axis=0, weights=masses[mask_loop])

        cmrel_positions = pos - cm
        cmrel_velocities = vels - vcm
        cmrel_R = np.sqrt(cmrel_positions[:,0]**2 + cmrel_positions[:,1]**2 + cmrel_positions[:,2]**2)
        cmrel_absvel = np.sqrt(cmrel_velocities[:,0]**2 + cmrel_velocities[:,1]**2 + cmrel_velocities[:,2]**2)

        mean = np.sqrt(np.sum(cmrel_absvel.mean(axis=0)**2))
        std = np.sqrt(np.sum(cmrel_absvel.std(axis=0)**2))

        mask_sigmavel = (np.abs(cmrel_absvel - mean) < std_lim*std)

        hzero = root_scalar(lambda r: masses[mask_loop & (halorel_R < r)].sum()/masses[mask_loop].sum() - fmass, method="brentq", bracket=[0, halo_Rvir])


        R90h = hzero.root * halorel_R.units
        mask_R90h = (halorel_R < frad*R90h)
        
        cmzero = root_scalar(lambda r: masses[mask_loop & (cmrel_R < r)].sum()/masses[mask_loop].sum() - fmass, method="brentq", bracket=[0, halo_Rvir])


        R90cm = cmzero.root * halorel_R.units
        mask_R90cm = (cmrel_R < frad*R90cm)
    
        mask_loop = mask_vmax & mask_R90cm & mask_R90h & mask_sigmavel
        new_mmass = masses[mask_loop].sum()

        if np.count_nonzero(mask_loop)==0:
            if verbose:
                print(f"Interations terminated on {i+1}. No coherent star structures where found.")
                
            return np.array([]), np.zeros_like(masses, dtype=bool), np.nan


        n_init = len(masses[mask_loop])
        delta_rel = np.abs(new_mmass - old_mmass)/old_mmass

        delta_mm.append(np.round(delta_rel, 2))            
        
        if verbose:
            print(f"\n\n### {i}-th iteration ###")
            print(f"\nCM: {cm}, Vcm: {vcm}")
            print(f"sigma_vel: {std}")
            print("\n  Halo R90      CM R90  ")
            print(f"  {R90h:.3e}  {R90cm:.3e}")
            print(f"\nOld mass: {old_mmass:.3f}, New mass: {new_mmass:.3f} ")
            print(f"Rel-Delta mass: {np.abs(new_mmass - old_mmass)/old_mmass:.2e}")
            print("\nroot_scalar diagnostics")
            print("Halo relative")
            print("-------------")
            print(hzero)
            print("CM relative")
            print("-----------")
            print(cmzero)
            
        if hzero.flag != "converged":
            raise Exception(f"Could not converge on halo-centered R90 on iteration {i}.")
        if cmzero.flag != "converged":
            raise Exception(f"Could not converge on cm-centered R90 on iteration {i}.")
        if delta_rel < np.maximum(0.01, 1/n_init):
            definite_mask = copy(mask_loop)
            indices = pindices[definite_mask]
            return indices, definite_mask, delta_rel
        if np.count_nonzero(mask_loop) < nmin:    
            definite_mask = copy(mask_loop)
            indices = pindices[definite_mask]
            return indices, definite_mask, delta_rel
        if i >= 2:
            if delta_mm[i] == delta_mm[i-2]:
                definite_mask = copy(mask_loop)
                indices = pindices[definite_mask]
                return indices, definite_mask, delta_rel

    
    definite_mask = copy(mask_loop)
    indices = pindices[definite_mask]
    
    print(f"No convergence. Iterations terminated. Initial mass {masses[mask_vmax].sum()}. Current mass {masses[definite_mask].sum()}.")
    print(f"Rel-Delta mass is {delta_rel}.\n")
    
    return indices, definite_mask, delta_rel
