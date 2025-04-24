import numpy as np
from unyt import unyt_array, unyt_quantity
from pytreegrav import Potential
from copy import deepcopy, copy

from .utils import softmax

import pprint
pp = pprint.PrettyPrinter(depth=4)

def create_subset_mask(E, subset, N):
    """Takes an array of energies E, and finds the N most bound particles that are inside the subset.
    The returned array has the same length as E. Usefull for selecting the most bound particles of a 
    certain type, when performing potential calculations with dm/stars/gas.

    Parameters
    ----------
    E : array
        Array with which to perform the selection.
    subset : array, bool
        Subset of elements that will be taken into account on selection.
    N : int
        Number of selected elements. If N==-1 all bound subset particles are selected.

    Returns
    -------
    mask : array, bool, shape=E.shape
    """
    if N == -1:
        mask = (E < 0) & subset
    else:
        E_flat = E[subset]
        if N > len(E_flat):  
            raise Exception(f"More particles to select than are available! You want {N} but only {len(E_flat)} are on the subset")
        if N > np.count_nonzero(E_flat < 0):  
            raise Exception(f"More particles to select than are bound! You want {N} but only {np.count_nonzero(E_flat < 0)} are bound inside the subset")

        smallest_indices = np.argpartition(E_flat, N)[:N]  
        original_indices = np.where(subset)[0][smallest_indices]    
        mask = np.zeros_like(E, dtype=bool)
        mask[original_indices] = True
    return mask




def bound_particlesBH(pos, 
                      vel, 
                      mass, 
                      softs=None,
                      extra_kin=None,
                      cm=None,
                      vcm=None,
                      refine=False,
                      delta=1E-5,
                      cm_subset=None,
                      weighting="softmax",
                      T=0.20,
                      f=0.1,
                      nbound=32,
                      theta = 0.5,
                      return_cm=False,
                      verbose=False
                     ):
    """Computes the bound particles of a halo/ensemble of particles using the Barnes-Hut algorithm implemented in PYTREEGRAV.
    The bound particles are defined as those with E= pot + kin < 0. The center of mass position is not required, as the potential 
    calculation is done with relative distances between particles. To compute the kinetic energy of the particles the *v_cm*
    is needed: you can provide one or the function make a crude estimates of it with all particles. 

    You can ask the function to refine the unbinding by iterating until the *cm* and *v_cm* vectors converge to within a user specified
    delta (default is 1E-5, relative). This is recomended if you dont specify the *v_cm*, as the initial estimation using all particles
    may be biased for anisotropic particle distributions. If you provide your own *v_cm*, taken from somewhere reliable e.g. Rockstar halo
    catalogues, it will be good enough. Note that if you still want to refine, the *v_cm* you provided will be disregarded after the
    first iteration.

    Typical run times are of 0.5-2s dependin if the result is refined or not. This is similar to the I/O overheads of yt.load, making it reasonable for
    use with a few halos (where you dont care if the total runtime is twice as long because it will be fast enough).

    Unit handling is done with unyt. Please provide arrays as unyt_arrays.
    
    Parameters
    ----------
    pos : array
        Position of the particles, with units. When calculating, units are converted to physical kpc.
    vel : array
        Velocity of the particles in pysical km/s.
    mass : array
        Masses of particles, with units. When calculatig, they are converted to Msun.
    softs : array, optional
        Softening of particles. Not required, results do not differ a lot.
    cm : array, optional
        Placeholder. Doesnt do anything if provided.
    vcm : array, optional
        Initial Center of mass velocity estimation.
    theta : float, optional
        theta parameter fir Barnes Hut algorithm, default is 0.7.
    refine, bool, optional
        Whether to refine the undinding. Default is false.
    delta : float, optional
        Relative tolerance for determined if the unbinding is refined. Default is 1E-5.
    verbose : bool, optional
        Whether to print when running. Work in progrees. default is False.
    weighting : str
        SOFTMAX or MOST-BOUND. Names are self, explanatory.
    nbound : int
        Controls how many particles are used when estimating CoM properties through MOST-BOUND.
    T : int
        Controls how many particles are used when estimating CoM properties through SOFTMAX.
    cm_subset : array, bool
        Boolean array determining which particles are to be used for the center-of-mass determination.
    """
    particle_subset = np.ones_like(mass, dtype=bool) if cm_subset is None else cm_subset

    cm = np.average(pos[particle_subset], axis=0, weights=mass[particle_subset]) if cm is None else cm
    vcm = np.average(vel[particle_subset], axis=0, weights=mass[particle_subset]) if vcm is None else vcm

    softenings = softs.in_units(pos.units) if softs is not None else None

    if verbose:
        print(f"Initial total mass: {mass.sum().to('Msun')}")
        print(f"Number of particles: {len(mass)}")
        print(f"Number of subset particles: {np.count_nonzero(particle_subset)}")
        print(f"Initial (subset) center-of-mass position: {cm}")
        print(f"Initial (subset) center-of-mass velocity: {vcm}")

    
    potential = Potential(
        pos, 
        mass, 
        softenings,
        parallel=True, 
        quadrupole=True, 
        G=unyt_quantity(4.300917270038e-06, "kpc/Msun * (km/s)**2").in_units(pos.units/mass.units * (vel.units)**2),   
        theta=theta
    )

    for i in range(100):
        abs_vel = np.sqrt( (vel[:,0]-vcm[0])**2 +
                           (vel[:,1]-vcm[1])**2 + 
                           (vel[:,2]-vcm[2])**2
                       )
    
        kin = 0.5 * mass * abs_vel**2
        kin += extra_kin.to(kin.units)
        
        pot = mass * unyt_array(potential, vel.units**2)
        E = kin + pot
        
        bound_mask = E < 0
        bound_subset_mask = E[particle_subset] < 0
        
        if np.all(E >= 0):
            return E, kin, pot, unyt_array([np.nan, np.nan, np.nan], pos.units), unyt_array([np.nan, np.nan, np.nan], vel.units)
        if np.all(E[particle_subset] >= 0):
            raise Exception(f"All the subset particles used for center-of-mass computations are unbound!!")
        
        if weighting.lower() == "most-bound":
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_subset_mask), nbound)))
            most_bound_mask = create_subset_mask(E, particle_subset, N)
            
            new_cm = np.average(
                pos[most_bound_mask], 
                axis=0, 
                weights=mass[most_bound_mask]
            )
            new_vcm = np.average(
                vel[most_bound_mask], 
                axis=0, 
                weights=mass[most_bound_mask]
            )  
        elif weighting.lower() == "softmax":
            subset_bound_mask = create_subset_mask(E, particle_subset, -1)
            
            w = E[subset_bound_mask]/E[subset_bound_mask].min()
            
            if T == "adaptative":
                T = np.abs(kin[subset_bound_mask].mean()/E[subset_bound_mask].min())
                
            new_cm = np.average(
                pos[subset_bound_mask], 
                axis=0, 
                weights=softmax(w, T)
            )
            new_vcm = np.average(
                vel[subset_bound_mask], 
                axis=0, 
                weights=softmax(w, T)
            )               
        else:
            raise ValueError("Weighting mode doesnt exist!")

        
        delta_cm = np.sqrt(np.sum((new_cm - cm)**2, axis=0)) / np.linalg.norm(cm) < delta
        delta_vcm =  np.sqrt(np.sum((new_vcm - vcm)**2, axis=0)) / np.linalg.norm(vcm) < delta        


        if verbose:
            print(f"\n\n### {i}-th iteration ###\n")
            if weighting=="softmax":
                print(f"Softmax parameters:")
                print(f"-------------------")
                print(f"   Temperature of softmax: {T}")
                print(f"   Max/Min softmax weight ratio: {w.max()/w.min()}")
            else:
                print(f"N-bound parameters:")
                print(f"-------------------")
                print(f"   Max number of particles: {nbound}")
                print(f"   Particle-fraction f: {f}")
                print(f"   Number of particles used: {N}")

            print(f"\nInfo:")
            print(f"-----")
            print(f"   NEW Center-of-mass position: {new_cm}")
            print(f"   NEW Center-of-mass velocity: {new_vcm}")
            print(f"   NEW Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
            print(f"   NEW Number of bound particles: {len(mass[bound_mask])}")
            print(f"   NEW Number of bound subset particles: {np.count_nonzero(bound_subset_mask)}")

        
        if not refine or (delta_cm and delta_vcm):
            if verbose:
                if not refine:
                    print(f"\nFinal Values:")
                    print(f"-------------")
                    print(f"   FINAL Center-of-mass position: {cm}")
                    print(f"   FINAL Center-of-mass velocity: {vcm}")
                    print(f"   FINAL Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
                    print(f"   FINAL Number of bound particles: {len(mass[bound_mask])}")
                    print(f"   FINAL Number of bound subset particles: {np.count_nonzero(bound_subset_mask)}")
                else:
                    print(f"\nFinal Values:")
                    print(f"-------------")
                    print(f"   FINAL Center-of-mass position: {new_cm}")
                    print(f"   FINAL Center-of-mass velocity: {new_vcm}")
                    print(f"   FINAL Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
                    print(f"   FINAL Number of bound particles: {len(mass[bound_mask])}")
                    print(f"   FINAL Number of bound subset particles: {np.count_nonzero(bound_subset_mask)}")
        
            if return_cm:
                return E, kin, pot, new_cm, new_vcm
            else:
                return E, kin, pot

        cm, vcm = copy(new_cm), copy(new_vcm)





def bound_particlesAPROX(pos, 
                         vel, 
                         mass, 
                         extra_kin=None,
                         cm=None,
                         vcm=None,
                         refine=False,
                         delta=1E-5,
                         cm_subset=None,
                         weighting="softmax",
                         T=0.20,
                         f=0.1,
                         nbound=32,
                         return_cm=False,
                         verbose=False
                        ):
    """Computes the bound particles by approximating the ensemble as a point source for ease of potential calculation. 
    The bound particles are defined as those with E= pot + kin < 0. The center of mass position is required, as the potential 
    calculation is done relative to the cm position with the following expression:
    
                                                pot = -G* Mtot * m_i / |x_i - x_cm|
    
    To compute the kinetic energy of the particles the *v_cm* is needed: you can provide one or the function make a crude estimates of it with all particles. 

    You can ask the function to refine the unbinding by iterating until the *cm* and *v_cm* vectors converge to within a user specified
    delta (default is 1E-5, relative). This is recomended if you dont specify the *v_cm*, as the initial estimation using all particles
    may be biased for anisotropic particle distributions. If you provide your own *v_cm*, taken from somewhere reliable e.g. Rockstar halo
    catalogues, it will be good enough. Note that if you still want to refine, the *v_cm* you provided will be disregarded after the
    first iteration.

    Refining the unbinding is advised. This method is somewhat less reliable than its BH counterpart.

    Typical run times are of 0.01-0.05s dependin if the result is refined or not. This is much faster to the I/O overheads of yt.load, making it reasonable for
    use with a large number of halos (where you do care if the total runtime is twice as long).

    Unit handling is done with unyt. Please provide arrays as unyt_arrays.
    
    Parameters
    ----------
    pos : array
        Position of the particles, with units. When calculating, units are converted to physical kpc.
    vel : array
        Velocity of the particles in pysical km/s.
    mass : array
        Masses of particles, with units. When calculatig, they are converted to Msun.
    soft : array, optional
        Softening of particles. Not required, results do not differ a lot.
    cm : array, optional
        Initial Center of mass velocity estimation.
    vcm : array, optional
        Initial Center of mass velocity estimation.
    refine, bool, optional
        Whether to refine the undinding. Default is false.
    delta : float, optional
        Relative tolerance for determined if the unbinding is refined. Default is 1E-5.
    verbose : bool, optional
        Whether to print when running. Work in progrees. default is False.
    weighting : str
        SOFTMAX or MOST-BOUND. Names are self, explanatory.
    nbound : int
        Controls how many particles are used when estimating CoM properties through MOST-BOUND.
    T : int
        Controls how many particles are used when estimating CoM properties through MOST-BOUND.
    """
    particle_subset = np.ones_like(mass, dtype=bool) if cm_subset is None else cm_subset

    cm = np.average(pos[particle_subset], axis=0, weights=mass[particle_subset]) if cm is None else cm
    vcm = np.average(vel[particle_subset], axis=0, weights=mass[particle_subset]) if vcm is None else vcm

    if verbose:
        print(f"Initial total mass: {mass.sum().to('Msun')}")
        print(f"Number of particles: {len(mass)}")
        print(f"Number of subset particles: {np.count_nonzero(particle_subset)}")
        print(f"Initial (subset) center-of-mass position: {cm}")
        print(f"Initial (subset) center-of-mass velocity: {vcm}")

    G = unyt_quantity(4.300917270038e-06, "kpc/Msun * km**2/s**2").in_units(pos.units/mass.units * (vel.units)**2)
    
    for i in range(100):
        radii = np.sqrt( (pos[:,0]-cm[0])**2 +
                         (pos[:,1]-cm[1])**2 + 
                         (pos[:,2]-cm[2])**2
                       )
        abs_vel = np.sqrt( (vel[:,0]-vcm[0])**2 +
                         (vel[:,1]-vcm[1])**2 + 
                         (vel[:,2]-vcm[2])**2
                       )
    
        kin = 0.5 * mass * abs_vel**2
        kin += extra_kin
        
        pot = -G * mass * mass.sum() / radii
        E = kin + pot
        
        bound_mask = E < 0
        bound_subset_mask = E[particle_subset] < 0

        if np.all(E >= 0):
            return E, kin, pot, unyt_array([np.nan, np.nan, np.nan], pos.units), unyt_array([np.nan, np.nan, np.nan], vel.units)
        if np.all(E[particle_subset] >= 0):
            raise Exception(f"All the subset particles used for center-of-mass computations are unbound!!")
        
        if weighting.lower() == "most-bound":
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_subset_mask), nbound)))
            most_bound_mask = create_subset_mask(E, particle_subset, N)
            
            new_cm = np.average(
                pos[most_bound_mask], 
                axis=0, 
                weights=mass[most_bound_mask]
            )
            new_vcm = np.average(
                vel[most_bound_mask], 
                axis=0, 
                weights=mass[most_bound_mask]
            )  
        elif weighting.lower() == "softmax":
            subset_bound_mask = create_subset_mask(E, particle_subset, -1)
            
            w = E[subset_bound_mask]/E[subset_bound_mask].min()
            
            if T == "adaptative":
                T = np.abs(kin[subset_bound_mask].mean()/E[subset_bound_mask].min())
                
            new_cm = np.average(
                pos[subset_bound_mask], 
                axis=0, 
                weights=softmax(w, T)
            )
            new_vcm = np.average(
                vel[subset_bound_mask], 
                axis=0, 
                weights=softmax(w, T)
            )               
        else:
            raise ValueError("Weighting mode doesnt exist!")
     
     
        delta_cm = np.sqrt(np.sum((new_cm - cm)**2, axis=0)) / np.linalg.norm(cm) < delta
        delta_vcm =  np.sqrt(np.sum((new_vcm - vcm)**2, axis=0)) / np.linalg.norm(vcm) < delta        

        if verbose:
            print(f"\n\n### {i}-th iteration ###\n")
            if weighting=="softmax":
                print(f"Softmax parameters:")
                print(f"-------------------")
                print(f"   Temperature of softmax: {T}")
                print(f"   Max/Min softmax weight ratio: {w.max()/w.min()}")
            else:
                print(f"N-bound parameters:")
                print(f"-------------------")
                print(f"   Max number of particles: {nbound}")
                print(f"   Particle-fraction f: {f}")
                print(f"   Number of particles used: {N}")

            print(f"\nInfo:")
            print(f"-----")
            print(f"   NEW Center-of-mass position: {new_cm}")
            print(f"   NEW Center-of-mass velocity: {new_vcm}")
            print(f"   NEW Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
            print(f"   NEW Number of bound particles: {len(mass[bound_mask])}")
            print(f"   NEW Number of bound subset particles: {np.count_nonzero(bound_subset_mask)}")

        
        if not refine or (delta_cm and delta_vcm):
            if verbose:
                if not refine:
                    print(f"\nFinal Values:")
                    print(f"-------------")
                    print(f"   FINAL Center-of-mass position: {cm}")
                    print(f"   FINAL Center-of-mass velocity: {vcm}")
                    print(f"   FINAL Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
                    print(f"   FINAL Number of bound particles: {len(mass[bound_mask])}")
                    print(f"   FINAL Number of bound subset particles: {bound_subset_mask}")
                else:
                    print(f"\nFinal Values:")
                    print(f"-------------")
                    print(f"   FINAL Center-of-mass position: {new_cm}")
                    print(f"   FINAL Center-of-mass velocity: {new_vcm}")
                    print(f"   FINAL Bound particle mass: {mass[bound_mask].sum().to('Msun')}")
                    print(f"   FINAL Number of bound particles: {len(mass[bound_mask])}")
                    print(f"   FINAL Number of bound subset particles: {np.count_nonzero(bound_subset_mask)}")
        
            if return_cm:
                return E, kin, pot, new_cm, new_vcm
            else:
                return E, kin, pot

        cm, vcm = copy(new_cm), copy(new_vcm)





