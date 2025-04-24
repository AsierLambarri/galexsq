import numpy  as np
from scipy.stats import binned_statistic

def density_profile(pos, 
                    mass, 
                    center = None, 
                    bins = None
                   ):
    """Computes the average radial density profile of a set of particles over a number of
    bins. The radii are the centers of the bins. 

    Returns errors on each bin based on the assumption that the statistics are poissonian.
    
    The routine works seamlessly for 3D and 2D projected data, as the particle mass is scalar,
    the only difference radicates in the meaning of "pos" (3D-->r vs 2D-->R_los).
    
    Parameters
    ----------
    pos : array
        Array of positions of the particles. Shape(N,3).
    mass : array
        Array of masses. Shape(N,).
    bins : array
        Bin edges to use. Recommended to provide externally and to perform logarithmic binning.
    center : array
        Center of particle distribution.
        
    Returns
    -------
    return_dict : dict
        Dictionary with radii, rho, e_rho, m_enclosed and center.
    """
    if center is not None:
        pass
    else:
        center = np.median(pos, axis=0)
        radii = np.linalg.norm(pos - center, axis=1)
        maskcen = radii < 0.5*radii.max()
        center = np.average(pos[maskcen], axis=0, weights=mass[maskcen])
        
        
    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)
    if bins is not None:
        mi, redges, bn = binned_statistic(radii, mass, statistic="sum", bins=bins)
        npbin, _, _ = binned_statistic(radii, mass, statistic="count", bins=bins)

    else:
        mi, redges, bn = binned_statistic(radii, mass, statistic="sum", bins=np.histogram_bin_edges(radii))
        npbin, _, _ = binned_statistic(radii, mass, statistic="count", bins=np.histogram_bin_edges(radii))
    
    redges = redges * coords.units
    mi = mi * mass.units

    if pos.shape[1] == 3:
        volumes = 4 * np.pi / 3 * ( redges[1:]**3 - redges[:-1]**3 )
    if pos.shape[1] == 2:
        volumes = np.pi * ( redges[1:]**2 - redges[:-1]**2 )

    
    rcoords = (redges[1:] + redges[:-1])/2
    dens = mi / volumes
    npbin[npbin == 0] = 1E20
    error = dens / np.sqrt(npbin)
    
    return_dict = {'r': rcoords,
                   'rho': dens,
                   'e_rho': error,
                   'm_enc': mi,
                   'center': center,
                   'dims': pos.shape[1]
                  }
    return return_dict


def velocity_profile(pos,
                     vel,
                     quantity=None,
                     mass=None,
                     center = None,
                     v_center = None,
                     bins = None,
                     projected=False,
                     average="bins"
                    ):
    """Computes the velocity dispersion profile for different radii. The radii are the centers of the bins. 
    
    As the projection of a vectorial quantity is not as straightforward as that of mass profiles, the projection
    is done inside the function and is controlled by the los argument

    Parameters
    ----------
    mass : array
        Array of masses of the particles. Shape(N,).
    vel : array
        Array of velocities. Shape(N,3).
    pos : array
        Array of positions of particles.
    bins : array
        Bin edges to use. Recommended to provide externally and to perform logarithmic binning.
    center : array
        Center of particle distribution.
    v_center : array
        Center of mass velocity. If none is provided, it is estimated with all the particles inside X kpc of center.
    average : str
        "bins" to average over bins in pos, "apertures" to average over filled apertures.
    
    Returns
    -------
    return_dict : dict
        Dictionary with radii, rho, e_rho, m_enclosed and center.
    """
    if center is not None:
        pass
    else:
        center = np.median(pos, axis=0)
        radii = np.linalg.norm(pos - center, axis=1)
        maskcen = radii < 0.5*radii.max()
        center = np.average(pos[maskcen], axis=0, weights=mass[maskcen])

    if v_center is not None:
        pass
    else:
        v_center = np.average(vel[np.linalg.norm(pos - center, axis=1) < 0.7], axis=0, weights=mass[np.linalg.norm(pos - center, axis=1) < 0.7])


    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)

    if bins is None:
        bins = np.histogram_bin_edges(np.log10(radii))

    if projected:
        quant = "rms" if quantity is None else quantity
        return_dict = _projected_velocity_profile(
            pos,
            vel,
            center,
            v_center,
            bins,
            quant,
            average            
        )
    else:
        quant = "rms" if quantity is None else quantity
        return_dict = _full_velocity_profile(
            pos,
            vel,
            center,
            v_center,
            bins,
            quant
        )
                               
    return return_dict


def _full_velocity_profile(pos,
                           vel,
                           center,
                           v_center,
                           bins,
                           quantity
                          ):
    """Computes the selected quantity: i.e. mean = mean(|v_i|), rms = sqrt(mean(v_i ** 2)) or dispersion at given radial bins
    for full velocities (i.e. velocities that are not projected, regardless of dimensionality of the data).
    """
    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)
    if quantity == "mean":
        stat = "mean"
        magvel = np.linalg.norm(vel - v_center, axis=1)
    elif quantity == "rms":
        stat = "mean"
        magvel = np.linalg.norm(vel - v_center, axis=1) ** 2
    elif quantity == "dispersion":
        stat = "std"
        magvel = np.linalg.norm(vel - v_center, axis=1)
    else:
        raise Exception(f"The quantity you provided is not available")


    vstat, redges, bn = binned_statistic(radii, magvel, statistic=stat, bins=bins)
    npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=bins)

    redges = redges * coords.units
    rcoords = (redges[1:] + redges[:-1])/2    

    if quantity == "rms":
        vstat = np.sqrt(vstat)
    if 0 in npart:
        npart[npart == 0] = np.inf

    vstat = vstat * vel.units        
    e_vstat = vstat / np.sqrt(npart)
    
    return {'r': rcoords,
            'v': vstat,
            'e_v': e_vstat,
            'center': center,
            'v_center': v_center
            }
    
def _projected_velocity_profile(pos,
                                pvels,
                                center,
                                v_center,
                                bins,
                                quantity,
                                average
                               ):
    """Computes the selected quantity: i.e. line-of-sight velocity mean or dispersion. The average can be computed for
    radial bins or filled apertures.
    """
    coords = pos - center
    radii = np.linalg.norm(coords, axis=1)

    if quantity == "mean":
        stat = "mean"
        magvel = np.abs(pvels - v_center)
    elif quantity == "rms":
        stat = "mean"
        magvel = np.abs(pvels - v_center) ** 2
    elif quantity == "dispersion":
        stat = "std"
        magvel = pvels - v_center
    else:
        raise Exception(f"The quantity you provided is not available")


    
    if average == "bins" :
        vstat, redges, bn = binned_statistic(radii, magvel, statistic=stat, bins=bins)
        npart, _, _ = binned_statistic(radii, magvel, statistic="count", bins=bins)
        

        redges = redges * coords.units
        rcoords = (redges[1:] + redges[:-1])/2
            
    elif average == "apertures":
        R_aperture = 0.5 * (np.array(bins[:-1]) + np.array(bins[1:]))
                
        cumulative_stats = []
        particle_counts = []
        for i in range(len(R_aperture)):
            r_ap = R_aperture[i]
            mask = radii <= r_ap
            mask_bin = (bins[i] <= radii ) & (radii <= bins[i+1])

            if stat == "mean":
                stat_in_aperture = np.mean(magvel[mask]) if np.any(mask_bin) else np.nan
            elif stat == "std":
                stat_in_aperture = np.std(magvel[mask]) if np.any(mask_bin) else np.nan
                
            cumulative_stats.append(stat_in_aperture)
            
            particle_count = np.sum(mask)
            particle_counts.append(particle_count)

        
        rcoords = R_aperture * coords.units
        vstat = np.array(cumulative_stats)
        npart = np.array(particle_counts)
        
    else:
        raise Exception(f"The averaging mode you provided is not available")

    
    if 0 in npart:
        npart[npart == 0] = np.inf

    if quantity == "rms":
        vstat = np.sqrt(vstat)
    vstat = vstat * pvels.units      
    e_vstat = vstat / np.sqrt(npart)
    
    return {'r': rcoords,
            'v': vstat,
            'e_v': e_vstat,
            'center': center,
            'v_center': v_center
           }



































