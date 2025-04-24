import numpy as np
from unyt import unyt_quantity

from .half_mass_radius import half_mass_radius
from .utils import easy_los_velocity

def center_of_mass_pos(pos,
                       mass
                       ):
    """Computes coarse CoM using all the particles as 
    
                CoM = sum(mass * pos) / sum(mass)
        
    Parameters
    ----------
    pos : array-like[float], shape(N,dims)
        Positions of particles. First dimension of the array corresponds to each particle.
        Second dimension correspond to each coordiante axis.
    mass : array-like, shape(N,)
        Masses of particles.

    Returns
    -------
    CoM_pos : array-like
    """
    return np.average(pos, axis=0, weights=mass)

def center_of_mass_vel(pos,
                       mass,
                       vel,
                       center=None,
                       R=(0.7, 'kpc'),
                      ):
    """Computes the center of mass velocity as 

                    CoM = sum(mass * vel) / sum(mass)
                    
    using only particles inside a specified R around the estimated CoM position.

    Parameters
    ----------
    pos : array-like[float], shape(N,dims)
        Positions of particles. First dimension of the array corresponds to each particle.
        Second dimension correspond to each coordiante axis.
    mass : array-like, shape(N,)
        Masses of particles.
    vel : array-like, shape(N,)
        Velocities of particles.
    center : array-like, shape(N,)
        CoM position
    R : tuple or unyt_*
        Radius for selecting particles. Default 0.7 'kpc'.
        
    Returns
    -------
    CoM_vel : array-like
    """
    if center is None:
        center = center_of_mass_pos(pos, mass)


    if isinstance(R, tuple) and len(R) == 2:
        r = unyt_quantity(*R)
    else:
        try:
            r = R * pos.units
        except:
            r = R
    
    mask = np.linalg.norm(pos - center, axis=1) < r
    return np.average(vel[mask], axis=0, weights=mass[mask])
    


        


def center_of_mass_vel_through_proj(pos,
                                    vel,
                                    center=None,
                                    rcyl=(1E4, "Mpc"),
                                    h=(1E4, "Mpc")
                                   ):
    """Computes the center of mass velocity by looking at the projected velocities through the x,y,z-axes of
    the provided coordinates. At each projection, only the particles within cylindrical distance rmax from the center
    and  a longitudinal distance h are taken into acocunt, to avoid picking up more than one kinematic component.
    Center-of-mass is taken to be the simple mean of each cylinder/projection.

    It is slightly different to center_of_mass_vel. Use not recomended.
    
    Parameters
    ----------
    pos : array
        Position array.
    vel : array
        Velocity array.
    center : array, optional
        Pre-computed center-of-mass.
    rcyl : tuple(float, str) or unyt_quantity
        Maximum radius.

    Returns
    -------
    cm_vel : array
    """
    mask_x = ( np.linalg.norm(pos[:,[1,2]] - center[1,2], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,0] - center[0]) < unyt_array(*h) )
    mask_y = ( np.linalg.norm(pos[:,[0,2]] - center[0,2], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,1] - center[1]) < unyt_array(*h) )
    mask_z = ( np.linalg.norm(pos[:,[0,1]] - center[0,1], axis=1) < unyt_array(*rcyl) ) & ( np.abs(pos[:,2] - center[2]) < unyt_array(*h) )

    los_velocities_x = np.mean(easy_los_velocity(vel[mask_x], [1,0,0]))
    los_velocities_y = np.mean(easy_los_velocity(vel[mask_y], [0,1,0]))
    los_velocities_z = np.mean(easy_los_velocity(vel[mask_z], [0,0,1]))
    
    try:
        return unyt_array([los_velocities_x, los_velocities_y, los_velocities_z], vel.units)
    except:
        return np.array([los_velocities_x, los_velocities_y, los_velocities_z])
     
    


def refine_6Dcenter(pos, 
                    mass, 
                    vel,
                    method="adaptative",
                    **kwargs
                   ):
                
    """Refined CM position estimation. 

    The CoM of a particle distribution is not well estimated by the full particle ensemble, 

    Parameters
    ----------
    pos : array
        Array of positions.
    mass : array
        Array of masses.
    vel : array
        Array of velocities.
    method : str, optional
        Method with which to refine the CoM: simple, interative or iterative-hm.
    delta : float
        Tolerance for stopping re-refinement. Default 1E-2.
    spacing : float
        Spacing between two consecutive shell radii when refining the CoM. A good guess is the softening length. Default is 0.08 in provided units.

    Returns
    -------
    centering_results : array
        Refined Center of mass and various quantities.


      
    """    
    lengthunit = pos.units
    velunit = vel.units
    pos = pos.value
    mass = mass.value    
    vel = vel.value


    rc_scale = 0.5 if "rc_scale" not in kwargs.keys() else kwargs['rc_scale']
    v_scale = 1.5 if "v_scale" not in kwargs.keys() else kwargs['v_scale']
    rsphere = unyt_quantity(4, 'kpc') if "rsphere" not in kwargs else unyt_quantity(*kwargs['rsphere']) if isinstance(kwargs['rsphere'], tuple) and len(kwargs['rsphere']) == 2 else unyt_quantity(kwargs['rsphere'], lengthunit)
    alpha = 0.7 if "alpha" not in kwargs.keys() else kwargs['alpha']
    nmin = 150 if "nmin" not in kwargs.keys() else kwargs['nmin']
    mfrac = 0.3 if "mfrac" not in kwargs.keys() else kwargs['mfrac']

    rsphere = rsphere.to(lengthunit).value
    
    if method == "radial-cut" or method == "rcc":
        centering_results = radial_cut_center(pos, mass, rc_scale=rc_scale)
        
    if method == "shrink-sphere" or method == "ssc":
        centering_results = shrink_sphere_center(pos, mass, rsphere=rsphere, alpha=alpha, nmin=nmin)  
        
    if method == "fractional-mass" or method == "fmc":
        centering_results = fractional_mass_center(pos, mass, mfrac=mfrac)
        
    if method == "adaptative":
        n = len(mass)
        if n < 2*nmin:
            centering_results = radial_cut_center(pos, mass, rc_scale=rc_scale)
        else:
            try:
                centering_results = shrink_sphere_center(pos, mass, rsphere=rsphere, alpha=alpha, nmin=nmin)   
            except Exception as e:
                centering_results = radial_cut_center(pos, mass, rc_scale=rc_scale)
                print(e)

            
        
    
    centering_results['r_vel'] = v_scale * centering_results['r_last']
    centering_results['velocity'] = center_of_mass_vel(
        pos,
        mass,
        vel,
        center=centering_results['center'],
        R=centering_results['r_vel']
    )
    centering_results['npart_vel'] = np.count_nonzero( np.linalg.norm(pos - centering_results['center'], axis=1) < centering_results['r_vel'] )

    
    centering_results['center'] *= lengthunit
    centering_results['velocity'] *= velunit
    
    centering_results['trace_cm'] *= lengthunit
    centering_results['trace_r'] *= lengthunit
    
    centering_results['r_last'] *= lengthunit
    centering_results['r_vel'] *= lengthunit
    
    return centering_results




def radial_cut_center(pos,
                      mass, 
                      rc_scale=0.5
                     ):

    """Computes the CoM by first finding the median of positions and then computing a refined CoM using
    only the particles inside r < 0.5*rmax. Usually is robust enough for estimations with precision of
    0.1-0.5 POS units.

    This approach has been inspired by Mark Grudic.
    """
    radii = np.linalg.norm(pos - np.median(pos, axis=0), axis=1)
    rmax = radii.max()
    center_new = np.average(pos[radii <= rc_scale * rmax], axis=0, weights=mass[radii <= rc_scale * rmax])
    
    centering_results = {'center': center_new ,
                         'r_last': rc_scale * rmax ,
                         'iters': 2,
                         'trace_cm': np.vstack((np.median(pos, axis=0), center_new)),
                         'trace_r': np.array([rmax, rc_scale * rmax]),
                         'npart_cen': len(pos[radii < rc_scale * rmax]),
                        }

    return centering_results

def shrink_sphere_center(pos,
                         mass,
                         rsphere=None,
                         alpha=0.7, 
                         nmin=100
                        ):
    """Iterative method where the center-of-mass is computed using the particles inside an ever-shrinking
    sphere, until the mnumber of particles inside the sphere reaches the user specified minimmum. This routine adapts
    the method described in Power et al. 2003: 2003MNRAS.338...14P.
    """
    center = np.median(pos, axis=0)
    
    for i in range(100 + 1): 
        if rsphere is None: rsphere = np.linalg.norm(pos - center, axis=1).max()

        if i == 0:
            trace_cm = np.array(center)
            trace_r = np.array([rsphere])
        else:
            trace_cm = np.vstack((trace_cm, center))
            trace_r = np.append(trace_r, rsphere)
            
        mask = np.linalg.norm(pos - center, axis=1) <= rsphere
        npart = len(mass[mask])
        
        if npart <= nmin:
            if i == 0: center = np.average(pos[mask], axis=0, weights=mass[mask])
            break
            
        if np.any(mask): center = np.average(pos[mask], axis=0, weights=mass[mask])
        
        rsphere *= alpha
        

    
    return {
        "center": center,
        "r_last": rsphere,
        "iters": i,
        "trace_cm": trace_cm,
        "trace_r": trace_r,
        "npart_cen": npart
    }



def fractional_mass_center(pos,
                           mass,
                           mfrac=0.5
                          ):
    """Computes the center of mass by computing the half mass radius iteratively until the center of mass
    of the enclosed particles converges to DELTA in M consecutive iterations. This method is less affected by particle 
    number because the CoM estimation is always done with half the particles of the ensemble, this way, we avoid
    the inconveniences of the iterative method in that the cm can be more robustly computed for ensembles with N PART
    much smaller: around 10 to 20 particles are enough to reliably estimate the CoM.

    The half-mass radius is computed with the ``half_mass_radius`` routine.

    As the CoM estimation gets more precise whe using the most central particles, but suffers when having few particles,
    you can change the mass fraction with which to estimate the CoM (i.e. you can choose if you want the half-mass radius,
    quarter-mass radius, 75-percent-mass radius etc.).

    The routine has a maximum of 100 iterations to avoid endless looping.
    """
    center = np.median(pos, axis=0)

    trace_r = np.array([])
    trace_cm = np.empty((0,pos.shape[1]))
    for i in range(100):
        rhalf = half_mass_radius(pos, mass,  center=center, mfrac=mfrac)
        mask = np.linalg.norm(pos - center, axis=1) <= rhalf
        
        center_new = np.average(pos[mask], axis=0, weights=mass[mask])

        diff = np.sqrt( np.sum((center_new - center)**2, axis=0) ) 
        npart = len(pos[mask])

        trace_r = np.append(trace_r, rhalf)
        trace_cm = np.vstack((trace_cm, center_new))
        
        if diff < 1E-5:
            break
            
        else:
            center = center_new

    centering_results = {'center': center_new ,
                         'r_last': rhalf ,
                         'iters': i ,
                         'trace_cm': trace_cm,
                         'trace_r': trace_r,
                         'npart_cen': npart,
                        }
    return centering_results





