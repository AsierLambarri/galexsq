import numpy as np
from scipy.spatial import KDTree
from numba import (
    vectorize,
    float32,
    float64,
    njit,
    jit,
    prange,
    get_num_threads,
    typed
)

from math import sqrt

@vectorize([float32(float32), float64(float64)])
def CSpline_3d(q):
    """Cubic spline in 3 dimensions.
    """
    if q <= 0.5:
        return (1 - 6 * q**2 + 6 * q**3) * 2.5464790894703264
    elif q <= 1.0:
        return (2 * (1 - q) ** 3) * 2.5464790894703264
    else:
        return 0.0
        
@vectorize([float32(float32), float64(float64)])
def Wendland2_3d(q):
    """Wendland C2 kernel in 3 dimensions.
    """
    if q <= 1.0:
        return 3.342253804929802 * (1 + 4*q) * (1 - q) * (1 - q) * (1 - q) * (1 - q)
    else:
        return 0.0
    
@vectorize([float32(float32), float64(float64)])
def Wendland4_3d(q):
    """Wendland C4 kernel in 3 dimensions.
    """
    if q <= 1.0:
        return 4.923856051905519 * (1 + 6*q + 35/3*q*q) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q)
    else:
        return 0.0
    
@vectorize([float32(float32), float64(float64)])
def Wendland6_3d(q):
    """Wendland C6 kernel in 3 dimensions.
    """
    if q <= 1.0:
        return 6.788953041263665 * (1 + 8*q + 25*q**2 + 32*q**3) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q) * (1 - q)
    else:
        return 0.0

@vectorize([float32(float32,float32,float32), float64(float64,float64,float64)])
def norm_vectorized(x,y,z):
    """Computes the norm of a vector. Equivalent to np.linalg.norm(x) or np.sqrt(np.sum(x**2)).
    """
    return sqrt(x*x + y*y + z*z)
    
@njit(fastmath=True, parallel=True)
def vector_norm(x):
    """Computes the norm of vector x along axis=1.
    """
    norms = np.empty(x.shape[0], dtype=x.dtype)
    for i in prange(x.shape[0]):
        norms[i] = np.sqrt(x[i,0]*x[i,0] + x[i,1]*x[i,1] + x[i,2]*x[i,2])
    return norms

@njit(fastmath=True, parallel=False)
def sum_density(pos, mass, smooth, gridpos, pindex, glist):
    """Sums densities for each gridpos point using the KDTree generated plist.
    """
    rho = np.zeros(gridpos.shape[0])
    #for i in range(len(plist)):
    #    if len(plist[i]) == 0:
    #        continue   
    #    else:
    #        kernel_values = Wendland2_3d(vector_norm(gridpos[plist[i]] - pos[i]) / smooth[i]) * 1 / (smooth[i] * smooth[i] * smooth[i])
    #        rho[plist[i]] += mass[i] * kernel_values
    
    for j in range(len(pindex)):
        i = pindex[j]
        gridpoints = glist[j]
        kernel_values = CSpline_3d(vector_norm(gridpos[gridpoints] - pos[i]) / smooth[i]) * 1 / (smooth[i] * smooth[i] * smooth[i])
        rho[gridpoints] += mass[i] * kernel_values
    
    return rho

    
def KDE(pos, mass, smooth, gridpos, tree=None):
    """Computes the Kernel Density Estimation on a grid `gridpos`, given by particles
    colocated at pos, with a certain mass an smoothing length smooth. The KDE is weighted by a
    custom compact suport kernel.

    The search is performed using KDTrees. As the compact kernels have density=0 for q>1, for each 
    grid position we are only interested in those particles whose distance |x_p - x_grid|<smooth_p. 
    
    We interchange the roles of gridpoints and particles: we first find all the gridpoints inside the softening
    length of each particle, producing a list of lists where the axis=0 has len(particles) and axis=1 has len(gridp).
    The i-th axis=0 element tells us to which gridpoints is the i-th particle contributing. By pre-creating a list of
    densities for each gridpoints, we can iterate over the axis=0 of the query and add the contribution fo each
    particle to the corresponding gridpoints

    Parameters
    ----------
    pos : array
        Positions of particles.
    mass : array
        Mass of particles.
    smooth : array
        Hsml of particles.
    gridpos : array
        Positions of gridpoints in which the KDE will be computed.
    kernel : func
        Compact kernel function with which the relative contribution of each particle will be wheighed. Default: kubic spline.
    tree : KDTree, optional
        Precomputed KDTree.

    Returns
    -------
    rho : array
        Densities computed at each gridpos point.
    """
    if tree is not None:
        pass
    else:
        tree = KDTree(gridpos)

    raw_plist = tree.query_ball_point(pos, smooth, workers=-1)
    #plist = [np.array(raw_plist[i]) for i in range(len(raw_plist))]
    plist = {i : np.array(raw_plist[i]) for i in range(len(raw_plist)) if len(raw_plist[i]) != 0}
    #print(plist)
    rho = sum_density(pos, mass, smooth, gridpos, list(plist.keys()), list(plist.values()))

    return rho



def __KDE__(pos, mass, smooth, gridpos, kernel=CSpline_3d, tree=None):
    
    if tree is not None:
        pass
    else:
        tree = KDTree(gridpos)

    plist = tree.query_ball_point(pos, smooth, workers=-1)

    rho = np.zeros(gridpos.shape[0])

    for i, particle in enumerate(plist):
        if len(particle) == 0:
            continue
        else:
            kernel_values = kernel( vector_norm(gridpos[particle] - pos[i]) / smooth[i] )
            rho[particle] += mass[i] * kernel_values
            
    return rho




    #for i in range(len(plist)):
    #    if len(plist[i]) == 0:
    #        continue
    #    particle_pos = pos[i]
    #    particle_mass = mass[i]
    #    particle_smooth = smooth[i]
    #    grid_pos = gridpos[plist[i]]
    #    distance = vector_norm(grid_pos - particle_pos)
    #    q = distance / particle_smooth
    #    if np.any(q>1):
    #        raise Exception("aaaaaa!")
    #
    #    kernel_value = kernel(q)
    #    rho[plist[i]] += particle_mass * kernel_value
        
       
        
        #for j in plist[i]:
        #    # Calculate the distance and kernel values
        #    distance_vector = gridpos[j] - particle_pos
        #    distance = vector_norm(distance_vector)[0]
        #    q = distance / particle_smooth
        #    
        #    if q < 1.0:
        #        kernel_value = kernel(q)
        #        rho[j] += particle_mass * kernel_value

def sample_sphere(radius, num_points):
    """Generates N random points on a sphere. Returns in cartesian coordinates.
    """
    np.random.seed()
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(-1, 1, num_points)
    phi = np.arccos(z)  
    
    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sqrt(1 - z**2) * np.cos(theta)
    y = radius * np.sqrt(1 - z**2) * np.sin(theta)
    z = radius * z
    
    return np.vstack((x, y, z)).T




