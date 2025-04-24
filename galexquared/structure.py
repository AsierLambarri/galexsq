import yt
import warnings
import numpy as np
from tqdm import tqdm
from copy import copy
from pNbody.Mockimgs import luminosities

from scipy.stats import binned_statistic

from .class_methods import half_mass_radius, refine_6Dcenter, easy_los_velocity, gram_schmidt, vectorized_base_change
from .class_methods import density_profile as densprof
from .class_methods import velocity_profile as velprof

def _collapse_to_longest_true(arr):
    """
    Given a boolean array, returns a new array where only the longest contiguous block 
    of True values is preserved, and all other True values are set to False.
    
    This function pads the array and uses np.diff to locate transitions.
    """
    arr = np.asarray(arr, dtype=bool)
    
    # If there are no True values, just return a copy.
    if not arr.any():
        return arr.copy()
    
    # Pad the array with False at both ends.
    padded = np.r_[False, arr, False]
    
    # Compute the differences between consecutive elements.
    # When converting booleans to int: False->0, True->1.
    d = np.diff(padded.astype(int))
    
    # A diff of 1 indicates a transition from False to True: these are our start indices.
    starts = np.where(d == 1)[0]
    
    # A diff of -1 indicates a transition from True to False: these are our end indices.
    # Since we padded on the left, subtract 1 to map back to original array indices.
    ends = np.where(d == -1)[0] - 1
    
    # Compute the lengths of each contiguous True block.
    lengths = ends - starts + 1
    
    # Identify the block with the maximum length.
    max_idx = np.argmax(lengths)
    longest_start = starts[max_idx]
    longest_end = ends[max_idx]
    
    # Build a new array: set only the indices between longest_start and longest_end to True.
    new_arr = np.zeros_like(arr, dtype=bool)
    new_arr[longest_start:longest_end+1] = True
    return new_arr



def refined_center6d(self, 
                     ptype,
                     method="adaptative",
                     **kwargs
                    ):
    """Refined center-of-mass position and velocity estimation. 

    The center of mass of a particle distribution is not well estimated by the full particle ensemble, since the outermost particles
    must not be distributed symmetrically around the TRUE center of mass. For that reason, a closer-to-truth value for th CM can be
    obtained by disregarding these outermost particles.
    
    Here we implement four methods in which a more refined CM estimation can be obtained. All of them avoid using gravitational 
    porentials, as these are not available to observers. Only positions and masses (analogous to luminosities in some sense)
    are used:
        
        1. RADIAL-CUT: Discard all particles outside of rshell = rc_scale * rmax. 
        2. SHRINK-SPHERE: An iterative version of the SIMPLE method, where rshell decreases in steps, with rshell_i+1 = alpha*rshell_i,
                   until an speficief minimun number of particles nmin is reached. Adapted from Pwer et al. 2003.
        3. FRAC-MASS: A variant of the RADIAL-CUT method. Here the cut is done in mass instead of radius. The user can specify a
                   X-mass-fraction and the X-mass radius and its center are computed iterativelly unitl convergence. 
        4. ADAPTATIVE: Performs `SHRINK-SPHERE` if the number of particles is larger than 2*nmin. Otherwise: `RADIAL-CUT`.

    The last radius; r_last, trace of cm positions; trace_cm, number of iterations; iters and final numper of particles; n_particles 
    are stored alongside the center-of-mass.
    
    Center-of-mass velocity estimation is done is done with particles inside v_scale * r_last.
    
    OPTIONAL Parameters
    -------------------

    method : str, optional
        Method with which to refine the CoM: radial-cut/rcc, shrink-sphere/ssc, fractional-mass/mfc od adaptative. Default: ADAPTATIVE
        
    rc_scaling : float
        rshell/rmax for radial-cut method. Must be between 0 and 1. Default: 0.5.
    alpha : float
        Shrink factor for shrink-sphere method. Default: 0.7.
    nmin : int
        Target number of particles for shrink-sphere method. Default: 250.
    mfrac : float
        Mass-fraction for fractional-mass method. Default: 0.3.
    v_scale : float
        Last Radius scale-factor for velocity estimation. Default: 1.5.
        
    Returns
    -------
    cm, vcm : array
        Refined Center of mass and various quantities.
    """
    if method.lower() == "pot-most-bound":
        bound_mask = self[ptype, "total_energy"] < 0
        f = 0.1 if "f" not in kwargs.keys() else kwargs["f"]
        nbound = 32 if "nbound" not in kwargs.keys() else kwargs["nbound"]
        
        N = int(np.rint(np.minimum(f * np.count_nonzero(bound_mask), nbound)))
        most_bound_ids = np.argsort(self[ptype, "total_energy"])[:N]
        most_bound_mask = np.zeros(len(self[ptype, "total_energy"]), dtype=bool)
        most_bound_mask[most_bound_ids] = True
        
        tmp_cm = np.average(self[ptype, "coords"][most_bound_mask], axis=0, weights=self[ptype, "mass"][most_bound_mask])
        tmp_vcm = np.average(self[ptype, "velocity"][most_bound_mask], axis=0, weights=self[ptype, "mass"][most_bound_mask]) 
        
    elif method.lower() == "pot-softmax":
        bound_mask = self[ptype, "total_energy"] < 0
        T = "adaptative" if "T" not in kwargs.keys() else kwargs["T"]
        
        w = self[ptype, "total_energy"][bound_mask]/self[ptype, "total_energy"][bound_mask].min()
        if T == "adaptative":
            T = np.abs(self[ptype, "kinetic_energy"][bound_mask].mean()/self[ptype, "total_energy"][bound_mask].min())
            
        tmp_cm = np.average(self[ptype, "coordinates"][bound_mask], axis=0, weights=softmax(w, T))
        tmp_vcm = np.average(self[ptype, "velocity"][bound_mask], axis=0, weights=softmax(w, T))
    
    else:                
        centering_results = refine_6Dcenter(
            self[ptype, "coordinates"],
            self[ptype, "mass"],
            self[ptype, "velocity"],
            method=method,
            **kwargs
        )

        tmp_cm = centering_results['center']
        tmp_vcm = centering_results['velocity']

    dq = {}
    dq["cm"] = tmp_cm
    dq["vcm"] = tmp_vcm
    
    return tmp_cm, tmp_vcm



def Xmass_radius(self, ptype, cm, mfrac=0.5, lines_of_sight=None, project=False, light=False):
    """By default, it computes 3D half mass radius of a given particle ensemble. If the center of the particles 
    is not provided, it is estimated by first finding the median of positions and then computing a refined CoM using
    only the particles inside r < 0.5*rmax.
    
    There is also an argument to compute other ALPHA_mass_radius of an arbitrary mass fraction. The desired ALPHA_mass_radius  
    is computed via rootfinding using scipy's implementation of brentq method.
    
    OPTIONAL Parameters
    -------------------
    mfrac : float
        Mass fraction of desired radius. Default: 0.5 (half, mass radius).
    project: bool
        Whether to compute projected quantity or not.
    
    Returns
    -------
    MFRAC_mass_radius : float
        Desired mfrac mass fraction radius estimation. Provided in same units as pos, if any.
    """
    if project is False:
        lines_of_sight = np.array([[1,0,0]])
    else:
        if lines_of_sight is None:
            raise ValueError(f"You haven't provided any lines of sight!!")
        elif np.array(lines_of_sight).ndim == 1:
            lines_of_sight = np.array([lines_of_sight])
        elif np.array(lines_of_sight).ndim == 2:
            lines_of_sight = np.array(lines_of_sight)
        else:
            raise ValueError(f"Lines of sight does not have the correct number of dimensions. It should have ndims=2, yours has {np.array(lines_of_sight).ndim}")

    tmp_rh_arr = self.ds.arr( -9999 * np.ones((lines_of_sight.shape[0])), self[ptype, "coordinates"].units)
    for i, los in enumerate(lines_of_sight):
        gs = gram_schmidt(los)
        new_coords = vectorized_base_change(np.linalg.inv(gs), self[ptype, "coordinates"])
        new_cm = np.linalg.inv(gs) @ cm
        
        tmp_rh_arr[i] = half_mass_radius(
            new_coords, 
            self[ptype, "mass"] if light is False else self[ptype, "luminosity"], 
            new_cm, 
            mfrac, 
            project=project
        )

    if project:
        return tmp_rh_arr
    else:
        return tmp_rh_arr[0]


def cumulative_mass(self, ptype, cm, bins=None):
    """Computes the radius at which dlog(rho)/dlog(r) = log_slope. This is done by looking at the cumulative mass profile
    for increased signal to noise ratio.
    """
    radii = np.linalg.norm(self[ptype, "coordinates"] - cm, axis=1)
    masses = self[ptype, "mass"]
    mass_sum, bin_edges, binnumber = binned_statistic(radii, masses, statistic='sum', bins=10**np.histogram_bin_edges(np.log10(radii), bins="fd") if bins is None else bins)
    npart, bin_edges, binnumber = binned_statistic(radii, masses, statistic='count', bins=10**np.histogram_bin_edges(np.log10(radii), bins="fd") if bins is None else bins)

    cumulative_mass = np.cumsum(mass_sum)
    cumulative_npart = np.cumsum(npart)
    e_cumulative_mass = cumulative_mass / np.sqrt(cumulative_npart)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    return {
        'encmass': cumulative_mass * masses.units, 
        'radii': bin_centers * radii.units, 
        'e_encmass': e_cumulative_mass * masses.units
    }


def fit_cumulative(radii, encmass, e_encmass, model="king", **kwargs):
    """Fits a Lowered isothermal model to the cumulative mass profile.
    """
    from limepy import limepy
    from lmfit import Model, Parameters, fit_report

    def LIMInterp(r, W0, g, M, rh, ra):
        """Produces a sample of a Lowered isothermal model with parameters W0, g, M and rh using LIMEPY 
        and interpolates the result for a specified r.
        """
        k = limepy(phi0=W0, g=g, M=M, rh=rh, ra=ra, G=4.300917270038e-06, project=True, ode_atol=1E-10, ode_rtol=1E-10)
        evals = np.interp(r, xp=k.r, fp=k.mc)
        return evals 
    
    if model=="king":
        g, W = 1, 5
    elif model=="plummer":
        g, W = 3.49, 0.01
    elif model=="wilson":
        g, W = 2, 4
    elif model=="woolley":
        g, W = 0, 3

    densModel = Model(LIMInterp, independent_vars=['r'])
    fit_params = densModel.make_params(
        W0={'value': W,'min': 0.01,'max': np.inf,'vary': True if model != "plummer" else False},
        g={'value': g, 'min': 0.01, 'max': 3.49,'vary': False},
        M={'value': encmass[-1].value, 'vary' : False},
        rh={'value': 1, 'min': 1E-4, 'max': 50, 'vary': True},
        ra={'value': 1E8, 'min': 0.1, 'max': 1E8, 'vary': False}
    )

    result = densModel.fit(
        r=radii,
        data=encmass,
        params=fit_params, 
        weights=1/e_encmass,
        nan_policy="omit"
    ) 

    k = limepy(
        phi0=result.params["W0"],
        g=result.params["g"],
        M=result.params["M"],
        rh=result.params["rh"],
        ra=result.params["ra"],
        G=4.300917270038e-06,
        ode_atol=kwargs.get("ode_atol", 1E-13), 
        ode_rtol=kwargs.get("ode_rtol", 1E-13),
        project=True
    )

    return k, result


def Slope_radius(self, ptype, cm, bins=None, model="plummer", log_slope=-3, method="density", **kwargs):
    """Uses cumulative_mass and fit_cumulative to estimate the radius at which the log_slope of the density is the one provided.
    Included models: plummer (for any log_slope) and King (for log_slope=-3)
    """
    from scipy.optimize import root_scalar
    profile = cumulative_mass(self, ptype, cm, bins=bins)
    limepy_model, result = fit_cumulative(
        profile["radii"], 
        profile["encmass"], 
        profile["e_encmass"],
        model=model,
        **kwargs
    )
    assert limepy_model.converged, "Limepy model creation did not converge!"
    assert result.success, "Least SQ fit did not succeed!"
    window = kwargs.get("window", int(min(5, 0.1 * limepy_model.nstep)) )

    if method == "cumulative":
        grad = np.convolve( 
            np.gradient(limepy_model.mc, limepy_model.r, edge_order=2),
            np.ones(window) / window,
            mode="same"
        )
        dgrad1 = np.convolve( 
            np.gradient(np.log10(grad), np.log10(limepy_model.r), edge_order=2),
            np.ones(window) / window,
            mode="same"
        )  
        p = 2
    elif method == "density":
        dgrad1 = np.convolve( 
            np.gradient(np.log10(limepy_model.rho), np.log10(limepy_model.r), edge_order=2),
            np.ones(window) / window,
            mode="same"
        )    
        p = 0


    mask = np.isfinite(dgrad1)
    mask = _collapse_to_longest_true(mask)

    r = limepy_model.r[mask]
    dgrad_def = dgrad1[mask]
    rmax = r[np.nanargmin(dgrad_def)]
    rmin = r[np.nanargmax(dgrad_def[:np.nanargmin(dgrad_def)])]
    root = root_scalar(lambda x: np.interp(x, xp=r, fp=dgrad_def) - p - log_slope, method="brentq", bracket=[rmin, rmax])

    assert root.converged, "Rootfinding did not converge!"

    return {
        "r_slope": root.root * profile["radii"].units,
        "r": limepy_model.r,
        "log_grad": dgrad1
    }
    



def los_dispersion_old(self, ptype, cm, lines_of_sight, rcyl):
    """Computes the line of sight velocity dispersion:  the width/std of f(v)dv of particles iside rcyl along the L.O.S. This is NOT the
    same as the dispersion velocity (which would be the rms of vx**2 + vy**2 + vz**2). All particles are used, including non-bound ones,
    given that observationally they are indistinguishable.

    OPTIONAL Parameters
    ----------
    rcyl : float, tuple[float, str] or unyt_quantity
        Axial radius of the cylinder. Default: 1 kpc.
    return_projections : bool
        Whether to return projected velocities. Default: False.

    Returns
    -------
    stdvel : unyt_quantity
    los_velocities : unyt_array
    """                
    if np.array(lines_of_sight).ndim == 1:
        lines_of_sight = np.array([lines_of_sight])
    elif np.array(lines_of_sight).ndim == 2:
        lines_of_sight = np.array(lines_of_sight)
    else:
        raise ValueError(f"Lines of sight does not have the correct number of dimensions. It should have ndims=2, yours has {np.array(lines_of_sight).ndim}")

    tmp_disp_arr = self.ds.arr( -9999 * np.ones((lines_of_sight.shape[0])), self[ptype, "velocity"].units)
    for i, los in enumerate(lines_of_sight):
        gs = gram_schmidt(los)
        cyl = self.ds.disk(cm, los, radius=rcyl, height=(np.inf, 'kpc'), data_source=self)
        tmp_disp_arr[i] = easy_los_velocity(cyl[ptype, "velocity"], los).std()

    return tmp_disp_arr


def los_dispersion(self, ptype, lines_of_sight=[[1,0,0], [0,1,0], [0,0,1]], rcyl="los-rh", cm=None, **kwargs):
    """Computes the line of sight velocity dispersion:  the width/std of f(v)dv of particles iside rcyl along the L.O.S. This is NOT the
    same as the dispersion velocity (which would be the rms of vx**2 + vy**2 + vz**2). All particles are used, including non-bound ones,
    given that observationally they are indistinguishable. The resulting value is weighted with luminosity.

    OPTIONAL Parameters
    ----------
    rcyl : float, tuple[float, str] or unyt_quantity
        Axial radius of the cylinder. Default: 1 kpc.
    return_projections : bool
        Whether to return projected velocities. Default: False.

    Returns
    -------
    stdvel : unyt_quantity
    los_velocities : unyt_array
    """    
    from photutils.aperture import CircularAperture, aperture_photometry
    from galexquared.utils import create_sph_dataset

    
    if np.array(lines_of_sight).ndim == 1:
        lines_of_sight = np.array([lines_of_sight])
    elif np.array(lines_of_sight).ndim == 2:
        lines_of_sight = np.array(lines_of_sight)

    nn = max(min(21, self["stars", "index"].shape[0]), 64)
    ds_sph = create_sph_dataset(self.ds, "stars", self, n_neighbours=nn, extra_fields=["luminosity_V"])
    ds_sph.add_field(
        ("io", "lumdens_V"),
        function=lambda field, data: data["io", "density"] * data["io", "luminosity_V"]/data["io", "particle_mass"],
        sampling_type="local",
        units="Lsun/kpc**3",
        force_override=True
    )
    ad = ds_sph.all_data()

    cen = cm if cm is not None else ad.center
    recompute = True if rcyl=="los-rh" else False
    if not recompute:
        rcyl = self.ds.quan(*rcyl) if isinstance(rcyl, tuple) else rcyl
        rcyl = rcyl.to("kpc")
        width = max( (ad.right_edge - ad.left_edge).to("kpc")[0]/2, 2*rcyl)

    tmp_disp_arr = self.ds.arr( -9999 * np.ones((lines_of_sight.shape[0])), self[ptype, "velocity"].units)
    for i, los in enumerate(lines_of_sight):
        if recompute:
            rcyl = Xmass_radius(self, ptype, cm=cm, mfrac=0.5, lines_of_sight=[los], project=True, light=kwargs.get("light", False))
            width = max( (ad.right_edge - ad.left_edge).to("kpc")[0]/2, 2*rcyl)
            print(rcyl)


        ad.set_field_parameter("axis" , los)
        dispersion = yt.ProjectionPlot(
            ds_sph,
            los,
            ("io", f"particle_velocity_los"),
            weight_field=("io", "mass"),
            moment=2,
            data_source=ad,
            center=cen,
            width=width
        )
        buff = np.array(dispersion.buff_size)
        size = (np.array(dispersion.width) * dispersion.width[0].units).to("kpc")
        pix_to_kpc = size / buff
        center = buff / 2
            
        rp = (rcyl / np.sqrt(pix_to_kpc[0] * pix_to_kpc[1])).to("").value
        surflum = yt.ProjectionPlot(
            ds_sph,
            los,
            ("io", f"lumdens_V"),
            data_source=ad,
            buff_size=buff,
            center=cen,
            width=width
        )
        
        vlos = dispersion.to_fits_data(("io", "particle_velocity_los")).get_data("particle_velocity_los_stddev").to("km/s").value
        lum = surflum.to_fits_data(("io", "lumdens_V")).get_data("lumdens_V").to("Lsun/kpc**2").value
        weighted_vlos = lum * vlos**2

        xp, yp = center
        positions = [(xp, yp)]
        aperture = CircularAperture(positions, r=float(rp))
        
        phot_table_lum = aperture_photometry(lum, aperture)
        total_lum = phot_table_lum['aperture_sum'][0]
        
        phot_table_weighted = aperture_photometry(weighted_vlos, aperture)
        total_weighted_sigma2 = phot_table_weighted['aperture_sum'][0]
        
        avg_sigma2 = total_weighted_sigma2 / total_lum
        
        tmp_disp_arr[i] = avg_sigma2


        ad.clear_data()

    return tmp_disp_arr


def enclosed_mass(self, ptype, center, r0):
    """Computes the enclosed mass on a sphere centered on center, and with radius r0.

    Parameters
    ----------
    r0 : unyt_quantity
        Radius
    center : unyt_array
        Center

    Returns
    -------
    encmass : unyt_quantity
    """
    mask = np.linalg.norm(self[ptype, "coordinates"] - center, axis=1) <= r0
    return self[ptype, "mass"][mask].sum()



def local_velocity_dispersion(pos, vel, **kwargs):
    """Computes the local 3D velocity dispersion for each member particle as the dispersion of the
    min(0.1 * Np, 7) nearest particles in 6D phase space. The metric reads:

                    Dij = |x_i - x_j|^2 + w_v*|v_i - v_j|^2
                    
    where w_v=sigma_pos^i / sigma_vel^i (where i is [x,y,z]).

    Parameters
    ----------
    pos, vel : unyt_array-like
        Position and velocity of member particles.
    - kwargs 
        centered : bool
            Whether the positions and velocities are corrected for center. Def: False.
        parallel : bool
            Whether to parallelize scipy.KDTREE. Def: False (might interfere with other parallelization).
        halo_params : dict[unyt_object]
            Dictionary containing center and center_vel keys. Required for non centered data. Def: None.

    Returns
    -------
    local_disp : unyt_array-like
        Local velocity dispersion for each particle, with units of vel.
    """
    from scipy.spatial import KDTree

    if not kwargs.get("centered", False):
        halo_params = kwargs.get("halo_params", None)
        assert halo_params is not None, "If data is not centered, you should provide halo params kwarg!"
        halo_center = halo_params['center'].to(pos.units)
        halo_center_vel = halo_params['center_vel'].to(vel.units)
        halorel_positions = (pos - halo_center)
        halorel_velocities = (vel - halo_center_vel)
    else:
        halorel_positions = pos
        halorel_velocities = vel


    n = int(max(0.01 * halorel_positions.shape[0], 7))
        
    wi = np.std(halorel_velocities, axis=0) / np.std(halorel_positions, axis=0)
    wi = wi.to(halorel_velocities.units / halorel_positions.units)

    halorel_velpos = halorel_velocities / wi
    assert halorel_velpos.units == halorel_positions.units

    data = np.zeros((halorel_positions.shape[0], halorel_positions.shape[1] * 2), dtype=float) * halorel_positions.units
    data[:, 0:3] = halorel_positions
    data[:, 3:] = halorel_velpos

    if kwargs.get("verbose", False):
        print(f"n: {n}")
        print(f"wi: {wi}")
        print(f"data:")
        print(f"{data}")
        
    kdtree = KDTree(data)
    _, i = kdtree.query(data, k=n, workers=-1 if kwargs.get("parallel", False) else 1)
    local_disp = -1 * np.ones((halorel_positions.shape[0],), dtype=float) * halorel_velocities.units
    for k, neighbour_index in enumerate(i):
        assert neighbour_index.shape[0] == n, "Wrong dimensions buddy!"
        assert neighbour_index.ndim == 1, "Wrong number of dimensions buddy!"

        local_disp[k] = np.std( np.linalg.norm(halorel_velocities[neighbour_index], axis=1) ) 

    return local_disp



































































































    
def density_profile(self,
                    ptype,
                    center,
                    bins=None,
                    project=None,
                    **kwargs
                   ):
    """Density profile
    """
    pos = self[ptype, "coordinates"]
    mass = self[ptype, "mass"]
    center = refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)

    if np.any(project):
        gs = gram_schmidt(project)
        pos = vectorized_base_change(np.linalg.inv(gs), pos)
        center = np.linalg.inv(gs) @ center

        pos = pos[:, :2]
        center = center[:2]


    
    if not np.any(bins):
        radii = np.linalg.norm(pos - center, axis=1)
        rmin, rmax, thicken, nbins = radii.min(), radii.max(), False, 10
        if "bins_params" in kwargs.keys():
            rmin = rmin if "rmin" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmin"]
            rmax = rmax if "rmax" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmax"]
            thicken = None if "thicken" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["thicken"]
            nbins = 10 if "bins" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["bins"]
            
        bins = np.histogram_bin_edges(
            np.log10(radii),
            bins=nbins,
            range=np.log10([rmin, rmax]) 
        )
        bins = 10 ** bins

        if thicken is not None:
            binindex = [i for i in range(len(bins)) if i==0 or i>thicken]
            bins = bins[binindex]

    result = densprof(
        pos,
        mass,
        center=center,
        bins=bins
    )
    result["bins"] = bins

    if np.any(project) and (result["dims"] != 2):
        raise Exception("You fucked up dimensions, bud!")

    return result


def velocity_profile(self, 
                     ptype,
                     center,
                     v_center,
                     bins=None,
                     project=None,
                     quantity="rms",
                     mode="bins",
                     **kwargs
                    ):
    """Velocity profile
    """
    pos = self[ptype, "coordinates"]
    vels = self[ptype, "velocity"]
    mass = self[ptype, "mass"]
    center = refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)
    v_center = refined_center6d(**kwargs["kw_center"])[1].to(vels.units) if v_center is None else v_center.to(vels.units)


    if np.any(project):
        gs = gram_schmidt(project)
        pos = vectorized_base_change(np.linalg.inv(gs), pos)
        vels = vectorized_base_change(np.linalg.inv(gs), vels)
        center = np.linalg.inv(gs) @ center
        v_center = np.linalg.inv(gs) @ v_center

        pos = pos[:, :2]
        center = center[:2]

        vels = vels[:, 0]
        v_center = v_center[0]



    if not np.any(bins):
        radii = np.linalg.norm(pos - center, axis=1)
        rmin, rmax, thicken, nbins = radii.min(), radii.max(), False, 10
        if "bins_params" in kwargs.keys():
            rmin = rmin if "rmin" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmin"]
            rmax = rmax if "rmax" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["rmax"]
            thicken = None if "thicken" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["thicken"]
            nbins = 10 if "bins" not in kwargs["bins_params"].keys() else kwargs["bins_params"]["bins"]
            
        bins = np.histogram_bin_edges(
            np.log10(radii), 
            bins=nbins,
            range=np.log10([rmin, rmax]) 
        )
        bins = 10 ** bins
        
        if thicken is not None:
            binindex = [i for i in range(len(bins)) if i==0 or i>thicken]
            bins = bins[binindex]

    
    if not np.any(project):
        result = velprof(
            pos,
            vels,
            center=center,
            v_center=v_center,
            bins=bins,
            projected=False,
            average=mode
        )
    if np.any(project) and mode=="bins":
        result = velprof(
            pos,
            vels,
            center=center,
            v_center=v_center,
            bins=bins,
            projected=True,
            average="bins",
            quantity=quantity
        )
    if np.any(project) and mode=="apertures":
        result = velprof(
            pos,
            vels,
            center=center,
            v_center=v_center,
            bins=bins,
            projected=True,
            average="apertures",
            quantity=quantity
        )

    result["bins"] = bins        
    return result
















































