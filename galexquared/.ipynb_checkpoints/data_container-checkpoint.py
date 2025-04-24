import yt
from copy import copy
import warnings
import numpy as np
from tqdm import tqdm

from .class_methods import half_mass_radius, refine_6Dcenter, easy_los_velocity, gram_schmidt, vectorized_base_change, density_profile, velocity_profile


#yt.utilities.logger.ytLogger.setLevel(40)


class DataContainer:
    def __init__(self, yt_region, particle_type, particle_ids=None, mask=None, old=None):
        """Initialize a new DataContainer.
        
        Parameters
        ----------
        yt_region : object
            A yt.Region-like object from which fields are lazily loaded.
        particle_type : str
            A string identifying the particle type (or gas field) for which this container is built.
        mask : numpy.ndarray or None
            A boolean array of length N (the total number of data points in yt_region)
            that selects which indices are active. If None, all data points are active.
        old : DataContainer or None
            For filtered containers, a reference to the full (unfiltered) container.
        particle_ids : array-like or None
            Optional. If provided, only particles whose IDs are in this list will be selected.
            It is assumed that the underlying region provides a field named f"{particle_type}_ids".
            
        Behavior regarding the “source” field:
          - If particle_ids is provided, a particle filter is registered and added to the dataset so
            that a new particle field f"{particle_type}_source" is created, containing only those
            particles with IDs in particle_ids.
          - If particle_ids is None, then f"{particle_type}_source" is simply taken to be the same
            as the original field (i.e. no additional filtering is done at the yt level).
        """            
        self.yt_region = yt_region
        self.ds = self.yt_region.ds
        self.arr = self.ds.arr
        self.quant = self.ds.quan
        self.particle_type = particle_type
        self._particle_ids = particle_ids
        self._mask = mask if mask is not None else None
        self._dq = {}
        self.old = old

        if self._particle_ids is not None:
            self._particle_mask = np.isin(self.yt_region[self.particle_type, "index"], self._particle_ids)
        else:
            self._particle_mask = np.ones_like(self.yt_region[self.particle_type, "index"].value, dtype=bool)

        self._fields = {}

    @property
    def dq(self):
        return self._dq
    @dq.setter
    def dq(self, keyval):
        if not isinstance(keyval, dict):
            raise TypeError("Expected a dictionary.")
        self._dq.update(keyval)    

    
    def __getitem__(self, field):
        """Retrieve a field by name. If not yet loaded, the field is loaded lazily
        from yt_region and then restricted using self._mask.
        
        Additionally, if the requested field matches the particle_type, the access is
        redirected to self.particle_type_field.
        """
        if field in self.dq:
            return self.dq[field]
        else:
            if field not in self._fields:
                full_data = self._load_field(field)
                if self._mask is None:
                    self._mask = np.ones(len(full_data), dtype=bool)
                
                self._fields[field] = full_data[self._mask]
            return self._fields[field]
            
        raise AttributeError(f"Dont know what you talking about, mate!")
    
    def _load_field(self, field):
        """Load a field from the underlying yt_region. Replace the code below with
        the appropriate call to your yt API as needed.
        """
        data = self.yt_region[self.particle_type, field]
        return data[self._particle_mask]

            

    def add_field(self, field_name, array):
        """Add a new field to the container.
        
        If this container is filtered (i.e. has an associated full container in self.old),
        then the field is also added to self.old. In that case, the new field data for the full
        dataset is built by padding with np.nan for indices that did not pass the filtering condition.
        
        Parameters
        ----------
        field_name : str
            The name of the field to add.
        array : numpy.ndarray
            The field data. Its length must equal the number of active (filtered) data points.
        """
        if self._mask is None:
            warnings.warn(f"Data length is still unknown. Make sure the array you provided has the correct length!")
        else:    
            current_length = np.sum(self._mask)
            if len(array) != current_length:
                raise ValueError(f"Length of provided array ({len(array)}) does not match "
                                 f"the current active data length ({current_length}).")

        self._fields[field_name] = copy(array)

        if self.old is not None:
            padded = np.full(len(self._mask), np.nan)
            padded[self._mask] = copy(array)
            self.old._fields[field_name] = padded

    def filter(self, condition):
        """Filter the data using the provided boolean condition and return a new DataContainer
        that only returns data for which the condition is True.
        
        The new filtered container will have an attribute 'old' that holds the full dataset.
        
        Parameters
        ----------
        condition : numpy.ndarray
            A boolean array whose length equals the current number of active data points.
        
        Returns
        -------
        DataContainer
            A new container instance containing only the data where condition is True.
        """
        if self._mask is None:
            warnings.warn(f"Data length is still unknown. Make sure the array you provided has the correct length!")
        else:    
            current_length = np.sum(self._mask)
            if len(condition) != current_length:
                raise ValueError(f"Length of provided array ({len(array)}) does not match "
                                 f"the current active data length ({current_length}).")

        active_indices = np.nonzero(self._mask)[0]
        new_mask = np.zeros_like(self._mask, dtype=bool)
        new_mask[active_indices] = condition

        full_container = self.old if self.old is not None else self

        new_container = DataContainer(
            self.yt_region.clone(),
            self.particle_type,
            particle_ids=self._particle_ids,
            mask=new_mask,
            old=full_container
        )
        new_container._particle_source = self.particle_type
        new_container._dq = self._dq
        
        for field, data in self._fields.items():
            new_container._fields[field] = data[condition]
        
        return new_container

    
    def extend(self, r):
        """Extends the size of the data container. Only works for yt.sphere
        """
        new_container = DataContainer(self.ds.sphere(self.yt_region.center, r).clone(), self.particle_type, particle_ids=self._particle_ids)
        new_container._dq = self._dq
        return new_container

    
    def empty(self):
        """Returns True if there is no data in the DataContainer instance.
        """
        if not bool(self._fields):
            return len(self["index"]) == 0
        else:
            return len(list(self._fields.values())[0]) == 0


    def close(self):
        self.ds.close()
        del self.ds
        del self.yt_region
        
        
    

    def refined_center6d(self, 
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
            bound_mask = self["total_energy"] < 0
            f = 0.1 if "f" not in kwargs.keys() else kwargs["f"]
            nbound = 32 if "nbound" not in kwargs.keys() else kwargs["nbound"]
            
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_mask), nbound)))
            most_bound_ids = np.argsort(self["total_energy"])[:N]
            most_bound_mask = np.zeros(len(self["total_energy"]), dtype=bool)
            most_bound_mask[most_bound_ids] = True
            
            tmp_cm = np.average(self["coords"][most_bound_mask], axis=0, weights=self["mass"][most_bound_mask])
            tmp_vcm = np.average(self["velocity"][most_bound_mask], axis=0, weights=self["mass"][most_bound_mask]) 
            
        elif method.lower() == "pot-softmax":
            bound_mask = self["total_energy"] < 0
            T = "adaptative" if "T" not in kwargs.keys() else kwargs["T"]
            
            w = self["total_energy"][bound_mask]/self["total_energy"][bound_mask].min()
            if T == "adaptative":
                T = np.abs(self["kinetic_energy"][bound_mask].mean()/self["total_energy"][bound_mask].min())
                
            tmp_cm = np.average(self["coordinates"][bound_mask], axis=0, weights=softmax(w, T))
            tmp_vcm = np.average(self["velocity"][bound_mask], axis=0, weights=softmax(w, T))
        
        else:                
            self._centering_results = refine_6Dcenter(
                self["coordinates"],
                self["mass"],
                self["velocity"],
                method=method,
                **kwargs
            )
    
            tmp_cm = self._centering_results['center']
            tmp_vcm = self._centering_results['velocity']

        self.dq = {"cm" : tmp_cm}
        self.dq = {"vcm" : tmp_vcm}     
        return tmp_cm, tmp_vcm

    def half_mass_radius(self, mfrac=0.5, lines_of_sight=None, project=False, light=False):
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

        tmp_rh_arr = self.arr( -9999 * np.ones((lines_of_sight.shape[0])), self["coordinates"].units)
        for i, los in enumerate(lines_of_sight):
            gs = gram_schmidt(los)
            new_coords = vectorized_base_change(np.linalg.inv(gs), self["coordinates"])
            new_cm = np.linalg.inv(gs) @ self["cm"]
            
            tmp_rh_arr[i] = half_mass_radius(
                new_coords, 
                self["mass"] if light is False else self["luminosity"], 
                new_cm, 
                mfrac, 
                project=project
            )

        if np.abs(mfrac - 0.5) <= 0.01:
            if project:
                self.dq = {"rh" : tmp_rh_arr.mean()}
                self.dq = {"e_rh" : tmp_rh_arr.std()}
            else:
                self.dq = {"rh3d" : tmp_rh_arr.mean()}

        if project:
            return tmp_rh_arr
        else:
            return tmp_rh_arr[0]

    def los_dispersion(self, lines_of_sight, rcyl=(1, 'kpc')):
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

        tmp_disp_arr = self.arr( -9999 * np.ones((lines_of_sight.shape[0])), self["velocity"].units)
        for i, los in enumerate(lines_of_sight):
            gs = gram_schmidt(los)
            cyl = self.ds.disk(self["cm"], los, radius=rcyl, height=(np.inf, 'kpc'), data_source=self.yt_region)
            tmp_disp_arr[i] = easy_los_velocity(cyl[self.particle_type, "velocity"], los).std()

        self.dq = {"sigma_los" : tmp_disp_arr.mean()}
        self.dq = {"e_sigma_los" : tmp_disp_arr.std()}
  
        return tmp_disp_arr

    
    def enclosed_mass(self, r0, center):
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
        mask = np.linalg.norm(self["coordinates"] - center, axis=1) <= r0
        return self["mass"][mask].sum()





    
    def density_profile(self, 
                        center=None,
                        bins=None,
                        project=None,
                        **kwargs
                       ):
        """Density profile
        """
        pos = self["coordinates"]
        mass = self["mass"]
        center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)

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

        result = density_profile(
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
                         center=None,
                         v_center=None,
                         bins=None,
                         project=None,
                         quantity="rms",
                         mode="bins",
                         **kwargs
                        ):
        """Velocity profile
        """
        pos = self["coordinates"]
        vels = self["velocity"]
        mass = self["mass"]
        center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units) if center is None else center.to(pos.units)
        v_center = self.refined_center6d(**kwargs["kw_center"])[1].to(vels.units) if v_center is None else v_center.to(vels.units)


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
            result = velocity_profile(
                pos,
                vels,
                center=center,
                v_center=v_center,
                bins=bins,
                projected=False,
                average=mode
            )
        if np.any(project) and mode=="bins":
            result = velocity_profile(
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
            result = velocity_profile(
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



   
            
    







        
     #   if self.old is None:    
     #       if self._particle_ids is not None:
     #           yt.add_particle_filter(
     #               self.particle_type, 
     #               function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], self._particle_ids), 
     #               filtered_type=self.particle_type, 
     #               requires=["index"]
     #   
     #           )
     #       else:
     #           yt.add_particle_filter(
     #               self.particle_type, 
     #               function=lambda pfilter, data: data[pfilter.filtered_type, "mass"] > 0, 
     #               filtered_type=self.particle_type, 
     #               requires=["mass"],    
     #           )
     #       self.ds.add_particle_filter(self.particle_type)

     #       self.particle_type = particle_type #f"{particle_type}_source"

