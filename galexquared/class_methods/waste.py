@classmethod
def format_value(cls, value):
    """Formats value using unyt if value != None, else returns none
    """
    if value is None:
        return None
        
    if type(value) == tuple:
        assert len(value) >= 1 and len(value) <= 2, f"Tuple must be of the formt (X,)==(X,'dimensionless') or (X,unit). Your provided {value}."
        if value[0] is None: return None
        else: return unyt_array(*value)
            
    else:
        return cls.format_value((value,))


@classmethod
def set_shared_attrs(cls, pt, kwargs):
    """Set class-level shared attributes for a specific particle type.
    """
    if pt not in cls._shared_attrs:
        raise ValueError(f"Unknown particle type: {pt}")
    elif kwargs is None:
        pass
    else:    
        for key, value in kwargs.items():
            for category in list(cls._shared_attrs[pt].keys()):
                if key in list(cls._shared_attrs[pt][category].keys()):
                    cls._shared_attrs[pt][category][key] = cls.format_value(value)
    

@classmethod
def get_shared_attr(cls, pt, key=None, cat=None):
    """Get a specific shared attribute for a particle type.
    """
    if pt not in cls._shared_attrs:
        raise ValueError(f"Unknown particle type: {pt}")

    if cat is not None and key is None:
            return cls._shared_attrs[pt].get(cat)
    elif cat is not None and key is not None:
            return cls._shared_attrs[pt][cat].get(key)            
    elif key is not None:
        for category in list(cls._shared_attrs[pt].keys()):
            if key in list(cls._shared_attrs[pt][category]):
                return cls._shared_attrs[pt][category].get(key)
    else:
        return cls._shared_attrs[pt]

@classmethod
def update_shared_attr(cls, pt, key, value):
    """Update a specific shared attribute for a particle type.
    """
    for category in list(cls._shared_attrs[pt].keys()):
        if  key in list(cls._shared_attrs[pt][category].keys()):
            cls._shared_attrs[pt][category][key] = value
            break

@classmethod
def list_shared_attributes(cls, pt, category):
    """List all shared attributes for a given particle type."""
    return list(cls._shared_attrs[pt].get(category, {}).keys())

@classmethod
def clean_shared_attrs(cls, pt):
    """Reset all shared attributes for a specific particle type to None."""
    if pt not in cls._shared_attrs:
        raise ValueError(f"Unknown particle type: {pt}")
    for category in list(cls._shared_attrs[pt].keys()):
        for key in list(cls._shared_attrs[pt][category].keys()):
            cls._shared_attrs[pt][category][key] = None

@classmethod
def set_shared_dataset(cls, dataset):
    """Set class-level shared attributes for a specific particle type.
    """
    cls._shared_dataset = dataset
    
@classmethod
def set_shared_datasource(cls, data_source):
    """Set class-level shared attributes for a specific particle type.
    """
    cls._shared_datasource = data_source
    
@classmethod        
def get_shared_data(cls):
    return cls._shared_dataset, cls._shared_datasource
    
@classmethod
def clean_shared_data(cls):
    """Set class-level shared attributes for a specific particle type.
    """
    cls._shared_datasource = None
    cls._shared_dataset = None     


def compute_stars_in_halo(self, 
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
    
        OPTIONAL Parameters
        ----------
        verbose : bool
            Wether to verbose or not. Default: False.        

        KEYWORD Parameters
        ----------
        center, center_vel : unyt_array
            Dark Matter halo center and center velocity.
        rvir : unyt_quantity
            virial radius
        vmax : unyt_quantity
            Maximum circular velocity of the halo
        vrms : unyt_quantity
            Disperion velocity of the halo
        max_radius : tuple[float, str] 
            Max radius to consider stellar particles. Default: 30 'kpc'
        imax : int
            Maximum number of iterations. Default: 200
            
        
            
        Returns
        -------
        indices : array
            Array of star particle indices belonging to the halo.
        mask : boolean-array
            Boolean array for masking quantities
        delta_rel : float
            Obtained convergence for selected total mass after imax iterations. >1E-2.
        """ 
        halo_params = {
            'center': self._shared_attrs["darkmatter"]["rockstar_center"],
            'center_vel': self._shared_attrs["darkmatter"]["rockstar_vel"],
            'rvir': self._shared_attrs["darkmatter"]["rvir"],
            'vmax': self._shared_attrs["darkmatter"]["vmax"],
            'vrms': self._shared_attrs["darkmatter"]["vrms"]                                                             
        }
        
        for key in halo_params.keys():
            halo_params[key] = halo_params[key] if key not in kwargs.keys() else kwargs[key]
        
        
        _, mask, delta_rel = compute_stars_in_halo(
            self.coords,
            self.masses,
            self.vels,
            self.IDs,
            halo_params=halo_params,
            max_radius=(30, 'kpc') if "max_radius" not in kwargs.keys() else kwargs["max_radius"],
            imax=200 if "imax" not in kwargs.keys() else int(kwargs["imax"]),
            verbose=verbose
        )
        
        self._starry_mask = mask
        self.delta_rel = delta_rel
        self.bound_method = "starry-halo"
        
        for key in list(self._fields_loaded.keys()):  
            if key.startswith("b"):
                del self._fields_loaded[key]
                
        return None        



def radius(self):
    if:        
        tmp_rh = half_mass_radius(
            self["coordinates"], 
            self["mass"], 
            self.q["cm"], 
            mfrac, 
            project=project
        )
    
    if project:
        self.update_shared_attr(
            self.ptype,
            "rh",
            tmp_rh            
        ) 
    else:
        self.update_shared_attr(
            self.ptype,
            "rh3d",
            tmp_rh            
        )             
    return tmp_rh
    
    
    
    mask = np.linalg.norm(self["coordinates"][:, 1:] - self.q["cm"][1:], axis=1) <= unyt_quantity(*rcyl)
    los_velocities = easy_los_velocity(self["velocity"][mask], [1,0,0])
    
    los_disp = np.std(self["velocity"][mask][:, 0])
    self.update_shared_attr(
        self.ptype,
        "sigma_los",
        los_disp            
    ) 
    if return_projections:
        return los_disp, los_velocities
    else:
        return los_disp







def compute_energies(self,
                     method="BH",
                     components=["stars","gas","darkmatter"],
                     cm_subset=["darkmatter"],
                     weighting="softmax",
                     verbose=False,
                     **kwargs
                    ):
    """Computes the kinetic, potential and total energies of the particles in the specified components and determines
    which particles are bound (un-bound) as E<0 (E>=0). Energies are stored as attributes. The gravitational potential
    canbe calculated in two distinct ways:

            1) Barnes-Hut tree-code implemented in pytreegrav, with allowance for force softening.
            2) By approximating the potential as for a particle of mass "m" located at "r" as:
                                     pot(r) = -G* M(<r) * m / |r - r_cm|
               this requires knowledge of the center-of-mass position to a good degree.

    Given that the initial center-of-mass position and velocity might be relativelly unknown (as the whole particle
    distribution is usually not a good estimator for these), the posibility of performing a refinement exist, where several
    iterationsa performed until the center-of-mass position and velocity converge `both` to delta. The enter-of-mass 
    position and velocity can be calculated using the N-most bound particles of using softmax weighting.

    OPTIONAL Parameters
    ----------
    method : str
        Method of computation. Either BH or APROX. Default: BH
    weighting : str
        SOFTMAX or MOST-BOUND. Names are self, explanatory. Default: softmax.
    components : list[str]
        Components to use: stars, gas and/or darkmatter. Default: all
    verbose : bool
        Verbose. Default: False

    KEYWORD ARGUMENTS
    -----------------
    cm, vcm : array
        Optional initial center-of-mass position and velocity.
    softenings : list[tuple[float, str]] or list[float]
        Softening for each particle type. Same shape as components. Default: [0,0,0]
    theta : float
        Opening angle for BH. Default: 0.7
    refine : bool
        Whether to refine. Default: False
    delta : float
        Converge tolerance for refinement. Default: 1E-5
    nbound : int
    Controls how many particles are used when estimating CoM properties through MOST-BOUND.
    T : int
    Controls how many particles are used when estimating CoM properties through SOFTMAX.
    parallel : bool
        Whether to parallelize BH computation. Default: True
    quadrupole : bool
        Whether to use quardupole approximation istead of dipole. Default: True

    Returns
    -------
    None
    """        
    if components == "particles":
        components = ["stars", "darkmatter"]
    elif components == "all":
        components = ["stars", "darkmatter", "gas"]

    if cm_subset == "particles":
        cm_subset = ["stars", "darkmatter"]
    elif cm_subset == "all":
        cm_subset = ["stars", "darkmatter", "gas"]
        
    masses = unyt_array(np.empty((0,)), "Msun")
    coords = unyt_array(np.empty((0,3)), "kpc")
    vels = unyt_array(np.empty((0,3)), "km/s")
    softenings = unyt_array(np.empty((0,)), "kpc")
    
    particle_types = np.empty((0,))
    particle_ids = np.empty((0,))

    for component in components:
        if not getattr(self, component).empty:                
            N = len(getattr(self, component)["mass"])
            masses = np.concatenate((
                masses, getattr(self, component)["mass"].to("Msun")
            ))
            coords = np.vstack((
                coords, getattr(self, component)["coordinates"].to("kpc")
            ))
            vels = np.vstack((
                vels, getattr(self, component)["velocity"].to("km/s")
            ))
            softenings = np.concatenate((
                softenings, getattr(self, component)["softening"].to("kpc")
            ))
            particle_ids = np.concatenate((
                particle_ids, getattr(self, component)["index"]
            ))
            particle_types = np.concatenate((
                particle_types, np.full(N, component)
            ))


    thermal_energy = unyt_array(np.zeros_like(masses).value, "Msun * km**2/s**2")
    if "gas" in components:
        thermal_energy[particle_types == "gas"] = self.gas["thermal_energy"].to("Msun * km**2/s**2")

    particle_subset = np.zeros_like(particle_types, dtype=bool)
    for sub in cm_subset:
        particle_subset = particle_subset | (particle_types == sub)


        
    if method.lower() == "bh":
        E, kin, pot, cm, vcm = bound_particlesBH(
            coords,
            vels,
            masses,
            softs=softenings,
            extra_kin=thermal_energy,
            cm=None if "cm" not in kwargs else unyt_array(*kwargs['cm']) if isinstance(kwargs['cm'], tuple) and len(kwargs['cm']) == 2 else kwargs['cm'],
            vcm=None if "vcm" not in kwargs else unyt_array(*kwargs['vcm']) if isinstance(kwargs['vcm'], tuple) and len(kwargs['vcm']) == 2 else kwargs['vcm'],
            cm_subset=particle_subset,
            weighting=weighting,
            refine=True if "refine" not in kwargs.keys() else kwargs["refine"],
            delta=1E-5 if "delta" not in kwargs.keys() else kwargs["delta"],
            f=0.1 if "f" not in kwargs.keys() else kwargs["f"],
            nbound=32 if "nbound" not in kwargs.keys() else kwargs["nbound"],
            T=0.22 if "T" not in kwargs.keys() else kwargs["T"],
            return_cm=True,
            verbose=verbose,

        )
    elif method.lower() == "aprox":
        E, kin, pot, cm, vcm = bound_particlesAPROX(
            coords,
            vels,
            masses,
            extra_kin=thermal_energy,
            cm=None if "cm" not in kwargs else unyt_array(*kwargs['cm']) if isinstance(kwargs['cm'], tuple) and len(kwargs['cm']) == 2 else kwargs['cm'],
            vcm=None if "vcm" not in kwargs else unyt_array(*kwargs['vcm']) if isinstance(kwargs['vcm'], tuple) and len(kwargs['vcm']) == 2 else kwargs['vcm'],
            cm_subset=particle_subset,
            weighting=weighting,
            refine=True if "refine" not in kwargs.keys() else kwargs["refine"],
            delta=1E-5 if "delta" not in kwargs.keys() else kwargs["delta"],
            f=0.1 if "f" not in kwargs.keys() else kwargs["f"],
            nbound=32 if "nbound" not in kwargs.keys() else kwargs["nbound"],
            T=0.22 if "T" not in kwargs.keys() else kwargs["T"],
            return_cm=True,
            verbose=verbose,

        )

    self.update_shared_attr("halo", "cm", cm)
    self.update_shared_attr("halo", "vcm", vcm)
   
    if "gas" in components:
        self._ds.add_field(
            (self.ptypes["gas"], "E"),
            function=lambda field, data: E[particle_types == "gas"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["gas"], "kin"),
            function=lambda field, data: kin[particle_types == "gas"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["gas"], "pot"),
            function=lambda field, data: pot[particle_types == "gas"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
    if "stars" in components:
        self._ds.add_field(
            (self.ptypes["stars"], "E"),
            function=lambda field, data: E[particle_types == "stars"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["stars"], "kin"),
            function=lambda field, data: kin[particle_types == "stars"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["stars"], "pot"),
            function=lambda field, data: pot[particle_types == "stars"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
    if "darkmatter" in components:
        self._ds.add_field(
            (self.ptypes["darkmatter"], "E"),
            function=lambda field, data: E[particle_types == "darkmatter"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["darkmatter"], "kin"),
            function=lambda field, data: kin[particle_types == "darkmatter"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )
        self._ds.add_field(
            (self.ptypes["darkmatter"], "pot"),
            function=lambda field, data: pot[particle_types == "darkmatter"],
            sampling_type="local",
            units='Msun*km**2/s**2',
            force_override=True
        )

    self._update_data()


























class DarkComponent(BaseHaloObject, BaseComponent):
    """ptype class that contains the particle data, for each particle type present/relevant to your analysis, in the simulation. 
    Fields are stored as attributes and have units thanks to unyt. The class might have as much fields as wished by the user,
    but a few are mandatory:
        
    Mandatory fields
    ----------------
    - coordinates : stored as self.coords
    - velocities : stored as self.vels
    - masses : stored as self.masses
    - IDs : stored as self.ids
    """
    def __init__(self,
                 data,
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()        
        self.ptype, self._base_ptype = "darkmatter", self.ptypes["darkmatter"]
        self._data = data
        
        self.clean_shared_attrs(self.ptype)
        self.set_shared_attrs(self.ptype, kwargs)
        
        del self.ptypes
          
    @property
    def q(self):
        return self.get_shared_attr(self.ptype, cat="quantities")
    @property
    def m(self):
        return self.get_shared_attr(self.ptype, cat="moments")



    def __getitem__(self, key):
        """Retrieve the value for the given key, dynamically computing it if it's a dynamic field.
        """
        return self._data[self._base_ptype, key]
    
    def info(self, 
             get_str=False
            ):
        """Returns a pretty information summary.
        
        Parameters
        ----------
        get_str : bool
            Return str instead of print. Default: False

        Returns
        -------
        info : str, optionally

        """
        output = []
        
        output.append(f"\n{self.ptype}")
        output.append(f"{'':-<21}")
        try:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'pos[0]':<20}: [{self.coords[0,0].value:.2f}, {self.coords[0,1].value:.2f}, {self.coords[0,2].value:.2f}] {self.units['length']}")
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            output.append(f"{'ID[0]':<20}: {self.IDs[0].value}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")
            output.append(f"{'len_ids':<20}: {len(self.coords)}")

        output.append(f"{'cm':<20}: {self.cm}")
        output.append(f"{'vcm':<20}: {self.vcm}")
        
        output.append(f"{'rh, rh3D':<20}: {self.rh}, {self.rh3d}")
        output.append(f"{'rvir, rs, c':<20}: {self.rvir}, {self.rs}, {self.c}")

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None

    















class GasComponent(BaseHaloObject, BaseComponent):
    """
    """
    def __init__(self,
                 data,
                 **kwargs
                 ):
        """Initializes the ptype class.
        """
        super().__init__()
        self.ptype, self._base_ptype = "gas", self.ptypes["gas"]
        self._data = data

        self.clean_shared_attrs(self.ptype)
        self.set_shared_attrs(self.ptype, kwargs)

        del self.ptypes

    @property
    def q(self):
        return self.get_shared_attr(self.ptype, cat="quantities")
    @property
    def m(self):
        return self.get_shared_attr(self.ptype, cat="moments")



    def __getitem__(self, key):
        """Retrieve the value for the given key, dynamically computing it if it's a dynamic field.
        """
        return self._data[self._base_ptype, key]

    def info(self, 
             get_str=False
            ):
        """Returns a pretty information summary.
        
        Parameters
        ----------
        get_str : bool
            Return str instead of print. Default: False

        Returns
        -------
        info : str, optionally

        """
        output = []
        
        output.append(f"\n{self.ptype}")
        output.append(f"{'':-<21}")
        try:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'pos[0]':<20}: [{self.coords[0,0].value:.2f}, {self.coords[0,1].value:.2f}, {self.coords[0,2].value:.2f}] {self.units['length']}")
            output.append(f"{'len_vel':<20}: {len(self.vels)}")
            output.append(f"{'vel[0]':<20}: [{self.vels[0,0].value:.2f}, {self.vels[0,1].value:.2f}, {self.vels[0,2].value:.2f}] {self.units['velocity']}")
            output.append(f"{'len_mass':<20}: {len(self.masses)}")
            output.append(f"{'mass[0]':<20}: {self.masses[0]}")
            output.append(f"{'len_ids':<20}: {len(self.IDs)}")
            
        except:
            output.append(f"{'len_pos':<20}: {len(self.coords)}")
            output.append(f"{'len_vel':<20}: {len(self.coords)}")
            output.append(f"{'len_mass':<20}: {len(self.coords)}")

        output.append(f"{'cm':<20}: {self.cm}")
        output.append(f"{'vcm':<20}: {self.vcm}")
        output.append(f"{'rh, rh3D':<20}: {self.rh}, {self.rh3d}")

        if get_str:
            return "\n".join(output)
        else:
            print("\n".join(output))
            return None














class BaseComponent:
    """BaseParticleType class that implements common methods and attributes for particle ensembles. These methods and attributes
    are accesible for all particle types and hence this class acts as a bridge between stars, darkmatter and gas, allowing 
    them to access properties of one another. This makes sense, as particles types in cosmological simulations are coupled to
    each other.
    
    It also simplifies the code, as a plethora of common methods are displaced to here.
    """
    
    def _default_center_of_mass(self):
        """Computes coarse CoM using all the particles as 

                CoM = sum(mass * pos) / sum(mass)        
        """
        if self.masses.sum() != 0:
            self.cm = center_of_mass_pos(
                self.coords, 
                self.masses
            )
            self.vcm = center_of_mass_vel(
                self.coords, 
                self.masses,
                self.vels,
                R=(1E4, "Mpc")
            )
        else:
            self.empty = True
            self.cm = None 
            self.vcm = None
        return None



    
    def knn_distance(self, center_point, k):
        """Returns the distance from center_point to the k nearest neighbour.

        Parameters
        ----------
        center_point : array
            center point of the knn search.
        k : int
            number of neighbours

        Returns
        -------
        dist : foat
        """        
        self._KDTree = KDTree(self.coords)        
        distances, _ = tree.query(center_point, k=k)
        
        return distances[-1] * self.coords.units

        
    def set_line_of_sight(self, los):
        """Sets the line of sight to the provided value. The default value for the line of sight is the x-axis: [1,0,0].
        By setting a new line of sight the coordinate basis in which vectorial quantities are expressed changes, aligning the
        old x-axis with the new provided line of sight. This way, a projected view of a set of particles when viewed 
        through los can be obtained by retrieving the new y,z-axes.
        
        The new coordinate system is obtained applying Gram-Schmidt to a preliminary non-orthogonal basis formed by los + identitiy.
        
        Parameters
        ----------
        los : array
            Unitless, non-normalized line of sight vector
            
        Returns
        -------
        None
        
        Changes basis and los instances in the class instance. All vectorial quantities get expressed in the new
        coordinate basis.
        """
        self._set_los(los)
        self._delete_vectorial_fields()
        if self.cm is not None:
            self.cm = self._old_to_new_base @ self.cm
        if self.vcm is not None:
            self.vcm = self._old_to_new_base @ self.vcm

        return None

        
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
        cm : array
            Refined Center of mass and various quantities.
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no center-of-mass to refine!")
            
        if method.lower() == "pot-most-bound":
            bound_mask = self.E < 0
            f = 0.1 if "f" not in kwargs.keys() else kwargs["f"]
            nbound = 32 if "nbound" not in kwargs.keys() else kwargs["nbound"]
            
            N = int(np.rint(np.minimum(f * np.count_nonzero(bound_mask), nbound)))
            most_bound_ids = np.argsort(self.E)[:N]
            most_bound_mask = np.zeros(len(self.E), dtype=bool)
            most_bound_mask[most_bound_ids] = True
            
            self.cm = np.average(self.coords[most_bound_mask], axis=0, weights=self.masses[most_bound_mask])
            self.vcm = np.average(self.vels[most_bound_mask], axis=0, weights=self.masses[most_bound_mask]) 
            
        elif method.lower() == "pot-softmax":
            bound_mask = self.E < 0
            T = "adaptative" if "T" not in kwargs.keys() else kwargs["T"]
            
            w = self.E[bound_mask]/self.E[bound_mask].min()
            if T == "adaptative":
                T = np.abs(self.kin[bound_mask].mean()/self.E[bound_mask].min())
                
            self.cm = np.average(self.coords[bound_mask], axis=0, weights=softmax(w, T))
            self.vcm = np.average(self.vels[bound_mask], axis=0, weights=softmax(w, T))               


        else:
            self._centering_results = refine_6Dcenter(
                self.bcoords,
                self.bmasses,
                self.bvels,
                method=method,
                **kwargs
            )
    
            self.cm = self._centering_results['center']
            self.vcm = self._centering_results['velocity']
        return self.cm, self.vcm
    
    def half_mass_radius(self, mfrac=0.5, project=False, only_bound=True):
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
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no half mass radius to compute!")
            
        if only_bound:
            if True in self._bmask:
                rh = half_mass_radius(
                    self.bcoords, 
                    self.bmasses, 
                    self.cm, 
                    mfrac, 
                    project=project
                )
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no half-mass radius to compute!")
        else:
            rh = half_mass_radius(
                self.coords, 
                self.masses, 
                self.cm, 
                mfrac, 
                project=project
            )
        
        if project:
            self.rh = rh
        else:
            self.rh3d = rh
            
        return rh


    def los_dispersion(self, rcyl=(1, 'kpc'), return_projections=False, only_bound=False):
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
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no line of sight velocity to compute")

        mask = np.linalg.norm(self.coords[:, 0:2] - self.cm[0:2], axis=1) < unyt_array(*rcyl)

        if only_bound:
            if True in self._bmask:
                los_velocities = easy_los_velocity(self.vels[mask & self._bmask], self.los)
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no line of sight velocity to compute!")

        else:
            los_velocities = easy_los_velocity(self.vels[mask], self.los)

        
        losvel = np.std(los_velocities)
        
        if return_projections:
            return losvel, los_velocities
        else:
            return losvel

    def enclosed_mass(self, r0, center, only_bound=False):
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
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no enclosed mass to compute!")
            
        if only_bound:
            if True in self._bmask:
                mask = np.linalg.norm(self.bcoords - center, axis=1) <= r0
                return self.bmasses[mask].sum()
            else:
                return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no enclosed mass to compute!")
        else:
            mask = np.linalg.norm(self.coords - center, axis=1) <= r0
            return self.masses[mask].sum()

    
    def density_profile(self, 
                        pc="bound",
                        center=None,
                        bins=None,
                        projected=False,                      
                        return_bins=False,
                        **kwargs
                       ):
        """Computes the average density profile of the particles. Returns r_i (R_i), rho_i and e_rho_i (Sigma_i, e_Sigma_i) for each bin. Center
        and bins are doanematically computed but can also be used specified. Profiles can be for all or bound particles. The smallest two bins
        can be combined into one to counteract lack of resolution. Density error is assumed to be poissonian.

        OPTIONAL Parameters
        ----------
        pc : str
            Particle-Component. Either bound or all.
        center : array
            Center of the particle distribution. Default: None.
        bins : array
            Array of bin edges. Default: None.
        projected : bool
            Whether to get the projected distribution at current LOS. Default: False.
        return_bins : bool
            Whether to return bin edges. Default: False
        
        Returns
        -------
        r, dens, e_dens, (bins, optional) : arrays of bin centers, density and errors (and bin edges)
        
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no density profile to compute!")

        
        if "new_data_params" in kwargs.keys():
            if "sp" in kwargs["new_data_params"].keys():
                sp = kwargs["new_data_params"]["sp"]
            else:
                sp = self._data.ds.sphere(
                    self._sp_center if "center" not in kwargs["new_data_params"].keys() else kwargs["new_data_params"]["center"], 
                    kwargs["new_data_params"]["radius"]
                )

            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["coords"]].in_units(self.coords.units)
            )
            mass = sp[self._base_ptype, self._dynamic_fields["masses"]].in_units(self.masses.units)
            
        else:
            if pc == "bound":
                if np.any(self._bmask) == False:
                    return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no density profile to compute!")
                else:
                    pos = self.bcoords
                    mass = self.bmasses
            elif pc == "all":
                pos = self.coords
                mass = self.masses
                
        
        if center is None:
            center = self.refined_center6d(**kwargs["kw_center"])[0].to(pos.units)

            
        else:
            if isinstance(center, tuple):
                center = unyt_array(*center).to(pos.units)
    

        
        if projected:
            pos = pos[:, :2]
            center = center[:2]

        
        if bins is None:
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

        if projected and (result["dims"] != 2):
            raise Exception("you fucked up dimensions, bud")

        if return_bins:
            return result["r"], result["rho"], result["e_rho"], bins
        else:
            return result["r"], result["rho"], result["e_rho"]

    
    def velocity_profile(self, 
                         pc="bound",
                         center=None,
                         v_center=None,
                         bins=None,
                         projected="none",
                         quantity="rms",
                         return_bins=False,
                         **kwargs
                        ):
        """Computes the average disperion velocity profile of the particles. Returns r_i (R_i), vrms_i and e_vrms_i for each bin. Center
        and bins are doanematically computed but can also be used specified. Profiles can be for all or bound particles. The smallest two bins
        can be combined into one to counteract lack of resolution. Density error is assumed to be poissonian.

        OPTIONAL Parameters
        ----------
        pc : str
            Particle-Component. Either bound or all.
        center, v_center : array
            Center of the particle distribution. Default: None.
        bins : array
            Array of bin edges. Default: None.
        projected : bool
            Whether to get the projected distribution at current LOS. Default: False.
        return_bins : bool
            Whether to return bin edges. Default: False
        
        Returns
        -------
        r, vrms, e_vrms, (bins, optional) : arrays of bin centers, density and errors (and bin edges)
        """
        if self.empty:
            return AttributeError(f"Compoennt {self.ptype} is empty! and therefore there is no velocity profile to compute!")
            
        if "new_data_params" in kwargs.keys():
            if "sp" in kwargs["new_data_params"].keys():
                sp = kwargs["new_data_params"]["sp"]
            else:
                sp = self._data.ds.sphere(
                    self._sp_center if "center" not in kwargs["new_data_params"].keys() else kwargs["new_data_params"]["center"], 
                    kwargs["new_data_params"]["radius"]
                )
                
            pos = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["coords"]].in_units(self.coords.units)
            )
            vels = vectorized_base_change(
                np.linalg.inv(self.basis), 
                sp[self._base_ptype, self._dynamic_fields["vels"]].in_units(self.vels.units)
            )
        else:
            if pc == "bound":
                if np.any(self._bmask) == False:
                    return AttributeError(f"Compoennt {self.ptype} has no bound mass! and therefore there is no velocity profile to compute!")
                else:    
                    pos = self.bcoords
                    vels = self.bvels
            elif pc == "all":
                pos = self.coords
                vels = self.vels

        
        
        if center is None:
            center, v_center = self.refined_center6d(**kwargs["kw_center"])
            center, v_center = center.to(pos.units), v_center.to(vels.units)
        else:
            if isinstance(center, tuple):
                center = unyt_array(*center).to(pos.units)
            if isinstance(v_center, tuple):
                v_center = unyt_array(*v_center).to(vels.units)
        

        if projected != "none":
            pos = pos[:, :2]
            vels = easy_los_velocity(vels - v_center, [1,0,0])
            center = center[:2]
            v_center = np.array([0])



        
        if bins is None:
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

        if projected == "none":
            result = velocity_profile(
                pos,
                vels,
                center=center,
                v_center=v_center,
                bins=bins,
                projected=False,
                average="bins"
            )
        if projected == "radial-bins" or projected == "bins" :
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
        elif projected == "apertures":
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

        
        if return_bins:
            return result["r"], result["v"], result["e_v"], bins
        else:
            return result["r"], result["v"], result["e_v"]



   
        























