import numpy as np

from pytreegrav import PotentialTarget

from .halo_base import HaloParticles

import gc

from time import time

class HaloParticlesInside_Grav(HaloParticles):
    """Class to handle tracking of halo (track with dm + keep stars around) when inside of host virial radius
    (host != main galaxy).
    """
    def __init__(self,
                 sp,
                 particle_names,
                 previous_particle_info
                 ):
        super().__init__(sp, particle_names)
        
        self._ppi = previous_particle_info
        self._prepare_particle_sets()
        gc.collect()

    
    def _prepare_particle_sets(self):
        """Modifies the data atribute to include *_prev filter for previously bound particles
        """
        assert len(self.data[self.nbody, "particle_index"]) == len(self._ppi["all"]), "YT particle filter didnt work correctly."
        assert False not in np.isin(self.data[self.nbody, "particle_index"], self._ppi["all"]), "Particle Indexes are not matching."
        
        self.data.add_bound_mask(self._ppi["bound"], suffix="_prev_bound")
        self.bound_nbody = self.nbody + "_prev_bound"
        self.bound_stars = self.stars + "_prev_bound"
        self.bound_dm = self.dm + "_prev_bound"

    
    def compute_potential(self):
        """Computes potential of all particles.
        """
        st = time()
        n = self.data[self.nbody, "particle_index"].shape[0]
        nbody_grav_potential = self.arr( PotentialTarget(
                pos_target=self.data[self.nbody, "particle_position"].to("kpc"), 
                softening_target=None, 

                pos_source=self.data[self.bound_nbody, "particle_position"].to("kpc"), 
                m_source=self.data[self.bound_nbody, "particle_mass"].to("Msun"),
                softening_source=None, 
            
                G=4.300917270038E-6, 
            
                theta=min( 0.7, 0.4 * (n / 1E3) ** 0.1 ),
                quadrupole=True,
                tree=self.octree,
            
                parallel=True
            ), 
            'km**2/s**2'
        )
        ft = time()
        print("Time for potential calculation",ft-st, "s")
        
        nbody_grav_energy = (self.data[self.nbody, "particle_mass"] * nbody_grav_potential).to("Msun * km**2/s**2")
        self.data.add_field(
            (self.nbody, "grav_potential"),
            nbody_grav_potential,
            self.data[self.nbody, "particle_index"]
        )
        self.data.add_field(
            (self.nbody, "grav_energy"),
            nbody_grav_energy,
            self.data[self.nbody, "particle_index"]
        )

        st_mask = np.isin( self.data[self.nbody, "particle_index"],  self.data[self.stars, "particle_index"])
        self.data.add_field(
            (self.stars, "grav_potential"),
            nbody_grav_potential[st_mask],
            self.data[self.stars, "particle_index"]
        )
        self.data.add_field(
            (self.stars, "grav_energy"),
            nbody_grav_energy[st_mask],
            self.data[self.stars, "particle_index"]
        )

        dm_mask = np.isin( self.data[self.nbody, "particle_index"],  self.data[self.dm, "particle_index"])
        self.data.add_field(
            (self.dm, "grav_potential"),
            nbody_grav_potential[dm_mask],
            self.data[self.dm, "particle_index"]
        )
        self.data.add_field(
            (self.dm, "grav_energy"),
            nbody_grav_energy[dm_mask],
            self.data[self.dm, "particle_index"]
        )
        
        
    def compute_kinetic(self, vcm=None):
        """Computes kinetic energy of all particles.
        """
        if vcm is not None:
            bulk_vel = vcm
        else:
            bulk_vel = np.average(self.data[self.bound_dm, "particle_velocity"], axis=0, weights=self.data[self.bound_dm, "particle_mass"])

        print(bulk_vel)
        nbody_kin_energy =  0.5 * self.data[self.nbody, "particle_mass"] * np.linalg.norm(self.data[self.nbody, "particle_velocity"] - bulk_vel, axis=1) ** 2
        self.data.add_field(
            (self.nbody, "kinetic_energy"),
            nbody_kin_energy.to("Msun * km**2/s**2"),
            self.data[self.nbody, "particle_index"]
        )

        st_mask = np.isin( self.data[self.nbody, "particle_index"],  self.data[self.stars, "particle_index"])
        self.data.add_field(
            (self.stars, "kinetic_energy"),
            nbody_kin_energy[st_mask].to("Msun * km**2/s**2"),
            self.data[self.stars, "particle_index"]
        )
        dm_mask = np.isin( self.data[self.nbody, "particle_index"],  self.data[self.dm, "particle_index"])
        self.data.add_field(
            (self.dm, "kinetic_energy"),
            nbody_kin_energy[dm_mask].to("Msun * km**2/s**2"),
            self.data[self.dm, "particle_index"]
        )

        
    def compute_energy(self, vcm=None, beta=0.95):
        """Computes potential, kinetic and total energy all at once.
        Boundness is determined using bulk velocity (see ROCKSTAR).
        """
        self.compute_kinetic(vcm=vcm)
        self.compute_potential()
        
        self.data.add_field(
            (self.nbody, "total_energy"),
            self.data[self.nbody, "grav_energy"] + self.data[self.nbody, "kinetic_energy"],
            self.data[self.nbody, "particle_index"]
        )
        self.data.add_field(
            (self.stars, "total_energy"),
            self.data[self.stars, "grav_energy"] + self.data[self.stars, "kinetic_energy"],
            self.data[self.stars, "particle_index"]
        )
        self.data.add_field(
            (self.dm, "total_energy"),
            self.data[self.dm, "grav_energy"] + self.data[self.dm, "kinetic_energy"],
            self.data[self.dm, "particle_index"]
        )
        mask = self.data[self.nbody, "grav_energy"] + beta * self.data[self.nbody, "kinetic_energy"] < 0
        bound_indices = self.data[self.nbody, "particle_index"][mask]
        self.data.add_bound_mask(bound_indices)

        self.bound_nbody = self.nbody + f"_bound"
        self.bound_stars = self.stars + f"_bound"
        self.bound_dm = self.dm + f"_bound"
        
        
    def double_pass_unbinding(self, vcm=None, beta=0.95, **kwargs):
        """Performs a single pass unbinding to determine particle belongings and compute interesting quantities
        """
        self.compute_energy(vcm, beta)
        cm, vcm = self.cm_softmax()
        self.compute_kinetic()
        
        

