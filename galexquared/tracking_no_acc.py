import os
import yt
import sys
import h5py
import warnings
import numpy as np
import pandas as pd
from copy import copy
from pathlib import Path

from scipy.optimize import root_scalar
from scipy.stats import binned_statistic

from pytreegrav import Potential, PotentialTarget, ConstructTree

from .config import config
from .mergertree import MergerTree
from .class_methods import load_ftable, softmax, half_mass_radius, refine_6Dcenter


import gc

class ParticleData:
    """Simple class to handle particle data, with an option to provide a mask
    """
    def __init__(self,
                 sp,
                 particle_names
                ):
        """Init function.
        """
        self.datasrc = sp 
        self.ds = sp.ds
        self.nbody = particle_names["nbody"]
        self.dm = particle_names["darkmatter"]
        self.stars = particle_names["stars"]

        self._data = {
            (self.nbody, "particle_position") : sp[self.nbody, "particle_position"].to("kpc"),
            (self.nbody, "particle_mass") :  sp[self.nbody, "particle_mass"].to("Msun"),
            (self.nbody, "particle_velocity") :  sp[self.nbody, "particle_velocity"].to("km/s"),
            (self.nbody, "particle_index") :  sp[self.nbody, "particle_index"],

            (self.stars, "particle_position") : sp[self.stars, "particle_position"].to("kpc"),
            (self.stars, "particle_mass") :  sp[self.stars, "particle_mass"].to("Msun"),
            (self.stars, "particle_velocity") :  sp[self.stars, "particle_velocity"].to("km/s"),
            (self.stars, "particle_index") :  sp[self.stars, "particle_index"],

            (self.dm, "particle_position") : sp[self.dm, "particle_position"].to("kpc"),
            (self.dm, "particle_mass") :  sp[self.dm, "particle_mass"].to("Msun"),
            (self.dm, "particle_velocity") :  sp[self.dm, "particle_velocity"].to("km/s"),
            (self.dm, "particle_index") :  sp[self.dm, "particle_index"],
        }
        self._masks = {}
        sp.clear_data()
        
    def __getitem__(self, key):
        return self._data[key]

    def _get_keys(self, selected_ptype):
        """Get keys for a given particle
        """
        return  [field for (ptype, field) in self._data.keys() if ptype == selected_ptype]

    def add_field(self, field_name, field_value, field_indices):
        """ Adds fields to data.
        """
        indexes = self._data[field_name[0], "particle_index"]

        sorter = np.argsort(field_indices)
        field_value_sorted = field_value[sorter]
        field_indices_sorted = field_indices[sorter]
        
        positions = np.searchsorted(field_indices_sorted, indexes)
        ordered_field_value = field_value_sorted[positions]

        self._data[field_name] = ordered_field_value

        
        
        
    def add_bound_mask(self, bound_indices, suffix="_bound"):
        """Adds fields for boundness
        """
        particles = [self.nbody, self.dm, self.stars]
        for ptype in particles:
            all_fields = self._get_keys(ptype)
            mask = np.isin(self._data[ptype, "particle_index"], bound_indices)
            self._masks[ptype] = mask
            for field in all_fields:
                self._data[ptype + suffix, field] = self._data[ptype, field][mask]

    
    def rename_particle(self, ptype, new_name):
        """Renames particle
        """
        all_fields = self._get_keys(ptype)
        for field in all_fields:
            self._data[new_name, field] = self._data.pop(ptype, field)

        gc.collect()


    def remove_particle(self, ptype):
        """Removes particle from data
        """
        all_fields = self._get_keys(ptype)
        for field in all_fields:
            self._data.pop(ptype, field)

        gc.collect()



class HaloParticles:
    """Class that contains the particles (stellar and dark) that conform the halo. Has utilities for computing boundness,
    virial, tidal radius and center of mass position and velocity.
    """
    def __init__(self,
                 data_source,
                 particle_names
                ):
        """Init function.
        """
        self.nbody = particle_names["nbody"]
        self.dm = particle_names["darkmatter"]
        self.stars = particle_names["stars"]
        self._particle_names = particle_names
        self._octree = None

        self.data = ParticleData(data_source, particle_names)
        self.ds = self.data.ds
        self.quan = self.ds.quan
        self.arr = self.ds.arr
        self.cosmo = self.ds.cosmology
        self.redshift = self.ds.current_redshift
        self.crit_dens = self.critical_density()
        self.virial_overdens = self.overdensity()

        self.b_a = {}
        self.c_a = {}
        self.I = {}
        self.shape_tensor = {}
        self.A = {}
        
        self.cm = {}
        self.vcm = {}
        
        self.bound_dm = self.dm
        self.bound_nbody = self.nbody
        self.bound_stars = self.stars
        
    @property
    def octree(self):
        if self._octree is None:
            self._octree = ConstructTree(
                self.data[self.nbody, "particle_position"].to("kpc"),
                m=self.data[self.nbody, "particle_mass"].to("Msun"),
                softening=None,
                quadrupole=True,

            )
        return self._octree
            

    def _cm_nmostbound(self, N=32, f=0.1):
        """Returns the CoM computed with the N=min(32, f * Ntot) most bound particles
        """
        Npart = int(np.rint(np.minimum(f * self.data[self.bound_dm, "particle_index"].shape[0], N)))
        most_bound_ids = np.argsort(self.data[self.bound_dm, "total_energy"])[:Npart]
        
        mask = np.zeros(len(self.data[self.bound_dm, "particle_index"]), dtype=bool)
        mask[most_bound_ids] = True
        
        self.neff = Npart

        tmp_cm = np.average(self.data[self.bound_dm, "particle_position"][mask], axis=0, weights=self.data[self.bound_dm, "particle_mass"][mask]).to("kpc")
        tmp_vcm = np.average(self.data[self.bound_dm, "particle_velocity"][mask], axis=0, weights=self.data[self.bound_dm, "particle_mass"][mask]).to("km/s")
        return tmp_cm, tmp_vcm
    
    def _cm_softmax(self, T="adaptative"):
        """Returns the CoM computed via softmax with wheights being:
                            w ~ e^-E/T 
        and T = | mean(kin) / min(E)| ~ 0.2
        """        
        w = self.data[self.bound_dm,"total_energy"]/self.data[self.bound_dm,"total_energy"].min()
        if T == "adaptative":
            T = np.abs(self.data[self.bound_dm, "kinetic_energy"].mean()/self.data[self.bound_dm,"total_energy"].min())
            
        self.neff = 1 / np.sum(softmax(w, T)**2)
        self.T = T
        tmp_cm = np.average(self.data[self.bound_dm, "particle_position"], axis=0, weights=softmax(w, T)).to("kpc")
        tmp_vcm = np.average(self.data[self.bound_dm, "particle_velocity"], axis=0, weights=softmax(w, T)).to("km/s")
        return tmp_cm, tmp_vcm
    
    def _cm_deepest_potential(self, N=32, f=0.1):
        """Computes the center as the average position of the deepest particles in the potential
        """
        Npart = int(np.rint(np.minimum(f * self.data[self.bound_dm, "particle_index"].shape[0], N)))
        most_bound_ids = np.argsort(self.data[self.bound_dm, "grav_potential"])[:Npart]
        
        mask = np.zeros(len(self.data[self.bound_dm, "particle_index"]), dtype=bool)
        mask[most_bound_ids] = True
        
        self.neff = Npart

        tmp_cm = np.average(self.data[self.bound_dm, "particle_position"][mask], axis=0, weights=self.data[self.bound_dm, "particle_mass"][mask]).to("kpc")
        tmp_vcm = np.average(self.data[self.bound_dm, "particle_velocity"][mask], axis=0, weights=self.data[self.bound_dm, "particle_mass"][mask]).to("km/s")
        return tmp_cm, tmp_vcm

    def _cm_no_potential(self, method, **kwargs):
        """Computes an approximate center without using potential energy information
        """
        self._cen_result = refine_6Dcenter(
            self.data[self.dm, "particle_position"].to("kpc"),
            self.data[self.dm, "particle_mass"].to("Msun"),
            self.data[self.dm, "particle_velocity"].to("km/s"),
            method=method,
            **kwargs
        )
        return self._cen_result["center"], self._cen_result["particle_velocity"]




    def cm_nmostbound(self, N=32, f=0.1):
        """Returns the CoM computed with the N=min(32, f * Ntot) most bound particles
        """
        tmp_cm, tmp_vcm = self._cm_nmostbound(N=N, f=f)
        
        self.cm["darkmatter"] = tmp_cm
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_cm, tmp_vcm
    
    def cm_softmax(self, T="adaptative"):
        """Returns the CoM computed via softmax with wheights being:
                            w ~ e^-E/T 
        and T = | mean(kin) / min(E)| ~ 0.2
        """        
        tmp_cm, tmp_vcm = self._cm_softmax(T=T)
        
        self.cm["darkmatter"] = tmp_cm
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_cm, tmp_vcm
    
    def cm_deepest(self, N=32, f=0.1):
        """Returns the CoM computed with the N=min(32, f * Ntot) most deepest potential particles
        """
        tmp_cm, tmp_vcm = self._cm_deepest_potential(N=N, f=f)
        
        self.cm["darkmatter"] = tmp_cm
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_cm, tmp_vcm
    
    def cm_nopotential(self, method="adaptative", **kwargs):
        """Returns the CoM computed with the N=min(32, f * Ntot) most deepest potential particles
        """
        tmp_cm, tmp_vcm = self._cm_no_potential(method=method, **kwargs)
        
        self.cm["darkmatter"] = tmp_cm
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_cm, tmp_vcm

    def rockstar_velocity(self):
        """Computes velocity like a rockstar.
        """
        assert "darkmatter" in self.cm, "You first need to compute the center of mass of the halo!"
        center = self.cm["darkmatter"]
        sp2 = self.ds.sphere(center, self.rvir * 0.1)
        tmp_vcm = sp2.quantities.bulk_velocity(use_gas=False, use_particles=True, particle_type=self.dm).to("km/s")
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_vcm







    def update_data(self, new_data):
        """Changes the sphere data source to a new one with center and radius
        specified as arguments
        """
        self.data = ParticleData(new_data, self._particle_names)
        self._octree = None
        self.bound_dm = self.dm
        self.bound_nbody = self.nbody
        self.bound_stars = self.stars





    def overdensity(self):
        """Computes the virial collapse overdensity.
        """        
        E = self.cosmo.hubble_parameter(self.redshift).to("km/s * 1/Mpc") / self.cosmo.hubble_constant
        omega_z = (1 + self.redshift)**3 * self.cosmo.omega_matter / E**2 
        x = omega_z - 1

        if self.cosmo.omega_radiation == 0:
            overdensity = 18 * np.pi**2 + 82 * x - 39 * x**2
        elif self.cosmo.omega_lambda == 0:
            overdensity = 18 * np.pi**2 + 60 * x - 32 * x**2
        return overdensity

    def critical_density(self):
        """Computes the critical density.
        """
        crit_dens = self.cosmo.critical_density(self.redshift)
        return crit_dens.to("Msun/kpc**3")

    def mean_density(self):
        """Computes the critical density.
        """
        E = self.cosmo.hubble_parameter(self.redshift).to("km/s * 1/Mpc") / self.cosmo.hubble_constant
        omega_z = (1 + self.redshift)**3 * self.cosmo.omega_matter / E**2 
        mean_dens = self.cosmo.critical_density(self.redshift) * omega_z
        return mean_dens.to("Msun/kpc**3")

    def overdensity_radius(self, mode="crit", overdensity="virial"):
        """Computes the virial radius of a halo as.  
        """  
        overdens_crit_ratio = lambda masses, radius: masses.to("Msun").sum() / (4/3 * np.pi * (radius)**3) / self.crit_dens / self.virial_overdens
        
        assert "darkmatter" in self.cm, "You first need to compute the center of mass of the halo!"
        center = self.cm["darkmatter"]
            
        masses = self.data[self.bound_nbody, "particle_mass"].to("Msun")
        radii = np.linalg.norm(self.data[self.bound_nbody, "particle_position"].to(center.units) - center, axis=1).to("kpc")        
        max_radius = radii.max()
        min_radius = radii.min()
        
        virialradius_zero = root_scalar(
            lambda radius: overdens_crit_ratio(masses[radii <= radius], radius * max_radius.units) - 1, 
            method="brentq", 
            bracket=[min_radius, max_radius]
        )

        self.rvir = self.quan(virialradius_zero.root, max_radius.units).to("kpc")
        masses = self.data[self.bound_dm, "particle_mass"].to("Msun")
        radii = np.linalg.norm(self.data[self.bound_dm, "particle_position"].to(center.units) - center, axis=1).to("kpc") 
        
        self.mvir = masses[radii <= self.rvir].sum().to("Msun")
        
        return self.rvir, self.mvir



    def X_mass_radius(self, X=0.5):
        """Computes the half mass radius
        """
        self.rhalf = half_mass_radius(
            self.data[self.bound_nbody, "particle_position"].to("kpc"), 
            self.data[self.bound_nbody, "particle_mass"].to("Msun"), 
            self.cm["darkmatter"], 
            X, 
            project=False
        )
        return self.rhalf

    def vmax_rmax(self, soft, nbins):
        """Computes the maximum circular velocity and its radius.
        """
        nbins = min(nbins, 1000)
        radii = np.linalg.norm(self.data[self.bound_dm, "particle_position"] - self.cm["darkmatter"], axis=1).to("kpc")
        bin_masses, bin_edges, bin_num = binned_statistic(
            radii,
            self.data[self.bound_dm, "particle_mass"].to("Msun"),
            statistic="sum",
            range=[2*soft.to("kpc").value, radii.max().to("kpc").value],
            bins=nbins
        )
        bin_radii = self.arr( 0.5 * (bin_edges[1:] + bin_edges[:-1]), 'kpc')
        bin_encmass = self.arr( np.cumsum(bin_masses), 'Msun')
        circular_velocity = np.sqrt(
            self.quan(4.300917270038e-06, "kpc/Msun * (km/s)**2") * bin_encmass / bin_radii
        )
        index_vmax = np.argmax(circular_velocity)
        self.vmax = circular_velocity[index_vmax].to("km/s")
        self.rmax = bin_radii[index_vmax].to("kpc")
        return self.rmax, self.vmax

    def rs_klypin(self):
        """Computes the scale radius of the dark matter halo using the approach described in Behroozi et al. 2011
        """
        f = lambda x: np.log(1 + x) - x / (1 + x)
        alpha = (self.vmax**2 * self.rvir) / (self.quan(4.300917270038e-06, "kpc/Msun * (km/s)**2") * self.mvir)
        g = lambda c: c / f(c) * f(2.1626) / 2.1626 - alpha

        conc_sweep = np.linspace(0.01, 10000, 10000)
        g_sweep = g(conc_sweep)

        max_indices, min_indices = np.where(g_sweep < 0)[0],  np.where(g_sweep > 0)[0]
        max_index, min_index = max_indices[-1] if max_indices.size > 0 else -1, min_indices[-1] if min_indices.size > 0 else -1
        neg_g, pos_g = conc_sweep[max_index], conc_sweep[min_index]

        conc_zero = root_scalar(g, method="brentq", bracket=[min(neg_g, pos_g), max(neg_g, pos_g)])
        self.conc = conc_zero.root
        self.rs = self.rvir / self.conc
        return self.conc, self.rs

    def inertia_tensor(self, pt):
        """Computes the inertia tensor of a set of particles.
        """    
        if pt == "stars":
            sp_pt = self.bound_stars
        elif pt == "darkmatter":
            sp_pt = self.bound_dm
            
        positions = self.data[sp_pt, "particle_position"].to("kpc") - self.cm[pt]
        masses = self.data[sp_pt, "particle_mass"].to("Msun")
        if positions.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3)")

        I_xx = np.sum(masses * (positions[:, 1]**2 + positions[:, 2]**2))
        I_yy = np.sum(masses * (positions[:, 0]**2 + positions[:, 2]**2))
        I_zz = np.sum(masses * (positions[:, 0]**2 + positions[:, 1]**2))
        
        I_xy = -np.sum(masses * positions[:, 0] * positions[:, 1])
        I_xz = -np.sum(masses * positions[:, 0] * positions[:, 2])
        I_yz = -np.sum(masses * positions[:, 1] * positions[:, 2])
        
        inertia_tensor = np.array([
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz]
        ]) * masses.units * positions.units ** 2
        self.I[pt] = inertia_tensor
        return inertia_tensor

    def mass_distribution_tensor(self, pt):
        """Computes the mass distribution tensor for a set of particles.
        """
        if pt == "stars":
            sp_pt = self.bound_stars
        elif pt == "darkmatter":
            sp_pt = self.bound_dm
            
        positions = self.data[sp_pt, "particle_position"].to("kpc") - self.cm[pt]
        if positions.shape[1] != 3:
            raise ValueError("Positions must have shape (N, 3)")
        
        N = positions.shape[0]  
        shape_tensor = (positions.T @ positions) / N 
        self.shape_tensor[pt] = shape_tensor
        return shape_tensor
    
    def axes_ratios(self, pt, method="shape-tensor"):
        """Computes the square of each axis as the sorted eigenvalues of the  MDT
        and computes b/a and c/a.
        """
        if method == "shape-tensor":
            if not hasattr(self, "shape_tensor"):
                shape_tensor = self.mass_distribution_tensor(pt)
            else:
                shape_tensor = self.shape_tensor[pt]
                
            eigenvals, eigenvecs = np.linalg.eig(shape_tensor)
        
            sort_idx = np.argsort(eigenvals)[::-1] 
            eigenvals_sorted = eigenvals[sort_idx]
            eigenvecs_sorted = eigenvecs[:, sort_idx]
        
            a = np.sqrt(eigenvals_sorted[0])
            b = np.sqrt(eigenvals_sorted[1])
            c = np.sqrt(eigenvals_sorted[2])
        
            b_over_a = b / a
            c_over_a = c / a
            largest_axis_vector = eigenvecs_sorted[:, 0] / np.linalg.norm(eigenvecs_sorted[:, 0])
            
        elif method == "inertia-tensor":
            if not hasattr(self, "I"):
                I_tensor = self.inertia_tensor(pt)
            else:
                I_tensor = self.I[pt]
                
            eigenvals, eigenvecs = np.linalg.eig(I_tensor)
        
            sort_idx = np.argsort(eigenvals)[::-1] 
            eigenvals_sorted = eigenvals[sort_idx]
            eigenvecs_sorted = eigenvecs[:, sort_idx]
            
            b_over_a = np.sqrt( (eigenvals_sorted[2] - eigenvals_sorted[1] + eigenvals_sorted[0]) / (eigenvals_sorted[1] - eigenvals_sorted[0] + eigenvals_sorted[2]) )
            c_over_a = np.sqrt( (eigenvals_sorted[0] - eigenvals_sorted[2] + eigenvals_sorted[1]) / (eigenvals_sorted[1] - eigenvals_sorted[0] + eigenvals_sorted[2]) )
            
            largest_axis_vector = eigenvecs_sorted[:, 0] / np.linalg.norm(eigenvecs_sorted[:, 0])
            
             
        self.b_a[pt] = b_over_a
        self.c_a[pt] = c_over_a
        self.A[pt] = largest_axis_vector
        return b_over_a, c_over_a, largest_axis_vector



































class HaloParticlesInside(HaloParticles):
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

    
    def _prepare_particle_sets(self):
        """Modifies the data atribute to include *_prev filter for previously bound particles
        """
        assert len(self.data[self.nbody, "particle_index"]) == len(self._ppi["all"]), "YT particle filter didnt work correctly."
        assert False not in np.isin(self.data[self.nbody, "particle_index"], self._ppi["all"]), "Particle Indexes are not matching."
        
        self.data.add_bound_mask(self._ppi["bound"], suffix="_prev_bound")
        self.bound_nbody = self.nbody + f"_prev_bound"
        self.bound_stars = self.stars + f"_prev_bound"
        self.bound_dm = self.dm + f"_prev_bound"

    
    def compute_potential(self):
        """Computes potential of all particles.
        """
        n = self.data[self.nbody, "particle_index"].shape[0]
        nbody_grav_potential = self.arr( PotentialTarget(
                pos_target=self.data[self.nbody, "particle_position"].to("kpc"), 
                softening_target=None, 

                pos_source=self.data[self.bound_nbody, "particle_position"].to("kpc"), 
                m_source=self.data[self.bound_nbody, "particle_mass"].to("Msun"),
                softening_source=None, 
            
                G=4.300917270038E-6, 
            
                theta=min( 0.6, 0.3 * (n / 1E3) ** 0.08 ),
                quadrupole=True,
                tree=self.octree,
            
                parallel=False
            ), 
            'km**2/s**2'
        )
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
            (self.nbody, "total_energy"),
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






















def _extract_particles(subtree, ds, pdir2, row, row2):
    """Returns particle indices that are bound (AGORA VII)
    """
    cen = row[["position_x", "position_y", "position_z"]].values[0]
    rvir = row["virial_radius"].values[0]
    sp = ds.sphere((cen, 'kpccm'), (2 * rvir, 'kpccm'))
    
    potential = ds.arr(Potential(
        pos=sp["nbody", "particle_position"].to("kpc"), 
        m=sp["nbody", "particle_mass"].to("Msun"),
        softening=None, 
        G=4.300917270038E-6, 
        parallel=False
    ), 
    'km**2/s**2'
    )
    grav_energy = potential * sp["nbody", "particle_mass"].to("Msun")
    
    vel = ds.arr(row[["velocity_x", "velocity_y", "velocity_z"]].values[0], 'km/s')
    kinetic = 0.5 * sp["nbody", "particle_mass"].to("Msun") * np.linalg.norm(sp["nbody", "particle_velocity"].to("km/s") - vel, axis=1)**2
    
    mask = (kinetic + grav_energy < 0)
    index_sel1 = sp["nbody", "particle_index"][mask].astype(int).value

    del sp, potential, grav_energy, kinetic, mask
    gc.collect()
    
    ds2 = yt.load(pdir2)
    cen = row2[["position_x", "position_y", "position_z"]].values[0]
    rvir = row2["virial_radius"].values[0]
    sp = ds2.sphere((cen, 'kpccm'), (rvir, 'kpccm'))
    index_sel2 = sp["nbody", "particle_index"].astype(int).value
    return np.unique(
            np.concatenate(( index_sel1, index_sel2 ))
        )

def _extract_particles_wrapper(task):
    return _extract_particles(*task)


class ParticleTracker:
    """Class that tracks halo particles after rockstar + consistent-trees looses the halos.
    Enables following the halos as they merge with much more massive halos. The main 
    purpose of this code is to track Dwarf Galaxies when they merge with MW-like galaxies.
    
    
    If a single halo is provided, it falls back to SingleTracker. If a list of halos is
    provided it tracks all at once. Gravitational computations are done with pytreegrav.
    
    When tracking, the code computes halo position and velocity using only DM particles, but 
    also computes center of stars, virial, tidal radius and bound fraction. The code checks
    for mergers by determining if |x_1 - x_2| < rh both with the main MW-like halo and 
    between tracked halos.
    
    Particles are selected as all those that satisfy: 1) Inside 2 * Rvir_sub or 
    2) particles inside of Rvir_sub when the halo was at 2*Rvis_main.
 
    The number of particles is fixed: mergers of subhalos can happen with subhalos present 
    in  the initial particle selection, but external particles cannot be accreted. For 
    dwarf galaxy tracking, this is a sensible approximation.
    """
    def __init__(self, 
                 sim_dir,
                 catalogue,
                 equivalence_table
                 ):
        """Init function.
        """
        self.sim_dir = sim_dir
        self.mergertree = MergerTree(catalogue)
        self.mergertree.set_equivalence(equivalence_table)

        self._prefix = os.path.commonprefix([os.path.basename(file) for file in self.equiv["snapname"].values])
        
        self._merge_info = None #self.mergertree.MergeInfo
        self._faulty = None     #self.mergertree.SpikeInfo
    
    @property
    def CompleteTree(self):
        return self.mergertree.CompleteTree
    @property
    def PrincipalLeaf(self):
        return self.mergertree.PrincipalLeaf
    @property
    def equiv(self):
        return self.mergertree.equivalence_table
        
    @property
    def max_snap(self):
        return self.mergertree.snap_max
    @property
    def min_snap(self):
        return self.mergertree.snap_min
    @property
    def max_index(self):
        return self._snap_to_index(self.max_snap)  
    @property
    def min_index(self):
        return self._snap_to_index(self.min_snap)
    @property
    def principal_subid(self):
        return self.mergertree.principal_subid
        
    @property
    def snap_z_t_dir(self):
        return self._snap_z_time
    @property
    def tracked_halo_trees(self):
        return self._thtrees
        
    @property
    def MergeInfo(self):
        return self._merge_info
    @property
    def SpikeInfo(self):
        return self._faulty

    @property
    def _files(self):
        return [self.sim_dir + "/" + file for file in self.equiv["snapname"].values]
    @property
    def ts(self):
       return yt.DatasetSeries(self._files) 


    def _snap_to_index(self, snap):
        """Translates snapshot to index.
        """
        return self.equiv[self.equiv["snapshot"] == snap].index.values[0]

    def _index_to_snap(self, index):
        """Translates index to snap.
        """
        return self.equiv.loc[index]["snapshot"]

    
    
    def _create_hdf5_file(self, fname): 
        """Creates an hdf5 file to store the data. 
        Each snapshot goes in a column.
        Each halo has a separate group.
        Each halo has four datasets: pIDs and Es for ds and stars.
        """
        self.f = h5py.File(fname, "a")
        
        for index, row in self.halos_tbt.sort_values("Sub_tree_id").iterrows():
            subgroup = f"sub_tree_{int(row['Sub_tree_id'])}"
            cols = self.max_snap - row["Snapshot"] + 1
            
            self.f.create_group(subgroup, track_order=True)
            self.f[subgroup].attrs["energy_norm"] = 1E9
            
            self.f[subgroup].create_dataset(
                "darkmatter_ids", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.int32,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "darkmatter_potential", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.float16,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "darkmatter_kinetic", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.float16,
                fillvalue=np.nan
            )

            
            self.f[subgroup].create_dataset(
                "stars_ids", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.int32,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "stars_potential", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.float16,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "stars_kinetic", 
                shape=(0, 0), 
                maxshape=(None, cols), 
                dtype=np.float16,
                fillvalue=np.nan
            )

            
            self.f[subgroup].create_dataset(
                "snapshot", 
                shape=(0,), 
                maxshape=(cols,), 
                dtype=np.int16,
                fillvalue=np.nan
            )
            self.f[subgroup].create_dataset(
                "R/Rvir_host", 
                shape=(0,), 
                maxshape=(cols,), 
                dtype=np.float32,
                fillvalue=np.nan
            )


    def _postprocess_trees(self, ids):
        self.mergertree.postprocess(ids)

        self._merge_info = self.mergertree.MergeInfo
        self._faulty = self.mergertree.SpikeInfo

    def _create_snapztdir(self):
        """Creates _snap_z_time and adds column with relevant particle file directories.
        """
        self.snap_names = self.equiv[ self.equiv["snapshot"] >= self.halos_tbt["Snapshot"].min() ]["snapname"].values
        self.snap_dirs = [self.sim_dir + snap for snap in self.snap_names]
        self._snap_z_time = self.PrincipalLeaf[ self.PrincipalLeaf["Snapshot"] >= self.halos_tbt["Snapshot"].min() ][["Snapshot", "Redshift", "Time"]].reset_index(drop=True)
        self._snap_z_time["pdir"] = [self.sim_dir + snap for snap in self.snap_names]

    
    def _create_thtrees(self):
        """Creates a dictionary whose keys are the sub_trees of each of the tracked trees in pd.DataFrame format.
        When tracking, statistics like the center of mass, virial radius, half mass radius etc are appended at each time.
        """
        self._thtrees = {}
        columns = ["Sub_tree_id", "Snapshot", "Redshift", "Time", "mass", "virial_radius", "scale_radius", "tidal_radius", "r_half", "vrms", "vmax", 
                  "position_x", "position_y", "position_z", "velocity_x", "velocity_y", "velocity_z", "R/Rvir", "stellar_mass", "peak_mass", 
                  "Sub_tree_id_host"
                 ]
        for sn, subtree in self.halos_tbt[["Snapshot", "Sub_tree_id"]].values:
            snapshot_numbers = np.sort(self.CompleteTree[self.CompleteTree["Snapshot"] >= sn]["Snapshot"].unique())
            times = np.sort(self.CompleteTree[self.CompleteTree["Snapshot"] >= sn]["Time"].unique())
            redshifts = np.sort(self.CompleteTree[self.CompleteTree["Snapshot"] >= sn]["Redshift"].unique())[::-1]
            rows = len(snapshot_numbers)

            subtree_host, peak_mass = self.MergeInfo[self.MergeInfo["Sub_tree_id"] == subtree].squeeze()[["Sub_tree_id_host", "peak_mass"]]
            
            self._thtrees[f"sub_tree_{int(subtree)}"] = pd.DataFrame(columns=columns, index=range(rows))
            
            self._thtrees[f"sub_tree_{int(subtree)}"]["Snapshot"] = snapshot_numbers
            self._thtrees[f"sub_tree_{int(subtree)}"]["Redshift"] = redshifts
            self._thtrees[f"sub_tree_{int(subtree)}"]["Time"] = times
            
            self._thtrees[f"sub_tree_{int(subtree)}"]["Sub_tree_id"] = int(subtree)
            self._thtrees[f"sub_tree_{int(subtree)}"]["Sub_tree_id_host"] = int(subtree_host)
            self._thtrees[f"sub_tree_{int(subtree)}"]["peak_mass"] = peak_mass


    def _sanitize_merge_info(self, ignore_faulty=True):
        """Sanitize merge info: if the merge host subtree is not the main tree and the merger happens at R/Rvir_main < 1,
        then we change the merger host subtree to the main tree. Furthermore, dubious subtrees are erased.
        """
        indices = self.MergeInfo[(self.MergeInfo["Sub_tree_id_host"] != self.principal_subid) & (self.MergeInfo["R/Rvir_merge"] < 1)].index
        self._merge_info.loc[indices, "Sub_tree_id_host"] = self.principal_subid

        for index, subtree in zip(indices, self._merge_info.loc[indices, "Sub_tree_id"].unique()):
            snap_crossing = self.CompleteTree[(self.CompleteTree["Sub_tree_id"] == subtree) & (self.CompleteTree["R/Rvir"] > 1)].sort_values("Snapshot", ascending=False)["Snapshot"].iloc[0]
            snap_crossing_2 = self.CompleteTree[(self.CompleteTree["Sub_tree_id"] == subtree) & (self.CompleteTree["R/Rvir"] > 2)].sort_values("Snapshot", ascending=False)["Snapshot"].iloc[0]
            
            #self._merge_info.loc[index, "Snapshot"] = snap_crossing
            self._merge_info.loc[index, "crossing_snap"] = snap_crossing
            self._merge_info.loc[index, "crossing_snap_2"] = snap_crossing_2


        if ignore_faulty:
            for index, row in self._faulty.iterrows():
                subtree = row["Sub_tree_id"]
                if subtree not in self._merge_info["Sub_tree_id"].unique():
                    warnings.warn("Faulty Tree is no present in merge infro for some reason!")
                else:
                    merge_row = self._merge_info[self._merge_info["Sub_tree_id"] == subtree].squeeze()
                    start_row = self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree].squeeze()
                    index_merge = merge_row.name
                    index_start = start_row.name
                    if row["Snapshot_spike"] < merge_row["crossing_snap_2"] and  row["Snapshot_spike"] < start_row["Snapshot"]:
                        if row["birth_R/Rvir"] <= 2:
                            self._merge_info = self.MergeInfo.drop(labels=index_merge)
                            self.halos_tbt = self.halos_tbt.drop(labels=index_start)
                        else:
                            continue
                    else:
                        self._merge_info = self.MergeInfo.drop(labels=index_merge)
                        self.halos_tbt = self.halos_tbt.drop(labels=index_start)
                        

    def _select_particles(self, inserted_halos, **kwargs):
        """Selects particles to track for a inserted subtree's.
        """
        import multiprocessing as mp

        from tqdm import tqdm

        yt.utilities.logger.ytLogger.setLevel(40)

        tasks = []
        for subtree in inserted_halos:
            subtree_table = self.mergertree.subtree(subtree)
            merge = self.MergeInfo[self.MergeInfo["Sub_tree_id"] == subtree]
            row = subtree_table.loc[subtree_table["Snapshot"] ==  merge["crossing_snap"].values[0]]
            row2 = subtree_table.loc[subtree_table["Snapshot"] ==  merge["crossing_snap_2"].values[0]]
            
            pdir2 = self.equiv[ self.equiv["snapshot"] == merge["crossing_snap_2"].values[0] ]["snapname"].values[0] 

            tasks.append((
                subtree,
                self.ds,
                self.sim_dir + "/" + pdir2,
                row,
                row2
            ))


        nproc = int(kwargs.get("parallel", 1))
        chsize = kwargs.get("chunksize", len(tasks) // (4*nproc))

        pool = mp.Pool(nproc)
        if nproc > 4: warnings.warn(f"Having nproc={nproc} may fuck up your cache.")

        chsize = kwargs.get("chunksize", len(tasks) // (4*nproc) + 1)
        for i, particle_list in enumerate(
            tqdm(pool.imap(_extract_particles_wrapper, tasks, chunksize=chsize), total=len(tasks), desc="Selecting particles...")
        ):
            self.particles_index_dict[tasks[i][0]] = particle_list

        pool.close()
        pool.join()
        del pool
        yt.utilities.logger.ytLogger.setLevel(20)




        
    def thinn_trees(self, thinned_snapshot):
        """Provides a way to thinn the equivalence table and MergerTrees to some snapshot_values 
        if the user doesn't have access to all the snapshots. Thinning to very coarse time steps might 
        lead to untrackability of halos.
        """
        self.mergertree = self.mergertree.thinn_catalogue(thinned_snapshot)
    
        if hasattr(self, "_snap_z_time"): 
            self._create_snapztdir()
        if hasattr(self, "_thtrees"):
            self._create_thtrees()

        if hasattr(self.mergertree, "MergeInfo"):
            self._merge_info = self.mergertree.MergeInfo
            self._faulty = self.mergertree.SpikeInfo
        
    
    def dump_to_hdf5(self, subtree, index, data):
        """Dumps particle IDs, its potential, kinetic and total energies.
        """
        subgroup = f"sub_tree_{int(subtree)}"
        group = self.f[subgroup]
        for field in ["darkmatter_ids", "stars_ids", "darkmatter_energies", "stars_energies"]:
            dset = group[field]
            dset.resize((len(data[field]), dset.shape[1] + 1))
            dset[:, index] = data[field]
            
        return None

    
    def dump_to_csv(self, subtree, snapnum, data):
        """Dumps a csv with halo position, velocity, virial radius etc with the same format as rockstar+consistent trees 
        files created using MergerTrees class
        """
        return None





    

        
    def track_halos(self, halos_tbt, output, **kwargs):
        """Tracks the halos_tbt (to-be-tracked, not throw-back-thursday) halos. Each halos starting poit can be a different snapshot
        """
        if isinstance(halos_tbt, pd.DataFrame):
            self.halos_tbt = halos_tbt.sort_values("Snapshot")
        else: 
            self.halos_tbt = pd.read_csv(halos_tbt).sort_values("Snapshot")

        self._postprocess_trees(self.halos_tbt["Sub_tree_id"].unique())
        
        self.output = output
        try:
            self.close()
        except:
            pass
            
        if not isinstance(self.CompleteTree, pd.DataFrame) or not isinstance(self.equiv, pd.DataFrame):
            raise ValueError("Either CompleteTree or equivalence_table are not pandas dataframes! or are not properly set!")
        
        self._sanitize_merge_info()
        self._create_snapztdir()
        self._create_thtrees()
        
        start_snapst = dict( zip(self.MergeInfo["Sub_tree_id"].values,  self.MergeInfo["Snapshot"].values) )
        file_path = Path(output)
        if file_path.is_file():
            outpùt2 = output[:-5] + "_v2" + output[-5:]
            warnings.warn(f"The file {output} already exists", RuntimeWarning)
            
        self._create_hdf5_file(fname=output)

        active_halos = set()
        terminated_halos = set()
        live_halos = set()

        ref_table = self.MergeInfo
        self.particles_index_dict = {}
        
        for index, row in self.snap_z_t_dir.iterrows():
            snapshot, redshift, time, pdir = row["Snapshot"], row["Redshift"], row["Time"], row["pdir"]
            
            inserted_halos = ref_table[ref_table["Snapshot"] == snapshot]["Sub_tree_id"].astype(int).values
            active_halos.update(inserted_halos)
                
            live_halos = active_halos - terminated_halos
            if not live_halos:
                continue
            if not active_halos:
                raise Exception(f"You dont have any active halos at this point! (snapshot {snapshot}). This means that something went wrong!")
            else:
                self.ds = yt.load(pdir)

            self._select_particles(inserted_halos, **kwargs)


            del self.ds

        self.f.close()
        return active_halos, live_halos, terminated_halos


        def close(self):
            os.remove(output)
            self.f.close()




        
            
               # for subtree in live_halos:
               # subtree_table = self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree]
               # if subtree in inserted_halos:
               #     if subtree_table["R/Rvir"] < 1:
               #         insert_row = 1
               #     else:
               #         insert_row = subtree_table
               #         
               #     sp = self.ds.sphere(
               #         self.ds.arr(subtree_table[["position_x", "position_y", "position_z"]].astype(float).values, 'kpccm'),    
               #         self.ds.quan(subtree_table["virial_radius"].astype(float).values, 'kpccm')
               #     )
               #     particles = HaloParticlesOutside(subtree, sp)
               #     
               # else:
               #     pass
                #pot + trackeo
                #add particles to self.f
                #compute statistics
                #add statistics to csv
#            for subtree in live_halos:
                #check for merging with main halo or merges between them using csvs. Now all of them are up to date.        