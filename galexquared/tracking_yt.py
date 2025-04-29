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


from dataclasses import dataclass


class ParticleData:
    """Simple class to handle particle data, with an option to provide a mask
    """
    def __init__(self,
                 sp,
                 particle_names
                ):
        """Init function.
        """
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
        self._masks = {
            self.nbody : None,
            self.dm : None,
            self.stars : None
        }
        
    def __getitem__(self, key):
        return self._data[key]

    def _get_keys(self, selected_ptype):
        """Get keys for a given particle
        """
        return  [field for (ptype, field) in self._data.keys() if ptype == selected_ptype]

    def add_field(self, field_name, field_value):
        """ Adds fields to data
        """
        self._data[field_name] = field_value
        
        
    def add_bound_mask(self, bound_indices):
        """Adds fields for boundness
        """
        particles = [self.nbody, self.dm, self.stars]
        for ptype in particles:
            all_fields = self._get_keys(ptype)
            mask = np.isin(self._data[ptype, "particle_index"], bound_indices)
            self._masks[ptype] = mask
            for field in all_fields:
                self._data[ptype + "_bound", field] = self._data[ptype, field][mask]

            # if "total_energy" in all_fields:
            #     assert len(self._data[ptype + "_bound", "particle_index"]) == np.count_nonzero(self._data[ptype, "total_energy"] < 0), "Boundness doesn't add up"
                
        






class HaloParticles:
    """Class that contains the particles (stellar and dark) that conform the halo. Has utilities for computing boundness,
    virial, tidal radius and center of mass position and velocity.
    """
    def __init__(self,
                 subtree,
                 data_source,
                 particle_names
                ):
        """Init function.
        """
        self.subtree = subtree
        self.nbody = particle_names["nbody"]
        self.dm = particle_names["darkmatter"]
        self.stars = particle_names["stars"]
        self._octree = None

        self.data = data_source #ParticleData(data_source, particle_names)
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
        tmp_vcm = sp2.quantities.bulk_velocity(use_gas=False, use_particles=True, particle_type=self.bound_nbody).to("km/s")
        self.vcm["darkmatter"] = tmp_vcm
        return tmp_vcm







    def change_sphere(self, center=None, radius=None):
        """Changes the sphere data source to a new one with center and radius
        specified as arguments
        """
        self.data = self.ds.sphere(
            center if center is not None else self.data.center, 
            radius if radius is not None else self.data.radius
        )






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
        from time import time

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
            bracket=[self.quan(0.1, 'kpc').to(max_radius.units), max_radius]
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


    def clear_data(self):
        self.data.clear_data()








class HaloParticlesOutside(HaloParticles):
    """Class to handle tracking of halo (track with dm + keep stars around) when outside of host virial radius
    (host != main galaxy).
    """
    def __init__(self,
                 subtree,
                 sp,
                 particle_names
                 ):
        super().__init__(subtree, sp, particle_names)
        
       
        
    def _add_dmstars_fields(self):
        """Extends particle fields to individual stars and dark-matter
        """
        yt.add_particle_filter(
            self.stars, 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], data[self.stars, "particle_index"]),
            filtered_type=self.nbody, 
            requires=["particle_index"]
        )
        self.ds.add_particle_filter(self.stars)
        
        yt.add_particle_filter(
            self.dm, 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], data[self.dm, "particle_index"]),
            filtered_type=self.nbody, 
            requires=["particle_index"]
        )
        self.ds.add_particle_filter(self.dm)
                
    def _add_energy_filter(self, beta):
        """Adds particle filters as stars_{# subtree} and darkmatter__{# subtree}
        """
        self.bound_nbody = self.nbody + f"_{self.subtree}"    #f"nbody_{self.subtree}"
        self.bound_stars = self.stars + f"_{self.subtree}"    #f"stars_{self.subtree}"
        self.bound_dm = self.dm + f"_{self.subtree}"          #f"darkmatter_{self.subtree}"
        
        yt.add_particle_filter(
            self.bound_nbody, 
            function=lambda pfilter, data: (data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0, 
            filtered_type=self.nbody, 
            requires=["grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(self.bound_nbody)
        
        yt.add_particle_filter(
            self.bound_stars, 
            function=lambda pfilter, data: (np.isin(data[pfilter.filtered_type, "particle_index"], data[self.stars, "particle_index"])) & ((data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0), 
            filtered_type=self.nbody, 
            requires=["particle_index", "grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(self.bound_stars)
        
        yt.add_particle_filter(
            self.bound_dm, 
            function=lambda pfilter, data:  (np.isin(data[pfilter.filtered_type, "particle_index"], data[self.dm, "particle_index"])) & ((data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0), 
            filtered_type=self.nbody, 
            requires=["particle_index", "grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(self.bound_dm)



        

        
        
    def compute_potential(self):
        """Computes potential of all particles.
        """
        def _pot(field, data, obj=self):
            n = data[obj.nbody, "particle_mass"].shape[0]
            octree = obj.octree
            return obj.arr( 
                Potential(
                    pos=data[obj.nbody, "particle_position"].to("kpc"), 
                    m=data[obj.nbody, "particle_mass"].to("Msun"), 
                    softening=None, 
                    G=4.300917270038E-6, 
                    theta=min( 0.6, 0.3 * (n / 1E3) ** 0.08 ),
                    parallel=False,
                    quadrupole=False,
                    tree=octree
                ), 
                'km**2/s**2'
            )

        
        self.ds.add_field(
            name=(self.nbody, "grav_potential"),
            function=_pot,
            sampling_type="local",
            units="km**2/s**2",
            force_override=True
        )
        self.ds.add_field(
            name=(self.nbody, "grav_energy"),
            function=lambda field, data: data[self.nbody, "grav_potential"] * data[self.nbody, "particle_mass"],
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        
    def compute_kinetic(self, vcm=None):
        """Computes kinetic energy of all particles.
        """
        if vcm is not None:
            bulk_vel = vcm
        else:
            bulk_vel = self.data.quantities.bulk_velocity(use_gas=False, use_particles=True, particle_type=self.dm).to("km/s")

        self.ds.add_field(
            name=(self.nbody, "kinetic_energy"),
            function=lambda field, data: 0.5 * data[self.nbody, "particle_mass"] * np.linalg.norm(data[self.nbody, "particle_velocity"] - bulk_vel, axis=1) ** 2,
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        
    def compute_energy(self, vcm=None, beta=0.95):
        """Computes potential, kinetic and total energy all at once.
        Boundness is determined using bulk velocity (see ROCKSTAR).
        """
        self.data.clear_data()
        self.compute_kinetic(vcm=vcm)
        self.compute_potential()
        self.ds.add_field(
            name=(self.nbody, "total_energy"),
            function=lambda field, data: data[self.nbody, "grav_energy"] + data[self.nbody, "kinetic_energy"],
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        self._add_dmstars_fields()
        self._add_energy_filter(beta=beta)
        
        













class HaloParticlesInside(HaloParticles):
    """Class to handle tracking of halo (track with dm + keep stars around) when inside of host virial radius
    (host != main galaxy).
    """
    def __init__(self,
                 subtree,
                 sp,
                 particle_names,
                 stars_index,
                 stars_bound_index,
                 dm_index,
                 dm_bound_index
                 ):
        super().__init__(subtree, sp, particle_names, "inside")
        
        self.prev_boundbody = f"prev_nbody_{subtree}"
        #self.prev_index = previous_index
        #self.prev_boundindex = previous_bound_index

        self.prev_index = {
            "stars" : stars_index,
            "darkmatter" : dm_index,
            "nbody" : np.concatenate((stars_index, dm_index))
        }
        self.prev_boundindex =  {
            "stars" : stars_bound_index,
            "darkmatter" : dm_bound_index,
            "nbody" : np.concatenate((stars_bound_index, dm_bound_index))
        }
        
        self.nbody = "nbody_static"
        self.dm = "darkmatter_static"
        self.stars = "stars_static"

        self.data.clear_data()
        
    
    def _initial_particle_filter(self):
        """Creates an initial particle filter where gravitational sources are 'prev_boundbody'
        """
        yt.add_particle_filter(
            "nbody_static", 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], self.prev_index["nbody"]),
            filtered_type="nbody", 
            requires=["particle_index"]
        )
        yt.add_particle_filter(
            "stars_static", 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], self.prev_index["stars"]) ,
            filtered_type="nbody_static", 
            requires=["particle_index"]
        )
        yt.add_particle_filter(
            "darkmatter_static", 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], self.prev_index["darkmatter"]),
            filtered_type="nbody_static", 
            requires=["particle_index"]
        )
        
        self.ds.add_particle_filter("nbody_static")
        self.ds.add_particle_filter("stars_static")
        self.ds.add_particle_filter("darkmatter_static")
       
        
        yt.add_particle_filter(
            self.prev_boundbody, 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], self.prev_boundindex["nbody"]),
            filtered_type="nbody_static", 
            requires=["particle_index"]
        )
        self.ds.add_particle_filter(self.prev_boundbody)
        
        

    def _add_energy_filter(self, beta):
        """Adds particle filters as stars_{# subtree} and darkmatter__{# subtree}
        """
        yt.add_particle_filter(
            f"stars_{self.subtree}", 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], data["stars_static", "particle_index"]) & ((data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0),  
            filtered_type="nbody_static", 
            requires=["index", "grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(f"stars_{self.subtree}")
        yt.add_particle_filter(
            f"darkmatter_{self.subtree}", 
            function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], data["darkmatter_static", "particle_index"]) & ((data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0), 
            filtered_type="nbody_static", 
            requires=["index", "grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(f"darkmatter_{self.subtree}")
        yt.add_particle_filter(
            f"nbody_{self.subtree}", 
            function=lambda pfilter, data: (data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0, 
            filtered_type="nbody_static", 
            requires=["grav_energy", "kinetic_energy"]
        )
        self.ds.add_particle_filter(f"nbody_{self.subtree}")
        
        self.nbody = f"nbody_{self.subtree}"
        self.stars = f"stars_{self.subtree}"
        self.dm = f"darkmatter_{self.subtree}"
        
        
    def compute_potential(self):
        """Computes potential of all particles.
        """
        
        self.ds.add_field(
            name=("nbody_static", "grav_potential"),
            function=lambda field, data: self.arr(
                PotentialTarget(
                    pos_target=data["nbody_static", "particle_position"].to("kpc"), 
                    pos_source=data[self.prev_boundbody, "particle_position"].to("kpc"), 
                    m_source=data[self.prev_boundbody, "particle_mass"].to("Msun"),
                    softening_target=data["nbody_static", "softening"].to("kpc"), 
                    softening_source=data[self.prev_boundbody, "softening"].to("kpc"),
                    G=4.300917270038E-6, 
                    theta=0.6,
                    parallel=True,
                    quadrupole=True
                ), 
                "km**2/s**2"
            ),
            sampling_type="local",
            units="km**2/s**2",
            force_override=True
        )
        self.ds.add_field(
            name=("nbody_static", "grav_energy"),
            function=lambda field, data: data["nbody_static", "grav_potential"] * data["nbody_static", "particle_mass"],
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        

    def compute_kinetic(self, vcm=None):
        """Computes kinetic energy of all particles.
        """
        if vcm is not None:
            bulk_vel = vcm
        else:
            bulk_vel = self.data.quantities.bulk_velocity(use_gas=False, use_particles=True, particle_type=self.dm).to("km/s")

        self.ds.add_field(
            name=("nbody_static", "kinetic_energy"),
            function=lambda field, data: 0.5 * data["nbody_static", "particle_mass"] * np.linalg.norm(data["nbody_static", "particle_velocity"] - bulk_vel, axis=1) ** 2,
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        
        
    def compute_energy(self, vcm=None, beta=0.95):
        """Computes potential, kinetic and total energy all at once.
        Boundness is determined using bulk velocity (see ROCKSTAR).
        """
        self.data.clear_data()
        self.compute_kinetic(vcm=vcm)
        self.compute_potential()
        self.ds.add_field(
            name=("nbody_static", "total_energy"),
            function=lambda field, data: data["nbody_static", "grav_energy"] + data["nbody_static", "kinetic_energy"],
            sampling_type="local",
            units="Msun * km**2/s**2",
            force_override=True
        )
        self._add_energy_filter(beta=beta)










        

class Tracker:
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
                 halos_tbt,
                 catalogue
                 ):
        """Init function.
        """
        self.sim_dir = sim_dir
        if halos_tbt.endswith(".csv"):
            self.halos_tbt = pd.read_csv(halos_tbt).sort_values("Snapshot")
        elif isinstance(halos_tbt, pd.DataFrame):
            self.halos_tbt = halos_tbt.sort_values("Snapshot")

        arbor = MergerTree(catalogue) 
        arbor.postprocess(self.halos_tbt["Sub_tree_id"].unique())
        self._merge_info = arbor.MergeInfo
        self._faulty = arbor.SpikeInfo
        self.set_catalogue(arbor.CompleteTree)
        
    @property
    def CompleteTree(self):
        return self._CompleteTree
    @property
    def PrincipalLeaf(self):
        if self._PrincipalLeaf is None:
            return None
        else:
            return self._PrincipalLeaf.sort_values("Snapshot", ascending=True)
    @property
    def equivalence_table(self):
        if self.PrincipalLeaf is None:
            return self._equiv
        else:
            return self._equiv[ self._equiv['snapshot'].isin(self.PrincipalLeaf['Snapshot']) ].sort_values("snapshot", ascending=True)
    @property
    def max_snap(self):
        return self._snap_max
    @property
    def min_snap(self):
        return self._snap_min
    @property
    def principal_subid(self):
        return self._principal_subid
    @property
    def snap_z_t_dir(self):
        return self._snap_z_time
    @property
    def tracked_halo_trees(self):
        return self._thtrees
    @property
    def MergeInfo(self):
        return self._merge_info

               

    
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

            
    def _create_snapztdir(self):
        """Creates _snap_z_time and adds column with relevant particle file directories.
        """
        self.snap_names = self.equivalence_table[ self.equivalence_table["snapshot"] >= self.halos_tbt["Snapshot"].min() ]["snapname"].values
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
                        

    def set_catalogue(self, complete_tree):
        """Sets trees by loading the complete tree and separating the principal-leaf, main and satellite-trees. 
        Computations are only even done with the complete-tree. Assumes that "Sub_tree"s and "TreeNum"s are already
        computed. 
    
        Parameters
        ----------
        complete_tree : str or pd.DataFrame
            Complete tree to be set in place
            
        """
        if isinstance(complete_tree, str):
            self._CompleteTree = pd.read_csv(complete_tree)
        elif isinstance(complete_tree, pd.DataFrame):
            self._CompleteTree = complete_tree

        self._snap_min, self._snap_max = int(self.CompleteTree['Snapshot'].values.min()), int(self.CompleteTree['Snapshot'].values.max())

        self._principal_subid, tree_num = self.CompleteTree.sort_values(['mass', 'Snapshot'], ascending = (False, True))[["Sub_tree_id", "TreeNum"]].values[0]
        
        self._PrincipalLeaf = self.CompleteTree[self.CompleteTree["Sub_tree_id"] == self.principal_subid].reset_index(drop=True)
        
                
    def set_equivalence(self, equiv):
        """Loads and sets equivalence table.
        """
        if isinstance(equiv, str):
            if not equiv.endswith("csv"):
                self._equiv = load_ftable(equiv)
            else:
                self._equiv = pd.read_csv(equiv)
                
        elif isinstance(equiv, pd.DataFrame):
            self._equiv = equiv

        else:
            raise AttributeError("Could not set equivalence table!")

    def thinn_trees(self, thinned_snapshot):
        """Provides a way to thinn the equivalence table and MergerTrees to some snapshot_values 
        if the user doesn't have access to all the snapshots. Thinning to very coarse time steps might 
        lead to untrackability of halos.
        """
        self.set_catalogue(self._CompleteTree[self._CompleteTree["Snapshot"].isin(thinned_snapshot)].sort_values("Snapshot").reset_index(drop=True))
        self.set_equivalence(self._equiv[self._equiv["snapshot"].isin(thinned_snapshot)].reset_index(drop=True)) 
        if hasattr(self, "_snap_z_time"): 
            self._create_snapztdir()
        if hasattr(self, "_thtrees"):
            self._create_thtrees()

        self._merge_info['Snapshot'] = self._merge_info['Snapshot'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))
        self._merge_info['crossing_snap'] = self._merge_info['crossing_snap'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))
        self._merge_info['crossing_snap_2'] = self._merge_info['crossing_snap_2'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))

    
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





    

        
    def track_halos(self, output, **kwargs):
        """Tracks the halos_tbt (to-be-tracked, not throw-back-thursday) halos. Each halos starting poit can be a different snapshot
        """
        self.output = output
        try:
            self.close()
        except:
            pass
            
        if not isinstance(self.CompleteTree, pd.DataFrame) or not isinstance(self.equivalence_table, pd.DataFrame):
            raise ValueError("Either CompleteTree or equivalence_table are not pandas dataframes! or are not properly set!")
        
        self._sanitize_merge_info()
        self._create_snapztdir()
        self._create_thtrees()
        
        start_snapst = dict( zip(self.halos_tbt["Sub_tree_id"].values,  self.halos_tbt["Snapshot"].values) )
        file_path = Path(output)
        if file_path.is_file():
            outpùt2 = output[:-5] + "_v2" + output[-5:]
            warnings.warn(f"The file {output} already exists", RuntimeWarning)
            
        sys.exit()
        self._create_hdf5_file(fname=output)

        active_halos = set()
        terminated_halos = set()
        live_halos = set()
        
        for index, row in self.snap_z_t_dir.iterrows():
            snapshot, redshift, time, pdir = row["Snapshot"], row["Redshift"], row["Time"], row["pdir"]
            
            inserted_halos = self.halos_tbt[self.halos_tbt["Snapshot"] == snapshot]["Sub_tree_id"].astype(int).values
            active_halos.update(inserted_halos)
                
            live_halos = active_halos - terminated_halos
            if not live_halos:
                continue
            if not active_halos:
                raise Exception(f"You dont have any active halos at this point! (snapshot {snapshot}). This means that something went wrong!")
            else:
                self.ds = config.loader(pdir)


            for subtree in live_halos:
                subtree_table = self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree]
                if subtree in inserted_halos:
                    if subtree_table["R/Rvir"] < 1:
                        insert_row = 1
                    else:
                        insert_row = subtree_table
                        
                    sp = self.ds.sphere(
                        self.ds.arr(subtree_table[["position_x", "position_y", "position_z"]].astype(float).values, 'kpccm'),    
                        self.ds.quan(subtree_table["virial_radius"].astype(float).values, 'kpccm')
                    )
                    particles = HaloParticlesOutside(subtree, sp)
                    
                else:
                    pass
                #pot + trackeo
                #add particles to self.f
                #compute statistics
                #add statistics to csv
#            for subtree in live_halos:
                #check for merging with main halo or merges between them using csvs. Now all of them are up to date.

            del self.ds

        self.f.close()
        return active_halos, live_halos, terminated_halos


        def close(self):
            os.remove(output)
            self.f.close()




        
            
            

    
        
        
        

        
    












        # self.ds.add_field(
        #     name=(self.stars, "grav_potential"),
        #     function=lambda field, data:  data[self.nbody, "grav_potential"][np.isin(data[self.nbody, "particle_index"], data[self.stars, "particle_index"])],
        #     sampling_type="particle",
        #     units="km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.stars, "grav_energy"),
        #     function=lambda field, data: data[self.stars, "grav_potential"] * data[self.stars, "particle_mass"],
        #     sampling_type="particle",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.stars, "kinetic_energy"),
        #     function=lambda field, data: data[self.nbody, "kinetic_energy"][np.isin(data[self.nbody, "particle_index"], data[self.stars, "particle_index"])], 
        #     sampling_type="particle",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.stars, "total_energy"),
        #     function=lambda field, data: data[self.stars, "grav_energy"] + data[self.stars, "kinetic_energy"],
        #     sampling_type="particle",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        
        
        # self.ds.add_field(
        #     name=(self.dm, "grav_potential"),
        #     function=lambda field, data:  data[self.nbody, "grav_potential"][np.isin(data[self.nbody, "particle_index"], data[self.dm, "particle_index"])],
        #     sampling_type="local",
        #     units="km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.dm, "grav_energy"),
        #     function=lambda field, data: data[self.dm, "grav_potential"] * data[self.dm, "particle_mass"],
        #     sampling_type="local",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.dm, "kinetic_energy"),
        #     function=lambda field, data: data[self.nbody, "kinetic_energy"][np.isin(data[self.nbody, "particle_index"], data[self.dm, "particle_index"])], 
        #     sampling_type="local",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        # self.ds.add_field(
        #     name=(self.dm, "total_energy"),
        #     function=lambda field, data: data[self.dm, "grav_energy"] + data[self.dm, "kinetic_energy"],
        #     sampling_type="local",
        #     units="Msun * km**2/s**2",
        #     force_override=True
        # )
        
        # (data[pfilter.filtered_type, "grav_energy"] + beta * data[pfilter.filtered_type, "kinetic_energy"]) < 0, 




















































