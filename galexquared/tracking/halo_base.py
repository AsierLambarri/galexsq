import os
import yt
import warnings
import numpy as np

from scipy.optimize import root_scalar
from scipy.stats import binned_statistic

from pytreegrav import ConstructTree

from ..class_methods import softmax, half_mass_radius, refine_6Dcenter
from .particle_data import ParticleData

import gc

class HaloParticles:
    """Class that contains the particles (stellar and dark) that conform the halo. Has utilities for computing boundness,
    virial, tidal radius and center of mass position and velocity.
    """
    def __init__(self,
                 data_source,
                 particle_names,
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
            T = 0.5 * np.abs(self.data[self.bound_dm, "kinetic_energy"].mean()/self.data[self.bound_dm,"total_energy"].min())
            
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
        self.cv = (2 * (self.vmax / self.rmax * 1/self.cosmo.hubble_constant)**2 ).to("")
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







