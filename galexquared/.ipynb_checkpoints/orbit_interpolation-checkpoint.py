import yt
import os
import gc
import psutil
import numpy as np
import pandas as pd

from unyt import unyt_quantity, unyt_array

import astropy.units as u

import gala.dynamics as gd
import gala.integrate as gi
import gala.potential as gp
from gala.units import UnitSystem

from pytreegrav import PotentialTarget, AccelTarget, Potential, ConstructTree

from .class_methods import load_ftable
from .mergertree import MergerTree
from .config import config

from copy import copy, copy

class ParticlePotential(gp.PotentialBase):
    m = gp.PotentialParameter("mass", physical_type="mass")
    pos = gp.PotentialParameter("pos", physical_type="length")
    soft = gp.PotentialParameter("soft", physical_type="length")
    ndim = 3
        
    def _compute_tree_ifnone(self):
        if not hasattr(self, "tree"):
            self.tree = ConstructTree(
                pos=self.parameters["pos"].value,
                m=self.parameters["m"].value,
                softening=self.parameters["soft"].value,
                quadrupole=True
            )
        else:
            if self.tree is None:
                self.tree = ConstructTree(
                    pos=self.parameters["pos"].value,
                    m=self.parameters["m"].value,
                    softening=self.parameters["soft"].value,
                    quadrupole=True
                )

    def _set_tree(self, tree):
        self.tree = tree
                
    def _energy(self, q, t=0):
        self._compute_tree_ifnone()
        pot = PotentialTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return pot

    def _acceleration(self, q, t=0):
        self._compute_tree_ifnone()
        accel = AccelTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return accel

    def _gradient(self, q, t=0):
        self._compute_tree_ifnone()
        accel = AccelTarget(
            pos_target=q, 
            pos_source=self.parameters["pos"].value, 
            m_source=self.parameters["m"].value,
            softening_source=self.parameters["soft"].value,
            G=self.units.get_constant("G"),
            parallel=False,
            quadrupole=True,
            tree=self.tree
        ) 
        return -1 * accel




class PhaseSpaceInstant:
    """Class to store Phase Space position and Time Stamp. It uses gala.dynamics.PhaseSpacePosition.
    """
    def __init__(
        self, 
        pos, 
        vel, 
        time, 
        redshift=None, 
        snapshot=None,
        units=None
    ):
        self.units = UnitSystem(units) if type(units) == list else units
        self.pos = self.units.decompose(pos)
        self.vel = self.units.decompose(vel)
        self.time = self.units.decompose(time)
        self.redshift = redshift
        self.snapshot = snapshot
        assert (time is not None) or (redshift is not None), "You must provide either time or redshift!"
        
        self.PhaseSpacePos =  gd.PhaseSpacePosition(
            pos=self.pos,
            vel=self.vel
        )

    def change_units(self, new_system):
        """Changes the units
        """
        new_units = UnitSystem(new_system) if type(new_system) == list else new_system
        return PhaseSpaceInstant(
            self.pos, 
            self.vel, 
            self.time, 
            redshift=self.redshift, 
            snapshot=self.snapshot, 
            units=new_system
        )

    def to_gala(self):
        """Changes to gala PhaseSpacePosition class
        """
        return gd.PhaseSpacePosition(
            pos=self.pos,
            vel=self.vel
        )


class CartesianOrbit:
    """Interpolated orbit containing phase space position, time and absolute magnitudes of xyz and vxyz in cartesian coordinates.
    """
    def __init__(
        self,
        xyz,
        vxyz,
        t,
        units=None
    ):
        self.units = UnitSystem(units) if type(units) == list else units

        self.t =  self.units.decompose(t)
        self.xyz =  self.units.decompose(xyz)
        self.vxyz =  self.units.decompose(vxyz)
        self.r = np.linalg.norm(self.xyz, axis=1)
        self.v = np.linalg.norm(self.vxyz, axis=1)

    def __add__(self, other):
        if not isinstance(other, CartesianOrbit):
            return NotImplemented

        t_all = np.concatenate((self.t, other.t))
        xyz_all = np.concatenate((self.xyz.T, other.xyz.T)).T
        vxyz_all = np.concatenate((self.vxyz.T, other.vxyz.T)).T

        sort_idx = np.argsort(t_all)
        t_sorted = t_all[sort_idx]
        xyz_sorted = xyz_all[:, sort_idx]
        vxyz_sorted = vxyz_all[:, sort_idx]

        unique_t, unique_indices = np.unique(t_sorted, return_index=True)
        unique_xyz = xyz_sorted[:, unique_indices]
        unique_vxyz = vxyz_sorted[:,unique_indices]

        return CartesianOrbit(
            unique_xyz, 
            unique_vxyz, 
            unique_t, 
            units=self.units
        )




def GetICSList(mergertree, subtree, snapshots, units):
    """Creates a PhaseSpaceInstant list for the specified snapshots and subtree
    """
    ics = []
    for snapshot in snapshots:
        sat_params = mergertree.get_halo_params(subtree, snapshot=snapshot)
        halo_params = mergertree.get_halo_params(mergertree.principal_subid, snapshot=snapshot)

        pos, vel = (sat_params["center"] - halo_params["center"]).to("kpc").value, (sat_params["center_vel"] - halo_params["center_vel"]).to("km/s").value

        tmp = PhaseSpaceInstant(
            pos=pos * u.kpc,
            vel=vel * u.km/u.s,
            time=(sat_params["time"].to("Gyr").value) * u.Gyr,
            units=units
        )
        ics.append(tmp)

    return ics

    
class Orbis:
    """Class for orbit interpolation, using data from simulations. Follows Richings et al. 2022
    """
    def __init__(
        self,
        pot_from_sim=None,
        file_table=None,
        pdir=None,
        pot_from_decomp=None,
        pot_model_list=None,
        units=None,
        **kwargs
    ):
        if not kwargs.get("logging", True):
            yt.utilities.logger.ytLogger.setLevel(40)
            
        self._pot_from_sim = True if pot_from_sim is not None else False
        self._pot_from_decomp = True if pot_from_decomp is not None else False
    
        if self._pot_from_sim:
            self._host = pot_from_sim
            assert pdir is not None, "You need to provide a directory for the simulation outputs"
            self._pdir = pdir
            assert file_table is not None, "You need to provide a file table containing columns SNAPSHOT NUMBER  |  REDSHIFT  |  FILE NAME"
            self._load_equivalence(file_table) 
            self._prefix = os.path.commonprefix([os.path.basename(file) for file in self._equiv["snapname"].values])
            self._files = [pdir + "/" + file for file in self._equiv["snapname"].values]
            self.ts = yt.DatasetSeries(self._files)
            self._octrees = np.empty(self._equiv["snapname"].shape, dtype=object)
            
        if self._pot_from_decomp:
            self._host = pot_from_decomp


        self._potentials = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.forward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.backward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.orbits = np.empty(self._equiv["snapname"].shape, dtype=object)

        self.units = UnitSystem(units) if type(units) == list else units

        self.rvir_factor = kwargs.get("rvir_factor", 2)
        
    @property
    def full_orbit(self):
        mask = (self.orbits == None)
        if np.all(mask): return None
        else:
            for i, orbit in enumerate(self.orbits[~mask]):
                if i == 0:
                    fo = orbit
                else:
                    fo += orbit

            return fo

            
    def _load_equivalence(self, equiv):
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


    def _load_pot_models(self):
        """Loads a list of potential models defined as the ones in pot_model_list.
        """
        return None


    def _get_host_params(self, redshift=None, snapshot=None):
        """Loads a halo identified by its sub_tree ID using the Halo class. A given redshift or snapshot number can be
        suplied to load the halo at a single redshift, or load all of its snapshots.
        """
        closest_value = lambda number, array: array[np.argmin(np.abs(np.array(array) - number))]

        if self._equiv is None:
            raise ValueError("There is no snapshot number <----> snapshot file equivalence table set!")

        if redshift is None:
            index = self._host[
                (self._host["Snapshot"] == snapshot)
            ].index
        elif snapshot is None:
            index = self._host[
                (self._host["Redshift"] == closest_value(redshift, self._host["Redshift"].values))
            ].index
        halo = self._host.loc[index]
        halo_params = {
            "redshift": halo["Redshift"].values[0].astype(float),
            "center": unyt_array(halo[["position_x", "position_y", "position_z"]].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "center_vel": unyt_array(halo[["velocity_x", "velocity_y", "velocity_z"]].values[0].astype(float), 'km/s'),
            "rvir": unyt_quantity(halo["virial_radius"].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "rs": unyt_quantity(halo["scale_radius"].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "vmax": unyt_quantity(halo["vmax"].values[0].astype(float), 'km/s'),
            "vrms": unyt_quantity(halo["vrms"].values[0].astype(float), 'km/s'),
            "mass": unyt_quantity(halo["mass"].values[0].astype(float), 'Msun'),
            "R/Rvir" : halo["R/Rvir"].values[0].astype(float)
        }
        return halo_params

    
    def _snap_to_index(self, snap):
        """Translates snapshot to index.
        """
        return self._equiv[self._equiv["snapshot"] == snap].index.values[0]

    def _index_to_snap(self, index):
        """Translates index to snap.
        """
        return self._equiv.loc[index]["snaphot"]

        
    def _find_snap(self, t, z, snap, prefer="nearest"):
        """Finds time series index for time t
        """
        if snap is not None:
            return self.ts[self._equiv["snapshot"] == snap]
        if z is not None:
            return self.ts.get_by_redshift(z, prefer=prefer)
        if t is not None:
            return self.ts.get_by_time((t.value, t.unit.to_string()), prefer=prefer)

    
    def _find_index(self, t, prefer="nearest"):
        """Finds index of closest for  time t.
        """
        time_diff = self._host['Time'] - t.to("Gyr")
        
        if prefer == "nearest":
            closest_index = time_diff.abs().idxmin()
        elif prefer == "smaller":
            negative_diffs = time_diff[time_diff <= 0]
            if negative_diffs.empty:
                raise ValueError("No 'time' values are smaller than the specified 't'.")
            closest_index = negative_diffs.idxmax()
        elif prefer == "larger":
            positive_diffs = time_diff[time_diff >= 0]
            if positive_diffs.empty:
                raise ValueError("No 'time' values are larger than the specified 't'.")
            closest_index = positive_diffs.idxmin()
        else:
            raise ValueError("Invalid 'prefer' argument. Choose from 'nearest', 'smaller', or 'larger'.")
        
        return self._snap_to_index(self._host.loc[closest_index]["Snapshot"].astype(int))
    
    
    def _prefabricate_particlePotenial(self, indices):
        """Constructs, in parallel, as many trees as needed for the interpolation.
        """
        for index in indices:
            if self._potentials[index] == None:
                rvir_factor = self.rvir_factor
                sn = self._equiv.loc[index]["snapshot"]
                halopars = self._get_host_params(snapshot=sn)
                ds = self.ts[index]
                sp = ds.sphere(halopars["center"], rvir_factor * halopars["rvir"])
                self._potentials[index] = ParticlePotential(
                    sp["nbody", "particle_mass"].to(self.units.decompose(1 * u.Msun).unit.to_string()),
                    (sp["nbody", "particle_position"] - sp.center).to(self.units.decompose(1 * u.kpc).unit.to_string()),
                    (unyt_quantity(0.08, 'kpc') * sp["nbody", "particle_ones"]).to(self.units.decompose(1 * u.kpc).unit.to_string()),
                    units=self.units
                )
                self._potentials[index]._compute_tree_ifnone()

                gc.collect()
                del ds
                del sp
            else:
                pass
        gc.collect()


    
    def _integrator(self, pot, ics, dt, nstep, **kwargs):
        """Integrator function using gala as backbone.
        """
        return gp.Hamiltonian(pot).integrate_orbit(
            ics.PhaseSpacePos, 
            dt=dt, 
            n_steps=nstep, 
            Integrator=kwargs.get("integrator", gi.LeapfrogIntegrator) 
        )

    
    def _interpolator_consecutive(self, index_1, index_2, time_pars):
        """Interpolates between two consecutive snapshots
        """
        t1, t2 = time_pars["t_ini"], time_pars["t_fini"]
        
        t = self.forward_orbits[index_1].t + t1
        xyz = self.forward_orbits[index_1].xyz * (t2 - t) / (t2 - t1) + self.backward_orbits[index_2].xyz[:, ::-1] * (t - t1) / (t2 - t1) 
        vxyz = self.forward_orbits[index_1].v_xyz * (t2 - t) / (t2 - t1) + self.backward_orbits[index_2].v_xyz[:, ::-1] * (t - t1) / (t2 - t1) 
        
        self.orbits[index_1] = CartesianOrbit(
            xyz,
            vxyz,
            t,
            units=self.units
        )


    def _clean_fwo(self, save_index=None):
        """Cleans forward and backward orbits to save memory. Some can be saved
        """
        if save_index is not None:
            saved_forbits = copy(self.forward_orbits[save_index])
            saved_worbits = copy(self.backward_orbits[[s+1 for s in save_index]])

        del self.forward_orbits, self.backward_orbits
        gc.collect()
        self.forward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.backward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)

        if save_index is not None:
            self.forward_orbits[save_index] = saved_forbits
            self.backward_orbits[[s+1 for s in save_index]] = saved_worbits


    def _clean_pot(self, save_index=None):
        """Cleans potential instances to save memory. Some can be saved.
        """
        if save_index is not None:
            saved_pots = copy(self._potentials[save_index])

        del self._potentials
        gc.collect()
        self._potentials = np.empty(self._equiv["snapname"].shape, dtype=object)

        if save_index is not None:
            self._potentials[save_index] = saved_pots

    
    def precompute_potential(self, t=None, snap=None, z=None):
        """Precomputes potential for given times.
        """
        if (t is None) & (z is None) & (snap is None):
            indexes = self._equiv.index.values
        elif snap is not None:
            indexes = self._equiv[np.isin(self._equiv["snapshot"].values, snap)].index.values
        else:
            if isinstance(t, list):
                indexes = []
                for time in t:
                    indexes.append(self._find_index(time, prefer="nearest"))
            else:
                indexes = [self._find_index(t, prefer="nearest")]

        self._prefabricate_particlePotenial(indexes)

        
    def interpolate_consecutive(self, ics, fcs, **kwargs):
        """Interpolates the orbit bewteen initial conditions ICS and final conditions FCS.
        """
        verbose = kwargs.get("verbose", False)
        ics, fcs = ics.change_units(self.units), fcs.change_units(self.units)
        
        time_diff = fcs.time - ics.time
        dt = self.units.decompose(kwargs.get("dt", 0.5 * u.Myr))
        nstep = np.ceil(time_diff / dt).astype(int)
        if nstep == 1:
            dt = time_diff
        else:
            dt = (time_diff / nstep).to(dt.unit)

        index_ics, index_fcs = self._find_index(ics.time, prefer="nearest"), self._find_index(fcs.time, prefer="nearest")
        fn_ics, fn_fcs = self._equiv.loc[index_ics]["snapname"], self._equiv.loc[index_fcs]["snapname"]

        t_ics, t_fcs = self._equiv.loc[index_ics]["time"] * u.Gyr, self._equiv.loc[index_fcs]["time"] * u.Gyr

        self._clean_pot(save_index=index_ics)

        self._prefabricate_particlePotenial([index_ics, index_fcs])

        if verbose:
            print(f"\n")
            print("Initial & Final Conditions for Interp:")
            print("--------------------------------------")
            print("")
            print(f"ics: -pos {ics.pos.astype('float16')}")
            print(f"ics: -vel {ics.vel.astype('float16')}")
            print(f"ics: -time {ics.time}")
            print("")
            print(f"fcs: -pos {fcs.pos.astype('float16')}")
            print(f"fcs: -vel {fcs.vel.astype('float16')}")
            print(f"fcs: -time {fcs.time}")
            print("")
            print(f"choose dt={dt:.3e}    nstep: {nstep}")
            print("")
            print(f"potential instances: {np.sum(self._potentials != None)}")
            print(f"\n")
            print(f"\n")

            
        self.forward_orbits[index_ics] = self._integrator(self._potentials[index_ics], ics, dt, nstep, **kwargs)            
        self.backward_orbits[index_fcs] = self._integrator(self._potentials[index_fcs], fcs, -1 * dt, nstep, **kwargs)            

        self._interpolator_consecutive(index_ics, index_fcs, {"t_ics": t_ics, "t_fcs": t_fcs, "t_ini": ics.time, "t_fini": fcs.time})

        self._clean_pot(save_index=index_fcs)
        self._clean_fwo()
        gc.collect()
        if kwargs.get("single", True):
            return self.orbits[index_ics]

    
    def interpolate(self, ics_list, **kwargs):
        """Interpolates over a list of snapshots, given a list of PhaseSpaceInstants
        """
        from tqdm import tqdm
        ics_list = [ics.change_units(self.units) for ics in ics_list]
        imax = len(ics_list)

        for i in tqdm(range(imax - 1), desc="Interpolating over snapshot pairs..."):
            self.interpolate_consecutive(ics_list[i], ics_list[i+1], single=False, **kwargs)

            if psutil.virtual_memory().percent > 65:
                self.clean_potentials
                self._clean_fwo()

            gc.collect()

        return self.full_orbit



    def clean_orbits(self):
        """Erases al orbits from memory.
        """
        del self.forward_orbits, self.backward_orbits, self.orbits
        
        self.forward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.backward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.orbits = np.empty(self._equiv["snapname"].shape, dtype=object)

        print("ALL ORBITS ERASED...")

        
    def clean_potentials(self):
        """Erases al potentials from memory.
        """
        del self._potentials
        
        self._potentials = np.empty(self._equiv["snapname"].shape, dtype=object)
        
        print("ALL POTENTIALS ERASED...")









#    def _find_snapshot_intersection(self, t_interval):
#        """Finds [t_ics, t_fcs] such that the in t_ics <= t_ini and t_fin <= t_fcs.
#        """
#        ds_interval_ics = self._find_snap(t[0], None, prefer="smaller")
#        ds_interval_fcs = self._find_snap(t[1], None, prefer="larger")
#        index_ics, index_fcs = self._find_index(ics.time, prefer="smaller"), self._find_index(fcs.time, prefer="larger")
#        
#        fn_ics, fn_fcs = self._equiv.loc[index_ics]["snapname"], self._equiv.loc[index_fcs]["snapname"]
#        
#        assert ds_interval_ics.filename.split("/")[-1] == fn_ics, "Bad ics!"
#        assert ds_interval_fcs.filename.split("/")[-1] == fn_fcs, "Bad fcs!"
#        
#        all_indices = list(range(index_ics, index_fcs + 1))
#        return all_indices 
#ds_ics, ds_fcs = self._find_snap(ics.time, ics.redshift, prefer="smaller"), self._find_snap(fcs.time, fcs.redshift, prefer="larger")
#assert ds_ics.filename.split("/")[-1] == fn_ics, "Bad ics!"
#assert ds_fcs.filename.split("/")[-1] == fn_fcs, "Bad fcs!"



        

#def ConstructTreeSphere(sp, unit_system):
#    """Wrapper for parallel execution of pytreegrav's ConstructTree
#    """
#    pos = unit_system.decompose((sp["nbody", "particle_position"] - sp.center).to("kpc").to_astropy())
#    mass = unit_system.decompose((sp["nbody", "particle_mass"]).to("Msun").to_astropy())
#    softs = None #unit_system.decompose((sp["nbody", "softening"]).to("kpc").to_astropy())
#    return ConstructTree(
#        pos,
#        mass,
#        softening=softs,
#        quadrupole=True
#    )
#
#def _prefabricate_particle_potential(index, orbis_instance):
#    sn = orbis_instance._equiv.loc[index]["snapshot"]
#    halopars = orbis_instance._get_host_params(snapshot=sn)
#    ds = orbis_instance.ts[index]
#    sp = ds.sphere(halopars["center"], 2 * halopars["rvir"])
#    orbis_instance._potentials[index] = ParticlePotential(
#        sp["nbody", "particle_mass"].to(orbis_instance.units.decompose(1 * u.Msun).unit.to_string()),
#        (sp["nbody", "particle_position"] - sp.center).to(orbis_instance.units.decompose(1 * u.kpc).unit.to_string()),
#        (unyt_quantity(0.08, 'kpc') * sp["nbody", "particle_ones"]).to(orbis_instance.units.decompose(1 * u.kpc).unit.to_string()),
#        units=orbis_instance.units
#    )
#    orbis_instance._potentials[index]._compute_tree_ifnone()
    


            

    #def _prefabricate_particlePotenial_v2(self, index):
    #    """Constructs, in parallel, as many trees as needed for the interpolation.
    #    """
    #    sn = self._equiv.loc[index]["snapshot"]
    #    halopars = self._get_host_params(snapshot=sn)
    #    ds = self.ts[index]
    #    sp = ds.sphere(halopars["center"], 2 * halopars["rvir"])
    #    self._potentials[index] = ParticlePotential(
    #        sp["nbody", "particle_mass"].to(self.units.decompose(1 * u.Msun).unit.to_string()),
    #        (sp["nbody", "particle_position"] - sp.center).to(self.units.decompose(1 * u.kpc).unit.to_string()),
    #        (unyt_quantity(0.08, 'kpc') * sp["nbody", "particle_ones"]).to(self.units.decompose(1 * u.kpc).unit.to_string()),
    #        units=self.units
    #    )
    #    self._potentials[index]._compute_tree_ifnone()
    #    return None


     #def _construct_particlePotential(self, indices):
    #    """Constructs potentials in parallel using one core per potential.
    #    """
    #    from concurrent.futures import ProcessPoolExecutor
    #
    #    
    #    with ProcessPoolExecutor(max_workers=14) as executor:
    #        futures = [
    #            executor.submit(_prefabricate_particle_potential, index, self)
    #            for index in indices
    #        ]
    #        result = [future.result() for future in futures]
    #
    #
    #def _construct_particlePotential(self, indices):
    #    from pathos.multiprocessing import ProcessPool
    #    with ProcessPool(nodes=14) as pool:
    #        pool.map(self._prefabricate_particlePotenial_v2, indices)   