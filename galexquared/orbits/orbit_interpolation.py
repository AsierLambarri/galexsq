import yt
import os
import gc
import psutil
import numpy as np
import pandas as pd

from unyt import unyt_quantity, unyt_array

import astropy.units as u

import gala.integrate as gi
import gala.potential as gp
from gala.units import UnitSystem


from ..class_methods import load_ftable


from .oorbit import CartesianOrbit
from .particle_potential import ParticlePotential
from .phase_instant import PhaseSpaceInstant
from .scf_potential import compute_and_filter_scf

from copy import copy



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
            snapshot=snapshot,
            units=units
        )
        ics.append(tmp)

    return ics

    
class Orbis:
    """Class for orbit interpolation, using data from simulations. Follows Richings et al. 2022
    """
    def __init__(
        self,
        *,
        host_tree=None,
        file_table=None,
        pdir=None,
        pot_mode="nbody",
        
        pot_model_list=None,
        units=None,
        **kwargs
    ):
        if not kwargs.get("logging", True):
            yt.utilities.logger.ytLogger.setLevel(40)
            
        self._host_tree = True if host_tree is not None else False
        self._host = host_tree
        self.mode = pot_mode.lower()
        
        if pot_mode.lower() in ["nbody", "scf"]:
            assert pdir is not None, "You need to provide a directory for the simulation outputs"
            self._pdir = pdir
            assert file_table is not None, "You need to provide a file table containing columns SNAPSHOT NUMBER  |  REDSHIFT  |  FILE NAME"
            self._load_equivalence(file_table) 
            self._prefix = os.path.commonprefix([os.path.basename(file) for file in self._equiv["snapname"].values])
            self._files = [pdir + "/" + file for file in self._equiv["snapname"].values]
            self.ts = yt.DatasetSeries(self._files)
            if pot_mode.lower() == "scf":
                pass
            else:
                self._octrees = np.empty(self._equiv["snapname"].shape, dtype=object)
            
        elif pot_mode.lower() in ["model"]:
            self._model_list = pot_model_list
            self.models = self._load_pot_models(pot_model_list)

            


        self._potentials = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.forward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.backward_orbits = np.empty(self._equiv["snapname"].shape, dtype=object)
        self.orbits = np.empty(self._equiv["snapname"].shape, dtype=object)

        self.units = UnitSystem(units) if type(units) == list else units

        self.rvir_factor = kwargs.get("rvir_factor", 3)
        
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


    def _load_pot_models(self, model_list):
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
        return self._equiv.loc[index]["snapshot"]

        
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
    
    
    def _prefabricate_potential(self, indices):
        """Constructs, in parallel, as many trees as needed for the interpolation.
        """
        if self.mode == "scf":
            for index in indices:
                if self._potentials[index] == None:
                    rvir_factor = self.rvir_factor
                    sn = self._equiv.loc[index]["snapshot"]
                    halopars = self._get_host_params(snapshot=sn)
                    ds = self.ts[index]
                    sp = ds.sphere(halopars["center"], rvir_factor * halopars["rvir"])
                    pool = mp.Pool(mp.cpu_count() - 2)
                    self._potentials[index] = compute_and_filter_scf(
                        (sp["nbody", "particle_position"] - sp.center).to(self.units.decompose(1 * u.kpc).unit.to_string()),       #position
                        sp["nbody", "particle_mass"].to(self.units.decompose(1 * u.Msun).unit.to_string()),                        #mass
                        10,                                                                                                        #nmax
                        4,                                                                                                         #lmax
                        halopars["rs"].to(self.units.decompose(1 * u.kpc).unit.to_string(),                                        #scale_radius for scf=nfw_rs
                        threshold=3,                                                                                               #snr threshold
                        pool=pool,                                                                                                 #pool
                        units=self.units                                                                                           #units
                    )
                    pool.join()
                    pool.close()
                    del pool

                    
        elif self.mode == "nbody":
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

        self._prefabricate_potential(indexes)

        
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
        #fn_ics, fn_fcs = self._equiv.loc[index_ics]["snapname"], self._equiv.loc[index_fcs]["snapname"]
        #t_ics, t_fcs = self._equiv.loc[index_ics]["time"] * u.Gyr, self._equiv.loc[index_fcs]["time"] * u.Gyr

        self._clean_pot(save_index=index_ics)

        self._prefabricate_potential([index_ics, index_fcs])

        if verbose:
            print("\n")
            print("Initial & Final Conditions for Interp:")
            print("--------------------------------------")
            print("")
            print("")
            print(f"Potential R/Rvir: {self.rvir_factor:.2f}")
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
            print("\n")
            print("\n")

            
        self.forward_orbits[index_ics] = self._integrator(self._potentials[index_ics], ics, dt, nstep, **kwargs)            
        self.backward_orbits[index_fcs] = self._integrator(self._potentials[index_fcs], fcs, -1 * dt, nstep, **kwargs)            

        self._interpolator_consecutive(index_ics, index_fcs, {"t_ini": ics.time, "t_fini": fcs.time})

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

            if psutil.virtual_memory().percent > 85:
                self.clean_potentials()
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
    


            

    #def _prefabricate_potential_v2(self, index):
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
    #        pool.map(self._prefabricate_potential_v2, indices)   