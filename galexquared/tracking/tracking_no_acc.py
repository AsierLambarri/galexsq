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

from ..config import config
from ..mergertree import MergerTree
from ..class_methods import load_ftable, softmax, half_mass_radius, refine_6Dcenter


import gc















def _extract_particles_infall(subtree, sp, row):
    """Returns particle indices that are bound (AGORA VII)
    """
    from unyt import unyt_array as unyt_arr
    from time import time

    st = time()
    pos = sp["nbody", "particle_position"].to("kpc")
    mass = sp["nbody", "particle_mass"].to("Msun")
    ft = time()
    potential = unyt_arr(Potential(
        pos=pos, 
        m=mass,
        softening=None, 
        G=4.300917270038E-6, 
        parallel=False
    ), 
    'km**2/s**2'
    )
    ft1 = time()
    n = len(sp["nbody", "particle_mass"])
    
    grav_energy = potential * sp["nbody", "particle_mass"].to("Msun")
    del potential
    vel = unyt_arr(row[["velocity_x", "velocity_y", "velocity_z"]].values[0], 'km/s')
    kinetic = 0.5 * sp["nbody", "particle_mass"].to("Msun") * np.linalg.norm(sp["nbody", "particle_velocity"].to("km/s") - vel, axis=1)**2
    
    mask = (kinetic + grav_energy < 0)
    index_sel1 = sp["nbody", "particle_index"][mask].astype(int).value
    
    del sp
    del grav_energy, kinetic, mask
    gc.collect()
    ft2 = time()

    print(f"{subtree}, data extraction: {ft-st}s, Potential for n={n}: {ft1 - ft}s, rest={ft2-ft1}s")
    return index_sel1


def _extract_particles_ptoinfall(subtree, file, row):
    """Returns particle indices that are bound (AGORA VII)
    """
    cen = row[["position_x", "position_y", "position_z"]].values[0]
    rvir = row["virial_radius"].values[0]

    ds = yt.load(file)
    sp = ds.sphere((cen, 'kpccm'), (rvir, 'kpccm'))
    
    return sp["nbody", "particle_index"].astype(int).value


def _extract_particles_infall_wrapper(task):
    return _extract_particles_infall(*task)
def _extract_particles_ptoinfall_wrapper(task):
    return _extract_particles_ptoinfall(*task)



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

        self.particles_index_dict_infall = {}
        self.particles_index_dict_prior = {}

    
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

    @property
    def particles_index_dict(self):
        assert np.unique(self.particles_index_dict_infall.keys()) == np.unique(self.particles_index_dict_prior.keys()), "Somehow, the selection skipped some infall/prior moments!"
        merged = {}
        for key in np.unique(self.particles_index_dict_infall.keys()):
            merged[key] = np.unique(
                np.concatenate(( self.particles_index_dict_infall[key], self.particles_index_dict_prior[key] ))
            )
        return merged


    
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

        #yt.utilities.logger.ytLogger.setLevel(40)

        nproc = min(int(kwargs.get("parallel", 1)), len(inserted_halos))
        pool = mp.Pool(nproc)
        if nproc > 6: warnings.warn(f"Having nproc={nproc} may fuck up your cache.")


        ds = self.ds
        
        tasks_infall = []
        tasks_prior = []
        for subtree in inserted_halos:
            merge = self.MergeInfo[self.MergeInfo["Sub_tree_id"] == subtree]
            row_infall = self.mergertree.subtree(subtree, snapshot=merge["crossing_snap"].values[0])
            row_prior = self.mergertree.subtree(subtree, snapshot=merge["crossing_snap_2"].values[0])

            assert int(self.equiv[self.equiv["snapshot"] == merge["crossing_snap"].values[0]].index[0]) == self._snap_to_index(merge["crossing_snap"].values[0]), "file selection fucked up"
            file_infall = self._files[self._snap_to_index(merge["crossing_snap"].values[0])]
            file_prior = self._files[self._snap_to_index(merge["crossing_snap_2"].values[0])]
 
            tasks_infall.append((
                subtree,
                file_infall,
                row_infall
            ))
            tasks_prior.append((
                subtree,
                file_prior,
                row_prior
            ))


            
        chsize = kwargs.get("chunksize", len(tasks_prior) // (4*nproc) + 1)

        for i, particle_list in enumerate(
            tqdm(pool.imap(_extract_particles_ptoinfall_wrapper, tasks_prior, chunksize=chsize), total=len(tasks_prior), desc="Selecting particles prior to infall...")
        ):
            self.particles_index_dict_prior[tasks_prior[i][0]] = particle_list
  
        for i, particle_list in enumerate(
            tqdm(pool.imap(_extract_particles_infall_wrapper, tasks_infall, chunksize=chsize), total=len(tasks_infall), desc="Selecting particles at infall...")
        ):
            self.particles_index_dict_infall[tasks_infall[i][0]] = particle_list

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