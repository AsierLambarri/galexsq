import os
import yt
import sys
import h5py
import warnings
import numpy as np
import pandas as pd

from ..config import config
from ..mergertree import MergerTree

import gc



class BaseTracker:
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
                 equivalence_table,
                 ptypes
                 ):
        """Init function.
        """
        self.ptypes = ptypes
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
        infall_keys = set(self.particles_index_dict_infall)
        prior_keys = set(self.particles_index_dict_prior)
        
        if infall_keys != prior_keys:
            missing_infall = prior_keys - infall_keys
            missing_prior = infall_keys - prior_keys
            raise RuntimeError(
                f"Mismatch in keys!  "
                f"Missing in infall: {missing_infall}, "
                f"Missing in prior:  {missing_prior}"
            )
    
        merged = {}
        for key in infall_keys:
            arr1 = np.asarray(self.particles_index_dict_infall[key])
            arr2 = np.asarray(self.particles_index_dict_prior[key])
            merged[key] = np.unique(np.concatenate((arr1, arr2)))
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


    def _sanitize_merge_info(self, ignore_faulty=False):
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


        if not ignore_faulty:
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
