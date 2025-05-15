import os
import gc

import yt
import numba
import numpy as np
import pandas as pd
from tqdm import tqdm

from time import time
import collections.abc

from ._helpers import _goofy_mass_scaling, _check_particle_uniqueness, _remove_duplicates
from .assignment import _assign_halo, _assign_halo_wrapper
from ..mergertree import MergerTree
from ..config import config



class AccretionHistory:
    """Constructs the Accretion History of the selected Sub_tree. Follows arxiv:2410.09144.
    
    Recomendation is to use only for z=zmin output.
    """
    def __init__(
        self,
        mergertree,
        file_table,
        pdir,
        **kwargs
    ):
        """Innit function.
        """
        self._type_list = {}
        self.mergertree = MergerTree(mergertree)
        self.mergertree.set_equivalence(file_table)

        if np.log2(self.mergertree.CompleteTree["Sub_tree_id"].max()) < 15:
            self._type_list["Sub_tree_id"] = np.int16
        elif np.log2(self.mergertree.CompleteTree["Sub_tree_id"].max()) < 31:
            self._type_list["Sub_tree_id"] = np.int32
        else:
            self._type_list["Sub_tree_id"] = np.int64

        if np.log2(self.mergertree.CompleteTree["Snapshot"].max()) < 16:
            self._type_list["Snapshot"] = np.uint16
        elif np.log2(self.mergertree.CompleteTree["Snapshot"].max()) < 32:
            self._type_list["Snapshot"] = np.uint32
        else:
            self._type_list["Snapshot"] = np.uint64

        self._type_list["Assigned_Halo_Mass"], self._type_list["mass_scale"] = _goofy_mass_scaling(self.mergertree.CompleteTree["mass"].max(), self.mergertree.CompleteTree["mass"].min())
        self._type_list["Time"] = np.float32

        vesc_approx = np.sqrt( 2 * 4.3E-6 * self.mergertree.CompleteTree["mass"])
        self._type_list["Assigned_Halo_Vel"], self._type_list["velocity_scale"] = _goofy_mass_scaling(vesc_approx.max(), vesc_approx.min())


        
        self._prefix = os.path.commonprefix([os.path.basename(file) for file in self.equiv["snapname"].values])
        self._files = [pdir + "/" + file for file in self.equiv["snapname"].values]
        self.ts = yt.DatasetSeries(self._files)

        yt.utilities.logger.ytLogger.setLevel(40)
        self._cosmo = self.ts[0].cosmology
        time = []
        if kwargs.get("precise_times", False):
            for ds in tqdm(self.ts, desc="Fetching Simulation Times to override MergerTree"):
                time.append(ds.current_time.to("Gyr").value.astype(float))
        self._time_override = np.array(time)
        yt.utilities.logger.ytLogger.setLevel(20)

    @property
    def equiv(self):
        return self.mergertree.equivalence_table
    @property
    def cosmo(self):
        return self._cosmo

    

    def _snap_to_index(self, snap):
        """Translates snapshot to index.
        """
        return self.equiv[self.equiv["snapshot"] == snap].index.values[0]

    def _index_to_snap(self, index):
        """Translates index to snap.
        """
        return self.equiv.loc[index]["snapshot"]

        
    def _find_snap(self, t, z, snap, prefer="nearest"):
        """Finds time series index for time t
        """
        if snap is not None:
            return self.ts[self.equiv["snapshot"] == snap]
        if z is not None:
            return self.ts.get_by_redshift(z, prefer=prefer)
        if t is not None:
            return self.ts.get_by_time((t.value, t.unit.to_string()), prefer=prefer)

    
    def _find_index(self, t, prefer="nearest"):
        """Finds index of closest for  time t.
        """
        time_diff = self.mergertree.PrincipalLeaf['Time'] - t.to("Gyr")
        
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
        
        return self._snap_to_index(self.mergertree.PrincipalLeaf.loc[closest_index]["Snapshot"].astype(int))
    
    def _add_creation_time(self, ds):
        """Adds creation time to dataset.
        """
        if config.code == "ART": 
            pass
        elif config.code == "GEAR":
            ds.add_field(
                ("PartType1", "particle_creation_time"),
                function=lambda field, data: ds.cosmology.t_from_a(data['PartType1', 'StarFormationTime']),
                sampling_type="local",
                units='Gyr'
            )
        return ds
    
    def _assign_t_snap(self, particle_creation_times):
        """Assigns the closest snapshot to each of the times
        """
        snapshot_ids = self.equiv["snapshot"].values
        mask = np.isin(self.mergertree.PrincipalLeaf["Snapshot"], self.equiv["snapshot"].values)
        snapshot_times = self.mergertree.PrincipalLeaf["Time"].values[mask]
        earliest_snapshot = self.equiv["snapshot"].min()
        lower_bound = self.equiv["time"].min()
        
        indices = np.searchsorted(snapshot_times, particle_creation_times)
        indices = np.clip(indices, 0, len(snapshot_times) - 1)
        
        closest_snapshots = snapshot_ids[indices]
        closest_snapshots[particle_creation_times < lower_bound] = earliest_snapshot
        return closest_snapshots

    def _delta_fitering(self, snap):
        """Applies Delta-Filtering to the output so that we save on-disk/run-time cache memory. This means
        that only particle host_subtree changes are kept track of, reducing the total number of rows in the
        accreted particles dataset.
        """
        prev_keys = [s for s in self._snap_particles.keys() if s < snap]
        if (not prev_keys) or (snap == self.snapshot_creation_times.max()):
            return self._snap_particles[snap]
            
        prev_df = pd.concat(
            [ self._snap_particles[s] for s in sorted(prev_keys) ],
            ignore_index=True,
            copy=False
        )
        del prev_keys
        
        last_assignment = (
            prev_df.sort_values("Snapshot")
            .drop_duplicates(subset="particle_index", keep="last")
            .loc[:, ["particle_index", "Sub_tree_id"]]
            .rename(columns={"Sub_tree_id": "subtree_prev"})
        )
        del prev_df

        merged = self._snap_particles[snap].merge(
            last_assignment,
            on="particle_index",
            how="left"
        )
        mask_new_or_changed = merged["subtree_prev"].isna() | (merged["Sub_tree_id"] != merged["subtree_prev"])
        filtered = merged.loc[mask_new_or_changed].copy()
        del merged, mask_new_or_changed

        return filtered.drop(columns=["subtree_prev"])


    
    def born_ids(self, snap):
        """Particles appearing on a given snapshot
        """
        return self.snap_particle_dict[snap]

    def born_between(self, ft, st=0, mode="snapshot"):
        """Particles appearing between stamps st and ft. These might be time in Gyr or snapshot numbers.
        """
        if mode.lower() == "time":
            mask = (st <= self.creation_times.value) & (self.creation_times.value <= ft)
            return self.particle_indexes[mask]
        elif mode.lower() == "snapshot":
            try:
                indices =  np.concatenate([self.snap_particle_dict[key] for key in range(int(st), int(ft) + 1) if key in self.snap_particle_dict])
            except: 
                indices = []
            return indices
    
    def select_accreted_particles(self, subtree, ptype, z=None, t=None, indices=None, **kwargs):
        """
        Creates Accretion History of the halo identified with subtree, starting from z/t and
        going backwards.

        Parameters
        ----------
        subtree : int
            Halo identifier.
        ptype : str
            Stellar particle identifier.
        z : float w/units, optional
            Starting redshift. Def: z=zmin.
        t : float w/units, optional
            Starting time. Def: t=tmax(=zmin).
        """
        self._ptype = ptype        
        if (subtree in ["na", "NA"]) or (subtree is None): 
            subtree_table = self.mergertree.subtree(self.mergertree.principal_subid)
        else: 
            subtree_table = self.mergertree.subtree(subtree)

            
        if z is not None:
            assert z >= subtree_table["Redshift"].min()
            index = self._find_index(self.cosmo.t_from_z(z))
            
        elif t is not None:
            assert t.to("Gyr") <= subtree_table["Time"].max()
            index = self._find_index(t.to("Gyr"))
            
        elif (z is None) & (t is None):
            z = subtree_table["Redshift"].min()
            index = self._find_index(self.cosmo.t_from_z(z))

        snap = self._index_to_snap(index)
        if (subtree in ["na", "NA"]) or (subtree is None):
            halo_params = self.mergertree.get_halo_params(self.mergertree.principal_subid, snapshot=snap)
        else:
            halo_params = self.mergertree.get_halo_params(subtree, snapshot=snap)
            
        ds = self.ts[index]
        ds = self._add_creation_time(ds)

        if (subtree in ["na", "NA"]) or (subtree is None):
            sp = ds.all_data()
        else:
            sp = ds.sphere(halo_params["center"], kwargs.get("rfac", 1) * halo_params["rvir"])
            
        if indices is None:
            self.particle_indexes = sp[self._ptype, "particle_index"].value.astype(int)
            mask = np.isin(sp[self._ptype, "particle_index"], self.particle_indexes)
            assert not False in mask, "Something went wrong!"
        else:
            self.particle_indexes = indices.astype(int)
            mask = np.isin(sp[self._ptype, "particle_index"], self.particle_indexes)
            
        self.npart = len(self.particle_indexes)
        self.creation_times = sp[self._ptype, "particle_creation_time"][mask].to("Gyr")
        self.snapshot_creation_times = self._assign_t_snap(self.creation_times.value)
        
        if np.log2(self.particle_indexes.max()) < 16:
            self._type_list["particle_index"] = np.uint16
        elif np.log2(self.particle_indexes.max()) < 32:
            self._type_list["particle_index"] = np.uint32
        else:
            self._type_list["particle_index"] = np.uint64

            
        self.snap_particle_dict = {
            snap : self.particle_indexes[self.snapshot_creation_times == snap]
            for snap in np.sort(np.unique(self.snapshot_creation_times))
        }
        if not _check_particle_uniqueness(self.snap_particle_dict):
            print("Particles being born twice have been detected... Fixing that.")
            self.snap_particle_dict = _remove_duplicates(self.snap_particle_dict)
            
        self.time_particle_dict = {
            snap : self.creation_times[np.isin(self.particle_indexes, indices)]
            for snap, indices in tqdm(self.snap_particle_dict.items(), total=len(np.unique(self.snap_particle_dict.keys())), desc="Finding birht of stars...")
        }
        del ds, sp, mask, subtree_table
        gc.collect()
        return self.snap_particle_dict

    
    def create_accretion_history(self, subtree, ptype, z=None, t=None, indices=None, mode="bipartite", **kwargs):
        """
        Creates Accretion History of the halo identified with subtree, starting from z/t and
        going backwards

        Parameters
        ----------
        subtree : int
            Halo identifier.
        ptype : str
            Stellar particle identifier.
        indices : dict[int : array[int]]
        """
        yt.utilities.logger.ytLogger.setLevel(40)

        self.accretion_id = subtree
        self._snap_particles = {}
        self.select_accreted_particles(subtree, ptype, z=z, t=t, indices=indices, **kwargs)
        
        if kwargs.get("verbose", False):
            print(f"Number of particles selected: {self.particle_indexes.shape[0]}")

        kwargs['type_list'] = self._type_list
            
        nproc = int(kwargs.get("parallel", 1))
        trajectories = kwargs.get("trajectories", False)
        trajectory_mode = kwargs.get("trajectory_mode", None)
        verbose = kwargs.get("verbose", False)
        
        #compute_potential = kwargs.get("compute_potential", True)
        #assignment = kwargs.get("assignment_mode", "most-bound")
        
        if trajectories: 
            snapshot_list = np.unique(np.clip(
                self.mergertree.equivalence_table["snapshot"].values.astype(int),
                a_min=np.unique(self.snapshot_creation_times).min().astype(int),
                a_max=np.inf
            ))
        else:
            snapshot_list = np.unique(self.snapshot_creation_times)
            
        if nproc == 1:
            for snapshot in tqdm(snapshot_list, total=len(snapshot_list)):
                self._snap_particles[snapshot] = _assign_halo(
                    snapshot,
                    self.born_between(snapshot) if trajectories else self.snap_particle_dict[snapshot],
                    self.mergertree,
                    self._ptype,
                    self._files,
                    mode,
                    **kwargs
                    #compute_potential=compute_potential,
                    #assignment=assignment,
                    #verbose=verbose,
                    #type_list=self._type_list
                )
                if trajectories and trajectory_mode in ["delta", "delta-filter"]:
                    self._snap_particles[snapshot] = self._delta_fitering(snapshot)

                if verbose:
                    print(f"{self._snap_particles[snapshot].memory_usage(deep=True)/1E6} MB")
                    print(self._snap_particles[snapshot].head())
                    print("")
                    print("")

        else:
            import multiprocessing as mp
            import warnings
            pool = mp.Pool(nproc)
            if nproc > 4: warnings.warn(f"Having nproc={nproc} may fuck up your cache.")
                
            tasks = []
            for snapshot in snapshot_list:
                particle_indexes = self.born_between(snapshot) if trajectories else self.snap_particle_dict[snapshot]
                tasks.append((
                    (snapshot,
                    particle_indexes,
                    self.mergertree,
                    self._ptype,
                    self._files,
                    mode),
                    kwargs
                    #compute_potential,
                    #assignment,
                    #verbose,
                    #self._type_list
                ))

            chsize = kwargs.get("chunksize", len(tasks) // (4*nproc))
            for i, particle_df in enumerate(
                tqdm(pool.imap(_assign_halo_wrapper, tasks, chunksize=chsize), total=len(tasks))
            ):
                snapshot = tasks[i][0][0]
                self._snap_particles[snapshot] = particle_df

                if trajectories and trajectory_mode in ["delta", "delta-filter"]:
                    self._snap_particles[snapshot] = self._delta_fitering(snapshot)

                if verbose:
                    print(f"{self._snap_particles[snapshot].memory_usage(deep=True)/1E6} MB")
                    print(self._snap_particles[snapshot].head())
                    print("")
                    print("")


            
            pool.close()
            pool.join()
            
            del pool

        self.accreted_particles = pd.concat([self._snap_particles[key] for key in self._snap_particles]).copy()
        del self._snap_particles
        return self.accreted_particles


    def apply_delta_filter(self):
        """Function that allows you to apply delta filter afterwards. Does not change internal state of class.
        """
        self._snap_particles = {}
        for snapshot in self.accreted_particles["Snapshot"].unique():
            self._snap_particles[snapshot] = self.accreted_particles[self.accreted_particles["Snapshot"] == snapshot]
            self._snap_particles[snapshot] = self._delta_fitering(snapshot)


    def _reconstruct_snapshot(self, snap):
        """Reconstruct subtree mapping at snapshot=snap by taking the last
        delta event ≤ snap for each particle.
        """
        short_snap = self.accreted_particles[self.accreted_particles["Snapshot"] <= snap]
    
        last = (
            short_snap
            .sort_values("Snapshot")                           # ascending
            .drop_duplicates("particle_index", keep="last")    # keep the largest‑Snapshot row
            .loc[:, ["particle_index", "Sub_tree_id"]]
        )
        
        last["Snapshot"] = snap
        last["Time"] = self.equiv[self.equiv["snapshot"] == snap]["time"].values[0]
        
        return last[["particle_index", "Snapshot", "Time", "Sub_tree_id"]]


    def _track_particle(self, pid_list):
        """Return the delta events for the given particle(s), in ascending snapshot order.
        """
        mapping = self.equiv.rename(columns={"snapshot": "Snapshot", "time": "Time"}).set_index("Snapshot")["Time"]
        short_track = self.accreted_particles[self.accreted_particles["particle_index"].isin(pid_list)]
        if short_track.empty:
            return pd.DataFrame(columns=["particle_index", "Snapshot", "Sub_tree_id"])
        else:
            track = []
            for pid in pid_list:
                pid_track = short_track[short_track["particle_index"] == pid]
                full_track = self.equiv[self.equiv["snapshot"] <= pid_track["Snapshot"].max()][["snapshot"]]
                full_track.rename(columns={"snapshot" : "Snapshot"}, inplace=True)
                for col in full_track.columns:
                    full_track[col] = full_track[col].astype(pid_track[col].dtype)
                merged = pd.merge_asof(
                    full_track.sort_values("Snapshot"),
                    pid_track.sort_values("Snapshot"),
                    on="Snapshot",
                    direction="backward"
                )
                merged["Time"] = merged["Snapshot"].map(mapping).fillna(merged["Time"])
                track.append(merged)
            return pd.concat(track).reset_index(drop=True)[["particle_index", "Snapshot", "Time", "Sub_tree_id"]]
                
        
    def query(self, *, snapshot=None, particle_index=None):
        """Query subtree assignments.

        Parameters
        ----------
        snapshot : int or list of int, optional
            One or more snapshot IDs.
        particle_index : int or list of int, optional
            One or more particle IDs.

        Returns
        -------
        pd.DataFrame with columns ["snapshot","particle_index","subtree"].
        """        
        if snapshot is None and particle_index is None:
            raise ValueError("At least one of 'snapshot' or 'particle_index' must be provided.")

        snaps = ([snapshot] if not isinstance(snapshot, collections.abc.Sequence) 
                  else list(snapshot)) if snapshot is not None else None
        pids  = ([particle_index] if not isinstance(particle_index, collections.abc.Sequence) 
                  else list(particle_index)) if particle_index is not None else None

        if snaps is not None and pids is not None:
            parts = [ self._reconstruct_snapshot(s) for s in snaps ]
            df = pd.concat(parts, ignore_index=True) if len(parts)>1 else parts[0]
            return df[df["particle_index"].isin(pids)].reset_index(drop=True)

        if snaps is not None:
            parts = [ self._reconstruct_snapshot(s) for s in snaps ]
            return pd.concat(parts, ignore_index=True).reset_index(drop=True)

        return self._track_particle(pids)
    
    def query_subtree(self, subtree, snapshot):
        """Returns all the particles inside a given subtree at snapshot snap
        """
        snaps = ([snapshot] if not isinstance(snapshot, collections.abc.Sequence) 
                  else list(snapshot)) if snapshot is not None else None
        parts = []
        for s in snaps:
            res = self._reconstruct_snapshot(s)
            if not res.empty:
                parts.append(res[res["Sub_tree_id"] == subtree])
        #parts = [res[res["Sub_tree_id"] == subtree] for s in snaps if (res := self._reconstruct_snapshot(s))]
        reconstructed_snap = pd.concat(parts, ignore_index=True).reset_index(drop=True)
        return  reconstructed_snap[reconstructed_snap["Sub_tree_id"] == subtree]

    
    def reduce_accretion(self):
        """
        For each particle in self.accreted_particles that ultimately ends up in
        self.accretion_id (or –1 treated as the same), find the last subtree it
        occupied *before* that final accretion into main/–1.
    
        Returns
        -------
        pd.DataFrame
          Columns: ['particle_index','preceding_subtree']
          One row per unique particle in self.accreted_particles.
        """
        main = self.accretion_id
        df = self.accreted_particles[['particle_index','Snapshot','Sub_tree_id']]
    
        # 1) Identify "final" states: main_subtree or -1
        finals = {main, -1}
        
        # 2) All events *not* in finals—these are candidate preceding states
        nonfinal = df[~df['Sub_tree_id'].isin(finals)]
        
        # 3) For each particle, find the *last* non-final snapshot (if any)
        last_nf = (
            nonfinal
            .groupby('particle_index', as_index=False)
            .agg(last_snap=('Snapshot','max'))
        )
        
        # 4) Pull out the subtree at that last non-final snapshot
        if not last_nf.empty:
            last_nf = last_nf.merge(
                df,
                left_on=['particle_index','last_snap'],
                right_on=['particle_index','Snapshot'],
                how='left'
            )
            preceding_nf = last_nf[['particle_index','Sub_tree_id']]\
                .rename(columns={'Sub_tree_id':'preceding_subtree'})
        else:
            preceding_nf = pd.DataFrame(columns=['particle_index','preceding_subtree'])
        
        # 5) Particles that never had a non-final event:
        all_pids = df['particle_index'].unique()
        pids_with_nf = preceding_nf['particle_index'].unique()
        pids_only_finals = np.setdiff1d(all_pids, pids_with_nf, assume_unique=True)
        
        if len(pids_only_finals):
            # 6) For those, look at their *first* event (birth):
            birth = (
                df[df['particle_index'].isin(pids_only_finals)]
                .sort_values(['particle_index','Snapshot'])
                .groupby('particle_index', as_index=False)
                .first()
            )
            # 7) If born unbound (-1) → preceding = -1, else → preceding = main
            birth['preceding_subtree'] = np.where(
                birth['Sub_tree_id'] == -1,
                -1,
                main
            )
            preceding_birth = birth[['particle_index','preceding_subtree']]
        else:
            preceding_birth = pd.DataFrame(columns=['particle_index','preceding_subtree'])
        
        # 8) Combine and return
        result = pd.concat([preceding_nf, preceding_birth], ignore_index=True)
        # ensure one row per particle:
        result = result.drop_duplicates('particle_index', keep='first').reset_index(drop=True)
        
        counts_df = (
            result
            .groupby('preceding_subtree', as_index=False)
            .size()
            .rename(columns={'preceding_subtree':'Sub_tree_id','size':'npart'})
        )
        return result, counts_df


    def save(self, name, format="hdf5", **kwargs):
        """Saves self.accreted_particles
        """
        if format == "csv":
            self.accreted_particles.to_csv(name, index=False)
        elif format == "hdf5":
            import h5py
            df = self.accreted_particles
            with h5py.File(name, "w") as f:
                structured_dtype = np.dtype([
                    (col, df[col].dtype) for col in df.columns
                ])
                structured_array = np.empty(len(df), dtype=structured_dtype)
                for col in df.columns:
                    structured_array[col] = df[col].values
        
                dset = f.create_dataset("data", data=structured_array)
        
                dset.attrs["columns"] = np.array(df.columns.values, dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            raise Exception("Non supported format")


    def load(self, name):
        """Loads self.accreted_particles
        """
        if name.endswith(".csv"):
            self.accreted_particles = pd.read_csv(name, index=False)
        elif name.endswith(".hdf5") or name.endswith(".h5") or name.endswith(".h") or name.endswith(".hdf"):
            import h5py
            with h5py.File(name, "r") as f:
                dset = f["data"]
                data = dset[:]
                columns = list(dset.attrs["columns"])
        
                self.accreted_particles = pd.DataFrame.from_records(data, columns=columns)
        else:
            raise Exception("Non supported format")













        
            
        


















    # def reduce_accretion(self):
    #     """
    #     For each particle whose last recorded Sub_tree_id == main_subtree,
    #     find the Sub_tree_id it occupied just before merging into main_subtree.
    
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Columns: ['particle_index', 'preceding_subtree']
    #     """
    #     main_subtree =  self.accretion_id
    #     df = self.accreted_particles
    
    #     # 1) Sort by particle and snapshot so we can shift correctly
    #     df_sorted = df.sort_values(['particle_index','Snapshot'], ignore_index=True)
    
    #     # 2) Compute the "previous subtree" for each event
    #     prev_sub = df_sorted.groupby('particle_index')['Sub_tree_id'].shift(1)
    
    #     # 3) Identify the rows where a particle enters main_subtree
    #     mask_merge = df_sorted['Sub_tree_id'] == main_subtree
    #     df_merge = df_sorted.loc[mask_merge, ['particle_index']].copy()
    
    #     # 4) For each of those, grab its preceding subtree (or main_subtree if none)
    #     df_merge['preceding_subtree'] = (
    #         prev_sub[mask_merge]
    #         .fillna(main_subtree)                 # if no prior, use main_subtree itself
    #         .astype(df_sorted['Sub_tree_id'].dtype)
    #         .values
    #     )
    
    #     return df_merge.reset_index(drop=True)











