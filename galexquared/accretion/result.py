import h5py
import numpy as np
import pandas as pd

import collections.abc


class AccretionHistoryResult:
    """Holds results and provides query utilities for accretion history."""
    def __init__(
        self,
        accretion_id=None,
        particle_indexes=None,
        creation_times=None,
        snapshot_creation_times=None,
        snap_particle_dict=None,
        time_particle_dict=None,
        npart=None,
        accreted_particles=None
    ):
        self.accretion_id = accretion_id
        self.particle_indexes = particle_indexes
        self.creation_times = creation_times
        self.snapshot_creation_times = snapshot_creation_times
        self.snap_particle_dict = snap_particle_dict
        self.time_particle_dict = time_particle_dict
        self.npart = npart
        self.accreted_particles = accreted_particles

    def born_ids(self, snap):
        """Particles appearing on a given snapshot."""
        return self.snap_particle_dict[snap]

    def born_between(self, ft, st=0, mode="snapshot"):
        """Particles appearing between stamps st and ft."""
        if mode.lower() == "time":
            mask = (st <= self.creation_times.value) & (self.creation_times.value <= ft)
            return self.particle_indexes[mask]
        elif mode.lower() == "snapshot":
            try:
                indices = np.concatenate([self.snap_particle_dict[key] for key in range(int(st), int(ft) + 1) if key in self.snap_particle_dict])
            except:
                indices = []
            return indices

    def apply_delta_filter(self):
        """Applies delta filter to the accreted particle events."""
        self._snap_particles = {}
        for snapshot in self.accreted_particles["Snapshot"].unique():
            self._snap_particles[snapshot] = self.accreted_particles[self.accreted_particles["Snapshot"] == snapshot]
            self._snap_particles[snapshot] = _delta_fitering(self, snapshot)

    def _reconstruct_snapshot(self, snap):
        """Reconstruct subtree mapping at given snapshot by last event."""
        short_snap = self.accreted_particles[self.accreted_particles["Snapshot"] <= snap]
        last = (
            short_snap
            .sort_values("Snapshot")
            .drop_duplicates("particle_index", keep="last")
            .loc[:, ["particle_index", "Sub_tree_id"]]
        )
        last["Snapshot"] = snap
        last["Time"] = self.creation_times.name == "time" and self.creation_times or None  # placeholder
        # original logic mapping
        mapping = None  # omitted placeholder to keep structure
        return last[["particle_index", "Snapshot", "Time", "Sub_tree_id"]]

    def _track_particle(self, pid_list):
        """Return the delta events for given particle(s)."""
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
        """Query subtree assignments from results."""
        if snapshot is None and particle_index is None:
            raise ValueError("At least one of 'snapshot' or 'particle_index' must be provided.")
        snaps = ([snapshot] if not isinstance(snapshot, collections.abc.Sequence) else list(snapshot)) if snapshot is not None else None
        pids  = ([particle_index] if not isinstance(particle_index, collections.abc.Sequence) else list(particle_index)) if particle_index is not None else None
        if snaps is not None and pids is not None:
            parts = [ self._reconstruct_snapshot(s) for s in snaps ]
            df = pd.concat(parts, ignore_index=True) if len(parts)>1 else parts[0]
            return df[df["particle_index"].isin(pids)].reset_index(drop=True)
        if snaps is not None:
            parts = [ self._reconstruct_snapshot(s) for s in snaps ]
            return pd.concat(parts, ignore_index=True).reset_index(drop=True)
        return self._track_particle(pids)

    def query_subtree(self, subtree, snapshot):
        """Returns all particles in a subtree at a given snapshot."""
        snaps = ([snapshot] if not isinstance(snapshot, collections.abc.Sequence) else list(snapshot)) if snapshot is not None else None
        parts = []
        for s in snaps:
            res = self._reconstruct_snapshot(s)
            if not res.empty:
                parts.append(res[res["Sub_tree_id"] == subtree])
        reconstructed_snap = pd.concat(parts, ignore_index=True).reset_index(drop=True)
        return reconstructed_snap[reconstructed_snap["Sub_tree_id"] == subtree]

    def reduce_accretion(self):
        """
        For each particle in accreted_particles, find the last subtree before final accretion.
        """
        main = self.accretion_id
        df = self.accreted_particles[['particle_index','Snapshot','Sub_tree_id']]
        finals = {main, -1}
        nonfinal = df[~df['Sub_tree_id'].isin(finals)]
        last_nf = (
            nonfinal
            .groupby('particle_index', as_index=False)
            .agg(last_snap=('Snapshot','max'))
        )
        if not last_nf.empty:
            last_nf = last_nf.merge(
                df,
                left_on=['particle_index','last_snap'],
                right_on=['particle_index','Snapshot'],
                how='left'
            )
            preceding_nf = last_nf[['particle_index','Sub_tree_id']].rename(columns={'Sub_tree_id':'preceding_subtree'})
        else:
            preceding_nf = pd.DataFrame(columns=['particle_index','preceding_subtree'])
        all_pids = df['particle_index'].unique()
        pids_with_nf = preceding_nf['particle_index'].unique()
        pids_only_finals = np.setdiff1d(all_pids, pids_with_nf, assume_unique=True)
        if len(pids_only_finals):
            birth = (
                df[df['particle_index'].isin(pids_only_finals)]
                .sort_values(['particle_index','Snapshot'])
                .groupby('particle_index', as_index=False)
                .first()
            )
            birth['preceding_subtree'] = np.where(
                birth['Sub_tree_id'] == -1,
                -1,
                main
            )
            preceding_birth = birth[['particle_index','preceding_subtree']]
        else:
            preceding_birth = pd.DataFrame(columns=['particle_index','preceding_subtree'])
        result = pd.concat([preceding_nf, preceding_birth], ignore_index=True)
        result = result.drop_duplicates('particle_index', keep='first').reset_index(drop=True)
        counts_df = (
            result
            .groupby('preceding_subtree', as_index=False)
            .size()
            .rename(columns={'preceding_subtree':'Sub_tree_id','size':'npart'})
        )
        return result, counts_df

    def save(self, name, format="hdf5", **kwargs):
        """Saves accreted_particles."""
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
        """Loads accreted_particles."""
        if name.endswith(".csv"):
            self.accreted_particles = pd.read_csv(name, index=False)
        elif any(name.endswith(ext) for ext in [".hdf5", ".h5", ".h", ".hdf"]):
            import h5py
            with h5py.File(name, "r") as f:
                dset = f["data"]
                data = dset[:]
                columns = list(dset.attrs["columns"])
                self.accreted_particles = pd.DataFrame.from_records(data, columns=columns)
        else:
            raise Exception("Non supported format")












































class AccretionHistoryResult:
    """Stores and allows to manipulate results created through accretion history.
    """

    def __init__(self, 
                 fn=None
                ):
        if fn is not None:
            self.load(fn)


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
                #dset.attrs["dtypes"] = np.array([str(df[col].dtype) for col in df.columns], dtype=h5py.string_dtype(encoding='utf-8'))
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