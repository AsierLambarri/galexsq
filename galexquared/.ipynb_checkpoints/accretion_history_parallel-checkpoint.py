import yt
import os
import gc
import numpy as np
import pandas as pd

from .class_methods import load_ftable
from .mergertree import MergerTree
from .config import config

from copy import copy, deepcopy
from tqdm import tqdm


def _nfw_potential(r, mvir, rs, c, G):
    x = r / rs
    A_nfw = np.log(1 + c) - c / (1 + c)
    return -G * mvir * rs / A_nfw * np.log(1 + x) / x 

def assign_birth_snapshot(snap, mergertree, particle_indexes, ptype, file_list, equiv):  
    """
    Process a single snapshot and assign a birth halo (birth_subtree)
    to the particles with indexes "particle_indexes" on that snapshot.
    
    Parameters
    ----------
    snap : int
        The snapshot number.
    particle_indexes : array[int]
        Array of particle indexes for this snapshot.
    
    Returns
    -------
    particles_df : pandas.DataFrame
        DataFrame with columns:
            - "particle_index"
            - "birth_time"
            - "birth_subtree"
    """
    from scipy.spatial import KDTree
    yt.utilities.logger.ytLogger.setLevel(40)
    # Get the yt dataset corresponding to the snapshot.
    fn = int(equiv[equiv["snapshot"] == snap].index[0])
    ds = yt.load(file_list[fn])

    # Select the merger tree halos for this snapshot.
    merger_df = mergertree.CompleteTree[mergertree.CompleteTree["Snapshot"] == snap]

    # Add a particle filter that selects particles with the given indexes.
    # yt.add_particle_filter(
    #     "born_stars", 
    #     function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], particle_indexes), 
    #     filtered_type=ptype, 
    #     requires=["particle_index", "particle_position", "particle_velocity"]
    # )
    # ds.add_particle_filter("born_stars")
    ad = ds.all_data()
    mask = np.isin(ad[ptype, "particle_index"].astype(int), particle_indexes)
    # Build a DataFrame for the particles in the snapshot.
    particles_df = pd.DataFrame({
        'particle_index': ad[ptype, "particle_index"][mask].astype(int),
        'birth_time': ad[ptype, "particle_creation_time"][mask].to("Gyr").value,
    })
    particles_df['birth_subtree'] = np.nan
    particles_df['Assigned_Halo_Mass'] = 0

    particle_positions = ad[ptype, "particle_position"][mask].to("kpccm").value
    particle_velocities = ad[ptype, "particle_velocity"][mask].to("km/s").value  

    particle_tree = KDTree(particle_positions)

    assert len(merger_df["Redshift"].unique()) == 1, "Mixed redshifts detected!"
    redshift = merger_df["Redshift"].values[0]

    for _, halo in merger_df.iterrows():
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']])
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']])
        radius = halo['virial_radius']
        halo_mass = halo['mass']
        halo_c = halo['virial_radius'] / halo['scale_radius']
        subtree = halo["Sub_tree_id"]

        # Query the KDTree for particles within the halo's virial radius.
        local_indices = particle_tree.query_ball_point(halo_center, r=radius)
        if len(local_indices) == 0:
            continue  

        pos_subset = particle_positions[local_indices]     # positions in kpccm
        vel_subset = particle_velocities[local_indices]      # velocities in km/s
        rel_positions = pos_subset - halo_center
        rel_velocities = vel_subset - halo_vel

        distances = np.linalg.norm(rel_positions, axis=1)  # in kpccm
        vel_mags = np.linalg.norm(rel_velocities, axis=1)   # in km/s


        denom = np.log(1 + halo_c) - halo_c / (1 + halo_c)
        if denom == 0:
            v_esc = np.ones_like(distances) * -1
        else:
            phi = _nfw_potential(
                distances / (1 + redshift),
                halo_mass,
                halo['scale_radius'] / (1 + redshift),
                halo_c,
                4.3E-6
                
                
            )
            v_esc = np.sqrt(2 * np.abs(phi))  # in km/s
        
        bound_mask = vel_mags < v_esc
        
        current_mass = particles_df.loc[local_indices, 'Assigned_Halo_Mass']
        update_mask = (halo_mass > current_mass) & bound_mask
        if np.any(update_mask):
            indices_to_update = np.array(local_indices)[update_mask]
            particles_df.loc[indices_to_update, 'birth_subtree'] = subtree
            particles_df.loc[indices_to_update, 'Assigned_Halo_Mass'] = halo_mass

    # Drop the temporary column.
    particles_df.drop(columns=['Assigned_Halo_Mass'], inplace=True)
    
    # Clean up the dataset and KDTree to free memory.
    ad.clear_data()
    del ds, ad, particle_tree
    import gc
    gc.collect()

    return particles_df

class AccretionHistory:
    """Constructs the Accretion History of the selected Sub_tree. Follows arxiv:2410.09144.
    
    Recomendation is to use only for z=zmin output.
    """
    def __init__(
        self,
        mergertree,
        file_table,
        pdir
    ):
        """Innit function.
        """
        
        self.mergertree = MergerTree(mergertree)
        self.mergertree.set_equivalence(file_table)
        
        self._prefix = os.path.commonprefix([os.path.basename(file) for file in self.equiv["snapname"].values])
        self._files = [pdir + "/" + file for file in self.equiv["snapname"].values]
        self.ts = yt.DatasetSeries(self._files)

        yt.utilities.logger.ytLogger.setLevel(40)
        self._cosmo = self.ts[0].cosmology
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
    
    def _add_creation_time(self):
        """Adds creation time to dataset.
        """
        if config.code == "ART": 
            pass
        elif config.code == "GEAR":
            self.ds.add_field(
                ("PartType1", "particle_creation_time"),
                function=lambda field, data: self.ds.cosmology.t_from_a(data['PartType1', 'StarFormationTime']),
                sampling_type="local",
                units='Gyr'
            )
    
    def _assign_t_snap(self, particle_creation_times):
        """Assigns the closest snapshot to each of the times
        """
        snapshot_times, snapshot_ids = self.mergertree.PrincipalLeaf["Time"].values, self.mergertree.PrincipalLeaf["Snapshot"].values
        earliest_snapshot = self.equiv["snapshot"].min()
        lower_bound = self.equiv["time"].min()
        
        indices = np.searchsorted(snapshot_times, particle_creation_times)
        indices = np.clip(indices, 0, len(snapshot_times) - 1)
        
        closest_snapshots = snapshot_ids[indices]
        closest_snapshots[particle_creation_times < lower_bound] = earliest_snapshot
        return closest_snapshots
        
    
    def select_accreted_particles(self, subtree, ptype, z=None, t=None, **kwargs):
        """
        Creates Accretion History of the halo identified with subtree, starting from z/t and
        going backwards

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
        fields = [
            "particle_position",
            "particle_position_x", 
            "particle_position_y", 
            "particle_position_z",
            
            "particle_velocity",
            "particle_velocity_x", 
            "particle_velocity_y", 
            "particle_velocity_z",
            
            "particle_index"
            ]
        fields = kwargs.get("fields", fields)
        
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
        halo_params = self.mergertree.get_halo_params(subtree, snapshot=snap)
        self.ds = self.ts[index]
        self._add_creation_time()
        
        sp = self.ds.sphere(halo_params["center"], kwargs.get("rfac", 1) * halo_params["rvir"])
        
        self.creation_times = sp[self._ptype, "particle_creation_time"].to("Gyr")
        self.particle_indexes = sp[self._ptype, "particle_index"].value.astype(int)
        self.snapshot_creation_times = self._assign_t_snap(self.creation_times.value)
        #indexes = np.array([self._snap_to_index(snap) for snap in snapshot_creation_times], dtype=int)
        
        self.snap_particle_dict = {
            snap : self.particle_indexes[self.snapshot_creation_times == snap]
            for snap in np.sort(np.unique(self.snapshot_creation_times))
        }
        self.time_particle_dict = {
            snap : self.creation_times[np.isin(self.particle_indexes, indices)]
            for snap, indices in tqdm(self.snap_particle_dict.items(), total=len(np.unique(self.snap_particle_dict.keys())), desc="Finding birht of stars...")
        }
        return self.snap_particle_dict




    def assign_birth_haloes(self, snap_index_list, snap_list):
        """
        Parallelizes the assignment of birth haloes over snapshots.
        
        Parameters
        ----------
        snap_index_list : dict[int: array[int]]
            Dictionary mapping snapshot number to particle indexes.
        snap_list : array[int]
            List/array of snapshot numbers.
            
        Returns
        -------
        final_particle_birth : pandas.DataFrame
            DataFrame with columns: "particle_index", "birth_time", "birth_subtree".
        """
        from joblib import Parallel, delayed
        from tqdm import tqdm
        from tqdm_joblib import tqdm_joblib
        # Process each snapshot in parallel.
        # The returned object is a dict: {snapshot_number: particles_df}
        with tqdm_joblib(tqdm(desc="Processing Snapshots", total=len(snap_list))):
            results_list = Parallel(n_jobs=15)(
                delayed(assign_birth_snapshot)(
                    snap,
                    self.mergertree,
                    self.snap_particle_dict[snap],
                    self._ptype, 
                    self._files, 
                    self.equiv
                )
                for snap in snap_list
            )
        
        self.results_dict = dict(zip(snap_list, results_list))
        
        self.final_particle_birth = pd.concat(results_list, ignore_index=True)
        return self.final_particle_birth










































































    def assign_birth_haloes_old(self, snap_index_list, snap_list):
        """
        Assigns a birth halo (Halo_ID, uid and Sub_tree_id) to the particles with indexes "indexes_in_snap",
        on their birth snapshots.
        
        Parameters
        ----------
        snap_index_list : dict[ int : array[int] ]
            Array of particle indexes for each snapshot. Must be ordered with snap_list
        snap_list : array[int]
            Birth snapshots.
        
        Returns
        -------
        None.
        """
        from scipy.spatial import KDTree
        
        yt.utilities.logger.ytLogger.setLevel(40)

        final_particle_birth = pd.DataFrame(columns=["particle_index", "birth_time", "birth_subtree"])

        for snap in tqdm(snap_list, total=len(snap_list)):
            particle_indexes = snap_index_list[snap]
            
            index = self._snap_to_index(snap)
            ds = self.ts[index]

            merger_df = self.mergertree.CompleteTree[self.mergertree.CompleteTree["Snapshot"] == snap]
            
            yt.add_particle_filter(
                "born_stars", 
                function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], particle_indexes), 
                filtered_type=self._ptype, 
                requires=["particle_index", "particle_position"]
            )
            ds.add_particle_filter("born_stars")
            ad = ds.all_data()
            
            particles_df = pd.DataFrame({
                'particle_index': ad["born_stars", "particle_index"].astype(int),
                'birth_time': ad["born_stars", "particle_creation_time"].to("Gyr").value,
            })
            particles_df['birth_subtree'] = np.nan
            particles_df['Assigned_Halo_Mass'] = 0
            
            particle_positions = ad["born_stars", "particle_position"].to("kpccm").value
            particle_velocities = ad["born_stars", "particle_velocity"].to("km/s").value  
            
            particle_tree = KDTree(particle_positions)
            
            assert len(merger_df["Redshift"].unique()) == 1, "Seems you have mixed redshifts in you catalogue!"
            redshift = merger_df["Redshift"].values[0]
            
            for _, halo in merger_df.iterrows():
                halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
                halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
                radius = halo['virial_radius']
                halo_mass = halo['mass']
                halo_rvir = halo['virial_radius']
                halo_c = halo['virial_radius'] / halo['scale_radius']
                subtree = halo["Sub_tree_id"]

                local_indices = particle_tree.query_ball_point(halo_center, r=radius)
                if not local_indices:
                    continue  
                
                pos_subset = particle_positions[local_indices]     # in kpccm 
                vel_subset = particle_velocities[local_indices]    # in km/s
                rel_positions = pos_subset - halo_center
                rel_velocities = vel_subset - halo_vel

                distances = np.linalg.norm(rel_positions, axis=1)  # in kpccm
                vel_mags = np.linalg.norm(rel_velocities, axis=1)  # in km/s
                
                
                denom = np.log(1 + halo_c) - halo_c / (1 + halo_c)
                if denom == 0:
                    v_esc = np.ones_like(distances) * -1
                else:
                    phi = _nfw_potential(
                        distances / (1 + redshift),
                        halo_mass,
                        halo['scale_radius'] / (1 + redshift),
                        halo_c,
                        4.3E-6
                        
                        
                    )
                    v_esc = np.sqrt(2 * np.abs(phi))  # in km/s

                # Determine which particles are bound: relative velocity < escape velocity
                bound_mask = vel_mags < v_esc

                # Update particles if this halo is more massive than the one already assigned
                current_mass = particles_df.loc[local_indices, 'Assigned_Halo_Mass']
                update_mask = (halo_mass > current_mass) & bound_mask
                if np.any(update_mask):
                    indices_to_update = np.array(local_indices)[update_mask]
                    particles_df.loc[indices_to_update, 'birth_subtree'] = subtree
                    particles_df.loc[indices_to_update, 'Assigned_Halo_Mass'] = halo_mass
            
            # Optionally, drop the temporary mass column
            particles_df.drop(columns=['Assigned_Halo_Mass'], inplace=True)
            print(particles_df.count())

            final_particle_birth = pd.concat([final_particle_birth, particles_df])
            final_particle_birth.reset_index(drop=True, inplace=True)

            #print(len(final_particle_birth["particle_index"]) , len(final_particle_birth["particle_index"].unique()) )
            #print(len(particles_df["particle_index"]) , len(particles_df["particle_index"].unique()) )

            assert len(final_particle_birth["particle_index"]) == len(final_particle_birth["particle_index"].unique()), f"Seems that a particle has been born twice!"
        
            ad.clear_data()
            del ds, ad, particle_tree
            gc.collect()
        
        return final_particle_birth
    
    
    def create_accretion_history(self, subtree, ptype, z=None, t=None, **kwargs):
        """
        Creates Accretion History of the halo identified with subtree, starting from z/t and
        going backwards

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
        self.select_accreted_particles(subtree, ptype, z=z, t=t, **kwargs)
        
        self.accreted_particles = self.assign_birth_haloes(self.snap_particle_dict, np.unique(self.snapshot_creation_times))
        return self.accreted_particles
        
        
        #trajs = self.ts.particle_trajectories(self.indices, fields=fields, suppress_logging=True)

