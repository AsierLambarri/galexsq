import gc

import yt
import numba
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.sparse import coo_array

from time import time
from collections import defaultdict

from ._helpers import _nfw_potential

def _greedy_assign(rows, cols, dists, n_particles, Dmax):
    """Sparse, RAM‑friendly greedy assignment of particles → halos. Returns a dictionary
    whose keys are halo_idx and values are particle_idx lists.
    """
    mask    = dists < Dmax
    rows_f  = rows[mask]
    cols_f  = cols[mask]
    dists_f = dists[mask]

    if rows_f.size == 0:
        return {}

    # 2) Sort by (row, then distance)
    order       = np.lexsort((dists_f, rows_f))
    sorted_rows = rows_f[order]
    sorted_cols = cols_f[order]

    # 3) Unique rows -> best halos
    unique_rows, first_idx = np.unique(sorted_rows, return_index=True)
    best_halos            = sorted_cols[first_idx]

    unique_halos = np.unique(best_halos)
    halo_to_particles = {
        int(h): unique_rows[best_halos == h].tolist()
        for h in unique_halos
    }

    return halo_to_particles


def assign_particle_positions_bipartite(
        merger_df, 
        particle_indices, 
        particle_positions, 
        particle_velocities, 
        Dmax=np.inf,
        type_list=None,
    ):
    """Checks the position of the particles in one snapshot and finds to which halo they belong

    Parameters
    ----------
    merger_df : TYPE
        Halo merger table. assumes to have only one snapshot in it
    particle_positions : TYPE, optional
        position of the particles.

    particle_tree : TYPE, optional
        scipy.kdtree created with particle positions. 
    halo_kdtree : TYPE, optional
        scipy.kdtree created with halo positions. Not used atm.

    Returns
    -------
    The subtree_id inside which the each particle is.
    """

    assert len(merger_df["Redshift"].unique()) == 1, "Seems you have mixed redshifts in you catalogue!"
    redshift = merger_df["Redshift"].values[0]

    regularization = 1E-6
    particle_tree = KDTree(particle_positions)

    if type_list is None:
        type_list = {
            "particle_index": np.uint64,
            "Time": np.float32,
            "Snapshot": np.uint32,
            "Sub_tree_id": np.int64,
            "Assigned_Halo_Mass": np.float64,  
            "mass_scale": 1,
            "velocity_scale": 1
        }

    n = particle_indices.size
    k = len(merger_df)

    halo_vel_means = []
    halo_cov_invs  = []
    rows = []
    cols = []
    dists = []
    
    t = time()
    for j, (_, halo) in enumerate(merger_df.iterrows()):
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']

        local_indices = np.array(particle_tree.query_ball_point(halo_center, r=halo['virial_radius']))
        if local_indices.size == 0:
            halo_cov_invs.append(np.eye(3))
            halo_vel_means.append(halo_vel)
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
                halo['mass'],
                halo['scale_radius'] / (1 + redshift),
                halo_c,
                4.3E-6,
                0.08
            )
            v_esc = np.sqrt(2 * np.abs(phi))  # in km/s

        bound_mask = vel_mags < v_esc        
        valid_indices = local_indices[bound_mask]

        if valid_indices.size > 0:
            vals = particle_velocities[valid_indices]
            cov = np.cov(vals, rowvar=False) + regularization * np.eye(3)
            cov_inv = np.linalg.inv(cov)
        else:
            cov_inv = np.eye(3)

        halo_cov_invs.append(cov_inv)
        halo_vel_means.append(halo_vel)

        vels = particle_velocities[valid_indices]
        dv   = vels - halo_vel[None, :]
        D2   = np.einsum('ij,ij->i', dv @ cov_inv, dv)

        rows.extend(valid_indices.tolist())
        cols.extend([j] * valid_indices.size)
        dists.extend(D2.tolist())

    del particle_tree
    t2 = time()
    print(f"halo kdtree loop and sparse matrix: {t2 -t}")


    
    t3 = time()
    rows  = np.array(rows, dtype=np.int32)
    cols  = np.array(cols, dtype=np.int32)
    dists = np.array(dists, dtype=np.float64)

    halo_to_particles = _greedy_assign(rows, cols, dists, n, Dmax**2)

    subtree = -np.ones(n, dtype=np.int32)
    for h, plist in halo_to_particles.items():
        subtree[plist] = merger_df['Sub_tree_id'].iat[h].astype(type_list['Sub_tree_id'])
    
    t4 = time()
    print(f"assignment {t4 - t3}")

    
    particles_df = pd.DataFrame({
        'particle_index': particle_indices.astype(type_list["particle_index"]),
        'Time': merger_df["Time"].values[0] * np.ones_like(particle_indices, dtype=type_list["Time"]),
        'Snapshot': merger_df["Snapshot"].values[0] * np.ones_like(particle_indices, dtype=type_list["Snapshot"]),
    
    })
    particles_df['Sub_tree_id'] = subtree
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])

    del rows, cols, dists, halo_to_particles, subtree
    gc.collect()
    return particles_df



    
def assign_particle_positions(
        merger_df, 
        particle_indices, 
        particle_positions, 
        particle_velocities, 
        compute_potential=True, 
        assignment="most-bound",                      
        type_list=None
    ):
    """Checks the position of the particles in one snapshot and finds to which halo they belong

    Parameters
    ----------
    merger_df : TYPE
        Halo merger table. assumes to have only one snapshot in it
    particle_positions : TYPE, optional
        position of the particles.

    particle_tree : TYPE, optional
        scipy.kdtree created with particle positions. 
    halo_kdtree : TYPE, optional
        scipy.kdtree created with halo positions. Not used atm.

    Returns
    -------
    The subtree_id inside which the each particle is.
    """
    from scipy.spatial import KDTree

    assert assignment in ("most-massive", "least-massive", "most-bound"), "assignment must be 'massive', 'lightest' or 'boundest'"
    assert len(merger_df["Redshift"].unique()) == 1, "Seems you have mixed redshifts in you catalogue!"
    redshift = merger_df["Redshift"].values[0]

    particle_tree = KDTree(particle_positions)

    if type_list is None:
        type_list = {
            "particle_index": np.uint64,
            "Time": np.float32,
            "Snapshot": np.uint32,
            "Sub_tree_id": np.int64,
            "Assigned_Halo_Mass": np.float64,  
            "Assigned_Halo_Vel" : np.float64,
            "mass_scale": 1,
            "velocity_scale": 1
        }
    particles_df = pd.DataFrame({
        'particle_index': particle_indices,
        'Time': merger_df["Time"].values[0] * np.ones_like(particle_indices, dtype=int),
        'Snapshot': merger_df["Snapshot"].values[0] * np.ones_like(particle_indices, dtype=int),

    })
    particles_df['Sub_tree_id'] = -1
    
    # Initialize the metric column
    if assignment == "most-massive":
        particles_df["Assigned_Halo_Mass"] = 0.0
    elif assignment == "least-massive":
        particles_df["Assigned_Halo_Mass"] = np.inf
    else: 
        compute_potential = True
        particles_df["Assigned_Halo_Mass"] = -np.inf
    
    #particles_df['Assigned_Halo_Mass'] = 0
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])
    
    if assignment == "most-bound":
        particles_df["Assigned_Halo_Mass"] = particles_df["Assigned_Halo_Mass"].astype(type_list["Assigned_Halo_Vel"])


    mass_scale = type_list["mass_scale"]
    vel_scale =  type_list["velocity_scale"]
    for _, halo in merger_df.iterrows():
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']

        local_indices = particle_tree.query_ball_point(halo_center, r=halo['virial_radius'])
        if not local_indices:
            continue  
        
        if compute_potential:
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
                    halo['mass'],
                    halo['scale_radius'] / (1 + redshift),
                    halo_c,
                    4.3E-6,
                    0.08
                )
                v_esc = np.sqrt(2 * np.abs(phi))  # in km/s
    
            # Determine which particles are bound: relative velocity < escape velocity
            bound_mask = vel_mags < v_esc
        else:
            bound_mask = np.ones_like(local_indices, dtype=bool)

        # compute this halo’s metric for each local particle
        if assignment == "most-massive":
            metric = halo['mass'] / mass_scale
        elif assignment == "least-massive":
            metric = halo['mass'] / mass_scale
        else:  # boundest
            metric = (v_esc**2 - vel_mags**2 ) / vel_scale**2
        metric_value = metric

        # compare & update
        current = particles_df.loc[local_indices, "Assigned_Halo_Mass"]
        if assignment == "most-massive":
            update_mask = (metric > current) & bound_mask
        elif assignment == "least-massive":
            update_mask = (metric < current) & bound_mask
        else:  
            update_mask = (metric > current) & bound_mask
            metric_value = metric[update_mask]

        if np.any(update_mask):
            indices_to_update = np.array(local_indices)[update_mask]
            particles_df.loc[indices_to_update, 'Sub_tree_id'] = halo["Sub_tree_id"]
            particles_df.loc[indices_to_update, 'Assigned_Halo_Mass'] = metric_value


    particles_df.drop(columns=['Assigned_Halo_Mass'], inplace=True)
    
    del particle_tree
    gc.collect()

    return particles_df




    



    
def _assign_halo(
        snap, 
        particle_indexes, 
        mergertree, 
        ptype, 
        file_list, 
        mode="bipartite",
        **kwargs
    ):  
    """Wrapper of assign_particle_positions for parallelization inside AccretionHistory class.
    """
    type_list = kwargs.get("type_list", None)
    verbose = kwargs.get("verbose", False)
    compute_potential = kwargs.get("compute_potential", True)
    assignment = kwargs.get("assignment", "most-bound")
    Dmax = kwargs.get("Dmax", 100000)
    
    st = time()
    fn = int(mergertree.equivalence_table[mergertree.equivalence_table["snapshot"] == snap].index[0])
    merger_df = mergertree.CompleteTree[mergertree.CompleteTree["Snapshot"] == snap]

    ds = yt.load(file_list[fn])
    

    #Add a particle filter that selects particles with the given indexes.
    yt.add_particle_filter(
        "born_stars", 
         function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], particle_indexes), 
         filtered_type=ptype, 
         requires=["particle_index", "particle_position", "particle_velocity"]
    )
    ds.add_particle_filter("born_stars")
    ad = ds.all_data()
    
    particle_indices = ad["born_stars", "particle_index"].astype(int).value
    particle_positions = ad["born_stars", "particle_position"].to("kpccm").value
    particle_velocities = ad["born_stars", "particle_velocity"].to("km/s").value
    ft = time()
    
    del ds, ad
    gc.collect()
    
    st2 = time()
    if mode.lower() == "fast":
        particle_snapshot_df = assign_particle_positions(
            merger_df,
            particle_indices,
            particle_positions,
            particle_velocities,
            compute_potential=compute_potential,
            assignment=assignment,
            type_list=type_list
        )
    elif mode.lower() == "bipartite":
        particle_snapshot_df = assign_particle_positions_bipartite(
            merger_df,
            particle_indices,
            particle_positions,
            particle_velocities,
            Dmax=Dmax,
            type_list=type_list
        )
    else:
        raise Exception(f"Mode follow accretion not valid. Must be 'fast' or 'bipartite'!")
    ft2 = time()
    
    gc.collect()

    if verbose:
        print("")
        print(f"SNAP:  {int(snap)}")
        print("------------------")
        print("")
        print(f"- path: {file_list[fn]}")
        print(f"- N: {int(len(particle_indices))}")
        print(f"- load time: {ft - st}")
        print(f"- process time: {ft2 - st2}")
    return particle_snapshot_df
    
def _assign_halo_wrapper(task):
    return _assign_halo(*task)



