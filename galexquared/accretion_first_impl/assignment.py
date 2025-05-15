import gc

import yt
import numba
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from time import time

from ._helpers import _nfw_potential



@numba.njit(parallel=False)
def _greedy_assign(particle_ix, halo_ix, dists, N):
    order = np.argsort(dists)
    assignment = -1 * np.ones(N, np.int32)
    seen = np.zeros(N, np.uint8)
    for k in range(order.shape[0]):
        idx = order[k]
        pi = particle_ix[idx]
        if seen[pi] == 0:
            assignment[pi] = halo_ix[idx]
            seen[pi] = 1
    return assignment




def assign_particle_positions(
        merger_df, 
        particle_indices, 
        particle_positions, 
        particle_velocities, 
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
    halo_vel_means  = []
    halo_cov_invs   = []
    candidate_list  = []
    t = time()
    for _, halo in merger_df.iterrows():
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']


        local_indices = np.array(particle_tree.query_ball_point(halo_center, r=halo['virial_radius']))
        if local_indices.size == 0:
            halo_cov_invs.append(np.eye(3))
            candidate_list.append(local_indices)
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
        candidate_list.append(valid_indices)

        if valid_indices.size > 0:
            vals = particle_velocities[valid_indices]
            cov = np.cov(vals, rowvar=False) + regularization * np.eye(3)
            cov_inv = np.linalg.inv(cov)
        else:
            cov_inv = np.eye(3)

        halo_cov_invs.append(cov_inv)
        halo_vel_means.append(halo_vel)

    del particle_tree

    t2 = time()
    print(f"halo kdtree loop: {t2 -t}")

    rows = []
    cols = []
    dists = []

    t3 = time()
    for j, valid in enumerate(candidate_list):
        mu   = halo_vel_means[j]    
        inv  = halo_cov_invs[j]     
        idxs = np.array(valid, dtype=int)
        if idxs.size == 0:
            continue
    
        vels = particle_velocities[idxs]       
        dv   = vels - mu[None,:]               
    
        D2 = np.einsum('ij,ij->i', dv.dot(inv), dv)  
    
        rows.extend(idxs)       
        cols.extend([j]*idxs.size)        
        dists.extend(D2) 

    t4 = time()
    print(f"Create sparse distance matrix {t4 - t3}")
    t5 = time()
    if rows:
        part_ix = np.array(rows, np.int32)
        halo_ix = np.array(cols, np.int32)
        dist_a  = np.array(dists, np.float64)
        assign  = _greedy_assign(part_ix, halo_ix, dist_a, n)
    else:
        assign  = -np.ones(n, np.int32)

    subtree = -np.ones(n, np.int32)
    for i in range(n):
        h = assign[i]
        if h >= 0:
            subtree[i] = merger_df["Sub_tree_id"].iat[h].astype(type_list["Sub_tree_id"])
          
            
    t6 = time()
    print(f"Assignment {t6 - t5}")

            
    particles_df = pd.DataFrame({
        'particle_index': particle_indices,
        'Time': merger_df["Time"].values[0] * np.ones_like(particle_indices, dtype=int),
        'Snapshot': merger_df["Snapshot"].values[0] * np.ones_like(particle_indices, dtype=int),
    
    })
    
    particles_df['Sub_tree_id'] = subtree
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])
        
    gc.collect()
    return particles_df







    
def _assign_halo(
        snap, 
        particle_indexes, 
        mergertree, 
        ptype, 
        file_list, 
        verbose, 
        type_list
    ):  
    """Wrapper of assign_particle_positions for parallelization inside AccretionHistory class.
    """
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
    particle_snapshot_df = assign_particle_positions(
        merger_df,
        particle_indices,
        particle_positions,
        particle_velocities,
        type_list=type_list
    )
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



