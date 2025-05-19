import gc

import yt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from time import time

from .sparse import _build_triplets_3v, _build_triplets_3d3v, _build_triplets_6dv, _build_covinv_3v, _build_covinv_3d3v, _build_covinv_6d, _greedy_assign
from ._helpers import _nfw_potential
from ._numba_helpers import numba_cov_inv
from ..tracking._helpers import best_dtype


def _halo_loop_3v(
    merger_df,
    particle_indices,
    particle_positions,
    particle_velocities
    ):
    """Organizer for halo loop only considering Dij with velocity.
    """
    # n = particle_indices.size
    # k = len(merger_df)
    # reg = 0.1 / (n * k)

    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
    #halo_cov_invs = []
    halo_scalings = []
    candidate_list = []
    
    for j, (_, halo) in enumerate(merger_df.iterrows()):
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']

        pos_scale = 2.16258 * halo['scale_radius']
        vel_scale = halo['vmax']
        
        halo_scalings.append((pos_scale, vel_scale))
        
        local_indices = np.array(particle_tree.query_ball_point(halo_center, r=halo['virial_radius']))
        if local_indices.size == 0:
            candidate_list.append(local_indices)
            halo_means.append(halo_vel)
            #halo_cov_invs.append(np.eye(3))
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
        
        # if valid_indices.size < 10000 and valid_indices.size > 1:
        #     cov_inv = np.linalg.inv(
        #         np.cov(particle_velocities[valid_indices] / vel_scale, rowvar=False) + reg * np.eye(3)
        #     )
        # elif valid_indices.size >= 10000:
        #     cov_inv = numba_cov_inv(particle_velocities[valid_indices] / vel_scale, reg=reg)
        # else:
        #     cov_inv = np.eye(3)


        # halo_cov_invs.append(cov_inv)
        halo_means.append(halo_vel)
    
    del particle_tree
    gc.collect()
    
    return halo_means, halo_scalings, candidate_list
    

def _halo_loop_3d3v(
    merger_df,
    particle_indices,
    particle_positions,
    particle_velocities
    ):
    """Organizer for halo loop only considering Dij with position and velocity separated.
    """
    # n = particle_indices.size
    # k = len(merger_df)
    # reg = 0.1 / (n * k)

    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
    #halo_cov_invs = []
    halo_scalings = []
    candidate_list = []
    
    for j, (_, halo) in enumerate(merger_df.iterrows()):
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']

        pos_scale = 2.16258 * halo['scale_radius']
        vel_scale = halo['vmax']
        
        halo_scalings.append((pos_scale, vel_scale))
        
        local_indices = np.array(particle_tree.query_ball_point(halo_center, r=halo['virial_radius']))
        if local_indices.size == 0:
            candidate_list.append(local_indices)
            halo_means.append((halo_center, halo_vel))
            #halo_cov_invs.append((np.eye(3), np.eye(3)))
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
        
        # if valid_indices.size >= 10000:
        #     cov_inv_x = numba_cov_inv(particle_positions[valid_indices] / pos_scale, reg=reg)
        #     cov_inv_y = numba_cov_inv(particle_velocities[valid_indices] / vel_scale, reg=reg)
        # elif valid_indices.size < 10000 and valid_indices.size > 1:
        #     cov_inv_x = np.linalg.inv(
        #         np.cov(particle_positions[valid_indices] / pos_scale, rowvar=False) + reg * np.eye(3)
        #     )
        #     cov_inv_y = np.linalg.inv(
        #         np.cov(particle_velocities[valid_indices] / vel_scale, rowvar=False) + reg * np.eye(3)
        #     )
        # else:
        #     cov_inv_x = np.eye(3)
        #     cov_inv_y = np.eye(3)


        # halo_cov_invs.append((cov_inv_x, cov_inv_y))
        halo_means.append((halo_center, halo_vel))
    
    del particle_tree
    gc.collect()
    
    return halo_means, halo_scalings, candidate_list


def _halo_loop_6d(
    merger_df,
    particle_indices,
    particle_positions,
    particle_velocities
    ):
    """Organizer for halo loop only considering Dij with 6d phase space.
    """
    # n = particle_indices.size
    # k = len(merger_df)
    # reg = 0.1 / (n * k)

    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
    #halo_cov_invs = []
    halo_scalings = []
    candidate_list = []
    
    for j, (_, halo) in enumerate(merger_df.iterrows()):
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c = halo['virial_radius'] / halo['scale_radius']

        pos_scale = 2.16258 * halo['scale_radius']
        vel_scale = halo['vmax']
        
        halo_scalings.append((pos_scale, vel_scale))
        
        local_indices = np.array(particle_tree.query_ball_point(halo_center, r=halo['virial_radius']))
        if local_indices.size == 0:
            candidate_list.append(local_indices)
            halo_means.append((halo_center, halo_vel))
            # halo_cov_invs.append((np.eye(3), np.eye(3)))
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
        
        # if valid_indices.size > 1:
        #     speeds = particle_velocities[valid_indices]
        #     cov_inv = numba_cov_inv(speeds, reg=reg)
        # else:
        #     cov_inv = np.eye(3)


        # halo_cov_invs.append(cov_inv)
        halo_means.append((halo_center, halo_vel))
    
    del particle_tree
    gc.collect()
    
    return halo_means, halo_scalings, candidate_list
    














    



def assign_particle_positions_bipartite(
    merger_df, 
    particle_indices, 
    particle_positions, 
    particle_velocities, 
    Dmax=np.inf,
    space="3v",
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
    time_stats = {}
    assert len(merger_df["Redshift"].unique()) == 1, "Seems you have mixed redshifts in you catalogue!"
    redshift = merger_df["Redshift"].values[0]

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
    
    regularization = 0.1 / (k * n)

    t1 = time()
    if space in ["3", "3v"]:
        halo_means, halo_scalings, candidate_list = _halo_loop_3v(
            merger_df, 
            particle_indices, 
            particle_positions, 
            particle_velocities
        )
        halo_cov_invs = _build_covinv_3v(candidate_list, halo_scalings, particle_velocities, regularization)
        t2 = time()
        
        rows, cols, dists = _build_triplets_3v(halo_means, halo_cov_invs, candidate_list, halo_scalings, particle_velocities)
        
    elif space in ["3d+3v", "3+3"]:
        halo_means, halo_scalings, candidate_list = _halo_loop_3d3v(
            merger_df, 
            particle_indices, 
            particle_positions, 
            particle_velocities
        )
        
        halo_cov_invs = _build_covinv_3d3v(candidate_list, halo_scalings, particle_positions, particle_velocities, regularization)
        t2 = time()
        
        rows, cols, dists = _build_triplets_3d3v(halo_means, halo_cov_invs, candidate_list, halo_scalings, particle_positions, particle_velocities)

    elif space in ["6d", "6"]:
        halo_means, halo_scalings, candidate_list = _halo_loop_6d(
            merger_df, 
            particle_indices, 
            particle_positions, 
            particle_velocities
        )
        halo_cov_invs = _build_covinv_6d(candidate_list, halo_scalings, particle_positions, particle_velocities, regularization)

        t2 = time()
    
        rows, cols, dists = _build_triplets_6dv(halo_means, halo_cov_invs, candidate_list, halo_scalings, particle_positions, particle_velocities)

    else:
        raise Exception("You didnt provide a valid phase space search mode!")

    
    rows  = np.array(rows, dtype=best_dtype(rows))
    cols  = np.array(cols, dtype=best_dtype(cols))
    dists = np.array(dists, dtype=best_dtype(dists, eps=1E-5))

    t3 = time()
    
    print(rows[:10], cols[:10], dists[:10])
    
    halo_to_particles = _greedy_assign(rows, cols, dists, n, Dmax**2)
    del rows, cols, dists
    
    keys = list(halo_to_particles.keys())
    
    print(keys[0], halo_to_particles[keys[0]][:10])
    print(keys[1], halo_to_particles[keys[1]][:10])
    
    t4 = time()
    
    subtree = -np.ones(n, dtype=np.int32)
    for h, plist in halo_to_particles.items():
        subtree[plist] = merger_df['Sub_tree_id'].iat[h].astype(type_list['Sub_tree_id'])
            
    t5 = time()

    particles_df = pd.DataFrame({
        'particle_index': particle_indices.astype(type_list["particle_index"]),
        'Time': merger_df["Time"].values[0] * np.ones_like(particle_indices, dtype=type_list["Time"]),
        'Snapshot': merger_df["Snapshot"].values[0] * np.ones_like(particle_indices, dtype=type_list["Snapshot"]),
    
    })
    particles_df['Sub_tree_id'] = subtree
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])

    del halo_to_particles, subtree
    gc.collect()
    
    t6 = time()
    
    time_stats["kdtree & halo loop"] = t2 - t1
    time_stats["sparse triplets"] = t3 - t2
    time_stats["greedy assignment"] = t4 - t3
    time_stats["subtree unpacking"] = t5 - t4
    time_stats["rest"] = t6 - t5
    
    return particles_df, time_stats



    
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
    time_stats = {}
    assert assignment in ("most-massive", "least-massive", "most-bound"), "assignment must be 'massive', 'lightest' or 'boundest'"
    assert len(merger_df["Redshift"].unique()) == 1, "Seems you have mixed redshifts in you catalogue!"
    redshift = merger_df["Redshift"].values[0]

    t1 = time()
    
    particle_tree = KDTree(particle_positions)
    
    t2 = time()
    
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
    
    if assignment == "most-massive":
        particles_df["Assigned_Halo_Mass"] = 0.0
    elif assignment == "least-massive":
        particles_df["Assigned_Halo_Mass"] = np.inf
    else: 
        compute_potential = True
        particles_df["Assigned_Halo_Mass"] = -np.inf
    
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])
    
    if assignment == "most-bound":
        particles_df["Assigned_Halo_Mass"] = particles_df["Assigned_Halo_Mass"].astype(type_list["Assigned_Halo_Vel"])

    t3 = time()
    
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
    
            bound_mask = vel_mags < v_esc
        else:
            bound_mask = np.ones_like(local_indices, dtype=bool)

        if assignment == "most-massive":
            metric = halo['mass'] / mass_scale
        elif assignment == "least-massive":
            metric = halo['mass'] / mass_scale
        else:  
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

    t4 = time()
    particles_df.drop(columns=['Assigned_Halo_Mass'], inplace=True)
    
    del particle_tree
    gc.collect()

    time_stats["kdtree build"] = t2 - t1
    time_stats["halo loop"] = t4 - t3
    time_stats["rest"] = t3 - t2

    return particles_df, time_stats




    

def _assign_halo(
        snap, 
        particle_indexes, 
        mergertree, 
        ptype, 
        file_list, 
        mode,
        **kwargs
    ):  
    """Wrapper of assign_particle_positions for parallelization inside AccretionHistory class.
    """
    type_list = kwargs.get("type_list", None)
    verbose = kwargs.get("verbose", False)
    compute_potential = kwargs.get("compute_potential", True)
    assignment = kwargs.get("assignment_mode", "most-bound")
    Dmax = kwargs.get("Dmax", 100000)
    space = kwargs.get("space", "3v")
    
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
    N = int(len(particle_indices))
    ft = time()
    
    del ds, ad
    gc.collect()
    
    st2 = time()
    if mode.lower() == "fast":
        particle_snapshot_df, time_stats = assign_particle_positions(
            merger_df,
            particle_indices,
            particle_positions,
            particle_velocities,
            compute_potential=compute_potential,
            assignment=assignment,
            type_list=type_list
        )
    elif mode.lower() == "bipartite":
        particle_snapshot_df, time_stats = assign_particle_positions_bipartite(
            merger_df,
            particle_indices,
            particle_positions,
            particle_velocities,
            Dmax=Dmax,
            space=space,
            type_list=type_list
        )
    else:
        raise Exception(f"Mode follow accretion not valid. Must be 'fast' or 'bipartite'!")
    ft2 = time()

    del particle_indices, particle_positions, particle_velocities
    gc.collect()

    if verbose:
        print("")
        print(f"SNAPSHOT:  {int(snap)}")
        print("------------------")
        print(f"-mode: {mode} - {assignment}")
        print(f"-path: {file_list[fn]}")
        print(f"-N: {N}")
        print(f"-load time: {(ft - st):.4f}s")
        print(f"-Processing Time: {np.array(list(time_stats.values())).sum():.4f}s")
        print(f"----------------")
        for key, val in time_stats.items():
            print(f"  -{key}: {val:.4f}s")

            
    return particle_snapshot_df
    
def _assign_halo_wrapper(task):
    args, kwargs = task
    return _assign_halo(*args, **kwargs)



