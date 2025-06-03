import gc
import warnings

import yt
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from time import time

from .sparse import _build_triplets_3v, _build_triplets_3d3v, _build_triplets_6dv, _build_covinv_3v, _build_covinv_3d3v, _build_covinv_6d, _greedy_assign
from ._helpers import potential, custom_load
from ..tracking._helpers import best_dtype


def _halo_loop_3v(
    merger_df,
    particle_indices,
    particle_positions,
    particle_velocities,
    pot_mode="kepler"
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
            phi = potential(
                distances / (1 + redshift),
                mode=pot_mode,
                mass=halo['mass'],
                rs=halo['scale_radius'] / (1 + redshift),
                c=halo_c,
                G=4.3E-6,
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
    particle_velocities,
    pot_mode="kepler"
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
            phi = potential(
                distances / (1 + redshift),
                mode=pot_mode,
                mass=halo['mass'],
                rs=halo['scale_radius'] / (1 + redshift),
                c=halo_c,
                G=4.3E-6,
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
    particle_velocities,
    pot_mode="kepler"
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
            phi = potential(
                distances / (1 + redshift),
                mode=pot_mode,
                mass=halo['mass'],
                rs=halo['scale_radius'] / (1 + redshift),
                c=halo_c,
                G=4.3E-6,
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
    dz_max = np.abs( merger_df["Redshift"].diff() ).max()
    if  dz_max > 1E-3: warnings.warn(f"Seems you have mixed redshifts in you catalogue! Have care... the maxmimum error is {dz_max:.3e}")

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




def _halo_loop(
    merger_df, 
    particle_indices, 
    particle_positions, 
    particle_velocities, 
    type_list,
    newborn,
    pot_mode="kepler"
    ):
    """Assigns particle positions!
    """
    time_stats = {}
    dz_max = np.abs( merger_df["Redshift"].diff() ).max()
    if  dz_max > 1E-3: warnings.warn(f"Seems you have mixed redshifts in you catalogue! Have care... the maxmimum error is {dz_max:.3e}")
        
    t1 = time()
    redshift      = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    n             = particle_indices.size
    t2 = time()
    

    particles_df = pd.DataFrame({
        'particle_index': particle_indices,
        'Time': merger_df["Time"].values[0] * np.ones_like(particle_indices, dtype=int),
        'Snapshot': merger_df["Snapshot"].values[0] * np.ones_like(particle_indices, dtype=int),

    })
    particles_df['Sub_tree_id'] = -1
    for col in particles_df.columns:
        particles_df[col] = particles_df[col].astype(type_list[col])
        
    if newborn:
        best_rnorm    = np.full(n, np.inf,   dtype=float)
        subtree_rnorm = -np.ones(n, dtype=int)
        best_bind     = np.full(n, -np.inf,  dtype=float)
        subtree_bind  = -np.ones(n, dtype=int)
    else:
        particles_df["Assigned_Halo_Mass"] = -np.inf    
        particles_df["Assigned_Halo_Mass"] = particles_df["Assigned_Halo_Mass"].astype(type_list["Assigned_Halo_Vel"])

    t3 = time()



    
    vel_scale  = type_list["velocity_scale"]
    for _, halo in merger_df.iterrows():
        halo_center = np.array([halo['position_x'], halo['position_y'], halo['position_z']]) 
        halo_vel    = np.array([halo['velocity_x'], halo['velocity_y'], halo['velocity_z']]) 
        halo_c      = halo['virial_radius'] / halo['scale_radius']
        halo_id = int(halo["Sub_tree_id"])
        
        local_indices = particle_tree.query_ball_point(halo_center, r=halo['virial_radius'])
        if not local_indices:
            continue  
        
        local_indices = np.array(local_indices, int)
        pos_subset     = particle_positions[local_indices]     # in kpccm 
        vel_subset     = particle_velocities[local_indices]    # in km/s
        rel_positions  = pos_subset - halo_center
        rel_velocities = vel_subset - halo_vel

        distances      = np.linalg.norm(rel_positions, axis=1)  # in kpccm
        vel_mags       = np.linalg.norm(rel_velocities, axis=1)  # in km/s
        
        
        denom = np.log(1 + halo_c) - halo_c / (1 + halo_c)
        if denom == 0:
            v_esc = np.ones_like(distances) * -1
        else:
            if np.isnan(halo["uid"]):
                phi = potential(
                    distances / (1 + redshift),
                    mode="kepler",
                    mass=halo['mass'],
                    rs=halo['scale_radius'] / (1 + redshift),
                    c=halo_c,
                    G=4.3E-6,
                )
            else:
                phi = potential(
                    distances / (1 + redshift),
                    mode="nfw",
                    mass=halo['mass'],
                    rs=halo['scale_radius'] / (1 + redshift),
                    c=halo_c,
                    G=4.3E-6,
                )
            v_esc = np.sqrt(2 * np.abs(phi))  # in km/s

        bound_mask  = vel_mags < v_esc
        metric      = (v_esc**2 - vel_mags**2 ) / vel_scale**2
        
        if newborn:
            rnorm = distances / max(halo['scale_radius'], 0.1 * halo['virial_radius'])   ########### aqui igual hay que cortarse un poco con 2 * rs
            bind_score = metric
            
            inside_mask = bound_mask & (rnorm <= 1.0)
            if inside_mask.any():
                idx_inside = local_indices[inside_mask]
                rn_inside  = rnorm[inside_mask]
                better_in  = rn_inside < best_rnorm[idx_inside]
                if better_in.any():
                    gi = idx_inside[better_in]
                    best_rnorm[gi]    = rn_inside[better_in]
                    subtree_rnorm[gi] = halo_id

            # outside scale radius
            outside_mask = bound_mask & (rnorm > 1.0)
            if outside_mask.any():
                idx_out   = local_indices[outside_mask]
                bs_out    = bind_score[outside_mask]
                better_out= bs_out > best_bind[idx_out]
                if better_out.any():
                    go = idx_out[better_out]
                    best_bind[go]    = bs_out[better_out]
                    subtree_bind[go] = halo_id
                    
                    
        else:
            current     = particles_df.loc[local_indices, "Assigned_Halo_Mass"]
            update_mask = (metric > current) & bound_mask
            if np.any(update_mask):
                indices_to_update = np.array(local_indices)[update_mask]
                particles_df.loc[indices_to_update, 'Sub_tree_id'] = halo["Sub_tree_id"]
                particles_df.loc[indices_to_update, 'Assigned_Halo_Mass'] = metric[update_mask].astype(particles_df['Assigned_Halo_Mass'].dtype)


    t4 = time()
    if newborn:
        use_in  = best_rnorm  < np.inf
        use_out = (~use_in) & (best_bind > -np.inf)
        particles_df.loc[use_in,  "Sub_tree_id"] = subtree_rnorm[use_in].astype(particles_df['Sub_tree_id'].dtype) #subtree_rnorm[use_in]
        particles_df.loc[use_out, "Sub_tree_id"] = subtree_bind[use_out].astype(particles_df['Sub_tree_id'].dtype) #subtree_bind[use_out]
    else:
        particles_df.drop(columns=['Assigned_Halo_Mass'], inplace=True)
    
    del particle_tree
    gc.collect()

    time_stats["kdtree build"] = t2 - t1
    time_stats["halo loop"] = t4 - t3
    time_stats["rest"] = t3 - t2
    
    return particles_df, time_stats

    
def assign_particle_positions(
    merger_df, 
    newborn_indices,
    particle_indices, 
    particle_positions, 
    particle_velocities, 
    type_list=None,
    newborn=False
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
    dz_max = np.abs( merger_df["Redshift"].diff() ).max()
    if  dz_max > 1E-3: warnings.warn(f"Seems you have mixed redshifts in you catalogue! Have care... the maxmimum error is {dz_max:.3e}")
    
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
    mask = np.isin(particle_indices, newborn_indices)
    

    if (~mask).any():
        existing_df, existing_stats = _halo_loop(
            merger_df, 
            particle_indices[~mask], 
            particle_positions[~mask], 
            particle_velocities[~mask], 
            type_list, 
            False
        ) 
    else:
        existing_stats = {}
        existing_df = pd.DataFrame(columns=["particle_index","Time","Snapshot","Sub_tree_id"])
        
        existing_stats["kdtree build"] = 0
        existing_stats["halo loop"] = 0
        existing_stats["rest"] = 0
        
    if mask.any():
        newborn_df, newborn_stats = _halo_loop(
            merger_df, 
            particle_indices[mask], 
            particle_positions[mask], 
            particle_velocities[mask], 
            type_list, 
            True
        )
    else:
        newborn_stats = {}
        newborn_df = pd.DataFrame(columns=["particle_index","Time","Snapshot","Sub_tree_id"])
        
        newborn_stats["kdtree build"] = 0
        newborn_stats["halo loop"] = 0
        newborn_stats["rest"] = 0


    particles_df = pd.concat(
        [existing_df, newborn_df],
        ignore_index=True,
        join='inner'
    ).reset_index(drop=True)
    for col, dt in zip(["particle_index","Time","Snapshot","Sub_tree_id"], [int, float, int, int]):
        particles_df[col] = particles_df[col].astype(dt)
        
    for key, val in newborn_stats.items():
        time_stats[key] = val + existing_stats[key]
        
    return particles_df, time_stats



def _assign_halo(
        snap, 
        newborn_indexes,
        particle_indexes, 
        mergertree, 
        ptype, 
        fields,
        file_list, 
        mode,
        **kwargs
    ):  
    """Wrapper of assign_particle_positions for parallelization inside AccretionHistory class.
    """
    type_list = kwargs.get("type_list", None)
    verbose = kwargs.get("verbose", False)
    Dmax = kwargs.get("Dmax", 100000)
    space = kwargs.get("space", "3+3")
    
    st = time()
    fn = int(mergertree.equivalence_table[mergertree.equivalence_table["snapshot"] == snap].index[0])
    merger_df = mergertree.CompleteTree[mergertree.CompleteTree["Snapshot"] == snap]

    ds = custom_load(file_list[fn], ptype)
    
    #Add a particle filter that selects particles with the given indexes.
    yt.add_particle_filter(
        "present_particles", 
         function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, fields["index"]], particle_indexes), 
         filtered_type=ptype, 
         requires=[fields["index"], fields["position"], fields["velocity"]]
    )
    ds.add_particle_filter("present_particles")

    
    ad = ds.all_data()
    
    particle_indices = ad["present_particles", fields["index"]].value.astype(int)
    particle_positions = ad["present_particles", fields["position"]].to("kpccm").value
    particle_velocities = ad["present_particles", fields["velocity"]].to("km/s").value
    N = int(len(particle_indices))
    ft = time()
    del ds, ad
    gc.collect()
    
    if mode.lower() == "fast":
        particle_snapshot_df, time_stats = assign_particle_positions(
            merger_df,
            newborn_indexes,
            particle_indices,
            particle_positions,
            particle_velocities,
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
        raise Exception("Mode follow accretion not valid. Must be 'fast' or 'bipartite'!")

    del particle_indices, particle_positions, particle_velocities
    gc.collect()

    if verbose:
        print("")
        print(f"SNAPSHOT:  {int(snap)}")
        print("------------------")
        print(f"-mode: {mode}")
        print(f"-path: {file_list[fn]}")
        print(f"-N: {N}")
        print(f"-load time: {(ft - st):.4f}s")
        print(f"-Processing Time: {np.array(list(time_stats.values())).sum():.4f}s")
        print("----------------")
        for key, val in time_stats.items():
            print(f"  -{key}: {val:.4f}s")

            
    return particle_snapshot_df
    
def _assign_halo_wrapper(task):
    args, kwargs = task
    return _assign_halo(*args, **kwargs)



