import numpy as np
from numba import njit, types
from numba.typed import List as NumbaList, Dict as NumbaDict

@njit
def _find(parent, u):
    # path compression
    while parent[u] != u:
        parent[u] = parent[parent[u]]
        u = parent[u]
    return u

@njit
def _union(parent, rank, u, v):
    ru = _find(parent, u)
    rv = _find(parent, v)
    if ru == rv:
        return
    if rank[ru] < rank[rv]:
        parent[ru] = rv
    elif rank[ru] > rank[rv]:
        parent[rv] = ru
    else:
        parent[rv] = ru
        rank[ru] += 1

@njit
def split_bipartite_subgraphs_numba(candidate_list, n_halos):
    """
    Numba‐JIT’d splitting of a bipartite graph into halo‐only connected components.
    candidate_list is a Numba List of 1D int32 arrays of particle indices per halo.
    """
    # initialize union-find
    parent = np.arange(n_halos, dtype=np.int32)
    rank   = np.zeros(n_halos,   dtype=np.int32)

    # map particle -> first halo seen
    # use a Numba Dict (particle_id -> halo_id)
    p2h = NumbaDict.empty(key_type=types.int64, value_type=types.int32)

    # for each halo j, union with any previous halo that shared a particle
    for j in range(n_halos):
        idxs = candidate_list[j]
        for k in range(len(idxs)):
            p = idxs[k]
            if p2h.get(p, -1) == -1:
                p2h[p] = j
            else:
                _union(parent, rank, j, p2h[p])

    # collect halos into components
    # key: root halo, value: NumbaList of members
    comp = NumbaDict.empty(key_type=types.int32, value_type=types.ListType(types.int32[:]))

    for j in range(n_halos):
        r = _find(parent, j)
        if comp.get(r, None) is None:
            lst = NumbaList.empty_list(types.int32[:])
            lst.append(j)
            comp[r] = lst
        else:
            comp[r].append(j)

    # extract as a Numba List of Lists
    result = NumbaList.empty_list(types.int32[:])
    for key in comp:
        result.append(comp[key])

    return result










def _halo_loop_3v(
    merger_df,
    particle_indices,
    particle_positions,
    particle_velocities
    ):
    """Organizer for halo loop only considering Dij with velocity.
    """
    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
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

        bound_mask = vel_mags < v_esc        
        valid_indices = local_indices[bound_mask]
        
        candidate_list.append(valid_indices)
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

    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
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

        bound_mask = vel_mags < v_esc        
        valid_indices = local_indices[bound_mask]
        
        candidate_list.append(valid_indices)
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

    redshift = merger_df["Redshift"].values[0]
    particle_tree = KDTree(particle_positions)
    
    halo_means = []
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
           
        bound_mask = vel_mags < v_esc        
        valid_indices = local_indices[bound_mask]
        
        candidate_list.append(valid_indices)
        halo_means.append((halo_center, halo_vel))
    
    del particle_tree
    gc.collect()
    
    return halo_means, halo_scalings, candidate_list







