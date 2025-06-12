import gc
import numpy as np

from ._numba_helpers import numba_cov, numba_cov_inv, numba_einsum_ijij_to_i


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
    best_halos             = sorted_cols[first_idx]

    unique_halos = np.unique(best_halos)
    halo_to_particles = {
        int(h): unique_rows[best_halos == h].tolist()
        for h in unique_halos
    }
    gc.collect()
    return halo_to_particles
    



def _build_triplets_3v(
        halo_vel_means, 
        halo_cov_invs, 
        candidate_list, 
        halo_scalings, 
        halo_ids,
        particle_indices, 
        particle_velocities
    ):
    """Build sparse matrix (row, col, dist) triplets for Mahalanobis distances.
    """
    rows, cols, dists = [], [], []

    for j, valid in enumerate(candidate_list):
        hoid = halo_ids[j]
        idxs = np.asarray(valid, dtype=int)
        mask = np.isin(particle_indices, idxs)
        if idxs.size == 0:
            continue
            
        _, vs = halo_scalings[j]
        _, mu = halo_vel_means[j]
        
        inv   = np.asarray(halo_cov_invs[j], dtype=float)
        vels  = particle_velocities[mask]              # (M_j, 3)
        dv    = (vels - mu[None, :]) / vs              # (M_j, 3)
        
        # Mahalanobis‑squared distance
        # D2    = np.einsum('ij,ij->i', dv.dot(inv), dv)  # (M_j,)
        D2 = np.einsum("ij,ij->i", dv @ inv, dv)
        
        rows.extend(idxs.tolist())
        cols.extend([hoid] * idxs.size)
        dists.extend(D2.tolist())
        
    return rows, cols, dists 




def _build_triplets_3d3v(
        halo_means, 
        halo_cov_invs, 
        candidate_list, 
        halo_scalings, 
        halo_ids,
        particle_indices, 
        particle_positions, 
        particle_velocities
    ):
    rows, cols, dists = [], [], []

    for j, valid in enumerate(candidate_list):
        hoid = halo_ids[j]
        idxs = np.asarray(valid, dtype=int)
        mask = np.isin(particle_indices, idxs)
        if idxs.size == 0:
            continue

        rs, vs = halo_scalings[j]
        mu_pos, mu_vel = halo_means[j]         # two (3,) arrays
        inv_pos, inv_vel = halo_cov_invs[j]    # two (3,3) arrays

        x = particle_positions[mask] 
        v = particle_velocities[mask] 

        dx = (x - mu_pos[None, :]) / rs
        dv = (v - mu_vel[None, :]) / vs

        D2_pos = np.einsum("ij,ij->i", dx @ inv_pos, dx)
        D2_vel = np.einsum("ij,ij->i", dv @ inv_vel, dv)
        D2 = 1 * (D2_pos + D2_vel)

        rows.extend(idxs.tolist())
        cols.extend([hoid] * idxs.size)
        dists.extend(D2.tolist())

    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)


    

def _build_triplets_6dv(
        halo_means, 
        halo_cov_invs, 
        candidate_list, 
        halo_scalings, 
        halo_ids,
        particle_indices, 
        particle_positions, 
        particle_velocities
    ):
    rows, cols, dists = [], [], []


    for j, valid in enumerate(candidate_list):
        hoid = halo_ids[j]
        idxs = np.asarray(valid, dtype=int)
        mask = np.isin(particle_indices, idxs)
        if idxs.size == 0:
            continue

        rs, vs = halo_scalings[j]
        mu_pos, mu_vel = halo_means[j]         # two (3,) arrays
        inv = halo_cov_invs[j]    

        x = particle_positions[mask] 
        v = particle_velocities[mask] 

        dx = (x - mu_pos[None, :]) / rs
        dv = (v - mu_vel[None, :]) / vs
        dx6 = np.hstack((dx, dv))
    
        D2 = np.einsum("ij,ij->i", dx6 @ inv, dx6)

        rows.extend(idxs.tolist())
        cols.extend([hoid] * idxs.size)
        dists.extend(D2.tolist())

    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)



















def _build_covinv_3v(candidate_list, halo_scalings, particle_indices, particle_velocities, reg):
    """Build inverse covariance matrix for 3v case.
    """
    halo_cov_invs = []
    for j, valid_indices in enumerate(candidate_list):
        _, vel_scale = halo_scalings[j]
        
        mask = np.isin(particle_indices, valid_indices)
        
        if valid_indices.size < 10000 and valid_indices.size > 1:
            cov_inv = np.linalg.inv(
                np.cov(particle_velocities[mask] / vel_scale, rowvar=False) + reg * np.eye(3)
            )
        elif valid_indices.size >= 10000:
            cov_inv = numba_cov_inv(particle_velocities[mask] / vel_scale, reg=reg)
        else:
            cov_inv = np.eye(3)
            
        halo_cov_invs.append(cov_inv)
    return halo_cov_invs

            
def _build_covinv_3d3v(candidate_list, halo_scalings, particle_indices, particle_positions, particle_velocities, reg):
    """Build inverse covariance matrix for 3d+3v case.
    """    
    halo_cov_invs = []
    for j, valid_indices in enumerate(candidate_list):
        pos_scale, vel_scale = halo_scalings[j]
        
        mask = np.isin(particle_indices, valid_indices)

        if valid_indices.size >= 10000:
            cov_inv_x = numba_cov_inv(particle_positions[mask] / pos_scale, reg=reg)
            cov_inv_y = numba_cov_inv(particle_velocities[mask] / vel_scale, reg=reg)
        elif valid_indices.size < 10000 and valid_indices.size > 1:
            cov_inv_x = np.linalg.inv(
                np.cov(particle_positions[mask] / pos_scale, rowvar=False) + reg * np.eye(3)
            )
            cov_inv_y = np.linalg.inv(
                np.cov(particle_velocities[mask] / vel_scale, rowvar=False) + reg * np.eye(3)
            )
        else:
            cov_inv_x = np.eye(3)
            cov_inv_y = np.eye(3)
            
        
            
        halo_cov_invs.append((cov_inv_x, cov_inv_y))
    return halo_cov_invs


def _build_covinv_6d(candidate_list, halo_scalings, particle_indices, particle_positions, particle_velocities, reg):
    """Build inverse covariance matrix for 6d case.
    """
    halo_cov_invs = []
    for j, valid_indices in enumerate(candidate_list):
        pos_scale, vel_scale = halo_scalings[j]
        
        mask = np.isin(particle_indices, valid_indices)

        if valid_indices.size >= 10000:
            x6 = np.hstack((particle_positions[mask] / pos_scale, particle_velocities[mask] / vel_scale))
            cov_inv = numba_cov_inv(x6, reg=reg)
        elif valid_indices.size < 10000 and valid_indices.size > 1:
            x6 = np.hstack((particle_positions[mask] / pos_scale, particle_velocities[mask] / vel_scale))
            cov_inv = np.linalg.inv(
                np.cov(x6, rowvar=False) + reg * np.eye(6)
            )

        else:
            cov_inv = np.eye(6)
            

        halo_cov_invs.append(cov_inv)
    return halo_cov_invs
















