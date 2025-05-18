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
    

def _build_triplets_3v(halo_vel_means, halo_cov_invs, candidate_list, halo_scalings, particle_velocities):
    """Build sparse matrix (row, col, dist) triplets for Mahalanobis distances.
    """
    rows, cols, dists = [], [], []

    for j, valid in enumerate(candidate_list):
        idxs = np.asarray(valid, dtype=int)
        if idxs.size == 0:
            continue
            
        _, vs = halo_scalings[j]
        mu    = np.asarray(halo_vel_means[j], dtype=float)
        inv   = np.asarray(halo_cov_invs[j], dtype=float)
        vels  = particle_velocities[idxs] / vs          # (M_j, 3)
        dv    = vels - mu[None, :]                      # (M_j, 3)
        
        # Mahalanobis‑squared distance
        #D2    = np.einsum('ij,ij->i', dv.dot(inv), dv)  # (M_j,)
        D2 = np.einsum("ij,ij->i", dv @ inv, dv)
        
        rows.extend(idxs.tolist())
        cols.extend([j] * idxs.size)
        dists.extend(D2.tolist())
        
        
    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)




def _build_triplets_3d3v(halo_means, halo_cov_invs, candidate_list, halo_scalings, particle_positions, particle_velocities):
    rows, cols, dists = [], [], []

    for j, valid in enumerate(candidate_list):
        idxs = np.asarray(valid, dtype=int)
        if idxs.size == 0:
            continue

        rs, vs = halo_scalings[j]
        mu_pos, mu_vel = halo_means[j]         # two (3,) arrays
        inv_pos, inv_vel = halo_cov_invs[j]    # two (3,3) arrays

        x = particle_positions[idxs] / rs
        v = particle_velocities[idxs] / vs

        dx = x - mu_pos[None, :]
        dv = v - mu_vel[None, :]

        D2_pos = np.einsum("ij,ij->i", dx @ inv_pos, dx)
        D2_vel = np.einsum("ij,ij->i", dv @ inv_vel, dv)
        D2 = D2_pos + D2_vel

        rows.extend(idxs.tolist())
        cols.extend([j] * idxs.size)
        dists.extend(D2.tolist())

    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)


    

def _build_triplets_6dv(halo_means, halo_covs, candidate_list, particle_positions, particle_velocities):
    rows, cols, dists = [], [], []

    for j, valid in enumerate(candidate_list):
        idxs = np.asarray(valid, dtype=int)
        if idxs.size == 0:
            continue

        mu = np.asarray(halo_means[j], dtype=float)         # shape (6,)
        inv = np.asarray(halo_covs[j], dtype=float)         # shape (6, 6)

        x = particle_positions[idxs]
        v = particle_velocities[idxs]
        x6 = np.hstack((x, v))                              # shape (N, 6)

        dx6 = x6 - mu[None, :]                              # (N, 6)
        D2 = np.einsum("ij,ij->i", dx6 @ inv, dx6)

        rows.extend(idxs.tolist())
        cols.extend([j] * idxs.size)
        dists.extend(D2.tolist())

        gc.collect()
        
    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)





#def _build_triplets_3d3v(halo_pos_means, halo_pos_covs, halo_vel_means, halo_vel_covs, candidate_list, particle_positions, particle_velocities):
#    rows, cols, dists = [], [], []

#    for j, valid in enumerate(candidate_list):
#        idxs = np.asarray(valid, dtype=int)
#        if idxs.size == 0:
#            continue

#        mu_pos = np.asarray(halo_pos_means[j], dtype=float)
#        mu_vel = np.asarray(halo_vel_means[j], dtype=float)
#        inv_pos = np.asarray(halo_pos_covs[j], dtype=float)
#        inv_vel = np.asarray(halo_vel_covs[j], dtype=float)

#        x = particle_positions[idxs]
#        v = particle_velocities[idxs]

#        dx = x - mu_pos[None, :]
#        dv = v - mu_vel[None, :]

#        D2_pos = np.einsum("ij,ij->i", dx @ inv_pos, dx)
#        D2_vel = np.einsum("ij,ij->i", dv @ inv_vel, dv)

#        D2 = D2_pos + D2_vel

#        rows.extend(idxs.tolist())
#        cols.extend([j] * idxs.size)
#        dists.extend(D2.tolist())

#        gc.collect()
        
#    return rows, cols, dists #np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32), np.array(dists, dtype=np.float64)

