#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 12:01:32 2025

@author: asier
"""
import numpy as np
from collections import Counter
from collections import defaultdict

from .sparse import _build_triplets_3v, _build_triplets_3d3v, _build_triplets_6dv, _build_covinv_3v, _build_covinv_3d3v, _build_covinv_6d, _greedy_assign

def find_groups_only_populated(candidates_list, positions, rvirs):
    # Step 0: Unify types
    positions = np.array(positions)
    rvirs = np.array(rvirs)

    # Step 1: Separate populated vs empty
    populated = np.array([i for i, c in enumerate(candidates_list) if np.any(c)])
    empty     = [i for i, c in enumerate(candidates_list) if not np.any(c)]

    M = populated.size
    if M == 0:
        return [empty] if empty else []

    # Step 2: Extract only populated data
    pos_pop = positions[populated]
    r_pop   = rvirs[populated]

    # Step 3: Compute pairwise distances (upper triangle only)
    diff = pos_pop[:, None, :] - pos_pop[None, :, :]
    dist2 = np.sum(diff ** 2, axis=2)
    r_sum2 = (r_pop[:, None] + r_pop[None, :]) ** 2

    iu, ju = np.triu_indices(M, k=1)
    mask = dist2[iu, ju] <= r_sum2[iu, ju]
    links_i = iu[mask]
    links_j = ju[mask]

    # Step 4: Union-Find
    parent = np.arange(M)
    rank = np.zeros(M, dtype=int)

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def union(u, v):
        ru, rv = find(u), find(v)
        if ru == rv:
            return
        if rank[ru] < rank[rv]:
            parent[ru] = rv
        elif rank[rv] < rank[ru]:
            parent[rv] = ru
        else:
            parent[rv] = ru
            rank[ru] += 1

    for i, j in zip(links_i, links_j):
        union(i, j)

    # Step 5: Collect groups
    groups_dict = defaultdict(list)
    for local_idx in range(M):
        global_idx = populated[local_idx]
        root = find(local_idx)
        groups_dict[root].append(global_idx)

    groups = list(groups_dict.values())

    # Step 6: Add all empty halos as one group
    if empty:
        groups.append(empty)

    return groups

def total_shared_particles(candidate_list):
    """
    Count how many unique particles appear in more than one halo.

    Parameters
    ----------
    candidate_list : list of array‑like[int]
        candidate_list[j] is the list/array of particle IDs for halo j.

    Returns
    -------
    int
        Number of particles that are assigned to two or more halos.
    """
    # Flatten all particle IDs and count occurrences
    counts = Counter()
    for lst in candidate_list:
        counts.update(lst)

    # Count how many particles appear in multiple halos
    return sum(1 for occ in counts.values() if occ > 1)





class BipartiteGraph:
    def __init__(self,
                 subtree_list,
                 candidate_list,
                 halo_means,
                 halo_scalings,
                 particle_indices,
                 particle_positions,
                 particle_velocities,
                 mode,
                 Dmax=1E7,
                 nmin=50,
                 regularization=1e-6):
        """
        candidate_list   : list of arrays of particle‐indices for each halo in this subgraph
        halo_means       : list of 3- or 6-element arrays (per‐halo mean vector)
        halo_scalings    : list of (r_scale, v_scale) tuples, one per halo
        particle_indices : 1D array of the unique particle IDs in this subgraph
        particle_positions, particle_velocities :
                         arrays of shape (n_sub,3) giving pos/vel for those particles
        Dmax, regularization :
                         thresholds for assignment and covariance stability
        """
        self.mode = mode
        
        self.sub_list  = subtree_list
        self.cand_list = candidate_list
        self.means     = halo_means
        self.scales    = halo_scalings
        self.pid       = particle_indices
        self.pos       = particle_positions
        self.vel       = particle_velocities
        self.Dmax      = Dmax
        self.reg       = regularization

        self.nmin = nmin
        self.nh   = len(candidate_list)
        self.np   = particle_positions.shape[0]
        self.shared = total_shared_particles(self.cand_list)
        
        assert self.nh > 0, "YOUR BIPARTITE GRAPH DOESNT HAVE HALOS!"

    def solve(self, maxiter=7, rtol=0.01, atol=100, initial_guess=None):
        """Solves the bipartite graph matching via greedy assignment.
        """
        if self.nh == 1:
            return {self.sub_list[0] : self.pid}
        elif self.np == 0:
            return {}

        if initial_guess is None:
            prelim_guess = self.cand_list
        else:
            prelim_guess = initial_guess


        for it in range(0, maxiter):
            if self.mode in ["3", "3v"]:
                halo_cov_invs = _build_covinv_3v(
                    prelim_guess, 
                    self.scales, 
                    self.pid, 
                    self.vel, 
                    self.reg
                )
                rows, cols, dists = _build_triplets_3v(
                    self.means, 
                    halo_cov_invs, 
                    self.cand_list, 
                    self.scales, 
                    self.sub_list,
                    self.pid,
                    self.vel,
                    self.nmin
                )
                
                
            elif self.mode in ["3d+3v", "3+3"]:
                halo_cov_invs = _build_covinv_3d3v(
                    prelim_guess, 
                    self.scales, 
                    self.pid, 
                    self.pos,
                    self.vel, 
                    self.reg
                )
                rows, cols, dists = _build_triplets_3d3v(
                    self.means, 
                    halo_cov_invs, 
                    self.cand_list, 
                    self.scales, 
                    self.sub_list,
                    self.pid,
                    self.pos,
                    self.vel,
                    self.nmin
                )
                
            
            elif self.mode in ["6d", "6"]:
                halo_cov_invs = _build_covinv_6d(
                    prelim_guess, 
                    self.scales, 
                    self.pid, 
                    self.pos,
                    self.vel, 
                    self.reg
                )
                rows, cols, dists = _build_triplets_6dv(
                    self.means, 
                    halo_cov_invs, 
                    self.cand_list, 
                    self.scales, 
                    self.sub_list,
                    self.pid,
                    self.pos,
                    self.vel,
                    self.nmin
                )
    
            else:
                raise Exception("You didnt provide a valid phase space search mode!")

    
            halo_to_particles = _greedy_assign(np.array(rows), np.array(cols), np.array(dists), self.np, self.Dmax**2)
            new_guess = [np.array(halo_to_particles[h]) if h in halo_to_particles else np.array([]) for h in self.sub_list]

            switched = 0
            atol = min(atol, self.shared + 1)
            total_old = self.shared
            for old, new in zip(prelim_guess, new_guess):                
                switched += len( set(old).symmetric_difference(set(new)))
            
            print(f"-{it}-th iteration (shared={total_old}): {switched}/{total_old} changed ({switched/total_old:.2e})")
            if total_old and ((switched / total_old) <= rtol or switched <= atol):
                print(f"Converged on iteration {it}!")
                break
            else:
                prelim_guess = new_guess
            
        return halo_to_particles        
       






















