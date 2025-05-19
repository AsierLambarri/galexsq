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


















