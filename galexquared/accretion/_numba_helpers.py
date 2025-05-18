from numba import jit, prange
import numpy as np


@jit(nopython=True, parallel=True)
def _numba_cov_core(X):
    """
    Core covariance: X shaped (n_obs, n_feat), rowvar=False.
    """
    n_obs, n_feat = X.shape
    # compute means
    means = np.empty(n_feat)
    for j in prange(n_feat):
        s = 0.0
        for i in range(n_obs):
            s += X[i, j]
        means[j] = s / n_obs
    # allocate covariance matrix
    cov = np.zeros((n_feat, n_feat))
    # compute covariance
    for i in prange(n_feat):
        for j in range(i, n_feat):
            s = 0.0
            for k in range(n_obs):
                s += (X[k, i] - means[i]) * (X[k, j] - means[j])
            val = s / (n_obs - 1)
            cov[i, j] = val
            cov[j, i] = val
    return cov

@jit(nopython=True)
def numba_cov(X, rowvar=False):
    """
    Compute covariance matrix of data X in parallel.
    If rowvar is True, each row represents a variable, observations are columns.
    Equivalent to np.cov(X, rowvar=rowvar, bias=False).
    """
    if rowvar:
        return _numba_cov_core(X.T)
    else:
        return _numba_cov_core(X)

@jit(nopython=True)
def numba_cov_inv(X, rowvar=False):
    """
    Compute the inverse of the covariance matrix of X.
    Returns the inverse covariance matrix.
    """
    cov = numba_cov(X, rowvar)
    return np.linalg.inv(cov)



@jit(nopython=True, parallel=True)
def numba_einsum_ijij_to_i(A, B):
    """
    Optimized for 'ij,ij->i' operations: computes a 1D array
    where out[i] = sum_j A[i, j] * B[i, j].
    Both A and B must have the same shape (n, m).
    """
    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        raise ValueError("Input arrays must have the same shape for 'ij,ij->i'.")
    n, m = A.shape
    out = np.zeros(n)
    for i in prange(n):
        s = 0.0
        for j in range(m):
            s += A[i, j] * B[i, j]
        out[i] = s
    return out