import numyp as np
from gala.potentials.scf import compute_coeffs_discrete, SCFPotential
import multiprocessing as mp


def compute_and_filter_scf(
            xyz,
            mass,
            nmax,
            lmax,
            r_s,
            skip_odd=False,
            skip_even=False,
            skip_m=False,
            compute_var=False,
            threshold=3,
            pool=None,
            units=None,
            **kwargs
    ):
    """
    Compute SCF coefficients (S, T) and covariance matrix, then
    zero out all modes with S/N <= threshold (using only S to
    compute the mask), applying the same mask to T.

    Parameters
    ----------
    xyz : array-like, shape (N,3)
        Particle positions.
    mass : array-like, shape (N,)
        Particle masses.
    nmax : int
        Maximum radial order.
    lmax : int
        Maximum angular order.
    r_s : float
        SCF scale radius.
    threshold : float
        Minimum required |S|/sigma_S for a mode to be kept.
    compute_var : bool, optional
        If True, returns covariance matrix from compute_coeffs_discrete().
    **kwargs : dict, optional
        Additional keyword args passed to compute_coeffs_discrete,
        e.g. `pool`, `compute_var=False` etc.

    Returns
    -------
    S_filtered : ndarray, shape (nmax+1, lmax+1, lmax+1)
    T_filtered : ndarray, shape (nmax+1, lmax+1, lmax+1)
    Cov        : ndarray, shape (M, M)
        The full covariance matrix for (S,T) if compute_var is True,
        else None.
    """
    if compute_var:
        S, T, Cov = compute_coeffs_discrete(
            xyz, mass, nmax=nmax, lmax=lmax, r_s=r_s,
            compute_var=True, pool=pool, skip_odd=False, 
            skip_even=False, skip_m=False, **kwargs
        )
    else:
        S, T = compute_coeffs_discrete(
            xyz, mass, nmax=nmax, lmax=lmax, r_s=r_s,
            compute_var=False, pool=pool, skip_odd=False, 
            skip_even=False, skip_m=False, **kwargs
        )
        Cov = None

    if Cov is not None:
        snr = np.sqrt(S**2 / (Cov[0, 0] + 1E-5))
    else:
        snr = 3 * threshold * np.ones_like(S)

    mask = snr > threshold

    S_filtered = np.where(mask, S, 0.0)
    T_filtered = np.where(mask, T, 0.0)

    return SCFPotential(Snlm=S_filtered, Tnlm=T_filtered, m=mass, r_s=r_s, units=units)





