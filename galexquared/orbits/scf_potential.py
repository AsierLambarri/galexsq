import numyp as np
from gala.potentials.scf import compute_coeffs_discrete, SCFPotential
import multiprocessing as mp


def compute_and_filter_scf(xyz, mass, nmax, lmax, r_s, threshold,
                           compute_var=True, **kwargs):
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
    # 1) Compute coefficients and covariance
    if compute_var:
        S, T, Cov = compute_coeffs_discrete(
            xyz, mass, nmax=nmax, lmax=lmax, r_s=r_s,
            compute_var=True, **kwargs
        )
    else:
        S, T = compute_coeffs_discrete(
            xyz, mass, nmax=nmax, lmax=lmax, r_s=r_s,
            compute_var=False, **kwargs
        )
        Cov = None

    # 2) Extract variance of S-coeffs: Cov[0:S.size,0:S.size] diag
    if Cov is not None:
        var_S = np.diag(Cov)[: S.size].reshape(S.shape)
        sigma_S = np.sqrt(var_S)
    else:
        # If no covariance, assume infinite S/N → keep all
        sigma_S = np.ones_like(S)

    # 3) Build S/N mask
    snr = np.abs(S) / sigma_S
    mask = snr > threshold

    # 4) Zero-out below-threshold modes
    S_filtered = np.where(mask, S, 0.0)
    T_filtered = np.where(mask, T, 0.0)

    return S_filtered, T_filtered, Cov





class scfPotential:
    """Wrapper fort the SCFPotential contained in Gala. This class takes the same arguments as gala's version and 
    returns a minimal SCF representation, filtering out expansions with small S/N.
    """
    
    def __init__(
            self,
            xyz,
            mass,
            nmax,
            lmax,
            r_s,
            skip_odd=False,
            skip_even=False,
            skip_m=False,
            compute_var=False,
            pool=None
            ):
        
        self.S, self.T, self.cov = compute_and_filter_scf(
            xyz,
            mass,
            nmax,
            lmax,
            r_s,
            skip_odd,
            skip_even,
            skip_m,
            compute_var,
            pool
        )
        self.potential = SCFPotential(Snlm=S, Tnlm=T, m=M, r_s=r_s)
