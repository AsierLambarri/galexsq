import yt
import numpy as np

from ..config import config

def _goofy_mass_scaling(max_mass, min_mass):
    log_mass_max = np.log10(max_mass)
    log_mass_min = np.log10(min_mass)
    
    scale_mass = 10**((log_mass_max + log_mass_min) / 2)
    
    normalized_max = max_mass / scale_mass
    normalized_min = min_mass / scale_mass
    
    max_value = max(normalized_max, normalized_min)
    
    bits_required = np.ceil(np.log2(max_value))
    
    if bits_required <= 16:
        dtype = np.float16
    elif bits_required <= 32:
        dtype = np.float32
    else:
        dtype = np.float64

    return dtype, scale_mass

    

def _check_particle_uniqueness(data):
    """Checks that particles are not born twice!
    """
    all_values = [value for sublist in data.values() for value in sublist]
    unique_values = set(all_values)
    
    return len(all_values) == len(unique_values)

def _remove_duplicates(data):
    """Removes duplicates leaving first appearences in snapshot order.
    """
    seen = set()
    new_data = {}
    for key in sorted(data.keys()):
        filtered_list = [x for x in data[key] if x not in seen and (seen.add(x) or True)]
        new_data[key] = np.array(filtered_list)
    return new_data


def best_dtype(arr, kind=None, eps=None):
    """
    Return the smallest NumPy dtype of the inferred (or overridden) kind
    that can represent `arr` with max absolute error ≤ eps.
    
    If kind is None, we infer:
      • if every value is integer-valued:
          – if all ≥ 0 → kind='uint'
          – else         → kind='int'
      • otherwise      → kind='float'
    
    For integer kinds, eps is forced to 0.
    
    Parameters
    ----------
    arr : array‑like
        Input data.
    kind : {'float','int','uint'} or None, optional
        Which family to consider; if None, inferred from values.
    eps : float or None, optional
        Max allowed absolute error (only meaningful for floats).
        If None and kind=='float', defaults to float32’s machine eps.
    
    Returns
    -------
    dtype : np.dtype
        The minimal dtype satisfying the constraint.
    
    Raises
    ------
    ValueError
        • Invalid kind
        • eps≠0 for integer kinds
        • non‑integer values when kind in {'int','uint'}
        • negative values when kind=='uint'
    """
    arr = np.asarray(arr, dtype=np.float64)
    # 1) Infer kind if not given
    if kind is None:
        is_int = np.all(arr == np.round(arr))
        if is_int:
            if np.all(arr >= 0):
                kind = 'uint'
            else:
                kind = 'int'
        else:
            kind = 'float'
    
    # 2) Sanity checks / defaults
    if kind not in ('float','int','uint'):
        raise ValueError("`kind` must be one of 'float','int','uint' or None.")
    if kind in ('int','uint'):
        if eps not in (0, None):
            raise ValueError("For integer kinds, eps must be 0 or None.")
        # require exact integers
        if not np.all(arr == np.round(arr)):
            raise ValueError("Array has non‑integer values; cannot cast losslessly.")
        if kind == 'uint' and np.any(arr < 0):
            raise ValueError("Array has negative entries; cannot cast to unsigned.")
        eps = 0.0
    else:  # float
        if eps is None:
            eps = np.finfo(np.float32).eps
    
    # 3) Candidate dtypes in increasing precision/size
    if kind == 'float':
        candidates = [np.float16, np.float32, np.float64]
    elif kind == 'int':
        candidates = [np.int8,  np.int16,  np.int32,  np.int64]
    else:  # uint
        candidates = [np.uint8, np.uint16, np.uint32, np.uint64]
    
    # 4) Test each: cast down & back up, measure max error
    for dt in candidates:
        back = arr.astype(dt).astype(np.float64)
        max_err = np.max(np.abs(back - arr))
        if max_err <= eps:
            return np.dtype(dt)
    
    # 5) fallback to highest-precision
    return np.dtype(candidates[-1])




def custom_load(fn, ptype):
    """Loads
    """
    if config.code in ["ART", "ART-I", "GEAR"]:
        return yt.load(fn)
    elif config.code in ["RAMSES", "VINTERGATAN"]:
        ds = yt.load(
            fn,
            extra_particle_fields=[
                ('particle_potential', 'd'), 
                ("conformal_birth_time", 'd'), 
                ('particle_metallicity0', 'd'), 
                ("particle_metallicity1", 'd'),
                ("particle_tag", 'd'), 
                ("particle_birth_time", 'd')
            ]
        )
        yt.add_particle_filter(
            ptype,
            function=lambda pfilter, data: (data[pfilter.filtered_type, "particle_birth_time"] > 0) & (data[pfilter.filtered_type, "particle_metallicity0"] > 0),
            requires=["particle_birth_time"],
            filtered_type="all"
        )
        ds.add_particle_filter(ptype)
        return ds
