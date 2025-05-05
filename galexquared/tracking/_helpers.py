import numpy as np

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