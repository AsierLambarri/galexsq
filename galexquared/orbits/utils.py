import numpy as np
from scipy.signal import savgol_filter, find_peaks, argrelextrema, argrelmax, argrelmin, peak_prominences
from scipy.interpolate import interp1d

def refind_pericenters_apocenters(time, radius, verbose=0):
    """
    Finds the pericenters and apocenters with argrelmax/min and peak_prominence. Prior to this, 
    a savitzky-golay filter is applied to upsampled data, to somewhat remove noise.
    
    Added Steps:
      1) First remove detections that do not pass the prominence threshold.
      2) Then, check the curvature via a local quadratic (parabolic) fit to the upsampled, 
         Savitzky–Golay data. For pericenters, we require d²r/dt² > 0; for apocenters, d²r/dt² < 0.
      3) Finally, remove early apocenters (those occurring before the first valid pericenter).
      If nothing is left after these cuts, the global minimum is taken as a tentative pericenter.
    
    Parameters
    ----------
    time : np.ndarray
        1D array of times.
    radius : np.ndarray
        1D array of radii (already converted to physical units if necessary).
    
    Returns
    -------
    peri_apo : dict
        dict["apo"] contains a list of dicts with keys ("time", "radius", "prominence").
        Same for dict["peri"].
    peri_apo_all : dict
        Same but for all found peri and apocenters prior to these additional cuts.
    """
    peri_apo = {}
    
    time_phys = time  # ts["Time"].values
    radius_phys = radius  # ts["Rhost"].values / (1 + ts["Redshift"].values)
    
    # ---------------------------------------------------------------------
    # Upsample and smooth the data.
    # ---------------------------------------------------------------------
    t_find = np.linspace(time_phys.min(), time_phys.max(), 1000)
    cs = interp1d(time_phys, radius_phys)
                     
    r_smooth = savgol_filter(
        cs(t_find), 
        window_length=int(t_find.shape[0] * 0.07) // 2 * 2 + 1, 
        polyorder=5
    )
    r_find = r_smooth
    
    # ---------------------------------------------------------------------
    # Determine candidate extrema (using argrelmax and argrelmin) and compute prominences.
    # ---------------------------------------------------------------------
    index_max = argrelmax(
        r_find, 
        order=int(t_find.shape[0] * 0.07 * 0.1) // 2 * 2 + 1,
    )[0]
    index_min = argrelmin(
        r_find, 
        order=int(t_find.shape[0] * 0.07 * 0.1) // 2 * 2 + 1,
    )[0]
    
    prominences_min = peak_prominences(-r_find, index_min)[0]  
    prominences_max = peak_prominences(r_find, index_max)[0]
    
    # Save the raw candidates (before further filtering)
    t_min, r_min = t_find[index_min], r_find[index_min]
    t_max, r_max = t_find[index_max], r_find[index_max]
    
    # ---------------------------------------------------------------------
    # 1) Remove detections that do not pass the prominence threshold.
    # ---------------------------------------------------------------------
    threshold = max(2.2, 0.03 * (radius_phys.max() - radius_phys.min()))
    filtered_min_indices = index_min[prominences_min > threshold]
    filtered_max_indices = index_max[prominences_max > threshold]
    
    # ---------------------------------------------------------------------
    # 2) Check curvature by fitting a quadratic (parabola) locally.
    #    We use the upsampled, Savitzky–Golay data (t_find, r_find).
    #    We'll use a window of 7 points (or as many as available near boundaries).
    #    For a quadratic fit y = a*x^2 + b*x + c, the curvature is 2*a.
    #    For a pericenter we require 2*a > 0, for an apocenter 2*a < 0.
    # ---------------------------------------------------------------------
    def local_curvature(idx, window=7):
        half = window // 2
        # Determine window bounds, taking care of boundaries.
        start = max(0, idx - half)
        end = min(len(t_find), idx + half + 1)
        x_window = t_find[start:end]
        y_window = r_find[start:end]
        # Fit quadratic: polyfit returns [a, b, c]
        if len(x_window) < 3:
            # Not enough points to fit a parabola; issue a warning.
            warnings.warn("Not enough points for quadratic fit.")
            return None
        coeffs = np.polyfit(x_window, y_window, 2)
        return coeffs  # curvature ~ 2*a

    curve_window = int(t_find.shape[0] * 0.07) // 2 * 2 + 1
    # Apply quadratic curvature check for minima (pericenters)
    keep_min_curv = []
    for i in filtered_min_indices:
        curv = tuple(local_curvature(i, window=curve_window))
        if curv[0] is not None and curv[0] > 0:
            keep_min_curv.append(i)
    keep_min_curv = np.array(keep_min_curv, dtype=int)
    
    # Apply quadratic curvature check for maxima (apocenters)
    keep_max_curv = []
    for i in filtered_max_indices:
        curv = tuple(local_curvature(i, window=curve_window))
        if curv[0] is not None and curv[0] < 0:
            keep_max_curv.append(i)
    keep_max_curv = np.array(keep_max_curv, dtype=int)
    
    # ---------------------------------------------------------------------
    # 3) Remove early apocenters: If a valid pericenter exists,
    #    drop any apocenter occurring before the first valid pericenter.
    # ---------------------------------------------------------------------
    if len(keep_min_curv) > 0:
        first_peri_time = t_find[keep_min_curv[0]]
        keep_max_curv = np.array([i for i in keep_max_curv if t_find[i] >= first_peri_time], dtype=int)
    
    # Override the original filtered indices with our new ones.
    filtered_min_indices = keep_min_curv
    filtered_max_indices = keep_max_curv
    
    # ---------------------------------------------------------------------
    # Map filtered indices to times, radii, and prominences.
    # ---------------------------------------------------------------------
    t_min_th, r_min_th = t_find[filtered_min_indices], r_find[filtered_min_indices]
    t_max_th, r_max_th = t_find[filtered_max_indices], r_find[filtered_max_indices]
    
    prominences_min_th = prominences_min[prominences_min > threshold]
    prominences_max_th = prominences_max[prominences_max > threshold]

    detection_min_th = np.ones_like(t_min_th, dtype=bool)
    detection_max_th = np.ones_like(t_max_th, dtype=bool)
    
    # ---------------------------------------------------------------------
    # If no valid detections remain, fallback to the global minimum as a tentative pericenter.
    # ---------------------------------------------------------------------
    if len(r_min_th) == 0 and len(r_max_th) == 0:
        index = np.argmin(r_find)
        t_min_th, r_min_th = [t_find[index]], [r_find[index]]
        t_max_th, r_max_th = [], []
        prominences_min_th = [999]
        prominences_max_th = []

        filtered_min_indices = [index]
        detection_min_th = np.append(detection_min_th, False)
        
    
    # ---------------------------------------------------------------------
    # Final assembly of peri_apo and peri_apo_all.
    # ---------------------------------------------------------------------
    pericenter_coeffs = []
    for i in index_min:
        pericenter_coeffs.append(tuple(local_curvature(i, window=curve_window)))
    pericenter_coeffs_th = []
    for i in filtered_min_indices:
        pericenter_coeffs_th.append(tuple(local_curvature(i, window=curve_window)))

    
    apocenter_coeffs = []
    for i in index_max:
        apocenter_coeffs.append(tuple(local_curvature(i, window=curve_window)))
    apocenter_coeffs_th = []
    for i in filtered_max_indices:
        apocenter_coeffs_th.append(tuple(local_curvature(i, window=curve_window)))

    assert len(t_min_th) == len(pericenter_coeffs_th), f"{ len(t_min_th)} {len(pericenter_coeffs_th)}"
    assert len(t_max) == len(apocenter_coeffs), f"{ len(t_max)} {len(apocenter_coeffs)}"

    peri_apo = {
        "peri" : [{"time": t, "radius": r, "prominence": p, "coeffs": pc, "detection": det} 
                  for t, r, p, pc, det in zip(t_min_th, r_min_th, prominences_min_th, pericenter_coeffs_th, detection_min_th)],
        "apo" :  [{"time": t, "radius": r, "prominence": p, "coeffs": ac, "detection": det} 
                  for t, r, p, ac, det in zip(t_max_th, r_max_th, prominences_max_th, apocenter_coeffs_th, detection_max_th)]
    }

    peri_apo_all = {
        "peri" : [{"time": t, "radius": r, "prominence": p, "coeffs": pc} 
                  for t, r, p, pc in zip(t_min, r_min, prominences_min, pericenter_coeffs)],
        "apo" :  [{"time": t, "radius": r, "prominence": p, "coeffs": ac} 
                  for t, r, p, ac in zip(t_max, r_max, prominences_max, apocenter_coeffs)]
    }
    if verbose == 1: pp = peri_apo
    elif verbose == 2: pp = peri_apo_all
    if verbose != 0:
        print(f"Adaptative Threshold: {threshold:.2f}")
        print("---------------------------")
        print("")
        print("Pericenters:")
        for value in pp["peri"]:
            print(f"-  t={value['time']:.3f}  r={value['radius']:.3f}  p={value['prominence']:.2f}  a={value["coeffs"][0]:.2e}")
        print("Apocenters:")
        for value in pp["apo"]:
            print(f"-  t={value['time']:.3f}  r={value['radius']:.3f}  p={value['prominence']:.2f}  a={value["coeffs"][0]:.2e}")
            

    return peri_apo, peri_apo_all

