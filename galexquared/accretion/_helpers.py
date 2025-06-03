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

def _nfw_potential(r, mvir, rs, c, G):
    #x = np.clip(r, 0 * 2 * soft, np.inf) / rs
    x = r / rs
    A_nfw = np.log(1 + c) - c / (1 + c)
    return -G * mvir * rs / A_nfw * np.log(1 + x) / x 

def _kepler_potential(r, mvir, G):
    return  -G * mvir / r


def potential(r, mode="kepler", **kwargs):
    if mode == "kepler":
        print("USING KEPLER")
        mvir, G = kwargs["mass"], kwargs["G"]
        return _kepler_potential(r, mvir, G)
    if mode == "nfw":
        print("USING NFW")
        mvir, rs, c, G = kwargs["mass"], kwargs["rs"], kwargs["c"], kwargs["G"]
        return _nfw_potential(r, mvir, rs, c, G)

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
            function=lambda pfilter, data: data[pfilter.filtered_type, "particle_birth_time"] > 0,
            requires=["particle_birth_time"],
            filtered_type="all"
        )
        ds.add_particle_filter(ptype)
        return ds
