import yt
import numpy as np


def load_custom_particles(data_source, subtree, particle_index, bound_index):
    """Loads particles in a "particle_dataset" for use in HaloParticlesInside
    """
    particle_name = f"nbody_{int(subtree)}"
    dm_name = f"darkmatter_{int(subtree)}"
    star_name =f"stars_{int(subtree)}"
    
    particle_mask = np.isin(data_source[particle_name, "index"], particle_index)
    
    data = {
        'particle_mass' :  data_source[particle_name, "mass"][particle_mask],
        
        'particle_position_x' :  data_source[particle_name, "particle_position_x"][particle_mask],
        'particle_position_y' :  data_source[particle_name, "particle_position_y"][particle_mask],
        'particle_position_z' :  data_source[particle_name, "particle_position_z"][particle_mask],
        
        'particle_velocity_x' :  data_source[particle_name, "particle_velocity_x"][particle_mask],
        'particle_velocity_y' :  data_source[particle_name, "particle_velocity_y"][particle_mask],
        'particle_velocity_z' :  data_source[particle_name, "particle_velocity_z"][particle_mask],
        
        'particle_softening' : data_source[particle_name, "softening"],
        
        'particle_index' :  data_source[particle_name, "index"]

    }
    
    ds_particles = yt.load_particles(data, data_source=data_source)
    ds_particles.current_redshift = data_source.ds.current_redshift
    ds_particles.cosmology = data_source.ds.cosmology
    ds_particles.current_time = data_source.ds.current_time
    ds_particles.unit_system = data_source.ds.unit_system
    ds_particles.unit_registry = data_source.ds.unit_registry

    ds_particles.add_field(
        ("nbody", "coordinates"),
        function=lambda field, data: data["nbody", "particle_position"],
        sampling_type="local",
        units='kpc'
    )
    ds_particles.add_field(
        ("nbody", "velocity"),
        function=lambda field, data: data["nbody", "particle_velocity"],
        sampling_type="local",
        units='km/s'
    )
    ds_particles.add_field(
        ("nbody", "mass"),
        function=lambda field, data: data["nbody", "particle_mass"],
        sampling_type="local",
        units='Msun'
    )
    ds_particles.add_field(
        ("nbody", "softening"),
        function=lambda field, data: data["nbody", "particle_softening"],
        sampling_type="local",
        units='kpc'
    )
    ds_particles.add_field(
        ("nbody", "index"),
        function=lambda field, data: data["nbody", "particle_index"].astype(int),
        sampling_type="local",
        units=''
    )
    


    yt.add_particle_filter(
        "stars", 
        function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], data_source[star_name, "index"]),
        filtered_type="nbody", 
        requires=["index"]
    )
    yt.add_particle_filter(
        "darkmatter", 
        function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], data_source[dm_name, "index"]),
        filtered_type="nbody", 
        requires=["index"]
    )
    
    yt.add_particle_filter(
        f"prev_nbody_{subtree}", 
        function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "index"], bound_index),
        filtered_type="nbody", 
        requires=["index"]
    )

    ds_particles.add_particle_filter("stars")
    ds_particles.add_particle_filter("darkmatter")
    ds_particles.add_particle_filter(f"prev_nbody_{subtree}")

    return ds_particles