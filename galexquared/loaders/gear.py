import yt
import numpy as np
from yt.units import dimensions
from pNbody.Mockimgs import luminosities

Z_Solar = 0.02041


def GEAR_loader(fn):
    """GEAR custom data loader: Adds correcly derived X_Metals and metallicity for stars and gas particles.
    
     Metalicities re-computed as:      PT0, Metals --->   PT0, correct_metallicity_Zsun = log10[ Metals[:,9] / Zsun(=0.0134) ]
                                   PT1, StarMetals --->   PT1, correct_metallicity_Zsun = log10[ StarMetals[:,9] / Zsun(=0.0134) ]

    additionally, metal fractions and metal masses are computed.

     Star Formation Times and Ages are computed from the provided StarFormationTime, which is, the scale factor o formation, as:

                                SFT --> t_from_a(ScaleFactor)
                               Ages --> current_time - SFT

    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.GEARDataset
    """
    lum = luminosities.LuminosityModel("BPASS230_JKC_V")


    if isinstance(fn, str):
        ds = yt.load(fn)
    else:
        ds = fn
    
    particle_type_aliases = {
        "PartType0": "gas",
        "PartType1": "stars",
        "PartType2": "darkmatter",
    }
    
    for field in ds.field_list:
        particle_type, field_name = field
        if particle_type in particle_type_aliases:
            alias_prefix = particle_type_aliases[particle_type]
            
            def alias_function(field, data, original_field=field):
                return data[original_field]
            
            ds.add_field(
                (alias_prefix, field_name), 
                function=alias_function, 
                units=str(ds.field_info[field].units),
                sampling_type=ds.field_info[field].sampling_type
            )
            
    for field in ds.derived_field_list:
        particle_type, field_name = field
        if particle_type in particle_type_aliases:
            alias_prefix = particle_type_aliases[particle_type]
            
            def alias_function(field, data, original_field=field):
                return data[original_field]
            
            ds.add_field(
                (alias_prefix, field_name), 
                function=alias_function, 
                units=str(ds.field_info[field].units),
                sampling_type=ds.field_info[field].sampling_type,
                force_override=True
            )        
        
    ### STARS ###
    ds.add_field(
        ("stars", "coordinates"),
        function=lambda field, data: data["PartType1", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "velocity"),
        function=lambda field, data: data["PartType1", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("stars", "mass"),
        function=lambda field, data: data["PartType1", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "initial_mass"),
        function=lambda field, data: data["PartType1", "InitialMass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["PartType1", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "index"),
        function=lambda field, data: data["PartType1", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "formation_time"),
        function=lambda field, data: ds.cosmology.t_from_a(data['PartType1', 'StarFormationTime']),
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("stars", "age"),
        function=lambda field, data: ds.current_time - data["stars", "formation_time"],
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("stars", "metal_mass_fraction"),
        function=lambda field, data: data["PartType1", "StarMetals"].in_units("") if len(data["PartType1", "StarMetals"].shape) == 1 else data["PartType1", "StarMetals"][:,9].in_units(""),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )
    ds.add_field(
        ("stars", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("stars", "MH"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("stars", "luminosity"),
        function=lambda field, data: ds.arr( lum.Luminosities(data["stars", "initial_mass"].to("Msun").value, data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=True), "Lsun"),
        sampling_type="local",
        units="Lsun"
    )
    ds.add_field(
        ("stars", "luminosity_V"),
        function=lambda field, data: ds.arr( lum.Luminosities(data["stars", "initial_mass"].to("Msun").value, data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=False), "Lsun"),
        sampling_type="local",
        units="Lsun"
    )
    ds.add_field(
        ("stars", "mass_to_light"),
        function=lambda field, data: ds.arr( lum.MassToLightRatio(data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=True), "Msun/Lsun"),
        sampling_type="local",
        units="Msun/Lsun"
    )
    ds.add_field(
        ("stars", "mass_to_light_V"),
        function=lambda field, data: ds.arr( lum.MassToLightRatio(data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=False), "Msun/Lsun"),
        sampling_type="local",
        units="Msun/Lsun"
    ) 




    ### GAS ###
    ds.add_field(
        ("gas", "coordinates"),
        function=lambda field, data: data["PartType0", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
        force_override=True
    )
    ds.add_field(
        ("gas", "velocity"),
        function=lambda field, data: data["PartType0", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity,
        force_override=True
    )
    ds.add_field(
        ("gas", "mass"),
        function=lambda field, data: data["PartType0", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "density"),
        function=lambda field, data: data["PartType0", "Density"],
        sampling_type="local",
        units='g/cm**3',
        dimensions=dimensions.mass / dimensions.length**3,
        force_override=True
    )   
    ds.add_field(
        ("gas", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["PartType0", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("gas", "metal_mass_fraction"),
        function=lambda field, data: data["PartType0", "Metals"].in_units("") if len(data["PartType0", "Metals"].shape) == 1 else data["PartType0", "Metals"][:,9].in_units(""),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("gas", "metal_mass"),
        function=lambda field, data: data["gas", "metal_mass_fraction"] * data["PartType0", "Masses"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "metallicity"),
        function=lambda field, data: np.log10(data["gas", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )    
    ds.add_field(
        ("gas", "thermal_energy"),
        function=lambda field, data: data["PartType0", "InternalEnergy"] * data["PartType0", "Masses"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass * dimensions.velocity**2,
    )



    
    ### DM ###
    ds.add_field(
        ("darkmatter", "coordinates"),
        function=lambda field, data: data["PartType2", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "velocity"),
        function=lambda field, data: data["PartType2", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("darkmatter", "mass"),
        function=lambda field, data: data["PartType2", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("darkmatter", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["PartType2", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "index"),
        function=lambda field, data: data["PartType2", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )


    ds.add_field(
        ("nbody", "coordinates"),
        function=lambda field, data: data["nbody", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("nbody", "velocity"),
        function=lambda field, data: data["nbody", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("nbody", "mass"),
        function=lambda field, data: data["nbody", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("nbody", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["nbody", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("nbody", "index"),
        function=lambda field, data: data["nbody", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    return ds








def light_GEAR_loader(fn, soft_kpc=None):
    """GEAR custom data loader: Adds correcly derived X_Metals and metallicity for stars and gas particles.
    
     Metalicities re-computed as:      PT0, Metals --->   PT0, correct_metallicity_Zsun = log10[ Metals[:,9] / Zsun(=0.0134) ]
                                   PT1, StarMetals --->   PT1, correct_metallicity_Zsun = log10[ StarMetals[:,9] / Zsun(=0.0134) ]

    additionally, metal fractions and metal masses are computed.

     Star Formation Times and Ages are computed from the provided StarFormationTime, which is, the scale factor o formation, as:

                                SFT --> t_from_a(ScaleFactor)
                               Ages --> current_time - SFT

    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.GEARDataset
    """
    # lum = luminosities.LuminosityModel("BPASS230_JKC_V")


    if isinstance(fn, str):
        ds = yt.load(fn)
    else:
        ds = fn
    
    
    if soft_kpc is not None:
        ds.add_field(
            ("PartType1", "softening"),
            function=lambda field, data: ds.arr(soft_kpc * np.ones_like(data["PartType1", "particle_mass"].value, dtype=float), 'kpc'),
            sampling_type="local",
            units='kpc',
            dimensions=dimensions.length,
        )
    ds.add_field(
        ("PartType1", "formation_time"),
        function=lambda field, data: ds.cosmology.t_from_a(data['PartType1', 'StarFormationTime']),
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("PartType1", "age"),
        function=lambda field, data: ds.current_time - data["stars", "formation_time"],
        sampling_type="local",
        units='Gyr',
        dimensions=dimensions.time,
    )
    ds.add_field(
        ("PartType1", "metal_mass_fraction"),
        function=lambda field, data: data["PartType1", "StarMetals"].in_units("") if len(data["PartType1", "StarMetals"].shape) == 1 else data["PartType1", "StarMetals"][:,9].in_units(""),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("PartType1", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )
    ds.add_field(
        ("PartType1", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("PartType1", "MH"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    # ds.add_field(
    #     ("stars", "luminosity"),
    #     function=lambda field, data: ds.arr( lum.Luminosities(data["stars", "initial_mass"].to("Msun").value, data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=True), "Lsun"),
    #     sampling_type="local",
    #     units="Lsun"
    # )
    # ds.add_field(
    #     ("stars", "luminosity_V"),
    #     function=lambda field, data: ds.arr( lum.Luminosities(data["stars", "initial_mass"].to("Msun").value, data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=False), "Lsun"),
    #     sampling_type="local",
    #     units="Lsun"
    # )
    # ds.add_field(
    #     ("stars", "mass_to_light"),
    #     function=lambda field, data: ds.arr( lum.MassToLightRatio(data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=True), "Msun/Lsun"),
    #     sampling_type="local",
    #     units="Msun/Lsun"
    # )
    # ds.add_field(
    #     ("stars", "mass_to_light_V"),
    #     function=lambda field, data: ds.arr( lum.MassToLightRatio(data["stars", "age"].to("Gyr").value, data["stars", "metallicity"], bolometric=False), "Msun/Lsun"),
    #     sampling_type="local",
    #     units="Msun/Lsun"
    # ) 


    return ds











