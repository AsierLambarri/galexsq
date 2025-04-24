import yt
import numpy as np
from yt.units import dimensions
from pNbody.Mockimgs import luminosities

Z_Solar = 0.02041

def ARTI_loader(fn):
    """ART-I custom data loader: adds correctly derived X_He, X_Metals and metallicity (Zsun units) to gas cells and stellar particles. 
    
    In frontends/art/fields.py X_He is not added, and the gas metallicity is computed as X_Metals/X_Hydrogen. For stars, the metallicity
    fields is not even computed and remains separate in metals1 and metals2.

    Metalicities re-computed as:    gas, metallicity          --->   gas, correct_metallicity_Zsun = log10[ metal_density / density / Zsun(=0.0134) ]
                                  stars, particle_metallicity ---> stars, correct_metallicity_Zsun = log10[ (metallicity1 + metallicity2) / Zsun(=0.0134) ]

    additionally, metal mass fraction and total metal mass are computed using straight forward relations, and Helium mass fraction is computed for the
    gas cells as Y = 1-X-Y (it is fixed at 0.245).



    ### TO ADD: PARTICLE FILTER FOR STAR CREATION TIME AND AGE:


    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.ARTDataset
    """
    lum = luminosities.LuminosityModel("BPASS230_JKC_V")

    def _gas_coordinates(field, data):
        x = data["gas", "x"]
        y = data["gas", "y"]
        z = data["gas", "z"]
        return np.column_stack((x, y, z))  
    def _gas_velocity(field, data):
        vx = data["gas", "velocity_x"]
        vy = data["gas", "velocity_y"]
        vz = data["gas", "velocity_z"]
        return np.column_stack((vx, vy, vz))  

    if isinstance(fn, str):
        ds = yt.load(fn)
    else:
        ds = fn
        
    ### GAS ###
    ds.add_field(
        ("gas", "coordinates"),
        function=_gas_coordinates,
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length
    )
    ds.add_field(
        ("gas", "velocity"),
        function=_gas_velocity,
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("gas", "mass"),
        function=lambda field, data: data['gas', 'cell_mass'],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass,
        force_override=True
    )
    ds.add_field(
        ("gas", "index"),
        function=lambda field, data: ds.arr(np.linspace(1, data["gas", "mass"].shape[0] - 1, num=data["gas", "mass"].shape[0], dtype=int), "") + np.maximum(data["stars", "particle_index"].max(), data["darkmatter", "particle_index"].max()),
        sampling_type="local",
        units='',
        dimensions=dimensions.dimensionless,
        force_override=False
    )
    ds.add_field(
        ("gas", "thermal_energy"),
        function=lambda field, data: data["gas", "cell_volume"] *  data["gas", "thermal_energy_density"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass * dimensions.velocity**2,
    )
    ds.add_field(
        ("gas", "metal_mass_fraction"),
        function=lambda field, data: (data["gas", "metal_ii_density"] + data["gas", "metal_ia_density"]) / data["gas", "density"] ,
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
        force_override=True
    )
    ds.add_field(
        ("gas", "metal_mass"),
        function=lambda field, data:  data["gas", "metal_mass_fraction"] * data["gas", "mass"] ,
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
        ("gas", "He_mass_fraction"),
        function=lambda field, data: 1 - data['gas', 'H_mass_fraction'] - data['gas', 'metal_mass_fraction'],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless
    )
    ds.add_field(
        ("gas", "cell_length"),
        function=lambda field, data: data["gas", "cell_volume"]**(1/3),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("gas", "softening"),
        function=lambda field, data: data["gas", "cell_length"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )


    ### STARS ###
    ds.add_field(
        ("stars", "coordinates"),
        function=lambda field, data: data["stars", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "velocity"),
        function=lambda field, data: data["stars", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("stars", "mass"),
        function=lambda field, data: data["stars", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "initial_mass"),
        function=lambda field, data: data["stars", "particle_mass_initial"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("stars", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["stars", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("stars", "index"),
        function=lambda field, data: data["stars", "particle_index"].astype(int),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass_fraction"),
        function=lambda field, data: data["stars", "particle_metallicity1"] + data["stars", "particle_metallicity2"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "MH"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "particle_mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )
    ds.add_field(
        ("stars", "creation_time"),
        function=lambda field, data: data["stars", "particle_creation_time"] ,
        sampling_type="local",
        units='Gyr'
    )
    ds.add_field(
        ("stars", "age"),
        function=lambda field, data: ds.current_time - data["stars", "creation_time"] ,
        sampling_type="local",
        units='Gyr'
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

    
    
    ### DM ###
    ds.add_field(
        ("darkmatter", "coordinates"),
        function=lambda field, data: data["darkmatter", "particle_position"],
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "velocity"),
        function=lambda field, data: data["darkmatter", "particle_velocity"],
        sampling_type="local",
        units='km/s',
        dimensions=dimensions.velocity
    )
    ds.add_field(
        ("darkmatter", "mass"),
        function=lambda field, data: data["darkmatter", "particle_mass"],
        sampling_type="local",
        units='Msun',
        dimensions=dimensions.mass
    )
    ds.add_field(
        ("darkmatter", "softening"),
        function=lambda field, data: ds.arr(0.08 * np.ones_like(data["darkmatter", "particle_mass"].value, dtype=float), 'kpc'),
        sampling_type="local",
        units='kpc',
        dimensions=dimensions.length,
    )
    ds.add_field(
        ("darkmatter", "index"),
        function=lambda field, data: data["darkmatter", "particle_index"].astype(int),
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


def light_ARTI_loader(fn, soft_kpc=None):
    """ART-I custom data loader: adds correctly derived X_He, X_Metals and metallicity (Zsun units) to gas cells and stellar particles. 
    
    In frontends/art/fields.py X_He is not added, and the gas metallicity is computed as X_Metals/X_Hydrogen. For stars, the metallicity
    fields is not even computed and remains separate in metals1 and metals2.

    Metalicities re-computed as:    gas, metallicity          --->   gas, correct_metallicity_Zsun = log10[ metal_density / density / Zsun(=0.0134) ]
                                  stars, particle_metallicity ---> stars, correct_metallicity_Zsun = log10[ (metallicity1 + metallicity2) / Zsun(=0.0134) ]

    additionally, metal mass fraction and total metal mass are computed using straight forward relations, and Helium mass fraction is computed for the
    gas cells as Y = 1-X-Y (it is fixed at 0.245).



    ### TO ADD: PARTICLE FILTER FOR STAR CREATION TIME AND AGE:


    Parameters
    ----------
    fn : str
        File path

    Returns
    -------
    yt.ARTDataset
    """
    # lum = luminosities.LuminosityModel("BPASS230_JKC_V")

    if isinstance(fn, str):
        ds = yt.load(fn)
    else:
        ds = fn
    
    if soft_kpc is not None:
        ds.add_field(
            ("stars", "softening"),
            function=lambda field, data: ds.arr(soft_kpc * np.ones_like(data["stars", "particle_mass"].value, dtype=float), 'kpc'),
            sampling_type="local",
            units='kpc',
            dimensions=dimensions.length,
        )
    
    ds.add_field(
        ("stars", "metal_mass_fraction"),
        function=lambda field, data: data["stars", "particle_metallicity1"] + data["stars", "particle_metallicity2"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metallicity"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "MH"),
        function=lambda field, data: np.log10(data["stars", "metal_mass_fraction"] / Z_Solar),
        sampling_type="local",
        units='auto',
        dimensions=dimensions.dimensionless,
    )
    ds.add_field(
        ("stars", "metal_mass"),
        function=lambda field, data: data["stars", "metal_mass_fraction"] * data["stars", "particle_mass"],
        sampling_type="local",
        units='auto',
        dimensions=dimensions.mass,
    )
    ds.add_field(
        ("stars", "creation_time"),
        function=lambda field, data: data["stars", "particle_creation_time"] ,
        sampling_type="local",
        units='Gyr'
    )
    ds.add_field(
        ("stars", "age"),
        function=lambda field, data: ds.current_time - data["stars", "creation_time"] ,
        sampling_type="local",
        units='Gyr'
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






















