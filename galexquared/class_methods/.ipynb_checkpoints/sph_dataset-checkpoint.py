import yt
import numpy as np
from unyt import unyt_array

def create_sph_dataset(ds, pt, 
                       data_source=None,
                       extra_fields=None, 
                       n_neighbours=32, 
                       kernel='wendland2',
                       use_norm=True,
                       rotation_matrix=None,
                       rotation_center=None,
                       fields_kw=None
                      ):
    """Creates an sph field using the provided particle field, from the data contained
    in the ds dataset. The smoothing is performed by .add_sph_field. Both the number of
    neighbours and kernel can be selected. The returned field has name 'io'.

    Parameters
    ----------
    data_source : yt.data_objects.selection_objects.region.YTRegion
        Original dataset.
    pt : str
        Particle type to be smoothened.
    extra_fields : list, optional
        Fields of the particles that will be added to the smoothened dataset.
        Default is None, which adds only mass, position and velocity of the particles.
        Any extra field passed, will be appended to this list. For now, they must be scalar
        to work properly with rotation matrices.
    n_neighbours : int, optional
        Number of neighbour particles to include in the smoothing. Default is 32.
    kernel : str, optional
        Kernel used in the smoothing. Default is 'wendland6' (see yt Documentation).
    use_norm : bool, optional
        Value of use_sph_normalization. Default is True.

    Returns
    -------
    ds_sph : yt.dataset
        Dataset containing the sph field. The returned field has name 'io'.
    """
    if data_source is None:
        data_source = ds.all_data()
    else:
        pass

    fields = ["particle_mass", "particle_position", "particle_velocity", "particle_index"]
    if extra_fields is not None:
        for e_field in extra_fields:
            fields.append(e_field)
                
    if rotation_matrix is None:
        data = {field: data_source[pt, field] for field in fields}
    else:
        if rotation_center is None:
            raise Exception("You must provide a rotation center along with the rotation matrix.")
        else:
            c = data_source[pt, "particle_position"] - rotation_center["cm"]
            v = data_source[pt, "particle_velocity"] - rotation_center["vcm"]
            
            rotated_coords = ds.arr([np.dot(rotation_matrix, c[i,:]) for i in range(len(c))], c.units) 
            rotated_vels = ds.arr([np.dot(rotation_matrix, v[i,:]) for i in range(len(v))], v.units)
            data = {
                'particle_mass' : data_source[pt, "particle_mass"],
                
                'particle_position_x' : rotated_coords[:,0],
                'particle_position_y' : rotated_coords[:,1],
                'particle_position_z' : rotated_coords[:,2],
                
                'particle_velocity_x' : rotated_vels[:,0],
                'particle_velocity_y' : rotated_vels[:,1],
                'particle_velocity_z' : rotated_vels[:,2],
                
                'particle_index' : data_source[pt, "particle_index"]
            }
            if extra_fields is not None:
                for field in extra_fields:
                    data[field] = data_source[pt, field]




    
    ds_sph = yt.load_particles(data, data_source=data_source, periodicity=(False, False, False))
    
    ds_sph.current_redshift = ds.current_redshift
    ds_sph.current_time = ds.current_time
    ds_sph.unit_system = ds.unit_system
    ds_sph.unit_registry = ds.unit_registry
    ds_sph.use_sph_normalization = use_norm
    
    ds_sph.add_sph_fields(n_neighbors=n_neighbours, kernel=kernel, sph_ptype='io')
    #ds_sph.add_deposited_particle_field(('io','particle_luminosity'), method="cic", kernel_name=kernel)
    return ds_sph