import yt
import re
import numpy as np
import pandas as pd
from astropy.table import Table
from unyt import unyt_quantity, unyt_array
from copy import copy, deepcopy





def load_halo_rockstar(
                    catalogue,
                    snapequiv,
                    path_to_sim,
                    ds = None,
                    snapnum = None,
                    subtree = None,
                    haloid = None,
                    uid = None,
                    **kwargs
                    ):
    """Loads the specified halo: it returns both the Merger catalogue row corresponding to said halo and a YT dataset
    of the region the halo occupies in the simulation. Various arguments can be passed as kwargs.

    Units are handled with unyt.
    
    Parameters
    ----------
    catalogue : pd.DataFrame, required
        Pandas DataFrame containing the Rockstar + Consistent Trees Merger Tree catalogue. List of fields can be passed as kwarg.
        
    snapequiv : pd.DataFrame/astropy.table/similar, required
        Table containing equivalences between snapshot number (i.e. first, second, thenth etc. snapshot of the sim-run), and snapshot name.
        It is used to identify from which snapshot to load the region occupied by the halo.
        
    path_to_sim : string, required
        Path where the output of the simulation, i.e. the snapshots, are.

    ds : yt.dataset, optional
        Pre-loaded yt-dataset containing (or not, if you mess up) the desired halo. Useful for avoiding the I/O  overhead of loading the same snapshot multiple times
        when you want to load a bunch of halos in the same snapshot.
        
    snapnum : int or float, optional BUT REQUIRED if loading with subtree or haloid
        Snapshot number of the snapshot where the halo is located.

    subtree, haloid, uid : int or float, optional but AT LEAST ONE is required
        Subtree; uid of the oldest progenitor of the branch identified in the merger tree, haloid; unique halo id at a specific snapshot and 
        universal id of the halo. Only one is required. IF LOADING WITH SUBTREE OR HALOID, the snapshot number is required.

    **kwargs: dictionary of keyword arguments that can be passed when loading the halo. They are the following:

        · catalogue_units : dict[key : str, value : str]
            Default are Rockstar units: mass : Msun, time : Gyr, length : kpccm, and velocity : km/s.

        · max_radius : (float, string)
                Max radius to cut out from the simulation and return to the user. The string provides the units.
        
        · catalogue_fields : dict[dict]
            · id_fields : dict[key : str, value : str]
                Default are Rockstar fields: subtree : Sub_tree_id, uid : uid, haloid : Halo_ID, snapnum : Snapshot.
                
            · position_fields : dict[key : str, value : str]
                Defaults are Rockstar fields: coord_i : position_i. Position of the center of the Halo.
    
            · radii_fields : dict[key : str, value : str]
                Defaults are scale_radius and virial_radius. For Scale and Virial NFW radii.
    
        · snapequiv_fields : dict[key : str, value : str]
            Only concerns snapnum and snapshot name columns. Defaults are "snapid" and "snapshot", respectively.
        

    Returns
    -------
    halo : pd.DataFrame

    sp : yt.region.sphere
    """
    catalogue_units = {
        'mass': "Msun",
        'time': "Gyr",
        'length':"kpccm",
        'velocity': "km/s"
    }
    catalogue_fields = {
        'id_fields': {'haloid': "Halo_ID", 'uid': "uid", 'snapnum': "Snapshot", 'subtree': "Sub_tree_id"},
        'position_fields': {'position_x': "position_x", 'position_y': "position_y", 'position_z': "position_z"},
        'radii_fields': {'rs': "scale_radius", 'rvir': "virial_radius"}
    }
    snapequiv_fields = {
        'snapnum': "snapid",
        'snapname': "snapshot"
    }

    #TBD poder cambiarlos.

    idfields = catalogue_fields['id_fields']
    posfields = list(catalogue_fields['position_fields'].values())
    if len(catalogue) == 1:
            halo = catalogue
            snapnum = halo[idfields['snapnum']].values[0]
    elif snapnum is None:
        if uid is not None:
            halo = catalogue[catalogue[idfields['uid']] == uid]
            snapnum = halo[idfields['snapnum']].values[0]
        else:
            raise Exception("SNAPNUM not provided!!")

    else:
        if uid is not None:
            halo = catalogue[catalogue[idfields['uid']] == uid]
        if subtree is not None:
            halo = catalogue[(catalogue[idfields['subtree']] == subtree) & (catalogue[idfields['snapnum']] == snapnum)]
        if haloid is not None:
            halo = catalogue[(catalogue[idfields['haloid']] == haloid) & (catalogue[idfields['snapnum']] == snapnum)]

    file = snapequiv[snapequiv[snapequiv_fields['snapnum']] == snapnum][snapequiv_fields['snapname']].values[0]
    
    if ds is not None:
        if ds.basename != file:
            raise Exception(f"Provided yt-dataset does not contain the selected halo! You provided {ds.basename} but the halo is contained in {file}")
    else:
        ds = yt.load(path_to_sim + f"/{file}")

    halocen = halo[posfields].values[0]
    halovir = halo[catalogue_fields['radii_fields']['rvir']].values[0] 

    if 'max_radius' in kwargs.keys(): sp = ds.sphere( (halocen, catalogue_units['length']), kwargs['max_radius'] )
    else: sp = ds.sphere( (halocen, catalogue_units['length']), (halovir, catalogue_units['length']) )


    return halo, sp, ds




    
 
def load_ftable(fn):
    """Loads astropy tables formated with ascii.fixed_width and sep='\t'. These tables are human readable but
    a bit anoying to read with astropy because of the necessary keyword arguments. This functions solves that.
    Useful for tables that need to be accessed in scripts but be readable (e.g. csv are NOT human readable).

    Equivalent to : Table.read(fn, format="ascii.fixed_width", delimiter='\t')

    Parameters
    ----------
    fn : string, required
        Path to Formated Table file.

    Returns
    -------
    ftable : pd.DataFrame
    """
    return Table.read(fn, format="ascii.fixed_width", delimiter="\t").to_pandas()



def parse_filename(filename, 
                   pattern = r"(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)"
                  ):
    """Finds snap numbers for format basename_number.format using regex on the provided pattern.
    
    Parameters
    ----------
    filename : str 
        Array of filenames.
    pattern : str, optional
        Patterns that the snapshots follow. Default: r'(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)'
        Corresponds to ~ basename_number.format.

    Returns
    -------
    basename, number, file_format : str, str, str
    
    """    
    match = re.match(pattern, filename)
    
    if match:
        basename = match.group('basename')
        number = int(match.group('number'))  
        file_format = match.group('format')
        return basename, number, file_format
    else:
        raise ValueError("Filename format not recognized")


def sort_snaps(file_list, 
               pattern = r"(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)"
              ):
    """Sorts files according to snapnumber using 'parse_filename' function.

    Parameters
    ----------
    file_list : array[str]
        List of file names.
    pattern : str, optional
        Patterns that the snapshots follow. Default: r"(?P<basename>.+?)_(?P<number>\d+)(?:\.\d+)?\.(?P<format>\w+)"
        Corresponds to ~ basename_number.format.

    Returns
    -------
    sorted_filenames : array[str]
    """
    return sorted(file_list, key=lambda x: parse_filename(x, pattern = pattern)[1])         





def load_halo_simple(catalogue,
                    snapequiv,
                    path_to_sim,
                    snapnum = None,
                    subtree = None,
                    haloid = None,
                    uid = None,
                    **kwargs
                    ):
    """
    More barebones version of load_halo_rockstar, it only returns the path to the snapshot containing the halo.

    Units are handled with unyt.
    
    Parameters
    ----------
    catalogue : pd.DataFrame, required
        Pandas DataFrame containing the Rockstar + Consistent Trees Merger Tree catalogue. List of fields can be passed as kwarg.
        
    snapequiv : pd.DataFrame/astropy.table/similar, required
        Table containing equivalences between snapshot number (i.e. first, second, thenth etc. snapshot of the sim-run), and snapshot name.
        It is used to identify from which snapshot to load the region occupied by the halo.
        
    path_to_sim : string, required
        Path where the output of the simulation, i.e. the snapshots, are.
        
    snapnum : int or float, optional BUT REQUIRED if loading with subtree or haloid
        Snapshot number of the snapshot where the halo is located.

    subtree, haloid, uid : int or float, optional but AT LEAST ONE is required
        Subtree; uid of the oldest progenitor of the branch identified in the merger tree, haloid; unique halo id at a specific snapshot and 
        universal id of the halo. Only one is required. IF LOADING WITH SUBTREE OR HALOID, the snapshot number is required.

    **kwargs: dictionary of keyword arguments that can be passed when loading the halo. They are the following:
    
         snapequiv_fields : dict[key : str, value : str]
            Only concerns snapnum and snapshot name columns. Defaults are "snapid" and "snapshot", respectively.
        

    Returns
    -------
    path_to_sim : str
    """
    snapequiv_fields = {'snapnum': "snapid", 'snapname': "snapshot"}
    idfields = catalogue_fields['id_fields']
    posfields = list(catalogue_fields['position_fields'].values())
    
    if len(catalogue) == 1:
            halo = catalogue
            snapnum = halo[idfields['snapnum']].values[0]
    elif snapnum is None:
        if uid is not None:
            halo = catalogue[catalogue[idfields['uid']] == uid]
            snapnum = halo[idfields['snapnum']].values[0]
        else:
            raise Exception("SNAPNUM not provided!!")
    
    else:
        pass
    
    file = snapequiv[snapequiv[snapequiv_fields['snapnum']] == snapnum][snapequiv_fields['snapname']].values[0]
    
    return path_to_sim + f"/{file}"
