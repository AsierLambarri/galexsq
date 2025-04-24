import yt
import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from yt.utilities.logger import ytLogger
from unyt import unyt_quantity, unyt_array

from copy import deepcopy
from pytreegrav import PotentialTarget


import concurrent.futures
import multiprocessing as mp

import galexquared as gal
from galexquared.class_methods import random_vector_spherical
from galexquared.utils import particle_unbinding_fire

def parse_args():
    parser = argparse.ArgumentParser(description="Script to analyze dwarf galaxies in different codes along their evolution/infall")
        
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')
    
    
    required.add_argument(
        "-i", "--input_file",
        type=str,
        help="Merger Tree file path. Must be csv. Tree must be constructed in csv format already.",
        required=True
    )
    required.add_argument(
        "-s", "--subtree_ids",
        type=str,
        help="Sub_tree_ids to analyze.",
        required=True
    )
    required.add_argument(
        "-eq", "--equivalence_table",
        type=str,
        help="Equivalence table path. Must have correct formatting.",
        required=True
    )
    required.add_argument(
        "--RRvir_max",
        type=float,
        help="Main value of R/Rvir to start analysis.",
        required=True
    ) 
    required.add_argument(
        "--zmax",
        type=float,
        help="Maximum value of radshift.",
        required=True
    ) 
    required.add_argument(
        "-c", "--code",
        type=str,
        help="Code of the simulation. Requred for selecting starry halos.",
        required=True
    )
    required.add_argument(
        "-pd", "--particle_data_folder",
        type=str,
        help="Location of particle data after tracking. Used for when the OG MergerTree lost the halos.",
        required=True
    )
    required.add_argument(
        "-sf", "--simulation_folder",
        type=str,
        help="Location of simulation data.",
        required=True
    )
    required.add_argument(
        "-o", "--output",
        type=str,
        help="Location of particle data. Requred for selecting starry halos.",
        required=True
    )
    
    
    opt.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose output. (Default False)"
    )
    
    return parser.parse_args()


def select_current(file_list, sn):
    b = [int(x.split("_")[2].split(".")[0]) for x in file_list]
    return file_list[np.array(b) == sn][0]
    
def select_first(file_list):
    b = [int(x.split("_")[2].split(".")[0]) for x in file_list]
    return file_list[np.argmin(b)]

def compute_boundness_outside(stars, dm, gas, halo_params, inside=False):
    """Computes boundness for halo.
    """
    stars.refined_center6d(method="adaptative", nmin=40)
    dm.refined_center6d(method="adaptative", nmin=100)    
    gas.dq = {"cm": dm["cm"]}
    gas.dq = {"vcm": dm["vcm"]}

    gas_pot, tree = PotentialTarget(
        pos_source=np.concatenate( (gas["coordinates"], stars["coordinates"], dm["coordinates"]) ).to("kpc"), 
        pos_target=gas["coordinates"].to("kpc"), 
        m_source=np.concatenate( (gas["mass"], stars["mass"], dm["mass"]) ).to("Msun"), 
        softening_target=gas["softening"].to("kpc"),
        softening_source=np.concatenate( (gas["softening"], stars["softening"], dm["softening"]) ).to("kpc"), 
        G=4.300917270038e-06,
        theta=0.6,
        parallel=False,
        quadrupole=True,
        return_tree=True
    )
    
    gas_pot = gas["mass"] * unyt_array(gas_pot, 'km**2/s**2')
    gas.add_field("grav_potential", gas_pot)
    
    gas_kin = 0.5 * gas["mass"] * np.linalg.norm(gas["velocity"] - dm["vcm"], axis=1) ** 2
    gas.add_field("kinetic_energy", gas_kin + gas["thermal_energy"].to(gas_kin.units))
    
    gas.add_field("total_energy", gas_pot + gas_kin)
    
    gas_bound = gas.filter(gas["total_energy"] < 0)


    if not inside:
        st_pot = stars["mass"] * unyt_array(
            PotentialTarget(
                pos_source=np.concatenate( (gas["coordinates"], stars["coordinates"], dm["coordinates"]) ).to("kpc"), 
                pos_target=stars["coordinates"].to("kpc"), 
                m_source=np.concatenate( (gas["mass"], stars["mass"], dm["mass"]) ).to("Msun"), 
                softening_target=stars["softening"].to("kpc"),
                softening_source=np.concatenate( (gas["softening"], stars["softening"], dm["softening"]) ).to("kpc"), 
                G=4.300917270038e-06,
                theta=0.6,
                parallel=False,
                quadrupole=True,
                tree=tree
            ), 
            'km**2/s**2'
        )
        dm_pot = dm["mass"] * unyt_array(
            PotentialTarget(
                pos_source=np.concatenate( (gas["coordinates"], stars["coordinates"], dm["coordinates"]) ).to("kpc"), 
                pos_target=dm["coordinates"].to("kpc"), 
                m_source=np.concatenate( (gas["mass"], stars["mass"], dm["mass"]) ).to("Msun"), 
                softening_target=dm["softening"].to("kpc"),
                softening_source=np.concatenate( (gas["softening"], stars["softening"], dm["softening"]) ).to("kpc"), 
                G=4.300917270038e-06,
                theta=0.6,
                parallel=False,
                quadrupole=True,
                tree=tree
            ), 
            'km**2/s**2'
        )
        stars.add_field("grav_potential", st_pot)
        dm.add_field("grav_potential", dm_pot)
        
        st_kin = 0.5 * stars["mass"] * np.linalg.norm(stars["velocity"] - dm["vcm"], axis=1) ** 2
        dm_kin = 0.5 * dm["mass"] * np.linalg.norm(dm["velocity"] - dm["vcm"], axis=1) ** 2
        
        stars.add_field("kinetic_energy", st_kin)
        dm.add_field("kinetic_energy", dm_kin)
        
        stars.add_field("total_energy", st_pot + st_kin)
        dm.add_field("total_energy", dm_pot + dm_kin)
    
    
        #pids, mask, _ = particle_unbinding_fire(
        #    stars["coordinates"],
        #    stars["mass"],
        #    stars["velocity"],
        #    stars["index"],
        #    halo_params
        #)
        stars_bound = stars.filter(stars["total_energy"] < 0)
        #stars_bound.refined_center6d(method="adaptative", nmin=40)
        
        dm_bound = dm.filter(dm["total_energy"] < 0)
        #dm_bound.refined_center6d(method="adaptative", nmin=100)

    if not inside:
        return stars_bound, dm_bound, gas_bound
    else:
        return gas_bound


def process_subtree(subtree, arbor):
    """Processes a lot of shit about the subtree in question
    """
    global sf, pdir, rvir_max, z_max, output
    
    satellite = deepcopy( arbor.CompleteTree[(arbor.CompleteTree["Sub_tree_id"] == subtree) & (arbor.CompleteTree["R/Rvir"] <= rvir_max)  & (arbor.CompleteTree["Redshift"] <= z_max)].sort_values("Snapshot") )
    
    
    
    #satellite['delta_rel'] = pd.Series()
    satellite['stellar_mass'] = pd.Series()
    satellite['dark_mass'] = pd.Series()
    satellite['gas_mass'] = pd.Series()
    
    satellite['rh_stars_physical'] = pd.Series()
    satellite['rh_dm_physical'] = pd.Series()
    satellite['e_rh_stars_physical'] = pd.Series()
    satellite['e_rh_dm_physical'] = pd.Series()
    
    satellite['rh3D_stars_physical'] = pd.Series()
    satellite['rh3D_dm_physical'] = pd.Series()

    satellite['Mhl'] = pd.Series()
    
    satellite['sigma*'] = pd.Series()
    satellite['e_sigma*'] = pd.Series()
    
    satellite['Mdyn'] = pd.Series()
    satellite['e_Mdyn'] = pd.Series()


    errors = []    
    for index, halorow in satellite.iterrows():
        print(f"\n#####   {halorow['Sub_tree_id']}:snapshot-{halorow['Snapshot']}   #####")
            
        lines_of_sight = random_vector_spherical(N=20)

        try:
            halo_params = arbor.get_halo_params(sub_tree=subtree, snapshot=halorow['Snapshot'])        
            if halorow["R/Rvir"] > 1:
                stars, dm, gas = arbor.load_halo(sf, sub_tree=subtree, snapshot=halorow['Snapshot'])
                stars_bound, dm_bound, gas_bound = compute_boundness_outside(stars, dm, gas, halo_params, inside=False)
            else:
                path = pdir + "/" + f"subtree_{int(subtree)}/"
                flist = np.array(os.listdir(path))
                stars, dm, gas = arbor.load_halo(sf, sub_tree=subtree, snapshot=halorow['Snapshot'], particle_ids=path + select_first(flist))
                stars_bound, dm_bound, _ = arbor.load_halo(sf, sub_tree=subtree, snapshot=halorow['Snapshot'], particle_ids=path + select_current(flist, subtree))
                
                gas_bound = compute_boundness_outside(stars, dm, gas, halo_params, inside=True)
            
    
            if stars.empty() or stars_bound.empty():
                errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no stars at all.")
                continue
            if gas.empty() or gas_bound.empty():
                errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no gas at all.")
                continue
                

            
            satellite.loc[index, "gas_mass"] =  gas_bound["mass"].sum().to("Msun").value
            satellite.loc[index, "dark_mass"] =  dm_bound["mass"].sum().to("Msun").value
            satellite.loc[index, "stellar_mass"] =  stars_bound["mass"].sum().to("Msun").value
            #satellite.loc[index, "delta_rel"] = float(stars.delta_rel)
    
            satellite.loc[index, 'rh3D_stars_physical'] = stars_bound.half_mass_radius().in_units("kpc").value
            satellite.loc[index, 'rh3D_dm_physical'] = dm_bound.half_mass_radius().in_units("kpc").value
    
            rh_stars = stars_bound.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            rh_dm = dm_bound.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            sigma = stars_bound.los_dispersion(lines_of_sight=lines_of_sight).to("km/s").value
            mdyn = 580 * 1E3 * rh_stars * sigma ** 2
            
            satellite.loc[index, "rh_stars_physical"] = rh_stars.mean()
            satellite.loc[index, "e_rh_stars_physical"] =  rh_stars.std()
            satellite.loc[index, "rh_dm_physical"] =  rh_dm.mean()
            satellite.loc[index, "e_rh_dm_physical"] = rh_dm.std()
            satellite.loc[index, "sigma*"] = sigma.mean()
            satellite.loc[index, "e_sigma*"] = sigma.std()
            satellite.loc[index, 'Mhl'] = stars_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) + dm_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) + gas_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"])
            
            satellite.loc[index, 'Mdyn'] = mdyn.mean()
            satellite.loc[index, 'e_Mdyn'] = mdyn.std()
        
        
        except Exception as e:
            errors.append(f"Strange error ocurred in {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']}")
            print(f"Error processing subtree {subtree}: {e}")
            import traceback
            traceback.print_exc()
            raise

        stars.close()
        dm.close()
        gas.close()

        stars_bound.close()
        dm_bound.close()
        gas_bound.close()
    satellite.to_csv(output + "/" + f"subtree_{int(subtree)}.csv", index=False)
    return f"Subtree {subtree} processed."

    
    






if __name__ == "__main__":
    args = parse_args()

    if not args.input_file.endswith(".csv"):
        raise Exception("The tree must be csv format.")
        
    arbor = gal.MergerTree(args.input_file)
    arbor.set_equivalence(args.equivalence_table)

    sub_tree_ids = pd.read_csv(args.subtree_ids)["Sub_tree_id"].unique()
    
    print(f"Selected sub_tree_ids to analyze: {sub_tree_ids}. \nThe provided R/Rvir range is: {args.RRvir_max} >")
    print(f"Simulation data folder: {args.simulation_folder}.")
    print(f"Master particle data folder: {args.particle_data_folder}.")

    sf = args.simulation_folder
    pdir = args.particle_data_folder
    equiv = arbor.equivalence_table
    output = args.output
    rvir_max = args.RRvir_max
    z_max = args.zmax
    
    gal.config.code = args.code

    max_workers = min(min(mp.cpu_count(), 10), len(sub_tree_ids))
    with mp.Pool(processes=max_workers) as pool:
        args_list = [(subtree, arbor) for subtree in sub_tree_ids]
        results = pool.starmap(process_subtree, args_list)

    for res in results:
        print(res)
    
    print(f"Finished. Bye!")











    