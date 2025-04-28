import os
import yt
import gc
import time
import argparse
import numpy as np
import pandas as pd
from yt.utilities.logger import ytLogger
from unyt import unyt_quantity, unyt_array

from copy import deepcopy
from pytreegrav import PotentialTarget, Potential


import concurrent.futures
import multiprocessing as mp

import galexquared as gal
from galexquared.class_methods import random_vector_spherical
from galexquared.utils import particle_unbinding_fire

from memory_profiler import profile

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
        "--threads",
        type=int,
        default=8,
        help="Number of threads to use. (Default 8)"
    )
    opt.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose output. (Default False)"
    )
    opt.add_argument(
        "--start_snap",
        type=int,
        default=None,
        help="Snapshot to start from. (Default None)"
    )
    
    return parser.parse_args()


def select_current(file_list, sn):
    b = [int(x.split("_")[2].split(".")[0]) for x in file_list]
    return file_list[np.array(b) == sn][0]
    
def select_first(file_list):
    b = [int(x.split("_")[2].split(".")[0]) for x in file_list]
    return file_list[np.argmin(b)]


def compute_boundness_outside(halo):
    """Computes boundness for halo.
    """
    stars = halo[0]
    dm = halo[1]
    gas = halo[2]
    
    dm.refined_center6d(method="adaptative", nmin=100)    
    stars.refined_center6d(method="adaptative", nmin=40)    
    gas.dq = {"cm": dm["cm"]}
    gas.dq = {"vcm": dm["vcm"]}
    
    positions = np.concatenate((
        stars["coordinates"], 
        dm["coordinates"],
        #gas[ "coordinates"]
    )).to("kpc")
    velocities = np.concatenate((
        stars["velocity"], 
        dm["velocity"],
        #gas[ "velocity"]
    )).to("km/s")
    masses = np.concatenate((
        stars["mass"], 
        dm["mass"],
        #gas[ "mass"]
    )).to("Msun")
    softs = np.concatenate((
        stars["softening"], 
        dm["softening"],
        #gas[ "softening"]
    )).to("kpc")
    indices = np.concatenate((
        stars["index"], 
        dm["index"],
        #gas["index"]
    )).to("")

    #gas_mask = np.isin(indices, gas["index"]) 
    stars_mask = np.isin(indices, stars["index"])
    dm_mask = np.isin(indices, dm["index"])

    
    potential = masses * unyt_array( 
        Potential(
            pos=positions, 
            m=masses, 
            softening=softs,
            G=4.300917270038E-6, 
            theta=0.6,
            parallel=True,
            quadrupole=True
        ), 
        'km**2/s**2'
    )

    kinetic = 0.5 * masses * np.linalg.norm(velocities - dm["vcm"], axis=1) ** 2
    #kinetic[gas_mask] += gas["thermal_energy"]
    total = kinetic + potential

    
    #gas.add_field("grav_potential", potential[gas_mask])
    stars.add_field("grav_potential", potential[stars_mask])
    dm.add_field("grav_potential", potential[dm_mask])

    #gas.add_field("kinetic_energy", kinetic[gas_mask])
    stars.add_field("kinetic_energy", kinetic[stars_mask])
    dm.add_field("kinetic_energy", kinetic[dm_mask])

    #gas.add_field("total_energy", total[gas_mask])
    stars.add_field("total_energy", total[stars_mask])
    dm.add_field("total_energy", total[dm_mask])

    
    gas_bound = gas #.filter(gas["total_energy"] < 0)
    dm_bound = dm.filter(dm["total_energy"] < 0)
    stars_bound = stars.filter(stars["total_energy"] < 0)

    del stars, dm, #gas
    del positions, velocities, masses, softs, indices
    del potential, kinetic, total
    del stars_mask, dm_mask, #gas_mask

    


    return stars_bound, dm_bound, gas_bound

def compute_boundness_inside(halo):
    stars = halo[0]
    dm = halo[1]
    gas = halo[2]
    
    dm.refined_center6d(method="adaptative", nmin=100)    
    stars.refined_center6d(method="adaptative", nmin=40)    
    gas.dq = {"cm": dm["cm"]}
    gas.dq = {"vcm": dm["vcm"]}
    
    positions = np.concatenate((
        stars["coordinates"], 
        dm["coordinates"],
        gas[ "coordinates"]
    )).to("kpc")
    velocities = np.concatenate((
        stars["velocity"], 
        dm["velocity"],
        gas[ "velocity"]
    )).to("km/s")
    masses = np.concatenate((
        stars["mass"], 
        dm["mass"],
        gas[ "mass"]
    )).to("Msun")
    softs = np.concatenate((
        stars["softening"], 
        dm["softening"],
        gas[ "softening"]
    )).to("kpc")
    indices = np.concatenate((
        stars["index"], 
        dm["index"],
        gas[ "index"]
    )).to("")

    
    potential = gas["mass"].to("Msun") * unyt_array( 
        PotentialTarget(
            pos_source=positions, 
            m_source=masses, 
            softening_source=softs,
            pos_target=gas["coordinates"].to("kpc"),
            softening_target=gas["softening"].to("kpc"),
            G=4.300917270038E-6, 
            theta=0.6,
            parallel=True,
            quadrupole=True
        ), 
        'km**2/s**2'
    )

    kinetic = 0.5 * gas["mass"].to("Msun") * np.linalg.norm(gas["velocity"].to("km/s") - dm["vcm"], axis=1) ** 2
    kinetic += gas["thermal_energy"]
    total = kinetic + potential

    gas.add_field("grav_potential", potential)
    gas.add_field("kinetic_energy", kinetic)
    gas.add_field("total_energy", total)
    
    gas_bound = gas.filter(gas["total_energy"] < 0)

    del stars, dm, gas
    del positions, velocities, masses, softs, indices
    del potential, kinetic, total
    
    return gas_bound


def analyze_halo(arbor, satellites, subtree, snapshot, kwargs):
    """Analyzes a halo.
    """
    errors = []    
    compute_gastats = False
    try:
        message = f" -Subtree {subtree} pocessed using saved particles."
        path = kwargs["pdir"] + "/" + f"subtree_{int(subtree)}/"
        flist = np.array(os.listdir(path))
        first_file = select_first(flist)
        fnumber = int(first_file.split("_")[2].split(".")[0])
    
        not_from_file = fnumber >= snapshot
    except:
        not_from_file = True
    print(f" -Processing Subtree {subtree}.")
    try:    
        halorow = satellites[(satellites["Snapshot"] == snapshot) & (satellites["Sub_tree_id"] == subtree)]
        if halorow.shape[0] != 1: raise Exception(f"Something got fucked up!")
        index = halorow.index
        halorow = halorow.squeeze()
    
        lines_of_sight = random_vector_spherical(N=20)
        halo_params = arbor.get_halo_params(sub_tree=subtree, snapshot=halorow['Snapshot'])        
        if not_from_file:
            message = f" -Subtree {subtree} pocessed."
            stars, dm, gas = arbor.load_halo(sub_tree=subtree, snapshot=halorow['Snapshot'])
            if stars.empty():
                errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no stars.")
                print(f" -Subtree {subtree} doesn't have stars.")
                return satellites[satellites["Snapshot"] == snapshot]
            
            stars_bound, dm_bound, gas_bound = compute_boundness_outside([stars, dm, gas])
        else:
            stars, dm, gas = arbor.load_halo(sub_tree=subtree, snapshot=halorow['Snapshot'], particle_ids=np.load(path + select_first(flist)))
            stars_bound, dm_bound, _ = arbor.load_halo(sub_tree=subtree, snapshot=halorow['Snapshot'], particle_ids=np.load(path + select_current(flist, snapshot)))
            
            gas_bound = gas #compute_boundness_inside([stars, dm, gas])
    
            dm_bound.refined_center6d(method="adaptative", nmin=100)    
            stars_bound.refined_center6d(method="adaptative", nmin=40) 

        
        pids, mask, _ = particle_unbinding_fire(
            stars_bound["coordinates"],
            stars_bound["mass"],
            stars_bound["velocity"],
            stars_bound["index"],
            halo_params
        )
        stars_bound = stars_bound.filter(mask)

        
        if stars_bound.empty():
            errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no bound stars.")
            print(f" -Subtree {subtree} doesn't have BOUND stars.")
            return satellites[satellites["Snapshot"] == snapshot]
    
        #if gas.empty():
        #    errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no bound gas.")
        #    print(f" -Subtree {subtree} doesn't have BOUND gas.")
        #else:
        #    compute_gastats = True
    
        if len(stars_bound["index"]) >= 10:
            satellites.loc[index, "gas_mass"] =  np.nan #gas_bound["mass"].sum().to("Msun").value
            
            satellites.loc[index, "dark_mass"] =   dm_bound["mass"].sum().to("Msun").value
            satellites.loc[index, 'rh3D_dm_physical'] = dm_bound.half_mass_radius().in_units("kpc").value
            
            rh_dm = dm_bound.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            
            satellites.loc[index, "rh_dm_physical"] =  rh_dm.mean()
            satellites.loc[index, "e_rh_dm_physical"] = rh_dm.std()
        
        
        
        
            satellites.loc[index, "stellar_mass"] =   stars_bound["mass"].sum().to("Msun").value
            satellites.loc[index, 'rh3D_stars_physical'] = stars_bound.half_mass_radius().in_units("kpc").value
        
            rh_stars = stars_bound.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            sigma = stars_bound.los_dispersion(lines_of_sight=lines_of_sight).to("km/s").value
            mdyn = 580 * 1E3 * rh_stars * sigma ** 2
        
            satellites.loc[index, "rh_stars_physical"] = rh_stars.mean()
            satellites.loc[index, "e_rh_stars_physical"] =  rh_stars.std()
        
            satellites.loc[index, "sigma*"] = sigma.mean()
            satellites.loc[index, "e_sigma*"] = sigma.std()
            
            satellites.loc[index, 'Mdyn'] = mdyn.mean()
            satellites.loc[index, 'e_Mdyn'] = mdyn.std()
    
            satellites.loc[index, 'LV'] = stars_bound["luminosity_V"].sum().to("Lsun").value
            satellites.loc[index, 'M/LV'] = stars_bound["mass_to_light_V"].sum().to("Msun/Lsun").value
            
            
            if compute_gastats:    
                satellites.loc[index, 'Mhl'] = stars_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) + dm_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) #+ gas_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"])
            else:
                satellites.loc[index, 'Mhl'] = stars_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) + dm_bound.enclosed_mass(stars_bound.half_mass_radius(), stars_bound["cm"]) 

        del stars, dm, gas        
        del stars_bound, gas_bound, dm_bound


    except Exception as e:
        print(e)

        
    satellites.to_csv(kwargs["output"] + "/" + f"{kwargs["code"]}_Satellites.csv", index=False)
    print(message)
    return satellites[satellites["Snapshot"] == snapshot]





    
def analyze_snapshot(arbor, satellites, snapshot, kwargs):
    """Processes a lot of shit about the subtree in question.
    """
    snapshot_satellites = satellites[satellites["Snapshot"] == snapshot]
    if not snapshot_satellites.empty:
        print(f"SNAPSHOT {snapshot}")
        print(f"Subtrees found in this snapshot: {snapshot_satellites["Sub_tree_id"].values.astype(int)}")
        
        arbor.set_snapshot(path=kwargs["sf"] + "/" + arbor.find_file(snapshot))
        print(f"Snapshot loaded.")
        for subtree in snapshot_satellites["Sub_tree_id"].unique():
            out = analyze_halo(arbor, satellites, subtree, snapshot, kwargs)
            gc.collect()
    else:
        pass

    print("\n", out, "\n")
    return f"SNAPSHOT {snapshot} pocessed.\n\n\n"

    
    






if __name__ == "__main__":
    cols = [
        'Halo_ID', 'Snapshot', 'Redshift', 'Time', 'uid', 'desc_uid', 'mass',
       'num_prog', 'virial_radius', 'scale_radius', 'vrms', 'vmax',
       'position_x', 'position_y', 'position_z', 'velocity_x', 'velocity_y',
       'velocity_z', 'A[x]', 'A[y]', 'A[z]', 'b_to_a', 'c_to_a', 'T_U',
       'Tidal_Force', 'Tidal_ID', 'Secondary', 'Sub_tree_id', 'Halo_at_z0',
       'TreeNum', 'R/Rvir', 'peak_mass'
    ]
    
    args = parse_args()

    if not args.input_file.endswith(".csv"):
        raise Exception("The tree must be csv format.")

    gal.config.code = args.code
    arbor = gal.MergerTree(args.input_file)
    arbor.set_equivalence(args.equivalence_table)

    sub_tree_ids = pd.read_csv(args.subtree_ids)["Sub_tree_id"].unique()
    if os.path.isfile(args.output + "/" + f"{args.code}_Satellites.csv"):
        satellites = pd.read_csv(args.output + "/" + f"{args.code}_Satellites.csv")
        if args.start_snap is not None:
            snapshots_to_run = satellites[satellites["Snapshot"] >= args.start_snap]["Snapshot"].unique()        
        else:
            min_snaps = satellites[~satellites["stellar_mass"].isnull()]["Snapshot"].max()    
            snapshots_to_run = satellites[satellites["Snapshot"] >= min_snaps]["Snapshot"].unique()        

    else:
        mask = (
            arbor.CompleteTree["Sub_tree_id"].isin(sub_tree_ids) & 
            (arbor.CompleteTree["Redshift"] <= args.zmax)
        )
    
        df = deepcopy(arbor.CompleteTree[mask][cols].sort_values("Snapshot", ascending=True))
    
        df['Priority'] = (df['R/Rvir'] > 4).astype(int)  
        
        df['R_diff'] = np.abs(df['R/Rvir'] - 4)
        
        df = df.sort_values(by=['Sub_tree_id', 'Redshift'], ascending=[True, False])
        
        df['crossings'] = (
            df.groupby('Sub_tree_id')['Priority']
            .transform(lambda x: (x != x.shift()).cumsum() - 1)
        )
        df['max_crossings'] = df.groupby('Sub_tree_id')['crossings'].transform('max')
        
        satellites = df[df['crossings'] == df['max_crossings']].sort_values("Snapshot", ascending=True).copy()
        satellites.drop(["Priority", "R_diff", "crossings", "max_crossings"], axis=1, inplace=True)
        
        satellites['delta_rel'] = pd.Series()
        satellites['stellar_mass'] = pd.Series()
        satellites['dark_mass'] = pd.Series()
        satellites['gas_mass'] = pd.Series()
        
        satellites['rh_stars_physical'] = pd.Series()
        satellites['rh_dm_physical'] = pd.Series()
        satellites['e_rh_stars_physical'] = pd.Series()
        satellites['e_rh_dm_physical'] = pd.Series()
        
        satellites['rh3D_stars_physical'] = pd.Series()
        satellites['rh3D_dm_physical'] = pd.Series()
    
        satellites['Mhl'] = pd.Series()
        
        satellites["sigma*"] = pd.Series()
        satellites["e_sigma*"] = pd.Series()
        
        satellites['Mdyn'] = pd.Series()
        satellites['e_Mdyn'] = pd.Series()

        satellites['LV'] = pd.Series()
        satellites['M/LV'] = pd.Series()

        snapshots_to_run = satellites["Snapshot"].unique()
    
        del df
        del mask
    
    
    kwargs = {
        "sf": args.simulation_folder,
        "pdir": args.particle_data_folder,
        "equiv": arbor.equivalence_table,
        "output": args.output,
        "R/Rvir_max": args.RRvir_max,
        "z_max": args.zmax,
        "code": args.code,
        "threads": args.threads
    }
    
    kwargs["min_snap"] = int(satellites["Snapshot"].min())
    kwargs["max_snap"] = int(satellites["Snapshot"].max()) 

    print(f"Selected sub_tree_ids to analyze: {sub_tree_ids.astype(int)}\n")
    print(f"Simulation data folder: {args.simulation_folder}")
    print(f"Master particle data folder: {args.particle_data_folder}")
    print(f"R/Rvir to start from: {args.RRvir_max}")
    print(f"Maximum redshift: {args.zmax}")
    print(f"Snapshot range in {kwargs['min_snap']}-{kwargs['max_snap']}")
    print(f"First and last files are: {arbor.find_file(kwargs['min_snap'])} and {arbor.find_file(kwargs['max_snap'])}")
    print(f"Output: {args.output}\n")
    print(f"Threads: {args.threads}\n")

    if os.path.isfile(args.output + "/" + f"{args.code}_Satellites.csv"):
        print(f"Previous ouput found... resuming operation from snapshot {np.min(snapshots_to_run)}")


    print(f"Sneak peak of satellites:")
    print(satellites[["Sub_tree_id", "Snapshot", "R/Rvir"]], "\n")
    


    for snap in snapshots_to_run:
        out = analyze_snapshot(arbor, satellites, snap, kwargs)
        print(out)
        gc.collect()
    
    print(f"Finished. Bye!")





