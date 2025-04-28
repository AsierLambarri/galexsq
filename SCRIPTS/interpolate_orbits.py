import gc
import time
import argparse
import numpy as np
import pandas as pd

import astropy.units as u

import galexquared as gal
from galexquared.class_methods import load_ftable
from galexquared import PhaseSpaceInstant, Orbis, GetICSList, CartesianOrbit


def parse_args():
    parser = argparse.ArgumentParser(description="Script to interpolate orbits using Orbis.")
        
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
    required.add_argument(
        "-c", "--code",
        type=str,
        help="Code of the simulation. Requred for selecting starry halos.",
        required=True
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


    
args = parse_args()




gal.config.code = args.code

MT = gal.MergerTree(args.input_file)
MT.set_equivalence(args.equivalence_table)
pdir = args.simulation_folder
candidates = pd.read_csv(args.subtree_ids)


units = [u.Gyr, u.kpc, u.Msun, u.deg, u.km/u.s]


if args.code == "ART":
    subtree_list = [
                    67126, 90652, 104106, 151846, 162050,
                    167667, 220710, 239354, 490284
                   ]
    #213, 236, 239, 262, 277, 987, 7788,
    #                14642, 14757, 23533, 29557, 29952, 30302, 32830,
    #                32806, 34701, 40092, 40554, 40578, 43679,
    #                53092, 54928, 56181, 58973, 59565, 59693,
    orbit_interpolator = Orbis(pot_from_sim=MT.PrincipalLeaf, pdir=pdir, file_table=args.equivalence_table, units=units)


else:
    #subtree_list = [410, 451, 899, 1059, 1071, 1199, 1209, 1274, 1304, 1311,
    #                1463, 1473, 17608, 28974, 31608, 42065, 50789, 77801,
    #                92193, 165594, 183069, 186762, 214096, 312003, 
    #                346178,  620178, 4273404
    #               ]
    subtree_list = [620179, 4273404]
    equiv = load_ftable(args.equivalence_table)
    equiv["snapshot"] = equiv.index
    orbit_interpolator = Orbis(pot_from_sim=MT.PrincipalLeaf, pdir=pdir, file_table=equiv, units=units)





for subtree in subtree_list:
    st = time.time()
    
    print(f"Initializing interpolation for SUBTREE {subtree}")
    
    infall_tree = MT.infall_from([subtree], 1.2, keep_ocurrences="first")
    
    max_radius = max(3, min(1.5 * args.RRvir_max, 1.5 * infall_tree["R/Rvir"].values.max()) )
    orbit_interpolator.rvir_factor = max_radius
    
    candidate_tree = MT.infall_from(
        [subtree], 
        args.RRvir_max, 
        redshift=[-1, args.zmax], 
        keep_ocurrences="first"
    )
    
    print(candidate_tree[["Halo_ID", "Sub_tree_id","Snapshot", "Redshift", "Time", "R/Rvir"]], "\n", f"Length of the tree is: {candidate_tree.shape}")
    ics_list = GetICSList(MT, subtree, candidate_tree["Snapshot"].unique(), units)
    
    full_orbit = orbit_interpolator.interpolate(ics_list, verbose=True)
    
    
    x,y,z = full_orbit.xyz.value.T[:, 0], full_orbit.xyz.value.T[:, 1], full_orbit.xyz.value.T[:, 2]
    vx,vy,vz = full_orbit.vxyz.value.T[:, 0], full_orbit.vxyz.value.T[:, 1], full_orbit.vxyz.value.T[:, 2]
    t = full_orbit.t.value
    
    data = {
        "Time" : t,
        
        "position_x": x,
        "position_y": y,
        "position_z": z,
        
        "velocity_x": vx,
        "velocity_y": vy,
        "velocity_z": vz
    }
    pd.DataFrame.from_dict(data).to_csv(args.output + "/" + f"Interpolated_orbit_{int(subtree)}.csv", index=False)
    
    
    gc.collect()
    orbit_interpolator.clean_orbits()
    
    del data, x, y, z, vx, vy, vz, t, full_orbit, ics_list, candidate_tree

    ft = time.time()
    
    print(f"\n{ft-st} seconds taken to interpolate SUBTREE {subtree}")
    print("\n\n")






    