import os
import yt
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from unyt import unyt_array
from copy import deepcopy
from multiprocessing import Pool, cpu_count

import galexquared as gal
from galexquared import config
from galexquared.utils import particle_unbinding_fire


def parse_args():
    parser = argparse.ArgumentParser(description="Script that searchs for halos that contain galaxies. To reduce the computational cost, galaxies are only searched for at the Peak Mass of each halo in a given redshift and R/Rvir range (This is done to avoid contamination when close to the host galaxy).")
        
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')
    
    
    required.add_argument(
        "-i", "--input_file",
        type=str,
        help="Merger Tree file path. Must be csv. If ytree readable, the tree will be constructed.",
        required=True
    )
    required.add_argument(
        "-eq", "--equivalence_table",
        type=str,
        help="Equivalence table path. Must have correct formatting.",
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
        help="Location of particle data. Requred for selecting starry halos.",
        required=True
    )
    
    opt.add_argument(
        "--zrange",
        nargs=2,
        type=float,
        default=[-0.1, 8],
        help="Redshift range to select crosisng halos. (Default [-0.1, 5])"
    )
    opt.add_argument(
        "--mrange",
        nargs=2,
        type=float,
        default=[1E8, np.inf],
        help="Peak Mass range to select crosisng halos in Msun. (Default [1E8, inf])"
    )
    opt.add_argument(
        "--secondary",
        action="store_true",
        default=False,
        help="Whether to take into account secondary halos (halos about to merge). (Default: False)"
    )
    
    opt.add_argument(
        "--stellarnmin",
        type=int,
        default=6,
        help="Minimum number of stellar particles for selection. (Default 6)"
    )
    opt.add_argument(
        "--RRvir_range",
        nargs="*",
        type=float,
        default=[1.5, np.inf],
        help="R/Rvir range to select. (Default [1.5, np.inf])"
    )
    
    
    opt.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose output. (Default False)"
    )
    
    return parser.parse_args()

    
def process_snapshot(snapnum):
    global df_filtered_final, snapequiv, pdir, args  

    if args.verbose:
        print(f"\n##########  {snapnum}  ##########")

    filtered_halos = df_filtered_final[df_filtered_final['Snapshot'] == snapnum]
    fn = snapequiv[snapequiv['snapshot'] == snapnum]['snapname'].values[0]

    if args.verbose:
        print(f"####  {fn}  #####")

    ds = config.loader(pdir + fn)
    results = []

    for index in filtered_halos.index:
        filtered_node = filtered_halos.loc[[index]]
        redshift = filtered_node['Redshift'].values[0]

        center = unyt_array(filtered_node[['position_x', 'position_y', 'position_z']].values[0] / (1+redshift), 'kpc')
        rvir = unyt_array(filtered_node['virial_radius'].values[0] / (1+redshift), 'kpc')
        cen_vel = unyt_array(filtered_node[['velocity_x', 'velocity_y', 'velocity_z']].values[0], 'km/s')
        vrms = unyt_array(filtered_node['vrms'].values[0],'km/s')
        vmax = unyt_array(filtered_node['vmax'].values[0],'km/s')

        halo_params = {'center': center, 'center_vel': cen_vel, 'rvir': rvir, 'vmax': vmax, 'vrms': vrms}
        sp = ds.sphere(center, rvir)

        _, mask_stars, _ = particle_unbinding_fire(
            sp['stars', 'coordinates'].to("kpc"),
            sp['stars', 'mass'].to("Msun"),
            sp['stars', 'velocity'].to("km/s"),
            sp['stars', 'index'],
            halo_params,
            verbose=args.verbose
        )

        hasstars = np.count_nonzero(mask_stars) != 0
        hassgal = np.count_nonzero(mask_stars) >= args.stellarnmin

        if hasstars:
            new_row = {
                col: filtered_node.iloc[0][col] if col in filtered_node.columns else np.nan
                for col in ["Sub_tree_id", "Snapshot", "Redshift", "Time", "mass", "peak_mass", "N_stars", "stellar_mass", "Halo_at_z0", "R/Rvir"]
            }
            new_row['N_stars'] = int(np.count_nonzero(mask_stars))
            new_row['stellar_mass'] = float(sp['stars', 'mass'][mask_stars].sum().in_units("Msun").value)
            results.append(new_row)

            print("\n", sp['stars', 'mass'][mask_stars].sum().in_units("Msun").value)
            print(f"uid: {filtered_node['uid'].values}, subtree: {filtered_node['Sub_tree_id'].values}.")
            print(f"Stars found: {hasstars}.")
            print(f"Galaxies found: {hassgal}. Np: {np.count_nonzero(mask_stars)}.")  

    ds.close()
    return results


if __name__ == "__main__":
    args = parse_args()


    arbor = gal.MergerTree(args.input_file)
    arbor.set_equivalence(args.equivalence_table)

    CompleteTree = deepcopy(arbor.CompleteTree)
    if "peak_mass" not in CompleteTree.columns:
        CompleteTree = arbor._compute_Mpeak(CompleteTree)

    zlow, zhigh = args.zrange
    mlow, mhigh = args.mrange
    keep_secondary = args.secondary
    Rvir_low, Rvir_high = args.RRvir_range

    df_filtered_initial = CompleteTree[
        (CompleteTree["Redshift"] > zlow)   & (CompleteTree["Redshift"] < zhigh)   &
        (CompleteTree["R/Rvir"] > Rvir_low) & (CompleteTree["R/Rvir"] < Rvir_high) &
        ((CompleteTree["Secondary"] == False) | (CompleteTree["Secondary"] == keep_secondary))
    ]

    peak_idx = df_filtered_initial.groupby("Sub_tree_id")["mass"].idxmax()
    df_peak = df_filtered_initial.loc[peak_idx].copy()
    df_filtered_final = df_peak[(df_peak["mass"] > mlow) & (df_peak["mass"] < mhigh)]
    df_filtered_final = df_filtered_final.sort_values(by=['Snapshot', 'Sub_tree_id'], ascending=[True, True])

    print(df_filtered_final)

    gal.config.code = args.code
    snapequiv = arbor.equivalence_table
    pdir = args.particle_data_folder

    snapunique = np.unique(df_filtered_final['Snapshot'].values)

    num_workers = min(min(cpu_count(), 8), len(snapunique))  
    with Pool(num_workers) as pool:
        results_list = list(tqdm(pool.imap(process_snapshot, snapunique), total=len(snapunique), desc="Processing snapshots in parallel"))

    starry_halos = pd.DataFrame([row for results in results_list for row in results if results])

    starry_halos.to_csv(f"Starry_halos_{args.code}.csv", index=False)
    print(f"Finished. Bye!")




