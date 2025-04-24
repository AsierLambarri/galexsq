import os
import yt
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from unyt import unyt_array, unyt_quantity
from copy import deepcopy

import galexquared as gal
from galexquared import config
from galexquared.utils import particle_unbinding_fire

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
    help="R/Rvir range to select. (Default None)"
)


opt.add_argument(
    "--verbose",
    type=bool,
    default=False,
    help="Verbose output. (Default False)"
)






args = parser.parse_args()

arbor = gal.MergerTree(args.input_file)
arbor.set_equivalence(args.equivalence_table)




CompleteTree = deepcopy(arbor.CompleteTree)
zlow, zhigh, mlow, mhigh, keep_secondary, Rvir_low, Rvir_high = -1, 1E4, 1E8, 1E20, False, 2, 1E5

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

cols = ["Sub_tree_id", "Snapshot", "Redshift", "Time", "mass", "peak_mass", "N_stars", "stellar_mass", "Halo_at_z0", "R/Rvir"]
starry_halos = pd.DataFrame(columns=cols)


snapunique = np.unique(df_filtered_final['Snapshot'].values)

for i in tqdm(range(len(snapunique)), desc=f"Finding stars... or not."):
    snapnum = snapunique[i]
    if args.verbose:
        print(f"\n##########  {snapnum}  ##########")
    filtered_halos  = df_filtered_final[(df_filtered_final['Snapshot'] == snapnum)]
    fn = snapequiv[snapequiv['snapshot'] == snapnum]['snapname'].values[0]
    if args.verbose:
        print(f"####  {fn}  #####")
    ds = config.loader(pdir + fn)

    
    for index in filtered_halos.index:
        filtered_node = filtered_halos.loc[[index]]
        redshift = filtered_node['Redshift'].values[0]
        
        center, rvir = unyt_array(filtered_node[['position_x', 'position_y', 'position_z']].values[0] / (1+redshift), 'kpc'), unyt_array(filtered_node['virial_radius'].values[0] / (1+redshift), 'kpc')
        cen_vel = unyt_array(filtered_node[['velocity_x', 'velocity_y', 'velocity_z']].values[0], 'km/s')
        vrms, vmax = unyt_array(filtered_node['vrms'].values[0],'km/s'), unyt_array(filtered_node['vmax'].values[0],'km/s')
        halo_params = {'center': center,
                       'center_vel': cen_vel,
                       'rvir': rvir,
                       'vmax': vmax,
                       'vrms': vrms
                      }
        sp = ds.sphere(center, rvir)
        _, mask_stars, delta_rel = particle_unbinding_fire(
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
            new_row = {}
            for col in cols:
                if col in filtered_node.columns:
                    new_row[col] = filtered_node.iloc[0][col]
                else:
                    new_row[col] = np.nan

            new_row['N_stars'] = int(np.count_nonzero(mask_stars))
            new_row['stellar_mass'] = float(sp['stars', 'mass'][mask_stars].sum().in_units("Msun").value)

            starry_halos = pd.concat([starry_halos, pd.DataFrame([new_row])], ignore_index=True)
            
            print("\n", sp['stars','mass'][mask_stars].sum().in_units("Msun").value)
            print(f"uid: {filtered_node['uid'].values}, subtree: {filtered_node['Sub_tree_id'].values}.")
            print(f"Stars found: {hasstars}.")
            print(f"Galaxies found: {hassgal}. Np: {np.count_nonzero(mask_stars)}.")  
            df_filtered_final.loc[index, 'stellar_mass'] = sp['stars','mass'][mask_stars].sum().in_units("Msun").value


    ds.close()


starry_halos.to_csv(f"Starry_halos_{args.code}.csv", index=False)

print(f"Finished. Bye!")



