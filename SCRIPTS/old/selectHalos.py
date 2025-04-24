import os
import yt
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from unyt import unyt_array, unyt_quantity

import galexquared as gal
from galexquared import config
from galexquared.utils import particle_unbinding_fire

parser = argparse.ArgumentParser(description="Script that uses galexquared's MergerTree class to select Crossing halos at R/Rvir values according to specified criteria for redshift, mass and merging history.")
    
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
    "--main_RRvir",
    type=float,
    help="Main value of R/Rvir to select crossing halos.",
    required=True
)


opt.add_argument(
    "--zrange",
    nargs=2,
    type=float,
    default=[-0.1, 5],
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
    "--select_stellar",
    action="store_true",
    default=False,
    help="Whether to trim the list for halos with stellar mass content. See stellarrange and stellarnmin. (Default: False)"
)
opt.add_argument(
    "--stellarrange",
    nargs=2,
    type=float,
    default=[4E5, np.inf],
    help="Stellar mass range to select crosisng halos in Msun. (Default [4E5, inf])"
)
opt.add_argument(
    "--stellarnmin",
    type=int,
    default=6,
    help="Minimum number of stellar particles for selection. (Default 6)"
)
opt.add_argument(
    "--extra_RRvir",
    nargs="*",
    type=float,
    default=None,
    help="Extra R/Rvir values to trace back the crossing halos. Must be bigger than main R/Rvir. (Default None)"
)

opt.add_argument(
    "-c", "--code",
    type=str,
    default=None,
    help="Code of the simulation. Requred for selecting starry halos.",
)
opt.add_argument(
    "-pd", "--particle_data_folder",
    type=str,
    default=None,
    help="Location of particle data. Requred for selecting starry halos.",
)


opt.add_argument(
    "-o", "--output",
    type=str,
    default="./selected_halos",
    help="Location of particle data. Requred for selecting starry halos.",
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

if not args.input_file.endswith(".csv"):
    print(f"First we need to construct the merger trees. This will take a while...")
    arbor.construct_df_forest()


print(f"Selecting crossing haloes at {args.main_RRvir}. The provided constraints are:")
if args.select_stellar:
    print(f"Range of Dark mass: {args.mrange}")
    print(f"Range of Redshifts: {args.zrange}")
    print(f"Range of Stellar mass: {args.stellarrange}")
    print(f"Minimum of Stellar Particles: {args.stellarnmin}")
    print(f"Keep Secondary: {args.secondary}")
else:
    print(f"Range of Dark mass: {args.mrange}")
    print(f"Range of Redshifts: {args.zrange}")
    print(f"Keep Secondary: {args.secondary}")

CrossingSats_Rmain, _ = arbor.select_halos(Rvir=args.main_RRvir, mass=args.mrange, redshift=args.zrange, keep_secondary=args.secondary, Rvir_tol=0.2*args.main_RRvir)
CrossingSats_Rmain.sort_values(["Sub_tree_id", "Snapshot"], ascending=[True, False])  

if args.verbose and args.select_stellar:
    print(f"Trimming halos for those containing stars...")


if args.select_stellar:
    gal.config.code = args.code
    snapequiv = arbor.equivalence_table
    pdir = args.particle_data_folder
    
    CrossingSats_Rmain['has_stars'] = pd.Series()
    CrossingSats_Rmain['has_galaxy'] = pd.Series()
    CrossingSats_Rmain['delta_rel'] = pd.Series()
    CrossingSats_Rmain['stellar_mass'] = pd.Series()
    snapunique = np.unique(CrossingSats_Rmain['Snapshot'].values)
    
    for i in tqdm(range(len(snapunique)), desc=f"Finding stars... or not."):
        snapnum = snapunique[i]
        if args.verbose:
            print(f"\n##########  {snapnum}  ##########")
        filtered_halos  = CrossingSats_Rmain[(CrossingSats_Rmain['Snapshot'] == snapnum)]
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
            
            CrossingSats_Rmain.loc[index, 'has_stars'] = hasstars 
            CrossingSats_Rmain.loc[index, 'has_galaxy'] = hassgal
            CrossingSats_Rmain.loc[index, 'delta_rel'] = float(delta_rel)
            if hasstars:
                print(sp['stars','mass'][mask_stars].sum().in_units("Msun").value)
                print(f"\n uid: {filtered_node['uid'].values}, subtree: {filtered_node['Sub_tree_id'].values}.")
                print(f"Stars found: {hasstars}.")
                print(f"Galaxies found: {hassgal}. Np: {np.count_nonzero(mask_stars)}.")  
                CrossingSats_Rmain.loc[index, 'stellar_mass'] = sp['stars','mass'][mask_stars].sum().in_units("Msun").value
    
    
        ds.close()


    CrossingSats_Rmain.to_csv(args.output + f"_{args.code}" + f"/{args.code}_AllCrossingHalos_RRvir{args.main_RRvir}.csv", index=False)    
    satellites = CrossingSats_Rmain[(CrossingSats_Rmain['stellar_mass'] >= 4E5) & (CrossingSats_Rmain['has_galaxy'])]

else:
    satellites = CrossingSats_Rmain


satellites.to_csv(args.output + f"_{args.code}" + f"/{args.code}_Satellites_RRvir{args.main_RRvir}.csv", index=False)

print(f"\nNumber of Satellites found at R/Rvir {args.main_RRvir}: {satellites.shape} \n")
print(satellites)


if bool(args.extra_RRvir):    
    
    tracedback_satellites = arbor.traceback_halos(Rvirs=args.extra_RRvir, halodf=satellites)
    
    for RRvir, df in tracedback_satellites.items():
        if args.select_stellar:
            df.to_csv(args.output + f"_{args.code}" + f"/{args.code}_Satellites_RRvir{RRvir}.csv", index=False)
        else:
            df.to_csv(args.output + f"_{args.code}" + f"/{args.code}_AllCrossingHalos_RRvir{RRvir}.csv", index=False)

        print(f"\nNumber of Satellites found at R/Rvir {RRvir}: {df.shape} \n")
        print(df)





print(f"Finished. Bye!")

