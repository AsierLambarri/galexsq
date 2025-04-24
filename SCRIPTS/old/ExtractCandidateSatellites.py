import os
import yt
import sys
import yaml
import shutil
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from lmfit import Model
from limepy import limepy
from copy import copy, deepcopy
from astropy.table import Table
from scipy.stats import binned_statistic
from unyt import unyt_array, unyt_quantity
import matplotlib.pyplot as plt
import smplotlib
import pprint
pp = pprint.PrettyPrinter(depth=4, width=10000)


from lib.loaders import load_ftable, load_halo_rockstar
from lib.mergertrees import compute_stars_in_halo, bound_particlesBH, bound_particlesAPROX
from lib.galan import half_mass_radius, density_profile, velocity_profile, refine_center, NFWc, random_vector_spherical, LOS_velocity

def clear_directory(directory_path):
    if not os.path.exists(directory_path):
        raise Exception("Directory does not exist.")
        
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)  
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    return None


def isolate_sequences(snap_numbers, index_list):
    differences = np.diff(snap_numbers)
    
    jump_indices = np.where(differences != 1)[0]
    
    sequences = []
    indices = []
    start_idx = 0
    for idx in jump_indices:
        sequences.append(snap_numbers[start_idx:idx + 1])
        indices.append(index_list[start_idx:idx + 1])
        start_idx = idx + 1
    
    sequences.append(snap_numbers[start_idx:])
    indices.append(index_list[start_idx:])
    
    return sequences, indices





parser = argparse.ArgumentParser(description='This script extracts suitable candidate galaxy information from a given simulation+Rockstar-ConsistentTrees dataset. The script finds candidate satelite galaxies as those at R/Rvir=1, and fitting the redshift and DM-mass criteria specified in the config yaml file.')
    
parser.add_argument('file', metavar='file.yml', type=str, help='Path to config yaml file. It must contain following entries: -dir_snapshots: snapshot directory, -RCT_paths: path RCT files for main and satellite trees, -snapequiv_path: path to equivalence between RCt snapshot numbers and snapshot ids. More information on the example config.yaml file that is provided.')
parser.add_argument('--verbose', action='store_true', help='Display while running for debugging or later visual inspection.')


args = parser.parse_args()






with open(args.file, "r") as file:
    config_yaml = yaml.safe_load(file)

code = config_yaml['code']
outdir = config_yaml['output_dir'] + "_" + code

skip_candidates = False
if 'candidates' in list(config_yaml.keys()):
    if config_yaml['candidates'] is not None:
        skip_candidates = True
    else:
        if os.path.isdir(outdir):
            clear_directory(outdir)
            os.mkdir(outdir + "/particle_data")
            os.mkdir(outdir + "/los_vel")
        else:
            os.mkdir(outdir)
            os.mkdir(outdir + "/particle_data")
            os.mkdir(outdir + "/los_vel")
else:
    if os.path.isdir(outdir):
        clear_directory(outdir)
        os.mkdir(outdir + "/particle_data")
        os.mkdir(outdir + "/los_vel")
    else:
        os.mkdir(outdir)
        os.mkdir(outdir + "/particle_data")
        os.mkdir(outdir + "/los_vel")




pdir = config_yaml['dir_snapshots']

snapequiv = load_ftable(config_yaml['snapequiv_path'])

MainTree = pd.read_csv(config_yaml['RCT_paths']['main'])
CompleteTree = deepcopy(MainTree)
for main_or_sat, path in config_yaml['RCT_paths'].items():
    if main_or_sat != 'main':   
        sat = pd.read_csv(path)
        CompleteTree = pd.concat([CompleteTree, sat])


mlow, mhigh = (
    float(config_yaml['constraints']['mlow']),
    float(config_yaml['constraints']['mhigh']) if str(config_yaml['constraints']['mhigh']) != "inf" else np.inf
)
stellar_low, stellar_high = (
    float(config_yaml['constraints']['stellar_low']),
    float(config_yaml['constraints']['stellar_high']) if str(config_yaml['constraints']['stellar_high']) != "inf" else np.inf
)
zlow, zhigh = (
    float(config_yaml['constraints']['zlow']),
    float(config_yaml['constraints']['zhigh']) if str(config_yaml['constraints']['zhigh']) != "inf" else np.inf
)
Rvir_extra = list([ R for R in config_yaml['Rvir'] if R != 1])


constrainedTree_R1 = CompleteTree[(np.abs(CompleteTree['R/Rvir'] - 1) < 0.10) & (mlow <= CompleteTree['mass']) & (CompleteTree['mass'] <= mhigh) &
                                  (zlow <= CompleteTree['Redshift']) & (CompleteTree['Redshift'] <= zhigh)].sort_values(by=['Snapshot', 'Sub_tree_id'], ascending=[True, True])

unique_subtrees = np.unique(constrainedTree_R1['Sub_tree_id'])

CrossingSats_R1 = pd.DataFrame(columns=constrainedTree_R1.columns)

if skip_candidates == False:

    verbose = False
    for subtree in tqdm(unique_subtrees):
        crossing_subtree = constrainedTree_R1[constrainedTree_R1['Sub_tree_id'] == subtree]
        crossing_subtree.sort_values("Snapshot")
    
        crossings_list, indexes_list = isolate_sequences(crossing_subtree['Snapshot'].values, crossing_subtree.index)
        
        cross_redshifts = []
        cross_indexes = []
        for cross_snapnums, indexes in zip(crossings_list, indexes_list):
            crossing_subtree_subseq = crossing_subtree.loc[indexes]
            delta_RRvir = crossing_subtree_subseq['R/Rvir'].values - 1
            redshifts = crossing_subtree_subseq['Redshift'].values
    
    
            if True in (delta_RRvir > 0):
                RRvir_plus = delta_RRvir[delta_RRvir > 0]
                RRvir_plus_minimum = RRvir_plus.min()
                ipos_min = np.where(delta_RRvir == RRvir_plus_minimum)[0][0]       
                cross_redshifts.append(redshifts[ipos_min])
                cross_indexes.append(indexes[ipos_min])
            else:
                pass
        
        crossings_df = crossing_subtree.loc[cross_indexes]
        CrossingSats_R1 = pd.concat([CrossingSats_R1, crossings_df])
        
        if verbose:
            print(f"\nSUBTREE: {subtree}. First Crossing Redshift: {cross_redshifts}, {cross_indexes}")
            pp.pprint(crossing_subtree[['Halo_ID','Snapshot','Redshift','uid','mass','R/Rvir']])
            print(crossings_list, indexes_list)
            print(crossing_subtree.loc[cross_indexes][['Halo_ID','Snapshot','Redshift','uid','mass','R/Rvir']])
    
    
    CrossingSats_R1['has_stars'] = pd.Series()
    CrossingSats_R1['has_galaxy'] = pd.Series()
    CrossingSats_R1['delta_rel'] = pd.Series()
    CrossingSats_R1['stellar_mass'] = pd.Series()
    
    
    for snapnum in np.unique(CrossingSats_R1['Snapshot'].values):
        filtered_halos  = CrossingSats_R1[(CrossingSats_R1['Snapshot'] == snapnum)]
        fn = snapequiv[snapequiv['snapid'] == snapnum]['snapshot'].value[0]
        ds = yt.load(pdir + fn)
        
        print(f"\n##########  {snapnum}  ##########")
        
        for index in filtered_halos.index:
            filtered_node = filtered_halos.loc[[index]]
            istars, mask_stars, sp, delta_rel = compute_stars_in_halo(filtered_node, ds, verbose=True)
            
            hasstars = np.count_nonzero(mask_stars) != 0
            hassgal = np.count_nonzero(mask_stars) >= 6
            
            CrossingSats_R1['has_stars'].loc[index] = hasstars 
            CrossingSats_R1['has_galaxy'].loc[index] = hassgal
            CrossingSats_R1['delta_rel'].loc[index] = float(delta_rel)
            if hasstars:
                print(sp['stars','particle_mass'][mask_stars].sum().in_units("Msun").value)
                CrossingSats_R1['stellar_mass'].loc[index] = sp['stars','particle_mass'][mask_stars].sum().in_units("Msun").value
    
            if hassgal:
                np.savetxt(outdir + "/" + f"particle_data/stars_{int(filtered_node['Sub_tree_id'].values)}.{int(snapnum)}.pids", istars, fmt="%i")            
    
    
    
            print(f"\n uid: {filtered_node['uid'].values}, subtree: {filtered_node['Sub_tree_id'].values}.")
            print(f"Stars found: {hasstars}.")
            print(f"Galaxies found: {hassgal}. Np: {np.count_nonzero(mask_stars)}.")  

        ds.close()
    
    
    CrossingSats_R1.to_csv(outdir + "/" + f"candidate_satellites_{code}.csv", index=False)
    CrossingSats_R1 = CrossingSats_R1[CrossingSats_R1['has_galaxy']]
    
    arrakihs_v2 = CrossingSats_R1[ (CrossingSats_R1['stellar_mass'] <= stellar_high) & (stellar_low <= CrossingSats_R1['stellar_mass']) ].sort_values("Redshift", ascending=False)
    arrakihs_v2 = arrakihs_v2[~arrakihs_v2.duplicated(['Sub_tree_id'], keep='first').values].sort_values("Sub_tree_id")
    arrakihs_v2.to_csv(outdir + "/" + f"ARRAKIHS_Infall_CosmoV18_{code}.csv", index=False)

else:
    print(f"ARRAKIHS table has been found!")
    arrakihs_v2 = pd.read_csv(config_yaml['candidates'])

arrakihs_v2['rh_stars_physical'] = pd.Series()
arrakihs_v2['rh_dm_physical'] = pd.Series()

arrakihs_v2['rhp_stars_physical'] = pd.Series()
arrakihs_v2['rhp_dm_physical'] = pd.Series()

arrakihs_v2['Mdyn'] = pd.Series()

arrakihs_v2['sigma*'] = pd.Series()
arrakihs_v2["e_sigma*"] = pd.Series()

N = 16
nx = int(np.ceil(np.sqrt(N)))
ny = int(N/nx)


def Gaussian(v, A, mu, sigma):
    return A * np.exp( -(v - mu)**2 / (2*sigma**2) )
    
gaussianModel = Model(Gaussian,independent_vars=['v'], nan_policy="omit")


for uid in arrakihs_v2['uid'].values:
    halo, sp, ds = load_halo_rockstar(arrakihs_v2, snapequiv, pdir, uid=uid)
    halocen, halovir = sp.center.in_units("kpccm"), sp.radius.in_units("kpccm")

    subid = int(halo['Sub_tree_id'].values[0])
    snap = int(halo['Snapshot'].values[0])

    st_pos = sp['stars','particle_position'].in_units("kpc")
    st_vel = sp['stars','particle_velocity'].in_units("km/s")
    st_mass = sp['stars','particle_mass'].in_units("Msun")
    st_ids = sp['stars', 'particle_index'].value

    dm_pos = sp['darkmatter','particle_position'].in_units("kpc")
    dm_vel = sp['darkmatter','particle_velocity'].in_units("km/s")
    dm_mass = sp['darkmatter','particle_mass'].in_units("Msun")
    dm_ids = sp['darkmatter', 'particle_index'].value

    bound, most_bound, iid, bound_iid = bound_particlesAPROX(dm_pos, dm_vel, dm_mass, ids=dm_ids, cm=halocen.in_units("kpc"), vcm=unyt_array(halo[['velocity_x', 'velocity_y', 'velocity_z']].values[0], 'km/s'), refine=False)
    
    print(f"Sub id: {subid}")
    print(f"-" * len(f"Sub id: {subid}" + "\n"))
    print(f"Total mass: {dm_mass[bound].sum()}.")
    print(f"CM is: {np.average(dm_pos[most_bound], axis=0, weights=dm_mass[most_bound]).in_units('kpccm')}")
    print(f"should be: {halocen}")
    
    np.savetxt(outdir + "/" + f"particle_data/dm_{int(subid)}.{int(snap)}.pids", np.array([iid, np.isin(iid, bound_iid) * 1]).T, fmt="%i", delimiter=",")            

    st_bids = np.loadtxt(outdir + "/" + f"particle_data/stars_{subid}.{snap}.pids")
    dm_bpts = np.loadtxt(outdir + "/" + f"particle_data/dm_{subid}.{snap}.pids", delimiter=",")
    dm_bids =  dm_bpts[:,0]

    dm_mask =  np.isin(dm_ids, dm_bids)
    st_mask =  np.isin(st_ids, st_bids)


    dm_rh, _ = half_mass_radius(dm_pos[dm_mask], dm_mass[dm_mask])
    st_rh, _ = half_mass_radius(st_pos[st_mask], st_mass[st_mask])
    arrakihs_v2['rh_dm_physical'].loc[halo.index] = dm_rh.in_units('kpc').value
    arrakihs_v2['rh_stars_physical'].loc[halo.index] = st_rh.in_units('kpc').value
    
    dm_center0 = refine_center(dm_pos[dm_mask], dm_mass[dm_mask], method="hm", delta=1E-2, m=2, mfrac=0.5)
    st_center0 = refine_center(st_pos[st_mask], st_mass[st_mask], method="hm", delta=1E-2, m=2, mfrac=0.5)
    dm_rhp, dm_center0 = np.nanmean(half_mass_radius(dm_pos[dm_mask, [0,1]], dm_mass[dm_mask], center=dm_center0[[0,1]])[0],
                                    half_mass_radius(dm_pos[dm_mask, [0,2]], dm_mass[dm_mask], center=dm_center0[ [0,2]])[0],
                                    half_mass_radius(dm_pos[dm_mask,[1,2]], dm_mass[dm_mask], center=dm_center0[[1,2]])[0]
                                   )
    st_rhp, st_center0 = np.nanmean(half_mass_radius(st_pos[st_mask, [0,1]], st_mass[st_mask], center=st_center0[[0,1]])[0],
                                    half_mass_radius(st_pos[st_mask, [0,2]], st_mass[st_mask], center=st_center0[ [0,2]])[0],
                                    half_mass_radius(st_pos[st_mask, [1,2]], st_mass[st_mask], center=st_center0[[1,2]])[0]
                                   )

    arrakihs_v2['rhp_dm_physical'].loc[halo.index] = dm_rhp.in_units('kpc').value
    arrakihs_v2['rhp_stars_physical'].loc[halo.index] = st_rhp.in_units('kpc').value

    dm_mdynMask = np.linalg.norm(dm_pos - st_center0, axis=1) <= st_rh
    st_mdynMask = np.linalg.norm(st_pos - st_center0, axis=1) <= st_rh

    dm_mdyn_cont = dm_mass[dm_mdynMask].sum()
    st_mdyn_cont = st_mass[st_mdynMask].sum()
    arrakihs_v2['Mdyn'].loc[halo.index] = dm_mdyn_cont + st_mdyn_cont


    print(f"Stellar mass: {st_mass[st_mask].sum():.3e}. rH: {st_rh:.2f}. Half mass:  {st_mass[st_mdynMask].sum():.3e}. Error: {(st_mdyn_cont/st_mass[st_mask].sum() - 0.5):.3e}")
    print(f"DM mass: {dm_mass[dm_mask].sum():.3e}. Rockstar mass: {halo['mass'].values[0]:.3e}. rH: {dm_rh:.2f}. Mass inside st_rH:  {dm_mdyn_cont/dm_mass[dm_mask].sum():.3e}")
    print(f"Mdyn: {(dm_mdyn_cont + st_mdyn_cont):.3e}")



    lines_of_sight = random_vector_spherical(N=N)
    
    center_stars = refine_center(st_pos[st_mask], st_mass[st_mask], method="iterative", delta=1E-2, m=2, nmin=20)
    if center_stars['converged'] is False or st_mass[st_mask].sum().value < 2E6:
        center_stars = refine_center(st_pos[st_mask], st_mass[st_mask], method="hm", delta=1E-2, m=2, mfrac=0.5)
    
    center = unyt_array(center_stars['center'], 'kpc')    
    print(f"Center of stars: {center}")
    sp_stars = ds.sphere(center, (0.64, 'kpc'))
    cm_vel = np.average(sp_stars['stars','particle_velocity'], axis=0, weights=sp_stars['stars', 'particle_mass']).in_units("km/s")

    sigma_los = []
    e_sigma_los = []
    
    fig, axes = plt.subplots(nx, ny, figsize=(ny * 8, nx * 8))
    plt.subplots_adjust(wspace=0.13, hspace=0.13)
    
    for i, los in enumerate(lines_of_sight):
        cyl = ds.disk(center, los, radius=(1, "kpc"), height=(np.inf, "kpc"), data_source=sp)
        pvels = LOS_velocity(cyl['stars', 'particle_velocity'].in_units("km/s") - cm_vel, los)

        fv_binned, binedges, _ = binned_statistic(pvels, np.ones_like(pvels), statistic="count", bins=np.histogram_bin_edges(pvels, bins="fd"))
        bincenters = 0.5 * (binedges[1:] + binedges[:-1])

        params = gaussianModel.make_params(A={'value': 1E3, 'min': 0, 'max': 1E10, 'vary': True},
                                           mu={'value': 0, 'min': -50, 'max': 50, 'vary': True},
                                           sigma={'value': 50, 'min': 0, 'max': 2E2, 'vary': True}
                                          )
        print(len(pvels), pvels, len(fv_binned), fv_binned, bincenters)
        result = gaussianModel.fit(fv_binned, v=bincenters, params = params)
        
        sigma_los.append(result.params['sigma'].value)
        e_sigma_los.append(result.params['sigma'].stderr)

        ax = axes.flatten()[i]
        v = np.linspace(-250, 250, 1000)
        evals = result.eval(v=v)
        
        ax.set_title(f"sigma={result.params['sigma'].value:.1f} ± {result.params['sigma'].stderr:.2f} km/s in L.O.S.={np.round(los,2)}")
        ax.text(0, 1.005 * fv_binned.max(), r"$N_{max}=$"+f"{fv_binned.max()}", ha="center", va="bottom", fontsize="small", color="black")
        ax.hist(pvels, bins=binedges)
        ax.axvline(0, color="blue")
        ax.plot(v, evals, color="red")
        
        ax.set_xlim(-230, 230)
        ax.set_yticklabels([])

        if i >= N-ny:
            ax.set_xlabel(r"$\sigma_{*, LOS}$ [km/s]")

    plt.savefig(outdir + "/los_vel" + f"/{subid}.{snap}.png")
    plt.close()

    sigma_los = np.array(sigma_los)
    e_sigma_los = np.array(e_sigma_los)

    arrakihs_v2['sigma*'].loc[halo.index] =  np.mean(sigma_los)  
    arrakihs_v2['e_sigma*'].loc[halo.index] = np.maximum(np.std(sigma_los), np.sqrt(1 / np.sum(1 / e_sigma_los**2)))

    print(f"Mean sigma LOS = {np.mean(sigma_los):.1f}, std sigma LOS = {np.std(sigma_los):.2f}, ext_sigma sigma LOS = {np.sqrt(1 / np.sum(1 / e_sigma_los**2)):.2f}")

    sp.clear_data()

arrakihs_v2.to_csv(outdir + "/" + f"ARRAKIHS_Infall_CosmoV18_{code}.csv", index=False)



if config_yaml['comparison_plot']:
    from astropy.table import Table, join, Column, conf
    from cmdstanpy import CmdStanModel


    candidates = Table.read(outdir + "/" + f"ARRAKIHS_Infall_CosmoV18_{code}.csv", format="csv")
    mccon = Table.read('./DwarfProperties/LocalGroup/McConnachie_properties_complete.dat', format = "ascii.fixed_width", delimiter="\t")
    mccon2 = Table.read('./DwarfProperties/LocalGroup/McConnachie_2012.csv')
    globc = Table.read("./DwarfProperties/GC_Harris1997.dat", format="ascii.csv",delimiter=";", comment="#")
    
    my_model = CmdStanModel(stan_file="linear_noErrors_robust.stan", cpp_options={'STAN_THREADS':'true'})

    dwarf_data = {
    "N": len(mccon['VMag'].value),
    "M": len(candidates['Mdyn'].value),
    "x": np.log10(mccon['Mdyn'].value*1E6).tolist(),
    "y": mccon['VMag'].value.tolist(),
    'xeval' : np.log10(candidates['Mdyn'].value).tolist()
    }
    
    fit = my_model.sample(data=dwarf_data, chains=4, iter_sampling=5000, show_console=False)
    df = fit.draws_pd()
    md = np.linspace(1E4, 2E10, 1000)
    trans_dict = {'dIrr/dSph': "p",
                  'dIrr': "*",
                  'dE/dSph': "h",
                  'cE': "D",
                  "Sc": "P",
                  'Irr': "s", 
                  '????': "x",
                  '(d)Irr?': "s",
                  'dSph' : "o",
                  'dSph?': "o"
                 }
    import matplotlib.gridspec as gridspec
    
    plt.rcParams['axes.linewidth'] = 1.1
    plt.rcParams['xtick.major.width'] = 1.1
    plt.rcParams['xtick.minor.width'] = 1.1
    plt.rcParams['ytick.major.width'] = 1.1
    plt.rcParams['ytick.minor.width'] = 1.1
    
    plt.rcParams['xtick.major.size'] = 7
    plt.rcParams['ytick.major.size'] = 7
        
    fig = plt.figure(figsize=(1.5*1.8*9,1.8*9))
    
    
    gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
    
    ax1 = fig.add_subplot(gs[0])  
    ax2 = fig.add_subplot(gs[1])  
    ax3 = fig.add_subplot(gs[4])  
    ax4 = fig.add_subplot(gs[3])  
    ax5 = fig.add_subplot(gs[2])  
    ax6 = fig.add_subplot(gs[5])  
    
    
    i = 0
    for key, value in trans_dict.items():
        morph_mccon = mccon2[mccon2['MType'] == key]
        
        if key != "dSph?" and key != "(d)Irr?":
            ax1.scatter([np.nan],[np.nan], marker=value, color="black", s=70, label=f"{key}") 
    
            
        if i==8:
            a1 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=70, label=f"Milky-Way group")
            a2 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=70, label=f"Andromeda gruop")
            a3 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70, label=f"Local group "+r"($D_{LG} \leq 1Mpc$)")
            a4 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70, label=f"Nearby galaxies "+r"($D_{LG} > 1Mpc$)")
        else:
            ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=70 )
            ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=70 )
            ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70 )
            ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70 ) 
    
    
        
        ax1.set_xscale("log")
        ax1.invert_yaxis()
        ax1.set_xlim(1,7000)
        ax1.set_ylim(0, -19)
        ax1.set_xlabel(r"$r_h$ [pc]")
        ax1.set_xticklabels([0, 1, 10, 100, 1000])
        ax1.set_ylabel(r"$M_V$ [mag]" )
        ax1.legend()
    
        i += 1
        
        
        
        
        y = morph_mccon['R2'] 
        x = morph_mccon['Mdyn'] * 1E6 
        ax3.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=70, )
        ax3.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=70, )
        ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70, )
        ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70, )
        
        ax3.set_ylabel(r"$r_h$ [pc]")
        ax3.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
        ax3.loglog()
        ax3.set_xlim(1E5, 1E10)
        ax3.set_ylim(ax1.get_xlim())
        ax3.set_yticklabels(ax1.get_xticklabels())
        
        
        
        
        
        
        y = morph_mccon['VMag']
        x = 1E6 * morph_mccon['Mdyn']
        ax2.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=70, )
        ax2.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=70, )
        ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70, )
        ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70, )
        
        
         
        
        
        ax2.set_xscale("log")
        ax2.invert_yaxis()
        ax2.set_ylim(ax1.get_ylim())
        ax2.set_ylabel(None)
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        ax2.set_xlabel(None)
        ax2.set_xlim(ax3.get_xlim())
    
    
    
        x = morph_mccon['sigma*']
        y = morph_mccon['R2']
        ax6.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=70, color="red")
        ax6.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=70, color="blue")
        #ax6.scatter(x[(morph_mccon['SubG'] == "Rest")], y[(morph_mccon['SubG'] == "Rest")], marker=value, s=70, color="green")
        ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70, )
        ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70, )
        
        ax6.set_xscale("log")
        ax6.set_yscale("log")
        ax6.set_xlim(2, 100)
        ax6.set_xlabel(r"$\sigma_*$ [km/s]")
        ax6.set_ylabel("")
        ax6.set_yticklabels([])
        ax6.set_ylim(ax3.get_ylim())
    
    
    
    
    
        x = morph_mccon['sigma*']
        y = morph_mccon['VMag']
        ax5.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=70, color="red")
        ax5.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=70, color="blue")
        ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=70, )
        ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=70, )    
        
        
        ax5.set_xscale("log")
        ax5.set_yscale("linear")
        ax5.set_xlim(ax6.get_xlim())
        ax5.set_xlabel("")
        ax5.set_ylabel("")
        ax5.set_yticklabels([])
        ax5.set_xticklabels([])
        ax5.set_ylim(ax1.get_ylim())
    
    
    legend_GC = ax1.scatter(60/206265 * globc['Rh']  * globc['Rsun'] * 1E3, globc['MVt'], s=20, label="MW globular clusters")
    
    
    for i in range(1,37):
        st = f"dpp[{i}]"
        ax1.errorbar(0.75 * candidates['rh_stars_physical'].value[i-1] * 1E3, df[st].mean(), yerr=df[st].std()*0, marker="^", color="black")
    
    
    legenc_candidates = ax3.scatter(candidates['Mdyn'], 0.75 * candidates['rh_stars_physical'] * 1E3, marker="^", label="Agora ART-I Satellites")
    
    
    bf = ax2.plot(md, df['beta0'].mean() + df['beta1'].mean()*np.log10(md), lw=2, zorder=10, label="Best Fit")
    
    for beta0, beta1 in df[['beta0', 'beta1']].values[::1]:
        ax2.plot(md, beta0 + beta1 * np.log10(md), lw=1, alpha=0.1, color="lightblue", zorder=-1)
    
    ax6.scatter(candidates['sigma*'] , 0.75 * candidates['rh_stars_physical'] * 1E3, marker="^")
    
    ax5.scatter(candidates['sigma*'] , df['beta0'].mean() + df['beta1'].mean() * np.log10(candidates['Mdyn']), marker="^")
    
    
    ax2.legend()
    ax3.legend()
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.8), fancybox=True)
    
    
    
    
    ax4.remove()
    plt.savefig(outdir + "/" +f"ARTI_sat_comparison.png")
    plt.close()
    print(f"FINISHED !")
else:
    print(f"FINISHED !")






