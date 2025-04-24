import os
import yt
import argparse
import numpy as np
import pandas as pd
from unyt import unyt_array, unyt_quantity
import matplotlib.pyplot as plt

import src.explorer as dge
from src.explorer.class_methods import load_ftable

from lmfit import Model, Parameters, fit_report
from limepy import limepy

from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
def KingProfileIterp(r, W0, g, M, rh):
    """Produces a sample of a Lowered isothermal model with parameters W0, g, M and rh using LIMEPY 
    and interpolates the result for a specified r.
    """
    k = limepy(phi0=W0, g=g, M=M, rh=rh, G=4.300917270038e-06)
    evals = np.interp(r, xp=k.r, fp=k.rho)
    return evals #in Msun/kpc**3

def KingProfileIterp_surf(r, W0, g, M, rh):
    """Produces a sample of a Lowered isothermal model with parameters W0, g, M and rh using LIMEPY 
    and interpolates the result for a specified r.
    """
    k = limepy(phi0=W0, g=g, M=M, rh=rh, G=4.300917270038e-06, project=True)
    evals = np.interp(r, xp=k.r, fp=k.Sigma)
    return evals #in Msun/kpc**3

def plummer(r, M, a):
    """Plummer model
    """
    return 3 * M / ( 4 * np.pi * a ** 3 ) * ( 1 + (r/a)**2 ) ** (-5/2)
    
def surface_plummer(R, M, a):
    """Surface density plummer
    """
    return ( M * a ** 2 ) / np.pi * 1 / ( a**2 + R**2 ) ** 2

#kw_center_all = {
#    "stars": {"method": "adaptative", "nmin": 30},
#    "darkmatter": {"method": "adaptative", "nmin": 300},
#    "gas": {"method": "rcc", "rc_scale": 0.5}
#}


def parse_args():
    parser = argparse.ArgumentParser(description="Process various input parameters for making profile plots of a given halo. You MUST provide the halo sub_tree_id and a list of snapshot numbers.\n The binning of the profiles is performed logarithmically over the (projected) radii from rmin to rmax. DM density profile is fit to a NFW (taken from Rockstar or any other) and Stellar density profiles can be fitted to a variety of Lowered Isothermal Models using limepy, and to a simple plummer profile.")
    
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')


    required.add_argument(
        "-i", "--input_file",
        type=str,
        help="Merger Tree file path. Must be csv.",
        required=True
    )
    required.add_argument(
        "-eq", "--equivalence_table",
        type=str,
        help="Equivalence table path. Must have correct formatting.",
        required=True
    )
    required.add_argument(
        "-st", "--sub_tree_id",
        type=int,
        help="Sub Tree ID of the halo.",
        required=True
    )
    required.add_argument(
        "-c", "--code",
        type=str,
        help="Code of the simulation. Required for GALEX^2.",
        required=True
    )
    required.add_argument(
        "-pd", "--particle_data_folder",
        type=str,
        help="Location of particle data.",
        required=True
    )



    opt.add_argument(
        "-sn", "--snapshot_numbers",
        nargs="*",
        type=int,
        default=None,
        help="List of snapshot numbers to plot. Maximum 4."
    )
    opt.add_argument(
        "-v", "--volumetric",
        nargs="*",
        type=str,
        default=["stars", "darkmatter", "gas"],
        help='Enable volumetric profiles with a list of components (default: ["stars", "darkmatter", "gas"]).'
    )
    opt.add_argument(
        "-s", "--surface",
        nargs="*",
        type=str,
        default=["stars", "darkmatter"],
        help='Enable surface profiles with a list of components (default: ["stars", "darkmatter"]).'
    )
    opt.add_argument(
        "-er", "--extra_radius",
        nargs=2,
        type=str,
        default=None,
        help="Extra radius as list of [float, str] (Default: halo virial radius extracted from data)."
    )
    opt.add_argument(
        "-n", "--Nproj",
        type=int,
        default=10,
        help="Number of projections to perform, uniformly, over half a sphere (Default: 10)."
    )
    opt.add_argument(
        "-pm", "--projection_mode",
        type=str,
        default="bins",
        help="Type of projection. Only affects projected velocity. 'bins' or 'apertures': 'bins' computes all quantities on radial bins while 'apertures does so in filled apertures' (Default: 'bins')."
    )
    opt.add_argument(
        "-g", "--gas_cm",
        type=str,
        default="darkmatter",
        help="Set center-of-mass properties of gas, given that its nature is clumpy and accurate values are hard to derive. (Default: darkmatter)."
    )

    
    opt.add_argument(
        "-o", "--output",
        type=str,
        default="./",
        help="Output folder (Default: ./)."
    )
    opt.add_argument(
        "-kf", "--king_fit",
        default=False,
        action="store_true",
        help="Wether to fit a King profile to the density profiles. (Default: False)."
    )
    opt.add_argument(
        "-pf", "--plummer_fit",
        default=False,
        action="store_true",
        help="Wether to fit a plummer profile to the density profiles. (Default: False)."
    )
    opt.add_argument(
        "-wf", "--woolley_fit",
        default=False,
        action="store_true",
        help="Wether to fit a Woolley profile to the density profile. Similar to plummer profile (Default: False)."
    )
    opt.add_argument(
        "-dbf", "--double_fit",
        default=False,
        action="store_true",
        help="Wether to perform King and Woolley fit to Volumetric and Surface profiles. By default, only Volumetric profiles are fitter. (Default: False)."
    )
    opt.add_argument(
        "-sg", "--set_g",
        default=None,
        help="Wether to vary 'g' parameter when performing profile fits. Incompatible with Woolley fit. (Default: False)."
    )
    opt.add_argument(
        "-fww", "--fit_with_weights",
        action="store_true"
        default=False,
        help="Perform fit with errorbars. (Default: False)."
    )

    opt.add_argument(
        "-rr", "--radii_range",
        nargs="*",
        type=float,
        default=[0.08, 50],
        help="Wether to use the same fit for Volumetric and Surface, or produce two different ones. (Default: Uses Volumetric throguhout)."
    )

    return parser.parse_args()

def plot_voldens(ax, halo, components):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    global extra_radius
    global gas_cm
    
    bins_params_all = {
        "stars": {"rmin": 0.08, "rmax": 50, "thicken": True},
        "darkmatter": {"rmin": 0.08, "rmax": 250, "thicken": False},
        "gas": {"rmin": 0.08, "rmax": 250, "thicken": False}
    }
    results = {}
    
    if components == "particles":
        components = ["stars", "darkmatter"]
    elif components == "all":
        components = ["stars", "darkmatter", "gas"]
        
    sp = halo._data.ds.sphere(
        halo.sp_center, 
        extra_radius
    )
    for component in components: 
        component_object = getattr(halo, component)
        bins = None

        if component_object.bmasses.sum() != 0:
            r_bound, rho_bound, e_rho_bound, bins = component_object.density_profile(
                pc="bound",
                center=component_object.cm,
                bins=bins,
                return_bins=True,
                bins_params=bins_params_all[component]
            )
        else:
           r_bound, rho_bound, e_rho_bound = np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


        r_all, rho_all, e_rho_all = component_object.density_profile(
            pc="all",
            center=component_object.cm,
            bins=bins,
            new_data_params={"sp": sp}
        )



        
        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(r_bound, rho_bound, yerr=e_rho_bound, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2, label=label)
        ax.errorbar(r_all, rho_all, yerr=e_rho_all, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)



        
        results[component] = {
            "bins": bins,
            "r_bound": r_bound,
            "rho_bound": rho_bound,
            "e_rho_bound": e_rho_bound,
            "r_all": r_all,
            "rho_all": rho_all,
            "e_rho_all": e_rho_all,
        }

    return results



def plot_dispvel(ax, halo, components, bins):
    """Adds average 3D velocity dispersion to plot.
    """
    global extra_radius
    global gas_cm

    bins_params_all = {
        "stars": {"rmin": 0.08, "rmax": 50, "thicken": True},
        "darkmatter": {"rmin": 0.08, "rmax": 250, "thicken": False},
        "gas": {"rmin": 0.08, "rmax": 250, "thicken": False}
    }
    
    results = {}
    
    if components == "particles":
        components = ["stars", "darkmatter"]
    elif components == "all":
        components = ["stars", "darkmatter", "gas"]

    sp = halo._data.ds.sphere(
        halo.sp_center, 
        extra_radius
    )
    for component in components: 
        component_object = getattr(halo, component)

                   
        if component_object.bmasses.sum() != 0:
            r_bound, vrms_bound, e_vrms_bound = component_object.velocity_profile(
                pc="bound",
                center=component_object.cm,
                v_center=component_object.vcm,
                bins=bins[component]
            )
       
        else:
            r_bound, vrms_bound, e_vrms_bound = np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan])

        r_all, vrms_all, e_vrms_all = component_object.velocity_profile(
            pc="all", 
            center=component_object.cm,
            v_center=component_object.vcm,
            bins=bins[component],
            new_data_params={"sp": sp}
        )

        
        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(r_bound, vrms_bound, yerr=e_vrms_bound, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2)
        ax.errorbar(r_all, vrms_all, yerr=e_vrms_all, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)    


        
        results[component] = {
            "bins": bins[component],
            "r_bound": r_bound,
            "vrms_bound": vrms_bound,
            "e_vrms_bound": e_vrms_bound,
            "r_all": r_all,
            "vrms_all": vrms_all,
            "e_vrms_all": e_vrms_all,
        }

    return results

def plot_surfdens(ax, halo, lines_of_sight, components):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    from tqdm import tqdm
    
    global extra_radius
    global gas_cm

    bins_params_all = {
        "stars": {"rmin": 0.08, "rmax": 50, "thicken": True},
        "darkmatter": {"rmin": 0.08, "rmax": 250, "thicken": False},
        "gas": {"rmin": 0.08, "rmax": 250, "thicken": False}
    }
    
    results = {}
    
    if components == "particles":
        components = ["stars", "darkmatter"]
    elif components == "all":
        components = ["stars", "darkmatter", "gas"]

    sp = halo._data.ds.sphere(
        halo.sp_center, 
        extra_radius
    )
    for component in components:
        rhos_bound_all = []
        rhos_all_all = []
        
        bins = None  
        component_object = getattr(halo, component)

        for i in tqdm(range(len(lines_of_sight)), desc=f"Projecting {component} for density"):
            los = lines_of_sight[i]
            halo.set_line_of_sight(los.tolist())
            
            if component_object.bmasses.sum() != 0:
                r_bound, rho_bound, e_rho_bound, bins = component_object.density_profile(
                    pc="bound",
                    center=component_object.cm,
                    bins=bins,
                    projected=True,
                    return_bins=True,
                    bins_params=bins_params_all[component]
                )
            else:
               r_bound, rho_bound, e_rho_bound = np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), np.array([np.nan, np.nan])
            

            rhos_bound_all.append(rho_bound)

            r_all, rho_all, e_rho_all = component_object.density_profile(
                pc="all",
                center=component_object.cm,
                bins=bins,
                projected=True,
                bins_params=bins_params_all[component],
                new_data_params={"sp": sp}
            )
            rhos_all_all.append(rho_all)
        
        # Convert to numpy arrays for averaging
        rhos_bound_all = np.array(rhos_bound_all)
        rhos_all_all = np.array(rhos_all_all)

        # Compute averages
        rho_bound_avg = np.mean(rhos_bound_all, axis=0)
        rho_all_avg = np.mean(rhos_all_all, axis=0)

        # Compute uncertainties (standard deviations)
        rho_bound_std = np.std(rhos_bound_all, axis=0)
        rho_all_std = np.std(rhos_all_all, axis=0)

        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(r_bound, rho_bound_avg, yerr=rho_bound_std, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2, label=label)
        ax.errorbar(r_all, rho_all_avg, yerr=rho_all_std, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)

        results[component] = {
            "bins": bins,
            "r_bound": r_bound,
            "rho_bound": rho_bound_avg,
            "rho_bound_std": rho_bound_std,
            "r_all": r_all,
            "rho_all": rho_all_avg,
            "rho_all_std": rho_all_std,
        }

    return results


def plot_losvel(ax, halo, lines_of_sight, components, bins, velocity_projection):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    from tqdm import tqdm

    global extra_radius
    global gas_cm

    bins_params_all = {
        "stars": {"rmin": 0.08, "rmax": 50, "thicken": True},
        "darkmatter": {"rmin": 0.08, "rmax": 250, "thicken": False},
        "gas": {"rmin": 0.08, "rmax": 250, "thicken": False}
    }
    
    results = {}
    
    if components == "particles":
        components = ["stars", "darkmatter"]
    elif components == "all":
        components = ["stars", "darkmatter", "gas"]

    sp = halo._data.ds.sphere(
        halo.sp_center, 
        extra_radius
    )
    for component in components:
        vlos_bound_all = []
        vlos_all_all = []
        
        component_object = getattr(halo, component)
        
        for i in tqdm(range(len(lines_of_sight)), desc=f"Projecting {component} for los-vel"):
            los = lines_of_sight[i]
            halo.set_line_of_sight(los.tolist())
            
            if component_object.bmasses.sum() != 0:
                r_bound, vlos_bound, _ = component_object.velocity_profile(
                    pc="bound",
                    center=component_object.cm,
                    v_center=component_object.vcm,
                    bins=bins[component],
                    projected=velocity_projection
                )
            else:
                r_bound, vlos_bound = np.array([np.nan, np.nan]), np.array([np.nan, np.nan])


            vlos_bound_all.append(vlos_bound)

            r_all, vlos_all, _ = component_object.velocity_profile(
                pc="all",
                center=component_object.cm,
                v_center=component_object.vcm,
                bins=bins[component],
                projected=velocity_projection,
                new_data_params={"sp": sp}
            )
            vlos_all_all.append(vlos_all)
        
        # Convert to numpy arrays for averaging
        vlos_bound_all = np.array(vlos_bound_all)
        vlos_all_all = np.array(vlos_all_all)

        # Compute averages
        vlos_bound_avg = np.mean(vlos_bound_all, axis=0)
        vlos_all_avg = np.mean(vlos_all_all, axis=0)

        # Compute uncertainties (standard deviations)
        vlos_bound_std = np.std(vlos_bound_all, axis=0)
        vlos_all_std = np.std(vlos_all_all, axis=0)

        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(r_bound, vlos_bound_avg, yerr=vlos_bound_std, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2)
        ax.errorbar(r_all, vlos_all_avg, yerr=vlos_all_std, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)

        results[component] = {
            "bins": bins[component],
            "r_bound": r_bound,
            "vlos_bound": vlos_bound_avg,
            "vlos_bound_std": vlos_bound_std,
            "r_all": r_all,
            "vlos_all": vlos_all_avg,
            "vlos_all_std": vlos_all_std,
        }

    return results


def limepy_fitAndPlot(model, fit_params, result_dens, mode):
    """Fits result_dens data to model, given fit_params initial values and constraints.
    """
    global fit_with_weights

    if fit_with_weights:
        if mode == "volumetric":
            w = 1 / result_dens["stars"]["e_rho_bound"]
        if mode == "surface":
            w = 1 / result_dens["stars"]["rho_bound_std"]
    else:
        w = np.ones_like(result_dens["stars"]["rho_bound"])

    result = model.fit(
            r=result_dens["stars"]["r_bound"], 
            data=result_dens["stars"]["rho_bound"], 
            params=fit_params, 
            weights=w,
            nan_policy="omit"
    ) 
    k = limepy(
        phi0=result_stars.params['W0'].value, 
        g=result_stars.params['g'].value, 
        M=result_stars.params['M'].value, 
        rh=result_stars.params['rh'].value, 
        G=4.300917270038e-06,
        project=True
    )


    return k, result







args = parse_args()

try:
    equivalence_table = load_ftable(args.equivalence_table)
except:
    equivalence_table = pd.read_csv(args.equivalence_table)

merger_table = pd.read_csv(args.input_file).sort_values("Snapshot")
snap_list = args.snapshot_numbers
sub_tree = args.sub_tree_id

if snap_list is not None:
    subtree_table = merger_table[(merger_table["Sub_tree_id"] == sub_tree) & (np.isin(merger_table["Snapshot"].values, snap_list))].sort_values("R/Rvir")
else:
    subtree_table = merger_table[(merger_table["Sub_tree_id"] == sub_tree)].sort_values("R/Rvir")






print(f"\nThe following ROWS where FOUND in the file you provided:")
print(f"--------------------------------------------------------")
print(subtree_table)

fns = []
for snapshot in subtree_table["Snapshot"].values:
    fn = equivalence_table[equivalence_table['snapshot'] == snapshot]['snapname'].values[0]
    fns.append(fn)

subtree_table["fn"] = fns
print(f"\nFiles corresponding to SELECTED ROWS: {fns}\n")




dge.config.code = args.code

densModel = Model(KingProfileIterp, independent_vars=['r'])
surfModel = Model(KingProfileIterp_surf, independent_vars=['r'])
plummerModel = Model(plummer, independent_vars=['r'])
surfplummerModel = Model(surface_plummer, independent_vars=['R'])


try:
    os.mkdir(args.output + f"subtree_{sub_tree}/")
except:
    pass












nmin_stars = 40
nmin_dm = 300
rc_gas = 0.5



import smplotlib
from src.explorer.class_methods import NFWc

plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.major.width'] = 1.1
plt.rcParams['xtick.minor.width'] = 1.1
plt.rcParams['ytick.major.width'] = 1.1
plt.rcParams['ytick.minor.width'] = 1.1

plt.rcParams['xtick.major.size'] = 7 * 1.5
plt.rcParams['ytick.major.size'] = 7 * 1.5

plt.rcParams['xtick.minor.size'] = 5 
plt.rcParams['ytick.minor.size'] = 5 


from src.explorer.class_methods import random_vector_spherical

double_fit = args.double_fit
N = args.Nproj
vol_components = args.volumetric
proj_components = args.surface
velocity_projection = args.projection_mode
gas_cm = args.gas_cm
fit_with_weights = args.fit_with_weights

    
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(len(fns)/2*12,12), sharex=True, sharey="row")
plt.subplots_adjust(hspace=0.05, wspace=0.05)
fig.suptitle(f"Averaged volumemetric density and velocity profiles, for sub_tree: {sub_tree}, at different host distances:", y=0.95, fontsize=29, ha="center")


fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(len(fns)/2*12,12), sharex=True, sharey="row")
plt.subplots_adjust(hspace=0.05, wspace=0.05)
fig2.suptitle(f"Averaged surface density and velocity profiles, for sub_tree: {sub_tree}, at different host distances:", y=0.95, fontsize=29, ha="center")


i = 0
for _, row in subtree_table.iterrows():
    halo_params = row

    redshift = halo_params['Redshift']
    center = (halo_params[['position_x', 'position_y', 'position_z']].values.astype(float) / (1 + redshift), 'kpc')
    center_vel = (halo_params[['velocity_x', 'velocity_y', 'velocity_z']].values.astype(float), 'km/s')
    rvir = (halo_params['virial_radius'] / (1 + redshift), 'kpc')
    rs = (halo_params['scale_radius'] / (1 + redshift), 'kpc')
    mass = (halo_params['mass'], 'Msun')
    vmax = (halo_params['vmax'], 'km/s')
    vrms = (halo_params['vrms'], 'km/s')
    

    try:
        rh3d_stars = (halo_params['rh3D_physical_stars'], 'kpc')
    except:
        rh3d_stars = (1, 'kpc')

    if args.extra_radius is None:
        extra_radius = rvir
    else:
        extra_radius = float(args.extra_radius[0]), str(args.extra_radius[1])
    
    axes[0, i].set_title(f"R/Rvir={halo_params['R/Rvir']:.2f}, z={redshift:.2f}")
  
    halo = dge.SnapshotHalo(args.particle_data_folder + halo_params["fn"], center=center, radius=rvir)

    halo.compute_bound_particles(
        components=["stars", "darkmatter", "gas"], 
        method="bh", 
        weighting="softmax", 
        T="adaptative", 
        verbose=False,  
        cm=unyt_array(*center), 
        vcm=unyt_array(*center_vel)
    )
    halo.darkmatter.refined_center6d(method="adaptative", nmin=nmin_dm)
    halo.stars.refined_center6d(method="adaptative", nmin=nmin_stars)
    if args.gas_cm == "darkmatter":
        halo.gas.cm = halo.darkmatter.cm
        halo.gas.vcm = halo.darkmatter.vcm
    elif args.gas_cm == "stars":
        halo.gas.cm = halo.stars.cm
        halo.gas.vcm = halo.stars.vcm
    else:
        halo.gas.refined_center6d(method="rcc", rc_scale=rc_gas)


    result_dens = plot_voldens(axes[0, i], halo, vol_components)
    bins = {c : result_dens[c]["bins"] for c in vol_components}
    result_vels = plot_dispvel(axes[1, i], halo, vol_components, bins)
    
    
    axes[0, i].axvspan(0.0001, 2 * 0.08, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes[1, i].axvspan(0.0001, 2 * 0.08, color="darkviolet", alpha=0.25, ls="--", lw=0.01)

    axes[0, i].axvline(2 * 0.08, color="darkviolet", ls="--", lw=2.5)
    axes[1, i].axvline(2 * 0.08, color="darkviolet", ls="--", lw=2.5)

    axes[0, i].text(0.08, 2E2, r"$\varepsilon=80$ pc" ,ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes[1, i].text(0.08, 11.5, r"$\varepsilon=80$ pc" ,ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes[1, i].text(0.18, 350, r"$M_*$="+f"{halo.stars.bmasses.sum().value:.3e}"+r" $M_\odot$"+"\n"+r"$M_{dm}$="+f"{halo.darkmatter.bmasses.sum().value:.3e}"+r" $M_\odot$"+f"{halo.gas.bmasses.sum():.3f}"+r" $M_\odot$" ,ha="left", va="top", color="black", rotation="horizontal", fontsize=14)

    axes[1, i].set_xlabel(f"r [kpc]", fontsize=20)

    if i==0:
        axes[0, i].set_ylabel(r"$\rho \ [M_\odot / kpc^3]$", fontsize=20)
        axes[1, i].set_ylabel(r"$v_{rms}$ [km/s]", fontsize=20)

    axes[0, i].loglog()
    axes[1, i].set_yscale("log")

    axes[0, i].set_xlim(0.05, 300)
    axes[0, i].set_ylim(1E2, 8E9)
    axes[1, i].set_ylim(10, 400)

    rho_h = unyt_quantity(*mass) / (4*np.pi/3 * unyt_quantity(*rvir).to("kpc")**3)
    c = (unyt_quantity(*rvir).to("kpc")/unyt_quantity(*rs).to("kpc")).value
    r = np.logspace(np.log10(0.05), np.log10(300), 200)
    rho_nfw = NFWc(r, rho_h, c, unyt_quantity(*rvir).to("kpc").value)
    
    axes[0, i].plot(r, rho_nfw, color="darkorange", label=f'NFW Rockstar Fit: c={c:.2f}, rvir={unyt_quantity(*rvir).to("kpc").value:.2f} kpc', zorder=9)


    if args.king_fit:
        fit_params_king = densModel.make_params(
            W0={'value': 5.5,'min': 0.1,'max': np.inf,'vary': True},
            g={'value': 1, 'vary': False},
            M={'value': halo.stars.bmasses.sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True}
        )
   
        k_king, fit_king =  limepy_fitAndPlot(densModel, fit_params_king, result_dens, "volumetric")

        axes[0, i].plot(
            k_king.r, 
            k_king.rho, 
            color="darkblue", 
            zorder=10, 
            label=f"Fit to King: W0={fit_king.params['W0'].value:.2f}±{fit_king.params['W0'].stderr:.0e},  rh={fit_king.params['rh'].value:.2f}±{fit_king.params['rh'].stderr:.0e} kpc"
        )
        axes[1, i].plot(
            k_king.r, 
            np.sqrt(k_king.v2), 
            color="darkblue", 
            zorder=10
        )

    if args.woolley_fit:
        fit_params_woolley = densModel.make_params(
            W0={'value': 5.5,'min': 0.1,'max': np.inf,'vary': True},
            g={'value': 0, 'vary': False},
            M={'value': halo.stars.bmasses.sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True}
        )
   
        k_woolley, fit_woolley =  limepy_fitAndPlot(densModel, fit_params_woolley, result_dens, "volumetric")

        axes[0, i].plot(
            k_woolley.r, 
            k_woolley.rho, 
            color="brown", 
            zorder=10, 
            label=f"Fit to Woolley: W0={fit_woolley.params['W0'].value:.2f}±{fit_woolley.params['W0'].stderr:.0e},  rh={fit_woolley.params['rh'].value:.2f}±{fit_woolley.params['rh'].stderr:.0e} kpc"
        )
        axes[1, i].plot(
            k_woolley.r, 
            np.sqrt(k_woolley.v2), 
            color="brown", 
            zorder=10
        )












































    
    axes[0, i].legend(loc="upper right", fontsize=12, markerfirst=False, reverse=False)






    

    



    axes2[0, i].set_title(f"R/Rvir={halo_params['R/Rvir']:.2f}, z={redshift:.2f}")

    lines_of_sight = random_vector_spherical(N, half_sphere=True)

    result_surf = plot_surfdens(axes2[0, i], halo, lines_of_sight, proj_components)
    bins = {c : result_surf[c]["bins"] for c in proj_components}
    plot_losvel(axes2[1, i], halo, lines_of_sight, proj_components, bins, velocity_projection)



    axes2[0, i].axvspan(0.0001, 2 * 0.08, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes2[0, i].axvline(2 * 0.08, color="darkviolet", ls="--", lw=2.5)
    axes2[1, i].axvspan(0.0001, 2 * 0.08, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes2[1, i].axvline(2 * 0.08, color="darkviolet", ls="--", lw=2.5)



    axes2[0, i].text(0.08, 2E2, r"$\varepsilon=80$ pc" ,ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes2[1, i].text(0.08, 1.15, r"$\varepsilon=80$ pc" ,ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes2[1, i].text(0.18, 150, r"$M_*$="+f"{halo.stars.bmasses.sum().value:.3e}"+r" $M_\odot$"+"\n"+r"$M_{dm}$="+f"{halo.darkmatter.bmasses.sum().value:.3e}"+r" $M_\odot$" ,ha="left", va="top", color="black", rotation="horizontal", fontsize=14)

    axes2[1, i].set_xlabel(f"R [kpc]", fontsize=20)

    if i==0:
        axes2[0, i].set_ylabel(r"$\Sigma \ [M_\odot / kpc^2]$", fontsize=20)
        axes2[1, i].set_ylabel(r"$v_{los}$ [km/s]", fontsize=20)

    axes2[0, i].loglog()
    axes2[1, i].set_yscale("log")

    axes2[0, i].set_xlim(0.05, 300)
    axes2[0, i].set_ylim(1E2, 8E9)
    axes2[1, i].set_ylim(1, 200)



    

    if double_fit:
        fit_params = surfModel.make_params(
            W0={'value': 5.5,'min': 0.1,'max': np.inf,'vary': True},
            g={'value': 1, 'vary': False},
            M={'value': halo.stars.bmasses.sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True}
        )
        result_stars = densModel.fit(r=result_surf["stars"]["r_bound"], data=result_surf["stars"]["rho_bound"], params=fit_params, nan_policy="omit") 
        k_stars = limepy(phi0=result_stars.params['W0'].value, 
                g=result_stars.params['g'].value, 
                M=result_stars.params['M'].value, 
                rh=result_stars.params['rh'].value, 
                G=4.300917270038e-06,
                project=True
                )
    
        axes2[0, i].plot(
            k_stars.r, 
            k_stars.Sigma, 
            color="darkblue", 
            zorder=10, 
            label=f"Fit to King: W0={result_stars.params['W0'].value:.2f}±{result_stars.params['W0'].stderr:.0e},  rh={result_stars.params['rh'].value:.2f}±{result_stars.params['rh'].stderr:.0e} kpc"
        )

    axes2[0, i].plot(
        k_stars.r, 
        k_stars.Sigma, 
        color="darkblue", 
        zorder=10, 
        label=f"Fit to King: rhp={k_stars.rhp:.2f} kpc"
    )

    
    axes2[0, i].legend(loc="upper right", fontsize=12, markerfirst=False, reverse=False)

    i += 1



fig.savefig(
    args.output + f"subtree_{sub_tree}/" + "volumetric_profiles.png",
    dpi=300, 
    bbox_inches="tight"
)
fig2.savefig(
    args.output + f"subtree_{sub_tree}/" + "projected_profiles.png",
    dpi=300, 
    bbox_inches="tight"
)

plt.close()
    



print(f"Finished. Bye!")






























