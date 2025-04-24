import os
import yt
import argparse
import numpy as np
import pandas as pd
from unyt import unyt_array, unyt_quantity
import matplotlib.pyplot as plt

import galexquared as gal
from galexquared.class_methods import load_ftable

from lmfit import Model, Parameters, fit_report
from lmfit.model import save_modelresult, load_modelresult

from limepy import limepy

from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
def KingProfileIterp(r, W0, g, M, rh, ra):
    """Produces a sample of a Lowered isothermal model with parameters W0, g, M and rh using LIMEPY 
    and interpolates the result for a specified r.
    """
    k = limepy(phi0=W0, g=g, M=M, rh=rh, ra=ra, G=4.300917270038e-06)
    evals = np.interp(r, xp=k.r, fp=k.rho)
    return evals #in Msun/kpc**3

def KingProfileIterp_surf(r, W0, g, M, rh, ra):
    """Produces a sample of a Lowered isothermal model with parameters W0, g, M and rh using LIMEPY 
    and interpolates the result for a specified r.
    """
    k = limepy(phi0=W0, g=g, M=M, rh=rh, ra=ra, G=4.300917270038e-06, project=True)
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

def einasto(r, rs, mu, M):
    """Einasto density profile
    """
    from scipy.special import gamma
    numerator = pow(2*mu, 3*mu) * M 
    denominator = 4 * np.pi * rs**3 * mu * np.exp(2*mu) * gamma(3*mu)
    return numerator / denominator * np.exp( -2*mu * ( pow(r/rs,1/mu) - 1 ) )


#kw_center_all = {
#    "stars": {"method": "adaptative", "nmin": 30},
#    "darkmatter": {"method": "adaptative", "nmin": 300},
#    "gas": {"method": "rcc", "rc_scale": 0.5}
#}
#bins_params_all = {
#    "stars": {"rmin": 0.08, "rmax": 50, "thicken": True},
#    "darkmatter": {"rmin": 0.08, "rmax": 250, "thicken": False},
#    "gas": {"rmin": 0.08, "rmax": 250, "thicken": False}
#}

def add_bins(bin_edges, max_value):
    """Extends the bin_edges array until max_value is reached, with a bin spacing of 
    delta log(r_e) = mean( log(r_i+1) - log(r_i) )
    """
    if bin_edges is None:
        return None
    bin_edges = np.asarray(bin_edges)
    
    if not np.all(bin_edges[:-1] < bin_edges[1:]):
        raise ValueError("bin_edges must be sorted in ascending order.")
    if np.any(bin_edges <= 0):
        raise ValueError("bin_edges must contain only positive values.")
    if max_value <= bin_edges[-1]:
        raise ValueError("max_value must be greater than the largest initial bin edge.")

    log_deltas = np.log(bin_edges[1:] / bin_edges[:-1])
    mean_log_delta = np.mean(log_deltas)
    
    current_edge = bin_edges[-1]
    extended_bins = [current_edge]
    
    while current_edge < max_value:
        next_edge = current_edge * np.exp(mean_log_delta)
        if next_edge >= max_value:
            next_edge = max_value
        extended_bins.append(next_edge)
        if next_edge == max_value:
            break
        current_edge = next_edge

    return np.concatenate([bin_edges, extended_bins[1:]])


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
        "-vm","--velocity_moment",
        type=str,
        default="rms",
        help="How to compute projected velocity quantity both for 3D and projected profiles: 'mean'. 'rms' or 'dispersion'. (Default: 'rms')."
    )
    opt.add_argument(
        "-g", "--gas_cm",
        type=str,
        default="darkmatter",
        help="Set center-of-mass properties of gas, given that its nature is clumpy and accurate values are hard to derive. (Default: darkmatter)."
    )
    opt.add_argument(
        "-ra", "--radial_anisotropy",
        action="store_true",
        default=False,
        help="Whether to account for radial anisotropy. (Default: False)."
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
        "-sg", "--set_g",
        default=None,
        help="Wether to vary 'g' parameter when performing profile fits. Incompatible with Woolley fit. (Default: False)."
    )
    opt.add_argument(
        "-dbf", "--double_fit",
        default=False,
        action="store_true",
        help="Wether to perform King and Woolley fit to Volumetric and Surface profiles. By default, only Volumetric profiles are fitter. (Default: False)."
    )
    opt.add_argument(
        "-fww", "--fit_with_weights",
        action="store_true",
        default=False,
        help="Perform fit with errorbars. (Default: False)."
    )

    opt.add_argument(
        "-rr", "--radii_range",
        nargs=2,
        type=float,
        default=[0.05, 170],
        help="Plot xlims. (Default: 0.05, 170)."
    )
    opt.add_argument(
        "-dr", "--density_range",
        nargs=2,
        type=float,
        default=[1E2, 8E9],
        help="Density plot ylims. (Default: 1E2, 8E9 Msun/kpc**3 or /kpc**2)."
    )
    opt.add_argument(
        "-vr", "--velocity_range",
        nargs=2,
        type=float,
        default=[9, 400],
        help="Velocity plot ylims. For projection, they are multiplied by 0.4 and 0.6 respectively. (Default: 5, 400 km/s)."
    )
    opt.add_argument(
        "-sf", "--softening",
        type=float,
        default=0.08,
        help="Softening of particles in kiloparsec. (Default: 0.08 kpc)."
    )

    return parser.parse_args()

def plot_voldens(ax, halo, components):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    global extra_radius
    global gas_cm
  
    bins_params_all = {
        "stars": {"rmin": 0.08, "thicken": 1, "bins": 10},
        "darkmatter": {"rmin": 0.08, "thicken": 0, "bins": 10},
        "gas": {"rmin": 0.08, "thicken": 0, "bins": 10}
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

        if not component_object.empty:
            halo.switch_to_bound()
            bound_profile = component_object.density_profile(
                center=component_object.q["cm"],
                bins=bins,
                bins_params=bins_params_all[component]
            )
            bins = bound_profile["bins"]
        else:
            bound_profile = {"r": np.array([np.nan, np.nan]),
                             "rho": np.array([np.nan, np.nan]),
                             "e_rho": np.array([np.nan, np.nan])
                            }

        halo.switch_to_all()
        all_profile = component_object.density_profile(
            center=component_object.q["cm"],
            bins=add_bins(bins, sp.radius.to("kpc").value),
            bins_params=bins_params_all[component],
            new_data_params={"sp": sp}
        )

        
        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(bound_profile["r"], bound_profile["rho"], yerr=bound_profile["e_rho"], fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2, label=label)
        ax.errorbar(all_profile["r"], all_profile["rho"], yerr=all_profile["e_rho"], fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)
        
        results[component] = {
            "bound": bound_profile,
            "all": all_profile
        }
    return results



def plot_dispvel(ax, halo, components, bins):
    """Adds average 3D velocity dispersion to plot.
    """
    global extra_radius
    global gas_cm
    global quant

    
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

                   
        if not component_object.empty:
            halo.switch_to_bound()
            bound_profile = component_object.velocity_profile(
                center=component_object.q["cm"],
                v_center=component_object.q["vcm"],
                bins=bins["bound"][component],
                quantity=quant
            )
        else:
            bound_profile = {"r": np.array([np.nan, np.nan]),
                             "v": np.array([np.nan, np.nan]),
                             "e_v": np.array([np.nan, np.nan])
                            }

        halo.switch_to_all()
        all_profile = component_object.velocity_profile(
            center=component_object.q["cm"],
            v_center=component_object.q["vcm"],
            bins=bins["all"][component],
            quantity=quant,
            new_data_params={"sp": sp}
        )

        
        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(bound_profile["r"], bound_profile["v"], yerr=bound_profile["e_v"], fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2)
        ax.errorbar(all_profile["r"], all_profile["v"], yerr=all_profile["e_v"], fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)
    
        results[component] = {
            "bound": bound_profile,
            "all": all_profile
        }
    return results

def plot_surfdens(ax, halo, lines_of_sight, components):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    from tqdm import tqdm
    
    global extra_radius
    global gas_cm

    bins_params_all = {
        "stars": {"rmin": 0.08, "thicken": 1, "bins": 10},
        "darkmatter": {"rmin": 0.08, "thicken": 0, "bins": 10},
        "gas": {"rmin": 0.08, "thicken": 0, "bins": 10}
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
            
            if not component_object.empty:
                halo.switch_to_bound()
                bound_profile = component_object.density_profile(
                    center=component_object.q["cm"],
                    bins=bins,
                    projected=True,
                    bins_params=bins_params_all[component]
                )
                bins = bound_profile["bins"]
            else:
                bound_profile = {"r": np.array([np.nan, np.nan]),
                                "rho": np.array([np.nan, np.nan]),
                                "e_rho": np.array([np.nan, np.nan])
                                }
            rhos_bound_all.append(bound_profile["rho"])

            halo.switch_to_all()
            all_profile = component_object.density_profile(
                center=component_object.q["cm"],
                bins=add_bins(bins, sp.radius.to("kpc").value),
                projected=True,
                bins_params=bins_params_all[component],
                new_data_params={"sp": sp}
            )
            rhos_all_all.append(all_profile["rho"])
        
        rhos_bound_all = np.array(rhos_bound_all)
        rhos_all_all = np.array(rhos_all_all)

        rho_bound_avg = np.mean(rhos_bound_all, axis=0)
        rho_all_avg = np.mean(rhos_all_all, axis=0)

        rho_bound_std = np.std(rhos_bound_all, axis=0)
        rho_all_std = np.std(rhos_all_all, axis=0)




        color = {"stars": "red", "darkmatter": "black", "gas": "green"}[component]
        marker = {"stars": "*", "darkmatter": ".", "gas": "s"}[component]
        label = {"stars": "stars (-bound, --all)", "darkmatter": "dark matter", "gas" : "gas"}[component]
        markersize = {"stars": 11, "darkmatter": 11, "gas": 7}[component]
        
        ax.errorbar(bound_profile["r"], rho_bound_avg, yerr=rho_bound_std, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2, label=label)
        ax.errorbar(all_profile["r"], rho_all_avg, yerr=rho_all_std, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)

        results[component] = {
            "bound": {
                "r": bound_profile["r"],
                "rho": rho_bound_avg,
                "rho_std": rho_bound_std,
                "bins": bins
            },
            "all": {
                "r": all_profile["r"],
                "rho": rho_all_avg,
                "rho_std": rho_all_std,
                "bins": add_bins(bins, sp.radius.to("kpc").value)
            }
        }
    return results


def plot_losvel(ax, halo, lines_of_sight, components, bins, velocity_projection):
    """Adds averaged surface density (both 'all' and 'bound') to plot over multiple lines of sight.
    """
    from tqdm import tqdm

    global extra_radius
    global gas_cm
    global quant
    
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
            
            if not component_object.empty:
                halo.switch_to_bound()
                bound_profile = component_object.velocity_profile(
                    center=component_object.q["cm"],
                    v_center=component_object.q["vcm"],
                    bins=bins["bound"][component],
                    quantity=quant,
                    projected=velocity_projection
                )
            else:
                bound_profile = {"r": np.array([np.nan, np.nan]),
                                "v": np.array([np.nan, np.nan]),
                                "e_v": np.array([np.nan, np.nan])
                                }
            vlos_bound_all.append(bound_profile["v"])


            halo.switch_to_all()
            all_profile = component_object.velocity_profile(
                center=component_object.q["cm"],
                v_center=component_object.q["vcm"],
                bins=bins["all"][component],
                quantity=quant,
                projected=velocity_projection,
                new_data_params={"sp": sp}
            )
            vlos_all_all.append(all_profile["v"])
        
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
        
        ax.errorbar(bound_profile["r"], vlos_bound_avg, yerr=vlos_bound_std, fmt=marker, ls="-", color=color, markersize=markersize, lw=1.2)
        ax.errorbar(all_profile["r"], vlos_all_avg, yerr=vlos_all_std, fmt=marker, ls="--", color=color, markersize=markersize, lw=1.2)

        results[component] = {
            "bound": {
                "r": bound_profile["r"],
                "v_bound_avg": vlos_bound_avg,
                "v_bound_std": vlos_bound_std,
                "bins": bins["bound"][component]
            },
            "all": {
                "r": all_profile["r"],
                "v_all": vlos_all_avg,
                "v_all_std": vlos_all_std,
                "bins": bins["all"][component]
            }
        }
    return results


def limepy_fitAndPlot(model, fit_params, result_dens, mode):
    """Fits result_dens data to model, given fit_params initial values and constraints.
    """
    global fit_with_weights

    if fit_with_weights:
        if mode == "volumetric":
            w = 1 / result_dens["stars"]["bound"]["e_rho"]
        if mode == "surface":
            w = 1 / result_dens["stars"]["bound"]["rho_std"]
    else:
        w = np.ones_like(result_dens["stars"]["bound"]["rho"])

    result = model.fit(
            r=result_dens["stars"]["bound"]["r"],
            data=result_dens["stars"]["bound"]["rho"],
            params=fit_params, 
            weights=w,
            nan_policy="omit"
    ) 
    k = limepy(
        phi0=result.params['W0'].value, 
        g=result.params['g'].value, 
        M=result.params['M'].value, 
        rh=result.params['rh'].value, 
        ra=result.params['ra'].value,
        G=4.300917270038e-06,
        project=True
    )


    return k, result

def plummer_fitAndPlot(model, fit_params, result_dens, mode):
    """Fits result_dens data to model, given fit_params initial values and constraints.
    """
    global fit_with_weights



    result = model.fit(
            r=result_dens["stars"]["bound"]["r"],
            data=result_dens["stars"]["bound"]["rho"],
            params=fit_params, 
            weights=w,
            nan_policy="omit"
    ) 

    return result

def dump_fit(result, path):
    import json

    params_with_stderr = {}
    for name, param in result.params.items():
        params_with_stderr[name] = {
            "value": param.value.tolist() if isinstance(param.value, np.ndarray) else param.value,
            "stderr": param.stderr.tolist() if isinstance(param.stderr, np.ndarray) else param.stderr,
        }

    with open(path, 'w') as f:
        json.dump(params_with_stderr, f, indent=4)

    return None





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




gal.config.code = args.code

densModel = Model(KingProfileIterp, independent_vars=['r'])
surfModel = Model(KingProfileIterp_surf, independent_vars=['r'])
plummerModel = Model(plummer, independent_vars=['r'])
surfplummerModel = Model(surface_plummer, independent_vars=['R'])
einastoModel = Model(einasto, independent_vars=['r'])

try:
    os.mkdir(args.output + f"subtree_{sub_tree}/")
    os.mkdir(args.output + f"subtree_{sub_tree}/fits/")

except:
    pass












nmin_stars = 40
nmin_dm = 100
rc_gas = 0.5



import smplotlib
from galexquared.class_methods import NFWc

plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.major.width'] = 1.1
plt.rcParams['xtick.minor.width'] = 1.1
plt.rcParams['ytick.major.width'] = 1.1
plt.rcParams['ytick.minor.width'] = 1.1

plt.rcParams['xtick.major.size'] = 7 * 1.5
plt.rcParams['ytick.major.size'] = 7 * 1.5

plt.rcParams['xtick.minor.size'] = 5 
plt.rcParams['ytick.minor.size'] = 5 

plt.rcParams['legend.handlelength'] = 1
plt.rcParams['legend.handletextpad'] = 0.7
plt.rcParams['legend.borderaxespad'] = 0.4

from galexquared.class_methods import random_vector_spherical

double_fit = args.double_fit
N = args.Nproj
vol_components = args.volumetric
proj_components = args.surface
velocity_projection = args.projection_mode
gas_cm = args.gas_cm
fit_with_weights = args.fit_with_weights
quant = args.velocity_moment

rmin, rmax = args.radii_range
densmin, densmax = args.density_range
pvmin, pvmax = args.velocity_range
soft = args.softening


fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(len(fns)/2*12,12), sharex=True, sharey="row")
plt.subplots_adjust(hspace=0.04, wspace=0.04)
fig.suptitle(f"Averaged volumemetric density and velocity profiles, for sub_tree: {sub_tree}, at different host distances:", y=0.95, fontsize=29, ha="center")


fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(len(fns)/2*12,12), sharex=True, sharey="row")
plt.subplots_adjust(hspace=0.04, wspace=0.04)
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

  
    halo = gal.SnapshotHalo(args.particle_data_folder + halo_params["fn"], center=center, radius=rvir)

    halo.darkmatter.refined_center6d(method="adaptative", nmin=nmin_dm)

    halo.compute_energy(refine=True)
    halo.compute_stars_in_halo(
        center=unyt_array(*center),
        center_vel=unyt_array(*center_vel),
        rvir=unyt_quantity(*rvir),
        vrms=unyt_quantity(*vrms),
        vmax=unyt_quantity(*vmax)
    )
    halo.switch_to_bound()
    halo.stars.refined_center6d(method="adaptative", nmin=nmin_stars)


    if args.gas_cm == "darkmatter":
        halo.gas.q["cm"] = halo.darkmatter.q["cm"]
        halo.gas.q["vcm"] = halo.darkmatter.q["vcm"]
    elif args.gas_cm == "stars":
        halo.gas.q["cm"] = halo.stars.q["cm"]
        halo.gas.q["vcm"] = halo.stars.q["vcm"]
    else:
        halo.gas.refined_center6d(method="rcc", rc_scale=rc_gas)


    result_dens = plot_voldens(axes[0, i], halo, vol_components)
    bins = {
        "bound": { c: result_dens[c]["bound"]["bins"] for c in vol_components},
        "all": { c: result_dens[c]["all"]["bins"] for c in vol_components}
    }
    print(bins)
    result_vels = plot_dispvel(axes[1, i], halo, vol_components, bins)
    
    axes[0, i].vlines(result_dens["darkmatter"]["bound"]["r"][-1], ymin=0, ymax=result_dens["darkmatter"]["bound"]["rho"][-1], zorder=-1, color="black", ls="-")
    axes[0, i].axvspan(0.0001, 2 * soft, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes[1, i].axvspan(0.0001, 2 * soft, color="darkviolet", alpha=0.25, ls="--", lw=0.01)

    axes[0, i].axvline(2 * soft, color="darkviolet", ls="--", lw=2.5)
    axes[1, i].axvline(2 * soft, color="darkviolet", ls="--", lw=2.5)

    halo.switch_to_bound()
    axes[0, i].text(soft, 3*densmin, r"$\varepsilon=80$ pc", ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes[1, i].text(soft, 0.9*pvmax, r"$\varepsilon=80$ pc", ha="left", va="top", color="darkviolet", rotation="vertical", fontsize=20)
    axes[1, i].text(2.1*soft, 0.9*pvmax, r"$M_*$="+f"{halo.stars['mass'].sum().value:.3e}"+r" $M_\odot$"+"\n"+r"$M_{dm}$="+f"{halo.darkmatter['mass'].sum().value:.3e}"+r" $M_\odot$"+"\n"+r"$M_{gas}=$"+f"{halo.gas['mass'].sum().value:.3e}"+r" $M_\odot$" ,ha="left", va="top", color="black", rotation="horizontal", fontsize=14)

    axes[1, i].set_xlabel(f"r [kpc]", fontsize=20)

    if i==0:
        axes[0, i].set_ylabel(r"$\rho \ [M_\odot / kpc^3]$", fontsize=20)
        if args.velocity_moment == "rms":
            axes[1, i].set_ylabel(r"$\sqrt{ \langle v^2 \rangle }$ [km/s]", fontsize=20)
        elif args.velocity_moment == "mean":
            axes[1, i].set_ylabel(r"$\langle v \rangle$ [km/s]", fontsize=20)
        elif args.velocity_moment == "dispersion":
            axes[1, i].set_ylabel(r"$\sigma$ [km/s]", fontsize=20)

    axes[0, i].loglog()
    axes[1, i].set_yscale("log")

    axes[0, i].set_xlim(rmin, rmax)
    axes[0, i].set_ylim(densmin, densmax)
    axes[1, i].set_ylim(pvmin, pvmax)

    if args.radial_anisotropy:
        ra_value = 5
        ra_vary = True
        print("radial_anisotropy")
    else:
        ra_value = 1E8
        ra_vary = False


    if args.set_g is not None:
        if args.set_g == "vary":
            gval = 1
            gvary = True

        else:
            gval = float(args.set_g)
            gvary = False

        fit_params_g = densModel.make_params(
            W0={'value': 5.5,'min': 0.01,'max': np.inf,'vary': True},
            g={'value': gval, 'min': 1E-4, 'max': 3.499, 'vary': gvary},
            M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
            ra={'value': ra_value, 'min': 0.1, 'max': 1E8, 'vary': ra_vary}
        )
   
        k_g, fit_g =  limepy_fitAndPlot(densModel, fit_params_g, result_dens, "volumetric")
        if fit_g.errorbars:
            if fit_g.params['W0'].stderr<fit_g.params["W0"].value:
                plot_g = True
                if gvary:
                    axes[0, i].plot(
                        k_g.r, 
                        k_g.rho, 
                        color="green", 
                        zorder=10, 
                        label=f"Fit to g={fit_g.params['g'].value:.1f}±{fit_g.params['g'].stderr:.1f}: W0={fit_g.params['W0'].value:.2f}±{fit_g.params['W0'].stderr:.0e},  rh={fit_g.params['rh'].value:.2f}±{fit_g.params['rh'].stderr:.0e} kpc" if fit_g.success else f"NOT CONVERGED"
                    )
                else:
                    axes[0, i].plot(
                        k_g.r, 
                        k_g.rho, 
                        color="green", 
                        zorder=10, 
                        label=f"Fit to {gval}: W0={fit_g.params['W0'].value:.2f}±{fit_g.params['W0'].stderr:.0e},  rh={fit_g.params['rh'].value:.2f}±{fit_g.params['rh'].stderr:.0e} kpc" if fit_g.success else f"NOT CONVERGED"
                    )

                axes[1, i].plot(
                    k_g.r, 
                    np.sqrt(k_g.v2), 
                    color="green", 
                    zorder=10
                )
            else:
                plot_g = False

        dump_fit(fit_g, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_g_volume_Rvir{round(halo_params['R/Rvir'], 1)}.json")


    if args.king_fit:
        fit_params_king = densModel.make_params(
            W0={'value': 5.5,'min': 0.01,'max': np.inf,'vary': True},
            g={'value': 1, 'vary': False},
            M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
            ra={'value': ra_value, 'min': 0.1, 'max': 1E8, 'vary': ra_vary}
        )
   
        k_king, fit_king =  limepy_fitAndPlot(densModel, fit_params_king, result_dens, "volumetric")
        if fit_king.errorbars:
            if fit_king.params['W0'].stderr<fit_king.params["W0"].value:
                plot_king = True
                axes[0, i].plot(
                    k_king.r, 
                    k_king.rho, 
                    color="darkblue", 
                    zorder=10, 
                    label=f"Fit to King: W0={fit_king.params['W0'].value:.2f}±{fit_king.params['W0'].stderr:.0e},  rh={fit_king.params['rh'].value:.2f}±{fit_king.params['rh'].stderr:.0e} kpc" if fit_king.success else f"King NOT CONVERGED"
                )
                axes[1, i].plot(
                    k_king.r, 
                    np.sqrt(k_king.v2), 
                    color="darkblue", 
                    zorder=10
                )
            else:
                plot_king = False

        dump_fit(fit_king, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_king_volume_Rvir{round(halo_params['R/Rvir'], 1)}.json")


    if args.plummer_fit:
        fit_params_plummer = densModel.make_params(
            W0={'value': 0.01,'min': 0.01,'max': np.inf,'vary': False},
            g={'value': 3.499, 'vary': False},
            M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
            rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
            ra={'value': 1E8, 'min': 0.1, 'max': 1E8, 'vary': False}
        )
   
        k_plummer, fit_plummer =  limepy_fitAndPlot(densModel, fit_params_plummer, result_dens, "volumetric")
        if fit_plummer.errorbars:
            if fit_plummer.params['W0'].stderr<fit_plummer.params["W0"].value:
                plot_plummer = True
                axes[0, i].plot(
                    k_plummer.r, 
                    k_plummer.rho, 
                    color="brown", 
                    zorder=10, 
                    label=f"Fit to Plummer: rh={fit_plummer.params['rh'].value:.2f}±{fit_plummer.params['rh'].stderr:.0e} kpc" 
                )
                axes[1, i].plot(
                    k_plummer.r, 
                    np.sqrt(k_plummer.v2), 
                    color="brown", 
                    zorder=10
                )
            else:
                plot_plummer = False

        dump_fit(fit_plummer, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_plummer_volume_Rvir{round(halo_params['R/Rvir'], 1)}.json")



    rho_h = unyt_quantity(*mass) / (4*np.pi/3 * unyt_quantity(*rvir).to("kpc")**3)
    c = (unyt_quantity(*rvir).to("kpc")/unyt_quantity(*rs).to("kpc")).value
    r = np.logspace(np.log10(0.05), np.log10(300), 200)
    rho_nfw = NFWc(r, rho_h, c, unyt_quantity(*rvir).to("kpc").value)
    

    fit_params_einasto = einastoModel.make_params(
        mu={'value': 2, 'min': 0.01, 'max': np.inf, 'vary': True},
        M={'value': halo.darkmatter["mass"].sum().to("Msun").value, 'vary' : False},
        rs={'value': unyt_quantity(*rs).to("kpc").value, 'min': 1E-4, 'max': 50, 'vary': True}
    )
   
    if fit_with_weights:
        w = 1 / result_dens["darkmatter"]["bound"]["e_rho"]
    else:
        w = np.ones_like(result_dens["darkmatter"]["bound"]["rho"])


    fit_einasto = einastoModel.fit(
            r=result_dens["darkmatter"]["bound"]["r"],
            data=result_dens["darkmatter"]["bound"]["rho"],
            params=fit_params_einasto, 
            weights=w,
            nan_policy="omit"
    )
    rho_einasto = fit_einasto.eval(r=r)



    axes[0, i].plot(r, rho_nfw, color="darkorange", label=f'NFW Rockstar Fit: c={c:.2f}, rvir={unyt_quantity(*rvir).to("kpc").value:.2f} kpc', zorder=9)
    axes[0, i].plot(r, rho_einasto, color="turquoise", label=f"Einasto Fit: rs={fit_einasto.params['rs'].value:.2f}±{fit_einasto.params['rs'].stderr:.2f}, " + r"$\mu=$" + f"{fit_einasto.params['mu'].value:.2f}±{fit_einasto.params['mu'].stderr:.2f} ", zorder=9)


    
    axes[0, i].legend(loc="upper right", fontsize=11, markerfirst=False, reverse=False)






    



    axes2[0, i].set_title(f"R/Rvir={halo_params['R/Rvir']:.2f}, z={redshift:.2f}")

    lines_of_sight = random_vector_spherical(N, half_sphere=False)

    result_surf = plot_surfdens(axes2[0, i], halo, lines_of_sight, proj_components)
    bins = {
        "bound": { c: result_surf[c]["bound"]["bins"] for c in proj_components},
        "all": { c: result_surf[c]["all"]["bins"] for c in proj_components}
    }
    plot_losvel(axes2[1, i], halo, lines_of_sight, proj_components, bins, velocity_projection)


    axes2[0, i].vlines(result_surf["darkmatter"]["bound"]["r"][-1], ymin=0, ymax=result_surf["darkmatter"]["bound"]["rho"][-1], zorder=-1, color="black", ls="-")
    axes2[0, i].axvspan(0.0001, 2 * soft, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes2[0, i].axvline(2 * soft, color="darkviolet", ls="--", lw=2.5)
    axes2[1, i].axvspan(0.0001, 2 * soft, color="darkviolet", alpha=0.25, ls="--", lw=0.01)
    axes2[1, i].axvline(2 * soft, color="darkviolet", ls="--", lw=2.5)


    halo.switch_to_bound()
    axes2[0, i].text(soft, 3*densmin, r"$\varepsilon=80$ pc" ,ha="left", va="bottom", color="darkviolet", rotation="vertical", fontsize=20)
    axes2[1, i].text(soft, 0.9*0.6*pvmax, r"$\varepsilon=80$ pc" ,ha="left", va="top", color="darkviolet", rotation="vertical", fontsize=20)
    axes2[1, i].text(2.1*soft, 0.9*0.6*pvmax, r"$M_*$="+f"{halo.stars['mass'].sum().value:.3e}"+r" $M_\odot$"+"\n"+r"$M_{dm}$="+f"{halo.darkmatter['mass'].sum().value:.3e}"+r" $M_\odot$"+r" $M_\odot$"+"\n"+r"$M_{gas}$="+f"{halo.gas['mass'].sum().value:.3e}"+r" $M_\odot$",ha="left", va="top", color="black", rotation="horizontal", fontsize=14)

    axes2[1, i].set_xlabel(f"R [kpc]", fontsize=20)

    if i==0:
        axes2[0, i].set_ylabel(r"$\Sigma \ [M_\odot / kpc^2]$", fontsize=20)
        if args.velocity_moment == "rms":
            axes2[1, i].set_ylabel(r"$\sqrt{\langle v_{los}^2 \rangle}$ [km/s]", fontsize=20)
        elif args.velocity_moment == "mean":
            axes2[1, i].set_ylabel(r"$\langle v_{los} \rangle$ [km/s]", fontsize=20)
        elif args.velocity_moment == "dispersion":
            axes2[1, i].set_ylabel(r"$\sigma_{los}$ [km/s]", fontsize=20)

    axes2[0, i].loglog()
    axes2[1, i].set_yscale("log")

    axes2[0, i].set_xlim(rmin, rmax)
    axes2[0, i].set_ylim(densmin, densmax)
    axes2[1, i].set_ylim(0.7*pvmin, 0.6*pvmax)


    

    if double_fit:
        if args.radial_anisotropy:
            ra_value = 5
            ra_vary = True
        else:
            ra_value = 1E8
            ra_vary = False


        if args.set_g is not None:
            if args.set_g == "vary":
                gval = 1
                gvary = True

            else:
                gval = float(args.set_g)
                gvary = False

            fit_params_g = surfModel.make_params(
                W0={'value': 5.5,'min': 0.01,'max': np.inf,'vary': True},
                g={'value': gval, 'min': 1E-4, 'max': 3.499, 'vary': gvary},
                M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
                rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
                ra={'value': ra_value, 'min': 0.1, 'max': 1E8, 'vary': ra_vary}
            )
       
            k_g, fit_g =  limepy_fitAndPlot(densModel, fit_params_g, result_surf, "surface")
            if fit_g.errorbars:
                if fit_g.params['W0'].stderr<fit_g.params["W0"].value:
                    if gvary:
                        axes2[0, i].plot(
                            k_g.r, 
                            k_g.Sigma, 
                            color="green", 
                            zorder=10, 
                            label=f"Fit to g={fit_g.params['g'].value:.1f}±{fit_g.params['g'].stderr:.1f}: W0={fit_g.params['W0'].value:.2f}±{fit_g.params['W0'].stderr:.0e},  rh={fit_g.params['rh'].value:.2f}±{fit_g.params['rh'].stderr:.0e} kpc" if fit_g.success else f"NOT CONVERGED"
                        )
                    else:
                        axes2[0, i].plot(
                            k_g.r, 
                            k_g.Sigma, 
                            color="green", 
                            zorder=10, 
                            label=f"Fit to {gval}: W0={fit_g.params['W0'].value:.2f}±{fit_g.params['W0'].stderr:.0e},  rh={fit_g.params['rh'].value:.2f}±{fit_g.params['rh'].stderr:.0e} kpc" if fit_g.success else f"NOT CONVERGED"
                        )

                    axes2[1, i].plot(
                        k_g.r, 
                        np.sqrt(k_g.v2p), 
                        color="green", 
                        zorder=10
                    )

            dump_fit(fit_g, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_g_surface_Rvir{round(halo_params['R/Rvir'], 1)}.json")


        if args.king_fit:
            fit_params_king = surfModel.make_params(
                W0={'value': 5.5,'min': 0.01,'max': np.inf,'vary': True},
                g={'value': 1, 'vary': False},
                M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
                rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
                ra={'value': ra_value, 'min': 0.1, 'max': 1E8, 'vary': ra_vary}
            )
       
            k_king, fit_king =  limepy_fitAndPlot(densModel, fit_params_king, result_surf, "surface")
            if fit_king.errorbars:
                if fit_king.params['W0'].stderr<fit_king.params["W0"].value:
                    axes2[0, i].plot(
                        k_king.r, 
                        k_king.Sigma, 
                        color="darkblue", 
                        zorder=10, 
                        label=f"Fit to King: W0={fit_king.params['W0'].value:.2f}±{fit_king.params['W0'].stderr:.0e},  rh={fit_king.params['rh'].value:.2f}±{fit_king.params['rh'].stderr:.0e} kpc" if fit_king.success else f"King NOT CONVERGED"
                    )
                    axes2[1, i].plot(
                        k_king.r, 
                        np.sqrt(k_king.v2p), 
                        color="darkblue", 
                        zorder=10
                    )

            dump_fit(fit_king, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_king_surface_Rvir{round(halo_params['R/Rvir'], 1)}.json")


        if args.plummer_fit:
            fit_params_plummer = surfModel.make_params(
                W0={'value': 0.01,'min': 0.01,'max': np.inf,'vary': False},
                g={'value': 3.499, 'vary': False},
                M={'value': halo.stars["mass"].sum().to("Msun").value, 'vary' : False},
                rh={'value': rh3d_stars[0], 'min': 1E-4, 'max': 50, 'vary': True},
                ra={'value': 1E8, 'min': 0.1, 'max': 1E8, 'vary': False}
            )
       
            k_plummer, fit_plummer =  limepy_fitAndPlot(densModel, fit_params_plummer, result_surf, "surface")
            if fit_plummer.errorbars:
                if fit_plummer.params['W0'].stderr<fit_plummer.params["W0"].value:
                    axes2[0, i].plot(
                        k_plummer.r, 
                        k_plummer.Sigma, 
                        color="brown", 
                        zorder=10, 
                        label=f"Fit to Plummer: rh={fit_plummer.params['rh'].value:.2f}±{fit_plummer.params['rh'].stderr:.0e} kpc" if fit_plummer.success else f"Plummer NOT CONVERGED"
                    )
                    axes2[1, i].plot(
                        k_plummer.r, 
                        np.sqrt(k_plummer.v2p), 
                        color="brown", 
                        zorder=10
                    )

            dump_fit(fit_plummer, args.output + f"subtree_{sub_tree}/fits/"+ f"fit_plummer_surface_Rvir{round(halo_params['R/Rvir'], 1)}.json")

        
    else:
        if args.plummer_fit:
            if plot_plummer:
                axes2[0, i].plot(
                 k_plummer.r,
                 k_plummer.Sigma,
                 color="brown", 
                 zorder=10,
                 label=f"Fit to Plummer: rhp={k_plummer.rhp:.2f} kpc"
                 )
                axes2[1, i].plot(
                 k_plummer.r,
                 np.sqrt(k_plummer.v2p),
                 color="brown", 
                 zorder=10
                )

        if args.set_g is not None:
            if plot_g:
                axes2[0, i].plot(
                    k_g.r, 
                    k_g.Sigma, 
                    color="green", 
                    zorder=10, 
                    label=f"Fit to g={fit_g.params['g'].value:.1f}: rhp={k_g.rhp:.2f}kpc"
                )
                axes2[1, i].plot(
                    k_g.r,
                    np.sqrt(k_g.v2p),
                    color="green", 
                    zorder=10
                )
        if args.king_fit:
            if plot_king:
                axes2[0, i].plot(
                    k_king.r, 
                    k_king.Sigma, 
                    color="darkblue", 
                    zorder=10, 
                    label=f"Fit to King: rhp={k_king.rhp:.2f} kpc"
                )
                axes2[1, i].plot(
                    k_king.r,
                    np.sqrt(k_king.v2p),
                    color="darkblue", 
                    zorder=10
                )
    
    axes2[0, i].legend(loc="upper right", fontsize=11, markerfirst=False, reverse=False)

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






























